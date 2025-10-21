import os
import sys
import json
import time
import shutil
import asyncio
import base64
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
import html
import logging

from typing import Any, Dict, List
from pathlib import Path
from PIL import Image

from contextlib import AsyncExitStack

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# =========================
# 0) 공통 설정
# =========================
ASSETS = Path("assets")
st.set_page_config(
    page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사",
    layout="wide",
    page_icon="📈",
)

# ✨ 제목/레이아웃 CSS: 한국어 줄바꿈, 반응형 크기, 잘림 방지
st.markdown("""
<style>
.block-container { padding-top: .8rem; padding-bottom: 1.2rem; max-width: 100%; }
.element-container, .stDataFrame { width: 100% !important; }

/* 타이틀: 한글 줄바꿈 + 반응형 + 잘림 방지 */
.app-title{
  font-weight: 800;
  line-height: 1.25;
  font-size: clamp(22px, 2.2vw + 14px, 40px);
  margin: .2rem 0 1.0rem 0;
  white-space: normal;
  word-break: keep-all;
  overflow-wrap: anywhere;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="app-title">신한카드 소상공인 비밀상담소 🔑</h1>', unsafe_allow_html=True)
# 로깅: 화면 대신 콘솔로만
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def user_warn(msg: str):
    st.warning(msg, icon="⚠️")

def user_err(msg: str):
    st.error(msg, icon="🛑")

# (현재 코드 유지—하드코드)
GOOGLE_API_KEY = "AIzaSyB8R3nurDOohfAvKXSgBUVRkoliXtfnTKo"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")
async_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------
# 에이전트(추론/리포팅) 로직
# ---------------------------
class InteractiveParallelAgent:
    """
    - 변환 → 문제정의 → 전략 → (통합)최종 보고서
    - 보고서는 통합 프롬프트(_GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT) 사용
    - 외부(run_full_pipeline)에서 xploration(시계열) 컨텍스트 주입
    """
        
    # 1) 변환
    _TRANSFORM_SYSTEM_PROMPT = """
당신은 사용자의 자연어 요청을 구조화된 데이터로 변환하는 전문가입니다.
사용자의 요청을 분석하여 다음 네 가지 JSON 항목으로 요약하세요.
- "target": 사용자가 전략을 요청한 대상은 무엇인가요? (가맹점 이름. 예: '성우**', '행*') ('매장', '가맹점'과 같은 단어를 포함하지 마세요)
- "challenge": 사용자가 직면한 어려움이나 해결해야 할 문제는 무엇인가요?
- "objective": 목표 달성을 위한 구체적인 목적은 무엇인가요?
- "solution_direction": 제안될 수 있는 해결책의 방향은 무엇인가요?
"""

    # 2) 문제 정의
    _DEFINE_PROBLEM_SYSTEM_PROMPT = """
당신은 데이터 분석가입니다. 제공된 가맹점의 압축된 통계 요약과 최초 사용자 요청을 기반으로, 문제 상황과 핵심 성과 지표(KPI)를 구체적으로 정의하세요.

**데이터 구조 설명:**
제공된 데이터는 시계열 통계 요약으로, 다음을 포함합니다:
1. 기본 정보 (meta): 가맹점ID, 업종, 상권, 분석기간, cluster 정보
2. 전체 평균 지표 (avg_metrics): 24개월 평균값
3. 트렌드 (trend): 초기 6개월 vs 최근 6개월 비교
4. 최근 3개월 상세 (recent_3months): 월별 상세 데이터
5. 주요 이벤트 (significant_events): 급격한 변화 시점
6. 구간 빈도 (range_frequency): 가장 자주 나타나는 성과 구간

**데이터 해석 가이드:**
- 매출금액/건수/고객수: 실제 값
- 취소율: 실제 비율
- 동일 업종 매출금액/건수 비율: 100% 기준, 100% 이상이면 평균 이상
- 동일 업종/상권 내 매출 순위 비율: 0%에 가까울수록 상위권
- cluster: 0(상위권 안정), 1(하위권 안정), 2(변동성 높음), 3(신규 가맹점)
- cluster_6: 0(여성40대+거주), 1(남성30-40대+신규), 2(2030+신규+유동), 3(60대이상), 4(남성40-60대), 5(직장+남성30대+재방문)

**특히 주목할 점:**
- 트렌드의 변화율(초기 대비 최근)
- 최근 3개월의 월별 추세
- 주요 이벤트(급등/급락) 시점과 크기

결과는 다음 JSON 형식이어야 합니다.
- "problem_statement": 현재 직면한 구체적인 이슈는 무엇인가요? (트렌드와 최근 데이터 기반)
- "kpis": 문제 해결을 위한 측정 가능한 핵심 지표는 무엇인가요? (문자열 목록)
"""

    # 4) 전략 제안
    _GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT = """
당신은 뛰어난 비즈니스 컨설턴트이자 전문 보고서 작성자입니다.
주어진 '문제 정의'와 압축된 데이터 분석 결과를 종합하여, 최종 사용자를 위한 포괄적인 보고서를 작성하세요.

보고서는 다음의 명확한 구조를 반드시 포함해야 합니다.

서론: 문제 배경 및 분석 목적을 간결하게 설명합니다.

본론:

데이터 기반 현황 분석: 트렌드와 최근 3개월 데이터를 중심으로 현재 상황을 분석합니다.

핵심 문제 정의: 분석을 통해 도출된 가장 시급하고 중요한 문제를 정의합니다.

구체적인 해결 전략: 정의된 문제를 해결하기 위한 구체적이고 실행 가능한 전략과 세부 실행 방안을 제시합니다. 또한, 왜 이 전략이 효과적일지에 대한 논리적 근거를 데이터 추세와 연결하여 설명합니다.

결론: 제안된 전략 실행 시 기대되는 효과를 요약합니다.

최종 결과는 반드시 "report"라는 단일 키를 가진 JSON 형식이어야 하며, 값에는 전체 보고서 내용이 마크다운 형식의 문자열로 포함되어야 합니다.
예시: {"report": "## 최종 보고서\n\n### 1. 개요\n...\n### 2. 현황 분석 및 문제 정의\n최근 3개월간 재방문 고객 비중이 지속적으로 감소하는 추세를 보였습니다...\n### 3. 해결 전략\n...\n"}
"""

    def __init__(self, df: pd.DataFrame, model):
        self.df = df
        self.model = model
        self.context = {} # 분석 과정을 저장할 컨텍스트

    def _generate_content_sync(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        LLM 호출을 담당하는 중앙 메서드. max_output_tokens 등 다양한 옵션 지원.
        """
        try:
            # ✅ 요구사항 반영: generation_config를 동적으로 설정
            generation_config = {"response_mime_type": "application/json"}
            if 'max_output_tokens' in kwargs:
                generation_config['max_output_tokens'] = kwargs['max_output_tokens']

            resp = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return json.loads(resp.text)
        except Exception as e:
            return {"error": f"LLM 동기 호출 실패: {e.__class__.__name__}: {e}"}

    def transform(self, initial_input: str):
        user = f"🔹 사용자 입력: {initial_input}"
        full = f"{self._TRANSFORM_SYSTEM_PROMPT}\n\n{user}"
        self.context["transformation"] = self._generate_content_sync(full)
        return self.context["transformation"]
    
    def set_target_store(self, merchant_id: str):
        """분석할 대상 가맹점의 ID를 받아 데이터를 필터링하고 컨텍스트에 저장합니다."""
        store_data = self.df[self.df['가맹점구분번호'].astype(str) == str(merchant_id)]
        if store_data.empty:
            self.context['error'] = "해당 ID의 가맹점 데이터를 찾을 수 없습니다."
            return
        # 분석에 사용될 가맹점의 시계열 데이터를 컨텍스트에 저장합니다.
        self.context['target_store_timeseries'] = store_data.sort_values(by='기준년월').reset_index(drop=True)

    def compress_store_data(self) -> Dict[str, Any]:
        """
        ✅ 요구사항 반영: 3단계: 가맹점 데이터 압축 요약 (사용자 제공 로직 통합)
        """
        print("3단계: 데이터 압축 요약 시작...")
        store_data = self.context.get('target_store_timeseries')
        if store_data is None or store_data.empty:
            self.context['error'] = "압축할 가맹점 데이터가 없습니다."
            return {"error": self.context['error']}

        # --- (사용자가 제공한 압축 로직 시작) ---
        first_row = store_data.iloc[0]
        meta = {
            '가맹점ID': str(first_row.get('가맹점구분번호')), '가맹점명': first_row.get('가맹점명'),
            '업종': first_row.get('업종'), '상권': first_row.get('상권'),
            '분석기간': f"{store_data['기준년월'].min().strftime('%Y-%m')} ~ {store_data['기준년월'].max().strftime('%Y-%m')}",
            '데이터포인트': len(store_data),
            'cluster': int(first_row['cluster']) if pd.notna(first_row.get('cluster')) else None,
            'cluster_6': int(first_row['cluster_6']) if pd.notna(first_row.get('cluster_6')) else None
        }
        numeric_cols = [
            '동일 업종 매출금액 비율', '동일 업종 매출건수 비율','동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율',
            '재방문 고객 비중', '신규 고객 비중', '거주 이용 고객 비율', '직장 이용 고객 비율', '유동인구 이용 고객 비율'
        ] # 필요한 컬럼만 선택
        
        store_data_clean = store_data.copy()
        for col in numeric_cols:
            if col in store_data_clean.columns:
                store_data_clean[col] = pd.to_numeric(store_data_clean[col], errors='coerce')
        
        avg_metrics = {col: round(store_data_clean[col].mean(), 2) for col in numeric_cols if col in store_data_clean.columns and store_data_clean[col].notna().any()}
        
        first_6 = store_data_clean.head(6)
        last_6 = store_data_clean.tail(6)
        trend = {}
        for col in numeric_cols:
            if col in store_data_clean.columns:
                first_val, last_val = first_6[col].mean(), last_6[col].mean()
                if pd.notna(first_val) and pd.notna(last_val) and first_val != 0:
                    change_pct = ((last_val - first_val) / abs(first_val)) * 100
                    trend[col] = {'초기_평균': round(first_val, 2), '최근_평균': round(last_val, 2), '변화율': f"{change_pct:+.1f}%"}
        # --- (사용자가 제공한 압축 로직 종료, 일부 간소화) ---

        exploration_result = {'meta': meta, 'avg_metrics': avg_metrics, 'trend': trend}
        self.context['exploration'] = exploration_result
        print("3단계: 데이터 압축 완료.")
        return exploration_result
    
    def define_problem(self) -> Dict[str, Any]:
        """
        ✅ 요구사항 반영: 4단계: 문제 정의 (압축 데이터와 max_tokens 사용)
        """
        print("4단계: 문제 정의 시작...")
        exploration = self.context.get('exploration')
        transformation = self.context.get('transformation')
        if not exploration or not transformation:
            return {"error": "분석 컨텍스트가 준비되지 않았습니다."}
            
        user_prompt = f"""
🔹 문맥(Context):
- 최초 요청: {json.dumps(transformation, indent=2, ensure_ascii=False)}
- 압축된 데이터 분석 결과: {json.dumps(exploration, indent=2, ensure_ascii=False)}
"""
        full_prompt = f"{self._DEFINE_PROBLEM_SYSTEM_PROMPT}\n\n{user_prompt}"
        
        response = self._generate_content_sync(
            full_prompt,
            max_output_tokens=4096  # ✅ max_tokens 적용
        )
        self.context['problem_definition'] = response
        print("4단계: 문제 정의 완료.")
        return response

    def generate_report(self) -> str:
        """5단계: 최종 보고서 생성"""
        print("5단계: 최종 보고서 생성 시작...")
        compact_context = {
            "transformation": self.context.get("transformation"),
            "problem_definition": self.context.get("problem_definition"),
            "exploration_summary": self.context.get("exploration"),
        }
        user_prompt = "🔹 전체 컨텍스트:\n" + json.dumps(compact_context, ensure_ascii=False, indent=2)
        full_prompt = f"{self._GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = self._generate_content_sync(full_prompt,
        max_output_tokens=8192, # (참고) Flash 모델의 최대 출력은 8192 토큰입니다.
        temperature=0.3
    )
        
        if "report" in response:
            print("5단계: 최종 보고서 생성 완료.")
            return response["report"]
        else:
            print("5단계: 최종 보고서 생성 실패 (Fallback).")
            # Fallback: 실패 시에도 컨텍스트를 보여줌
            return f"# 보고서 생성 실패\n\nAI가 보고서를 생성하지 못했습니다. 아래는 분석에 사용된 데이터 요약입니다.\n\n```json\n{json.dumps(compact_context, ensure_ascii=False, indent=2)}\n```"

# ---------------------------
# Streamlit UI
# ---------------------------
system_prompt = (
    "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, "
    "분석 결과를 바탕으로 적절한 마케팅 방법과 채널, 마케팅 메시지를 추천합니다. "
    "결과는 짧고 간결하게, 분석 결과에는 가능한 표를 사용하여 알아보기 쉽게 설명해주세요."
)
greeting = "마케팅이 필요한 가맹점을 알려주세요."

# ===== 세션 상태 안전 초기화 =====
def ensure_state():
    ss = st.session_state
    if "messages" not in ss:
        ss.messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=greeting),
        ]

    ss.setdefault("awaiting_candidate", False)
    ss.setdefault("candidates", [])
    ss.setdefault("exploration", None)
    ss.setdefault("transformation", None)
    ss.setdefault("selected_merchant_id", None)
    ss.setdefault("user_query", "")

    # ✅ 요구사항 반영: 차트 관련 상태(chart_b64, used_kpis) 제거
    if "pipeline" not in ss:
        ss.pipeline = {
            "running": False,
            "cancel_requested": False,
            "step": "idle",
            "merchant_id": None,
            "user_query": "",
            "trans": None,
            "candidates": [],
            "exploration": None,
            "pd_out": None,
            "report_md": None,
        }

ensure_state()

@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)

def clear_chat_history():
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting),
    ]
    st.session_state.awaiting_candidate = False
    st.session_state.candidates = []
    st.session_state.exploration = None
    st.session_state.transformation = None
    st.session_state.selected_merchant_id = None
    st.session_state.user_query = ""
    # ✅ 요구사항 반영: 파이프라인 초기화
    st.session_state.pipeline = {
        "running": False, "cancel_requested": False, "step": "idle",
        "merchant_id": None, "user_query": "", "trans": None,
        "candidates": [], "exploration": None, "pd_out": None,
        "report_md": None,
    }

def handle_followup(user_question: str) -> str:
    """
    현재 세션의 exploration / transformation / report를 컨텍스트로 사용해
    사용자의 추가 질문에 답합니다. 새 파이프라인을 돌리지 않습니다.
    """
    exp = st.session_state.pipeline.get("exploration")
    trans = st.session_state.pipeline.get("trans")
    report_md = st.session_state.pipeline.get("report_md")

    ctx = {
        "store_identity": (exp or {}).get("store_identity"),
        "time_series_tail": (exp or {}).get("time_series_analysis_data", [])[-12:] if exp else [],
        "transformation": trans,
        "report": report_md,
    }
    prompt = (
        "당신은 분석 컨설턴트입니다. 아래 컨텍스트를 바탕으로 한국어로 간결하게 답하세요.\n"
        "숫자는 표/불릿으로 명확히, 모르는 정보는 추측하지 말고 부족하다고 밝혀주세요.\n\n"
        f"### 컨텍스트\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        f"### 사용자의 질문\n{user_question}\n\n"
        "### 답변"
    )
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"답변 중 오류가 발생했습니다: {e}"

# 후보 표시 포맷
def _id_of(cand: dict):
    return cand.get("가맹점구분번호")

def _fmt_cand_with_id(c: dict) -> str:
    id_val = _id_of(c) or "UNKNOWN_ID"
    area = c.get("상권", "상권 정보 없음")
    name = c.get("가맹점명", "?")
    industry = c.get("업종", "?")
    return f"[{id_val}] {name} / {industry} / {area}"

def _set_step(step: str):
    st.session_state.pipeline["step"] = step

def _cancel_requested() -> bool:
    return bool(st.session_state.pipeline.get("cancel_requested"))

def _abort_if_cancelled():
    if _cancel_requested():
        st.info("⏹️ 전략 제안을 중지했습니다.")
        st.session_state.pipeline["running"] = False
        _set_step("idle")
        raise st.stop()

# === 세션 상태 초기화 (기존 블록에 추가) ===
# if "pipeline" not in st.session_state:
#     st.session_state.pipeline = {
#         "running": False,            # 파이프라인 동작 여부
#         "cancel_requested": False,   # 중지 요청 플래그
#         "step": "idle",              # 현재 단계
#         "merchant_id": None,
#         "user_query": "",
#         # 산출물 캐시(재실행 시 이어가기 위함)
#         "trans": None,
#         "candidates": [],
#         "exploration": None,
#         "pd_out": None,
#         "chart_b64": None,
#         "used_kpis": None,
#         "report_md": None,
#     }

# =========================
# 5) 사이드바
# =========================
with st.sidebar:
    logo = ASSETS / "shc_ci_basic_00.png"
    if logo.exists():
        st.image(load_image("shc_ci_basic_00.png"), use_container_width=True)
    st.markdown("<p style='text-align: center;'>2025 Big Contest • AI DATA 활용</p>", unsafe_allow_html=True)
    st.button("Clear Chat History", on_click=clear_chat_history)

    # 🔴 진행 중 중지 버튼 (재실행되어도 'running' 유지)
    col_stop = st.container()
    if st.session_state.pipeline["running"]:
        if col_stop.button("분석 중단", type="secondary"):
            st.session_state.pipeline["cancel_requested"] = True
        
# --- 히스토리 렌더 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# --- 시계열 렌더 ---
@st.cache_data(show_spinner=False)
def _prepare_ts(ts_list: list) -> pd.DataFrame:
    df = pd.DataFrame(ts_list)
    if "기준년월" in df.columns:
        try:
            df["기준년월"] = pd.to_datetime(df["기준년월"])
        except Exception:
            pass
        df = df.sort_values("기준년월")
    return df.reset_index(drop=True)

def render_timeseries(exploration: dict, merchant_id: str | None = None):
    ts = exploration.get("time_series_analysis_data") or []
    if not ts:
        st.info("시계열 데이터가 없습니다.")
        return

    df = _prepare_ts(ts)
    st.subheader("📊 최근 12개월 데이터 요약")
    st.dataframe(df.tail(12), use_container_width=True, key=f"ts_tail_{merchant_id or 'na'}")

# --- 전체 파이프라인 ---
# ✅ 데이터 로딩 및 Agent 초기화 (앱 실행 시 한 번만)
@st.cache_resource
def load_data_and_init_agent():
    # 데이터 경로는 실제 환경에 맞게 조정하세요.
    try:
        # 이 부분은 mcp_server의 _load_df 로직과 유사하게 데이터를 로드합니다.
        file_path = "./data/labeling_no_preprocessing.csv"
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df["기준년월"] = pd.to_datetime(df["기준년월"].astype(str).str.zfill(6), format="%Y%m")
        agent = InteractiveParallelAgent(df, model)
        return agent, df
    except FileNotFoundError:
        st.error(f"데이터 파일({file_path})을 찾을 수 없습니다. 앱을 종료합니다.")
        st.stop()

agent, df = load_data_and_init_agent()

# ✅ mcp_server를 호출하는 대신, Agent가 직접 검색하도록 변경
def search_merchant_local(store_name: str) -> Dict[str, Any]:
    search_term = (store_name or "").replace("*", "")
    hits = df[df["가맹점명"].astype(str).str.contains(search_term, case=False, na=False)]
    if hits.empty:
        return {"status": "error", "message": f"'{search_term}'로 검색된 가맹점이 없습니다."}
    
    cand_cols = ["가맹점구분번호", "가맹점명", "업종", "상권", "가맹점주소"]
    cands = hits[cand_cols].drop_duplicates().to_dict(orient="records")

    if len(cands) == 1:
        return {"status": "single_candidate", "candidate": cands[0]}
    return {"status": "clarification_needed", "candidates": cands}


def run_full_pipeline_resumable(user_text: str | None = None, merchant_id: str | None = None):
    P = st.session_state.pipeline
    
    if not P["running"]: # 최초 실행
        if user_text is None: return
        P.update({"running": True, "cancel_requested": False, "step": "transform", "user_query": user_text})

    if P["step"] == "transform":
        trans = agent.transform(P["user_query"])
        # ✅ 내용 검증: 'target' 키가 있는지 확인
        if "error" in trans or not trans.get("target"):
            user_err("요청을 분석하는 데 실패했습니다. 더 명확한 가맹점 이름을 포함하여 다시 시도해주세요.")
            P["running"] = False; _set_step("idle"); return
        P["trans"] = trans
        _set_step("search")

    if P["step"] == "search":
        target = P["trans"]["target"]
        resp = search_merchant_local(target) # 로컬 검색 함수 호출
        
        if resp["status"] == "error":
            user_err(resp["message"])
            P["running"] = False; _set_step("idle"); return

        if resp["status"] == "clarification_needed":
            st.session_state.candidates = resp["candidates"]
            st.session_state.awaiting_candidate = True
            return # 사용자 선택 대기

        if resp["status"] == "single_candidate":
            P["merchant_id"] = resp["candidate"]["가맹점구분번호"]
            _set_step("analysis")

    if P["step"] == "analysis":
        with st.spinner("⏳ AI가 데이터를 분석하고 있습니다..."):
            agent.set_target_store(P["merchant_id"])
            exploration = agent.compress_store_data()
            if "error" in exploration:
                user_err(f"데이터 요약 중 오류: {exploration['error']}")
                P["running"] = False; _set_step("idle"); return

            pd_out = agent.define_problem()
            # ✅ 내용 검증: 'problem_statement' 키가 있는지 확인
            if "error" in pd_out or not pd_out.get("problem_statement"):
                user_err("AI가 문제점을 정의하는 데 실패했습니다. 잠시 후 다시 시도해주세요.")
                P["running"] = False; _set_step("idle"); return

            report_md = agent.generate_report()
            # ✅ 내용 검증: 결과가 비어있지 않은지 확인
            if not report_md or "보고서 생성 실패" in report_md:
                user_err("최종 보고서를 생성하는 데 실패했습니다. 컨텍스트를 확인하세요.")
                # 실패 시에도 생성된 report_md(Fallback 메시지)는 그대로 표시
            
            P["report_md"] = report_md
        _set_step("done")

    if P["step"] == "done":
        st.subheader("📘 최종 보고서 (상세)")
        st.markdown(P["report_md"], unsafe_allow_html=True)
        # 요약 메시지는 생략하고 최종 보고서만 바로 표시
        P["running"] = False
        _set_step("idle")

# --- 입력창 ---
# --- 입력 및 제어 로직 ---
user_query = st.chat_input("가맹점 이름이나 고민을 입력하세요.")
if user_query:
    st.session_state.user_query = user_query
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.write(user_query)

    if (not st.session_state.pipeline["running"]) and st.session_state.pipeline.get("report_md"):
        with st.spinner("답변을 생성 중입니다..."):
            followup_response = handle_followup(user_query)
        st.session_state.messages.append(AIMessage(content=followup_response))
        with st.chat_message("assistant"):
            st.markdown(followup_response)
    elif not st.session_state.pipeline["running"]:
        run_full_pipeline_resumable(user_query, merchant_id=None)

if st.session_state.awaiting_candidate and st.session_state.candidates:
    st.info("이름이 유사한 가맹점이 여러 개입니다. 아래에서 하나를 선택하세요.")
    cands = st.session_state.candidates
    idx = st.radio("가맹점을 선택하세요",
                   options=list(range(len(cands))),
                   format_func=lambda i: _fmt_cand_with_id(cands[i]),
                   key="cand_radio",
                   label_visibility="collapsed")

    if st.button("선택 확정", key="cand_confirm", type="primary"):
        sel = cands[idx]
        picked_id = str(sel.get("가맹점구분번호"))
        st.session_state.awaiting_candidate = False
        st.session_state.pipeline["merchant_id"] = picked_id
        _set_step("analysis")
        st.rerun()

    st.stop()

if st.session_state.pipeline["running"] and st.session_state.pipeline.get("cancel_requested"):
    st.session_state.pipeline.update({"running": False, "step": "idle", "cancel_requested": False})
    st.info("⏹️ 분석을 중지했습니다.")
    st.rerun()

if st.session_state.pipeline["running"] and not st.session_state.awaiting_candidate:
    run_full_pipeline_resumable()