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
import re, difflib
import matplotlib.pyplot as plt
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

def _llm_json_sync(gen_model, prompt: str) -> Dict[str, Any]:
    try:
        time.sleep(0.6)  # 과호출 방지용 짧은 지연
        resp = gen_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(resp.text)
    except Exception as e:
        return {"error": f"LLM 동기 호출 실패: {e.__class__.__name__}: {e}"}

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
- "target": 사용자가 전략을 요청한 가맹점명은 무엇인가요? (가맹점 이름. 예: '성우**')
- "challenge": 사용자가 직면한 어려움이나 해결해야 할 핵심 문제는 무엇인가요?
- "objective": 목표 달성을 위한 구체적인 목표은 무엇인가요?
- "solution_direction": 제안될 수 있는 해결책의 방향은 무엇인가요?
"""

    # 2) 문제 정의
    _DEFINE_PROBLEM_SYSTEM_PROMPT = """
당신은 데이터 분석가입니다. 다음의 지침에 따라 '문제 정의'를 수행하세요. :
```
1. 제공된 가맹점의 시계열 데이터와 최초 사용자 요청을 기반으로, 문제 상황과 핵심 성과 지표(KPI)를 구체적으로 정의하세요.
특히, 시간의 흐름에 따른 데이터의 변화와 추세(예: 재방문율 하락, 특정 고객층 비중 감소 등)를 명확하게 파악하고, 이를 문제 정의에 반영해야 합니다.
2. 최근의 변화에 더 큰 가중치를 두어 문제를 정의하세요.
3. 제공된 데이터는 상대적이고 비율을 표현하는 수치임을 유의하세요.
```

데이터에 대한 설명은 아래와 같습니다. :
```
- 가맹점 운영개월수 구간,매출금액 구간,매출건수 구간,유니크 고객 수 구간,객단가 구간 : 6개 구간으로 (10%이하, 10-25%, 25-50%, 50-75%, 75-90%, 90%초과) 0%에 가까울 수록 상위.
- 취소율 구간 : 1구간에 가까울 수록 상위. (취소율 낮음)
- 동일 업종 매출금액 비율 : 동일 업종 매출 금액 평균 대비 해당 가맹점 매출 금액 비율 (평균과 동일 : 100%)
- 동일 업종 매출건수 비율 : 동일 업종 매출 건수 평균 대비 해당 가맹점 매출 건수 비율 (평균과 동일 : 100%)
- 동일 업종 내 매출 순위 비율 : 업종 내 순위 / 업종 내 전체 가맹점 * 100 (0에 가까울수록 상위)
- 동일 상권 내 매출 순위 비율 : 상권 내 순위 / 상권 내 전체 가맹점 * 100 (0에 가까울수록 상위)
```

결과는 다음 JSON 형식이어야 합니다.
```
- "problem_statement": 시계열 변화에 기반한 문제 정의
- "kpis": 측정 가능한 핵심 지표 목록 (문자열 목록으로 제공)
```
"""

    # 4) 전략 제안
    _GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT = """
당신은 비즈니스 전략 컨설턴트입니다.
주어진 '문제 정의'와 모든 컨텍스트를 종합하여, 마케팅 전략 제안 보고서를 작성하세요.

보고서는 다음의 명확한 구조를 반드시 포함해야 합니다. :
```
1. 서론: 문제 배경 및 분석 목적
2. 본론:
   - 데이터 기반 현황 분석: 데이터의 시계열적 추세(예: 최근 3개월간 재방문 고객 비중 감소)를 명확히 언급하며 현재 상황을 분석
   - 문제 정의: 분석을 통해 도출된 가장 시급하고 중요한 핵심 문제 정의
   - 해결 전략 제안: 정의된 문제를 해결하기 위한 구체적이고 실행 가능한 전략 및 세부 실행 방안을 제시
   - 근거 제시: 제안한 전략이 효과적인 이유를 데이터 추세와 연결하여 논리적으로 설명
3. 결론: 제안된 전략 실행 시 기대되는 효과 제시
```

**최종 보고서 출력 형식**
    - 반드시 "report"라는 단일 키를 가진 JSON 형식이어야 함
    - 값에는 전체 보고서 내용이 마크다운 형식의 문자열로 포함되어야 함
    - 예시: {"report": "## 최종 보고서\\n\\n### 1. 개요\\n...\\n### 2. 현황 분석 및 문제 정의\\n최근 3개월간 재방문 고객 비중이 지속적으로 감소하는 추세를 보였습니다...\\n### 3. 해결 전략\\n...\\n"}
"""

    def __init__(self, sync_model):
        self.model = sync_model
        self.context: Dict[str, Any] = {}

    def transform(self, initial_input: str):
        user = f"🔹 사용자 입력: {initial_input}"
        full = f"{self._TRANSFORM_SYSTEM_PROMPT}\n\n{user}"
        self.context["transformation"] = _llm_json_sync(self.model, full)
        return self.context["transformation"]

    def define_problem(self, exploration: Dict[str, Any], transformation: Dict[str, Any]):
        ts_tail = (exploration.get("time_series_analysis_data") or [])[-12:]
        slim_ctx = {
            "최초요청": transformation,
            "시계열_tail": ts_tail,
            "store_identity": exploration.get("store_identity")
        }
        user = "🔹 문맥:\n" + json.dumps(slim_ctx, ensure_ascii=False, indent=2)
        full = f"{self._DEFINE_PROBLEM_SYSTEM_PROMPT}\n\n{user}"
        self.context["problem_definition"] = _llm_json_sync(self.model, full)
        return self.context["problem_definition"]


    def _compact_context(self, max_rows: int = 12) -> Dict[str, Any]:
        exp = self.context.get("exploration") or {}
        ts = (exp.get("time_series_analysis_data") or [])[-max_rows:]
        return {
            "transformation": self.context.get("transformation"),
            "problem_definition": self.context.get("problem_definition"),
            "exploration_summary": {
                "store_identity": exp.get("store_identity"),
                "time_series_tail": ts,
                "tail_note": f"최근 {min(max_rows, len(ts))}개월만 포함",
            },
        }

    def generate_strategy_and_report(self) -> str:
        compact = self._compact_context(max_rows=12)
        user = "🔹 전체 컨텍스트:\n" + json.dumps(compact, ensure_ascii=False, indent=2)
        out = _llm_json_sync(self.model, f"{self._GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT}\n\n{user}")

        if isinstance(out, dict) and out.get("report"):
            return out["report"]

        raw = out.get("raw") if isinstance(out, dict) else None
        if raw:
            return raw if raw.strip().startswith("#") else f"# 최종 보고서(자동 복구)\n\n{raw}"

        return "보고서 생성에 실패했습니다."

class _McpSingleton:
    def __init__(self):
        self.inited = False
        self.stack: AsyncExitStack | None = None
        self.session = None
        self.tool_map = None

    async def _ainit(self):
        if self.inited:
            return

        server_path = (Path(__file__).parent / "mcp_server_v3.py").resolve()
        uv_path = shutil.which("uv")
        command = uv_path if uv_path else sys.executable
        cmd_args = ["run", str(server_path)] if uv_path else ["-u", str(server_path)]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        server_params = StdioServerParameters(command=command, args=cmd_args, env=env)

        # ⚠️ 핵심: async context manager를 AsyncExitStack으로 진입
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()

        # stdio_client는 async generator context manager입니다.
        read, write = await self.stack.enter_async_context(stdio_client(server_params))
        # ClientSession도 async context manager입니다.
        self.session = await self.stack.enter_async_context(ClientSession(read, write))

        await self.session.initialize()
        tools = await load_mcp_tools(self.session)
        self.tool_map = {t.name: t for t in tools}
        self.inited = True

    async def aclose(self):
        if self.stack is not None:
            await self.stack.aclose()
        self.inited = False
        self.stack = None
        self.session = None
        self.tool_map = None

    async def ainvoke(self, tool_name: str, args: dict):
        if not self.inited:
            await self._ainit()
        if tool_name not in self.tool_map:
            return {"error": f"툴 '{tool_name}' 없음", "available": list(self.tool_map.keys())}
        return await self.tool_map[tool_name].ainvoke(args)

_mcp_singleton = _McpSingleton()

# ---------------------------
# MCP 호출 유틸
# ---------------------------
@st.cache_resource(show_spinner=False)
def _get_mcp_client():
    # Streamlit은 동기 함수이므로 여기서 event loop를 보장해 줍니다.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_mcp_singleton._ainit())
    return _mcp_singleton

def _normalize_mcp_result(result):
    """
    MCP 툴 반환을 안전하게 dict로 표준화.
    - JSON string이면 파싱
    - list면 {"items": [...]}
    - None/기타는 {"raw": ...}로 래핑
    """
    if result is None:
        return {"status": "error", "message": "empty result"}
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            obj = json.loads(result)
            if isinstance(obj, dict):
                return obj
            else:
                return {"raw": obj}
        except Exception:
            return {"raw": result}
    if isinstance(result, list):
        return {"items": result}
    # 기타 타입
    return {"raw": result}

def mcp_call(tool_name: str, args: dict) -> dict:
    client = _get_mcp_client()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        res = loop.run_until_complete(client.ainvoke(tool_name, args))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(client.ainvoke(tool_name, args))
    except Exception as e:
        # 네트워크/프로세스 예외도 dict로 통일
        return {"status": "error", "message": f"MCP invoke exception: {e.__class__.__name__}: {e}"}

    return _normalize_mcp_result(res)

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
    # 대화/에이전트
    if "messages" not in ss:
        ss.messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=greeting),
        ]
    if "agent" not in ss:
        ss.agent = InteractiveParallelAgent(sync_model=model)

    # 조회/선택/산출물
    ss.setdefault("awaiting_candidate", False)
    ss.setdefault("candidates", [])
    ss.setdefault("exploration", None)
    ss.setdefault("transformation", None)
    ss.setdefault("selected_merchant_id", None)
    ss.setdefault("last_chart_b64", None)
    ss.setdefault("user_query", "")

    # 파이프라인(이미 만드셨지만, 혹시 없을 때를 대비)
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
            "chart_b64": None,
            "used_kpis": None,
            "report_md": None,
        }

# ✅ 반드시 사이드바/화면 렌더 전에 호출
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
    st.session_state.last_chart_b64 = None
    st.session_state.user_query = ""

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
        "time_series_tail": (exp or {}).get("time_series_analysis_data")[-12:] if exp else [],
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
    area = c.get("상권", None)
    area_txt = "(상권 없음)" if (area is None) else str(area)
    name = c.get("가맹점명", "?")
    industry = c.get("업종", "?")
    return f"[{id_val}] {name} / {industry} / {area_txt}"

# === 세션 상태 초기화 (기존 블록에 추가) ===
if "pipeline" not in st.session_state:
    st.session_state.pipeline = {
        "running": False,            # 파이프라인 동작 여부
        "cancel_requested": False,   # 중지 요청 플래그
        "step": "idle",              # 현재 단계
        "merchant_id": None,
        "user_query": "",
        # 산출물 캐시(재실행 시 이어가기 위함)
        "trans": None,
        "candidates": [],
        "exploration": None,
        "pd_out": None,
        "chart_b64": None,
        "used_kpis": None,
        "report_md": None,
    }

def _set_step(step: str):
    st.session_state.pipeline["step"] = step

def _cancel_requested() -> bool:
    return bool(st.session_state.pipeline.get("cancel_requested"))

def _abort_if_cancelled():
    if _cancel_requested():
        st.info("⏹️ 전략 제안을 중지했습니다.")
        st.session_state.pipeline["running"] = False
        _set_step("idle")
        raise st.stop()  # 현재 렌더 중단 (다음 재실행 때는 idle 상태)

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
        if col_stop.button("전략 제안 종료", type="secondary"):
            st.session_state.pipeline["cancel_requested"] = True
        
# --- 히스토리 렌더 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# --- KPI 이름 정규화/매핑 유틸 ---
def _normalize_kpi_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)  # 괄호 설명 제거
    synonyms = {
        "동일 업종 매출금액 비율": "동일 업종 내 매출 금액 비율",
        "동일 업종 매출 금액 비율": "동일 업종 내 매출 금액 비율",
        "동일 업종 매출건수 비율": "동일 업종 내 매출 건수 비율",
        "동일 업종 매출 건수 비율": "동일 업종 내 매출 건수 비율",
        "동일 상권 매출 순위 비율": "동일 상권 내 매출 순위 비율",
        "취소율": "취소율 구간",
    }
    s = synonyms.get(s, s)
    s = re.sub(r"\s+", " ", s)
    return s

def _map_kpis_to_available(requested: list[str], available: list[str]) -> list[str]:
    if not requested:
        return []
    req_norm = [_normalize_kpi_name(k) for k in requested if k]
    avail_norm = [re.sub(r"\s+", " ", str(a).strip()) for a in available]

    chosen = []
    def pick(c):
        if c and c not in chosen:
            chosen.append(c)

    # 완전일치
    for r in req_norm:
        for a in avail_norm:
            if r == a:
                pick(a)

    # 부분 포함
    for r in req_norm:
        if any(r == c for c in chosen):
            continue
        hits = [a for a in avail_norm if r in a or a in r]
        if hits:
            pick(hits[0])

    # 유사도
    for r in req_norm:
        if any(r == c or r in c or c in r for c in chosen):
            continue
        near = difflib.get_close_matches(r, avail_norm, n=1, cutoff=0.5)
        if near:
            pick(near[0])

    return chosen[:5]


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

    # ② 전체
    st.subheader("② 시계열 전체(최근 12개월)")
    st.dataframe(df.tail(12), use_container_width=True, key=f"ts_tail_{merchant_id or 'na'}")

# --- 전체 파이프라인 ---
def run_full_pipeline_resumable(user_text: str | None = None, merchant_id: str | None = None):
    P = st.session_state.pipeline
    agent = st.session_state.agent

    # 최초 진입: running=false → true 전환
    if not P["running"]:
        if user_text is None:
            return
        P.update({
            "running": True,
            "cancel_requested": False,
            "step": "transform",
            "user_query": user_text,
            "merchant_id": merchant_id,
            "trans": None,
            "candidates": [],
            "exploration": None,
            "pd_out": None,
            "chart_b64": None,
            "used_kpis": None,
            "report_md": None,
        })

    # ---- 1) 변환 ----
    if P["step"] == "transform":
        _abort_if_cancelled()
        trans = agent.transform(P["user_query"])
        st.session_state.transformation = trans
        P["trans"] = trans
        _set_step("search")
        st.toast("변환 완료", icon="✅")

    # ---- 2) 가맹점 검색/선택 ----
    if P["step"] == "search":
        _abort_if_cancelled()
        target = (P["trans"] or {}).get("target") or P["user_query"]
        resp = mcp_call("search_merchant", {"store_name": target})

        if not isinstance(resp, dict):
            logger.error(f"Unexpected MCP response type: {type(resp)} -> {resp}")
            resp = _normalize_mcp_result(resp)

        status = resp.get("status")
        if "error" in resp:
            user_err(f"가맹점 검색 오류: {resp.get('message', resp.get('error'))}")
            P["running"] = False; _set_step("idle"); return

        status = resp.get("status")
        if status == "clarification_needed":
            st.session_state.candidates = resp.get("candidates", [])
            P["candidates"] = resp.get("candidates", [])
            st.session_state.awaiting_candidate = True
            st.info("🔎 유사 가맹점 다수 — 좌측 리스트에서 선택하세요.")
            # 선택될 때까지 대기 (재실행 시 다시 여기로 옴)
            return

        if status == "single_candidate":
            P["merchant_id"] = resp["candidate"]["가맹점구분번호"]
            _set_step("timeseries")
        else:
            user_err("검색 실패 또는 알 수 없는 상태입니다.")
            P["running"] = False; _set_step("idle"); return

    # ---- 3) 시계열 조회/표시 ----
    if P["step"] == "timeseries":
        _abort_if_cancelled()
        ts_resp = mcp_call("get_store_timeseries", {"merchant_id": str(P["merchant_id"])})
        if ts_resp.get("status") != "success":
            user_err(ts_resp.get("message", "시계열 조회 실패"))
            P["running"] = False; _set_step("idle"); return

        data = ts_resp["data"]
        exploration = {
            "store_identity": data.get("store_identity"),
            "time_series_analysis_data": data.get("time_series_data"),
        }
        st.session_state.exploration = exploration
        st.session_state.selected_merchant_id = P["merchant_id"]
        agent.context["exploration"] = exploration
        P["exploration"] = exploration

        render_timeseries(exploration, merchant_id=P["merchant_id"])
        _set_step("problem")

    # ---- 4) 문제 정의 ----
    if P["step"] == "problem":
        _abort_if_cancelled()
        pd_out = agent.define_problem(P["exploration"], P["trans"])
        P["pd_out"] = pd_out
        _set_step("chart")

    # ---- 5) KPI 차트 ----
    if P["step"] == "chart":
        _abort_if_cancelled()
        kpis_from_llm = (P["pd_out"] or {}).get("kpis") or []
        chart_b64, used_kpis = None, None

        if P["merchant_id"] and kpis_from_llm:
            chart_resp = mcp_call("render_kpi_chart", {
                "merchant_id": str(P["merchant_id"]),
                "kpi_keys": kpis_from_llm,
                "title": f"KPI Trend – {P['exploration']['store_identity'].get('name')}",
                "normalize_0_1": False,
            })
            if chart_resp.get("status") != "success":
                avail = chart_resp.get("available_kpis", []) or []
                # 매핑 재시도
                mapped = _map_kpis_to_available(kpis_from_llm, avail)
                if mapped:
                    chart_resp = mcp_call("render_kpi_chart", {
                        "merchant_id": str(P["merchant_id"]),
                        "kpi_keys": mapped,
                        "title": f"KPI Trend – {P['exploration']['store_identity'].get('name')}",
                        "normalize_0_1": False,
                    })

            if chart_resp.get("status") == "success":
                chart_b64 = (chart_resp.get("image") or {}).get("base64")
                used_kpis = chart_resp.get("used_kpis", [])
                st.session_state.last_chart_b64 = chart_b64

        P["chart_b64"], P["used_kpis"] = chart_b64, used_kpis
        _set_step("report")

    # ---- 6) 최종 보고서 ----
    if P["step"] == "report":
        _abort_if_cancelled()
        report_md = agent.generate_strategy_and_report()
        if not report_md or "보고서 생성에 실패" in report_md:
            si = (P["exploration"] or {}).get("store_identity", {})
            report_md = "\n".join([
                f"# {si.get('name','가맹점')} 컨설팅 요약",
                "## 문제 정의", "```json",
                json.dumps(P["pd_out"], ensure_ascii=False, indent=2), "```",
                "## 전략 제안", "- 지역 타겟 메시지 강화",
                "- 신규/재방문 캠페인 동시 운영",
                "- 상권 내 경쟁 대비 프로모션 차별화",
                "## 기대 효과", "- 매출/방문 지표 개선 기대", "- 신규 유입 및 재방문 확대",
            ])
        P["report_md"] = report_md
        _set_step("done")

    # ---- 7) 출력 ----
    if P["step"] == "done":
        si = (P["exploration"] or {}).get("store_identity") or {}
        # ① 채팅엔 요약만
        brief = []
        brief.append(f"**가맹점:** {si.get('name')} / {si.get('industry')} / {si.get('commercial_area')}")
        brief.append("### 컨설팅 요약")
        # 핵심 문장만 추려서 표시
        core_stmt = (P["pd_out"] or {}).get("problem_statement") or "문제 정의 결과를 확인했습니다."
        brief.append(f"- **핵심 문제:** {core_stmt}")
        kpis_from_llm = (P["pd_out"] or {}).get("kpis") or []
        if kpis_from_llm:
            brief.append("- **주요 KPI:** " + ", ".join(kpis_from_llm[:5]))
        brief.append("- 아래 **최종 보고서** 섹션에서 상세 내용을 확인하세요.")
        brief_reply = "\n\n".join(brief)

        st.session_state.messages.append(AIMessage(content=brief_reply))
        with st.chat_message("assistant"):
            st.markdown(brief_reply)

        # ② KPI 차트는 채팅 메시지 밖(아래) — 가로폭 문제 방지
        if P["chart_b64"]:
            with st.container():
                st.image(base64.b64decode(P["chart_b64"]), use_container_width=True)
                st.caption("자동 생성 KPI 추세 차트")
                if P["used_kpis"]:
                    st.caption("사용된 KPI: " + ", ".join(map(str, P["used_kpis"])))

        # ③ 최종 보고서는 채팅 아래의 큰 섹션에 전체 마크다운으로
        st.markdown("----")
        st.subheader("📘 최종 보고서 (상세)")
        st.markdown(P["report_md"], unsafe_allow_html=True)

        # 종료
        P["running"] = False
        _set_step("idle")

# --- 입력창 ---
user_query = st.chat_input("가맹점 이름이나 고민을 입력하세요.")
if user_query:
    st.session_state.user_query = user_query
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # 1) 파이프라인이 돌고 있지 않고,
    # 2) 이미 보고서가 존재한다면 → '팔로업 질의'로 처리
    if (not st.session_state.pipeline["running"]) and st.session_state.pipeline.get("report_md"):
        handle_followup(user_query)
    else:
        # 그 외에는 새 분석으로 판단 (가맹점 검색 포함)
        if not st.session_state.pipeline["running"]:
            run_full_pipeline_resumable(user_query, merchant_id=None)

# 후보 선택 시에도 이어서 실행
if st.session_state.awaiting_candidate and st.session_state.candidates:
    st.info("이름이 유사한 가맹점이 여러 개입니다. 아래에서 하나를 선택하세요.")
    cands = st.session_state.candidates
    idx = st.radio("가맹점을 선택하세요",
                   options=list(range(len(cands))),
                   format_func=lambda i: _fmt_cand_with_id(cands[i]),
                   key="cand_radio")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("선택 확정", key="cand_confirm"):
            sel = cands[idx]
            picked_id = str(sel.get("가맹점구분번호"))
            st.session_state.awaiting_candidate = False
            # 재개: running이 켜져 있으면 merchant만 주입하고 다음 단계 진행
            st.session_state.pipeline["merchant_id"] = picked_id
            _set_step("timeseries")
            run_full_pipeline_resumable()
    with col2:
        if st.button("선택 취소", key="cand_cancel"):
            st.session_state.awaiting_candidate = False
            st.session_state.candidates = []
            st.session_state.exploration = None

        # 🔒 이 화면이 떠 있는 동안 뒤 코드를 그리지 않도록 강제 종료
    st.stop()
    
# 사용자가 '전략 제안 종료'를 눌렀다면 즉시 종료
if st.session_state.pipeline["running"] and st.session_state.pipeline.get("cancel_requested"):
    st.session_state.pipeline.update({"running": False, "step": "idle"})
    st.info("⏹️ 전략 제안을 중지했습니다.")

# 화면이 재실행되어도, running=True면 계속 진행
if st.session_state.pipeline["running"] and not st.session_state.awaiting_candidate:
    run_full_pipeline_resumable()