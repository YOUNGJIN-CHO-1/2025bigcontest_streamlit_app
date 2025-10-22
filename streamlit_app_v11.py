import os
import sys
import json
import time
import shutil
import streamlit as st
import google.generativeai as genai
from google.generativeai import protos
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from PIL import Image
import plotly.express as px
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
from dotenv import load_dotenv

# =========================
# 0) 공통 설정
# =========================
ASSETS = Path("assets")
st.set_page_config(
    page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사",
    layout="wide",
    page_icon="📈",
)

# 그래프 대상 컬럼 정의
PLOTTING_NUMERIC_COLS = [
    '동일 업종 매출금액 비율', '동일 업종 매출건수 비율', '동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율',
    '재방문 고객 비중', '신규 고객 비중', '거주 이용 고객 비율', '직장 이용 고객 비율', '유동인구 이용 고객 비율'
]

@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)

# 헤더
st.title("신한카드 소상공인 비밀상담소 🔑")
st.image(load_image("KMWL.png"), width= 500, caption="고민하지 말고, AI비밀상담사에게 물어보세요!")
st.write("")

# 로깅 및 UI 헬퍼
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
console_logger = logging.getLogger(__name__)

def user_warn(msg: str): st.warning(msg, icon="⚠️")
def user_err(msg: str, details: dict = None):
    st.error(msg, icon="🛑")
    if details:
        with st.expander("오류 상세 정보"): st.json(details)

# 1. st.secrets에서 get (1순위 - Streamlit 배포 시 표준 방식)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# 2. st.secrets에 값이 없다면 .env 활용
if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY가 st.secrets에 설정되지 않았습니다. .env를 사용합니다.")
    load_dotenv() # .env 파일의 변수를 환경 변수로 로드
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# 3. .env에도 값이 없다면 하드코딩 
if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY가 st.secrets 또는 .env에 설정되지 않았습니다. 하드코딩된 키를 사용합니다.")
    GOOGLE_API_KEY = "AIzaSyB8R3nurDOohfAvKXSgBUVRkoliXtfnTKo"

# --- 최종 확인 및 API 설정 ---
if GOOGLE_API_KEY:
    # st.success("API 키 로드 성공.") # (디버깅 용도로 사용)
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # 1, 2, 3 순서 모두 실패한 경우
    st.error("🛑 치명적 오류: GOOGLE_API_KEY를 찾을 수 없습니다. 앱 실행을 중지합니다.")
    st.stop() # 앱 실행 중지

# =========================
# 1) Streamlit용 로거
# =========================
class StreamlitLogHandler:
    def __init__(self):
        if "log_entries" not in st.session_state: st.session_state.log_entries = []
    
    def log(self, message: str, details: dict = None, expander_label: str = "상세 결과 보기", update: bool = False):
        entry = {"message": f"- {message}", "details": details, "expander_label": expander_label}
        if update and st.session_state.log_entries: st.session_state.log_entries[-1] = entry
        else: st.session_state.log_entries.append(entry)
        time.sleep(0.5)
    
    def render(self, container):
        container.empty()
        for entry in st.session_state.log_entries:
            container.markdown(entry["message"])
            if entry["details"]:
                with container.expander(entry["expander_label"], expanded=False): st.json(entry["details"])

# ---------------------------
# 에이전트(추론/리포팅) 로직
# ---------------------------
class InteractiveParallelAgent:
    """
    외부 환경 데이터를 통합하고 토큰 효율적인 데이터 압축 요약을 적용한
    대화형 (가맹점 선택 시) 비동기 처리 에이전트 (문제 정의 단계)
    """

    # --- ✨ 1단계: 변환 스키마 및 프롬프트 ---
    _TRANSFORM_SCHEMA = protos.Schema(
        type=protos.Type.OBJECT,
        properties={
            'target': protos.Schema(type=protos.Type.STRING, description="전략 요청 대상 (가맹점 이름)"),
            'challenge': protos.Schema(type= protos.Type.STRING, description="사용자가 직면한 어려움이나 문제"),
            'objective': protos.Schema(type=protos.Type.STRING, description="목표 달성을 위한 구체적인 목적"),
            'solution_direction': protos.Schema(type=protos.Type.STRING, description="제안될 수 있는 해결책의 방향"),
        },
        required=['target', 'challenge', 'objective', 'solution_direction']
    )
    _TRANSFORM_SYSTEM_PROMPT = """
당신은 사용자의 자연어 요청을 구조화된 데이터로 변환하는 전문가입니다.
사용자의 요청을 분석하여 다음 네 가지 항목으로 요약하세요.
- "target": 사용자가 전략을 요청한 대상은 무엇인가요? (가맹점 이름. 예: '성우**', '행*') ('매장', '가맹점'과 같은 단어를 포함하지 마세요)
- "challenge": 사용자가 직면한 어려움이나 해결해야 할 문제는 무엇인가요?
- "objective": 목표 달성을 위한 구체적인 목적은 무엇인가요?
- "solution_direction": 제안될 수 있는 해결책의 방향은 무엇인가요?
"""

    # --- ✨ 3단계: 문제 정의 스키마 및 프롬프트 (외부 데이터 포함) ---
    _DEFINE_PROBLEM_SCHEMA = protos.Schema(
        type=protos.Type.OBJECT,
        properties={
            'problem_statement': protos.Schema(type=protos.Type.STRING, description="데이터(내부+외부) 기반으로 정의된 구체적인 이슈"),
            'kpis': protos.Schema(
                type=protos.Type.ARRAY,
                items=protos.Schema(type=protos.Type.STRING),
                description="문제 해결을 위한 측정 가능한 핵심 성과 지표 목록"
            ),
        },
        required=['problem_statement', 'kpis']
    )
    _DEFINE_PROBLEM_SYSTEM_PROMPT = """
당신은 데이터 분석가입니다. 제공된 가맹점의 내부 성과 데이터(압축된 통계 요약)와 외부 환경 데이터, 그리고 최초 사용자 요청을 종합적으로 분석하여, 문제 상황과 핵심 성과 지표(KPI)를 구체적으로 정의하세요.

**데이터 구조 설명:**
1.  **내부 성과 데이터 (압축 요약):** 최근 시계열 통계 요약
    * 기본 정보 (meta): 가맹점ID, 업종, 업종_분류, 상권, 분석기간, cluster 정보 등
    * 전체 평균 지표 (avg_metrics): 장기 평균값
    * 트렌드 (trend): 초기 대비 최근 성과 변화
    * 최근 3개월 상세 (recent_3months): 월별 상세 데이터
    * 주요 이벤트 (significant_events): 급격한 변화 시점
    * 구간 빈도 (range_frequency): 가장 자주 나타나는 성과 구간
2.  **외부 환경 데이터:** 해당 가맹점이 속한 행정동 및 업종 분류의 상권 분석 정보 (영업환경지수, 잠재고객, 경쟁강도 등) - 제공된 경우 참고

**데이터 해석 가이드:**
- 내부 데이터 해석 가이드 (기존과 동일):
    - 매출금액/건수/고객수 구간: 1(상위)~6(하위)
    - 취소율 구간: 1(낮음, 좋음)~5(높음, 나쁨)
    - 동일 업종/상권 비교 지표: 100% 기준, 순위는 0%에 가까울수록 상위
    - cluster: 0(상위권 안정), 1(하위권 안정), 2(변동성 높음), 3(신규)
    - cluster_6: 고객 세그먼트 (0: 여성40대+거주, 1: 남성30-40대+신규, ...)
- 외부 데이터 해석:
    - 영업환경지수(BSENV_NIDX): 낮을수록 좋음 (1:최우수 ~ 9:위험)
    - 잠재고객지수(PRSPT_NIDX): 낮을수록 좋음 (1:최우수 ~ 9:위험)
    - 경쟁강도지수(CPITS_NIDX): 낮을수록 좋음 (1:최우수 ~ 9:위험, 99:정보없음)
    - 각 지수의 진단 코멘트(DGNS_CTT) 참고

**특히 주목할 점:**
- 내부 데이터: 트렌드 변화율, 최근 3개월 추세, 주요 이벤트(급등/급락), combination 지표(cluster와 cluster_6 정보를 결합한 고객 및 가맹점 세그먼트 특성)
- 외부 데이터: 가맹점의 영업 환경 수준 (양호, 보통, 관찰, 주의 등), 잠재고객 특성, 경쟁 강도 진단 내용

[중요] KPIs 선정 지침
'kpis' 목록을 생성할 때, 반드시 다음의 **[유효한 KPI 컬럼 목록]**에 있는 이름과
**정확히 일치하는** 단어만 사용해야 합니다.
목록에 없는 단어(예: '객단가', '성장률')를 KPI 이름으로 지어내지 마세요.

[유효한 KPI 컬럼 목록]
- '동일 업종 매출금액 비율'
- '동일 업종 매출건수 비율'
- '동일 업종 내 매출 순위 비율'
- '동일 상권 내 매출 순위 비율'
- '재방문 고객 비중'
- '신규 고객 비중'
- '거주 이용 고객 비율'
- '직장 이용 고객 비율'
- '유동인구 이용 고객 비율'

결과로 현재 직면한 구체적인 이슈(problem_statement)와 문제 해결을 위한 측정 가능한 핵심 지표(kpis)를 도출하세요. **외부 환경 요인을 문제 정의에 반드시 반영하세요.**
"""

    # --- ✨ 4단계: 리포트 생성 스키마 및 프롬프트 ---
    _GENERATE_STRATEGY_REPORT_SCHEMA = protos.Schema(
        type=protos.Type.OBJECT,
        properties={
            'report': protos.Schema(type=protos.Type.STRING, description="마크다운 형식의 전체 최종 보고서 내용"),
        },
        required=['report']
    )
    _GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT = """
    당신은 뛰어난 비즈니스 컨설턴트이자 전문 보고서 작성자입니다.
주어진 '문제 정의'(내부 데이터와 외부 환경 요인 포함)와 압축된 데이터 분석 결과를 종합하여, 최종 사용자를 위한 포괄적인 보고서를 작성하세요.

**사용자의 주요 요청 방향 ('solution_direction'):** {solution_direction}

보고서는 위 **요청 방향에 부합하는** 구체적인 전략과 실행 계획을 포함해야 합니다.
데이터와 직접 연결된 구체적인 실행 방안을 제시하세요. 외부 환경 데이터에서 도출된 인사이트도 전략에 반영해야 합니다.

**주의사항:**
- 가맹점의 이름을 임의로 바꾸지 마세요.
- 16384 토큰 제한을 고려하여, 불필요한 반복 없이 간결하고 명확하게 작성하세요.
- 전략적 제안 및 실행 계획에서 모든 제안과 답변에는 근거가 표시되어야 합니다. 단, KPI와 일반적인 지식은 근거로 표시하지 않습니다.
- 제공된 데이터에 근거하지 않은 추상적이거나 일반적인 조언은 피하세요.
- 단순히 데이터를 나열하는 것이 아니라, 데이터를 해석하여 실행 가능한 인사이트를 도출해야 합니다.
- 보고서 서두에 불필요한 인사말이나 자기소개는 포함하지 마세요.
- 외부 환경에 대한 데이터에 대한 출처를 다음과 같이 표기하세요 : (출처 : 신용보증기금 BASA 소상공인 상권분석 데이터)
- 초기 6개월 평균과 최근 6개월 평균 성과를 비교 분석할 때는 **반드시 Markdown 테이블 형식**을 사용하여 명확하게 제시합니다. 테이블에는 '지표', '초기 6개월 평균', '최근 6개월 평균', '변화율', '인사이트' 컬럼을 포함하세요. 최근 3개월 동향과 외부 환경 분석 결과를 요약합니다.

최종 결과는 전체 보고서 내용을 마크다운 형식의 문자열로 포함해야 합니다.
**당신의 최종 보고서는 반드시 '사용자의 주요 요청 방향': '{solution_direction}'을(를) 충족하는 전략을 상세하게 다루어야 합니다.**

"""

    def __init__(self, sync_model: Any, csv_path: str, json_data: List[Dict[str, Any]]):
        self.model = sync_model
        self.context = {}
        self.json_data = json_data
        self._last_external_data = None

        try:
            self.df = pd.read_csv(csv_path)
            self.df.columns = self.df.columns.str.strip()

            if "기준년월" in self.df.columns:
                self.df["기준년월"] = pd.to_datetime(self.df["기준년월"].astype(str).str.zfill(6), format="%Y%m", errors='coerce')
            console_logger.info(f"✅ CSV 데이터 로드 성공: {csv_path} ({len(self.df)} rows)")
            required_csv_cols = ['가맹점명', '업종', '업종_분류', '상권', '행정동', '기준년월']
            
            if not all(col in self.df.columns for col in required_csv_cols):
                console_logger.warning(f"⚠️ CSV 파일에 일부 필수 컬럼이 없을 수 있습니다.")
        
        except FileNotFoundError:
            console_logger.error(f"❌ 오류: CSV 파일 '{csv_path}'을(를) 찾을 수 없습니다.")
            self.df = pd.DataFrame()
        
        except Exception as e:
            console_logger.error(f"❌ CSV 데이터 로드 중 오류 발생: {e}")
            self.df = pd.DataFrame()

    def _generate_content_sync(self, prompt: str, response_schema: Optional[protos.Schema] = None, generation_config: Optional[Dict[str, Any]] = None, retries: int = 3, delay: int = 5) -> Dict[str, Any]:
        last_exception = None
        for i in range(retries):
            try:
                final_config_dict = generation_config.copy() if generation_config else {}
                if response_schema:
                    final_config_dict["response_schema"] = response_schema
                    final_config_dict["response_mime_type"] = "application/json"
                
                # protos.GenerationConfig 객체로 변환
                final_gen_config = genai.types.GenerationConfig(**final_config_dict)

                time.sleep(1)
                response = self.model.generate_content(prompt, generation_config=final_gen_config)

                if response.candidates and response.candidates[0].finish_reason != 1:
                    reason_map = {0: "UNKNOWN", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                    reason_str = reason_map.get(response.candidates[0].finish_reason, "기타")
                    raise ValueError(f"API 호출 비정상 종료 (Reason: {reason_str})")
                
                if not response.text:
                    safety_ratings = response.candidates[0].safety_ratings if response.candidates else "N/A"
                    raise ValueError(f"API 응답 내용 비어 있음 (Safety: {safety_ratings})")
                return json.loads(response.text)
            
            except (json.JSONDecodeError, ValueError, Exception) as e:
                last_exception = e
                if i == retries - 1: break
                wait_time = delay * (2 ** i)
                console_logger.warning(f"⚠️ LLM 동기 호출 실패 ({e}). {wait_time}초 후 재시도... (시도 {i + 1}/{retries})")
                time.sleep(wait_time)
        return {"error": f"LLM 동기 호출 최종 실패 ({retries}번 재시도): {last_exception}"}

    def _transform(self, initial_input: str):
        console_logger.info("1단계: 사용자 요청 변환 시작...")
        target_name_guess = initial_input.split()[0] if initial_input else "알 수 없는 가맹점"
        user_prompt = f"🔹 사용자 입력:\n대상 (가맹점 이름): '{target_name_guess}'\n요청 내용: {initial_input}"
        full_prompt = f"{self._TRANSFORM_SYSTEM_PROMPT}\n\n{user_prompt}"
        self.context['transformation'] = self._generate_content_sync(full_prompt, response_schema=self._TRANSFORM_SCHEMA, generation_config={"max_output_tokens": 1024})
        console_logger.info("1단계: 변환 완료.")
    
    def _find_and_clarify_store(self):
        console_logger.info("2-1단계: 가맹점 검색 시작...")
        transformation_result = self.context.get('transformation', {})
        target_description = transformation_result.get('target', '')
        if "error" in transformation_result:
            self.context['error'] = f"1단계 변환 결과 오류: {transformation_result['error']}"; return
        if not target_description:
            self.context['error'] = "1단계 변환 결과에서 'target' 정보를 찾을 수 없습니다."; return
        search_term = target_description.replace('*', '').strip()
        required_cols = ['가맹점명', '업종', '업종_분류', '상권', '행정동'] # ✅ '가맹점구분번호' 추가 필요
        if self.df.empty or not all(col in self.df.columns for col in required_cols + ['가맹점구분번호']):
             self.context['error'] = "CSV 데이터프레임이 비어 있거나 필수 컬럼 부족"; return
        try:
            candidate_stores = self.df[self.df['가맹점명'].str.contains(search_term, na=False)]
            # ✅ 가맹점 ID도 포함하여 중복 제거
            unique_candidates = candidate_stores[required_cols + ['가맹점구분번호']].drop_duplicates().to_dict('records')
        except Exception as e:
             self.context['error'] = f"가맹점 검색 중 오류 발생: {e}"; return

        if not unique_candidates:
            self.context['error'] = f"'{search_term}'(을)를 포함하는 가맹점을 찾을 수 없습니다."
        elif len(unique_candidates) == 1:
            self.context['target_store_info'] = unique_candidates[0]
            console_logger.info(f"✅ 가맹점 유일 식별: {unique_candidates[0]['가맹점명']}")
        else:
            self.context['clarification_needed'] = unique_candidates
            console_logger.warning(f"⚠️ '{search_term}'에 대해 {len(unique_candidates)}개 후보 발견")
        console_logger.info("2-1단계: 가맹점 검색 완료.")

    def _compress_store_data(self):
        console_logger.info("2-2단계: 데이터 압축 요약 시작...")
        store_info = self.context.get('target_store_info')
        if not store_info: self.context['error'] = "가맹점 정보 미확정"; return
        try:
            store_data = self.df[
                (self.df['가맹점명'] == store_info['가맹점명']) &
                (self.df['업종'] == store_info['업종']) &
                (self.df['업종_분류'] == store_info['업종_분류']) &
                (self.df['상권'] == store_info['상권']) &
                (self.df['행정동'] == store_info['행정동']) # 행정동 조건 추가
            ].sort_values(by='기준년월').reset_index(drop=True).copy()
            self.context['target_store_timeseries'] = store_data
        except KeyError as e: self.context['error'] = f"필요 컬럼({e}) 없음"; return
        except Exception as e: self.context['error'] = f"데이터 필터링 오류: {e}"; return
        if store_data.empty: self.context['error'] = "해당 가맹점 데이터 없음"; return

        # --- 압축 로직 ---
        meta, avg_metrics, trend = {}, {}, {}
        recent_3_detailed, significant_events, range_frequency = [], [], {}
        try:
            first_row = store_data.iloc[0]
            meta = {
                '가맹점ID': str(first_row.get('가맹점구분번호')), '가맹점명': first_row.get('가맹점명'),
                '업종': first_row.get('업종'), '업종_분류': first_row.get('업종_분류'),
                '상권': first_row.get('상권'), '행정동': first_row.get('행정동'),
                '분석기간': f"{store_data['기준년월'].min().strftime('%Y-%m')} ~ {store_data['기준년월'].max().strftime('%Y-%m')}",
                '데이터포인트': len(store_data),
                'cluster': int(first_row['cluster']) if 'cluster' in first_row and pd.notna(first_row['cluster']) else None,
                'cluster_6': int(first_row['cluster_6']) if 'cluster_6' in first_row and pd.notna(first_row['cluster_6']) else None
            }
            # target_store_info 업데이트 (가맹점ID 포함)
            self.context['target_store_info'].update({'가맹점ID': meta['가맹점ID'], '행정동': meta['행정동']})

            potential_numeric_cols = PLOTTING_NUMERIC_COLS + ['동일 업종 내 해지 가맹점 비중', '동일 상권 내 해지 가맹점 비중'] # 해지 비중 포함
            numeric_cols = [col for col in potential_numeric_cols if col in store_data.columns]
            store_data_clean = store_data[numeric_cols].apply(pd.to_numeric, errors='coerce').replace([-999999.9, -999999], np.nan)

            avg_metrics = {col: round(store_data_clean[col].mean() or 0, 2) for col in numeric_cols if store_data_clean[col].notna().any()}
            
            if len(store_data_clean) >= 12:
                first_6, last_6 = store_data_clean.head(6), store_data_clean.tail(6)
                for col in numeric_cols:
                    first_val, last_val = first_6[col].mean(), last_6[col].mean()
                    if pd.notna(first_val) and pd.notna(last_val) and abs(first_val) > 1e-6:
                        change_pct = ((last_val - first_val) / abs(first_val)) * 100
                        trend[col] = {'초기_평균': round(first_val, 2), '최근_평균': round(last_val, 2), '변화율': f"{change_pct:+.1f}%"}
            
            # 기준년월 컬럼을 int로 변환 시도
            if '기준년월' in store_data.columns:
                 store_data['기준년월_int'] = store_data['기준년월'].dt.strftime('%Y%m').astype(int)

            recent_3 = store_data.tail(3)
            for _, row in recent_3.iterrows():
                 month_key = int(row['기준년월_int']) if '기준년월_int' in row and pd.notna(row['기준년월_int']) else f"Row_{_}"
                 month_data = {'기준년월': month_key}
                 month_data.update({col: round(row[col], 2) for col in numeric_cols if col in row and pd.notna(row[col])})
                 recent_3_detailed.append(month_data)

            event_cols = [col for col in ['동일 업종 매출금액 비율', '재방문 고객 비중', '유동인구 이용 고객 비율'] if col in numeric_cols]
            if '기준년월_int' in store_data.columns:
                for col in event_cols:
                    values = store_data[[col, '기준년월_int']].dropna().copy()
                    if len(values) >= 2:
                        values[col] = pd.to_numeric(values[col], errors='coerce').dropna()
                        if len(values) >= 2:
                            values['pct_change'] = values[col].pct_change() * 100
                            events = values[abs(values['pct_change']) > 15].copy()
                            for idx in events.index:
                                if idx > values.index.min():
                                    prev_idx = values.index[values.index.get_loc(idx) - 1]
                                    prev_row, event_row = values.loc[prev_idx], values.loc[idx]
                                    significant_events.append({'지표': col,'시점': int(event_row['기준년월_int']),'변화': f"{prev_row[col]:.1f} → {event_row[col]:.1f} ({event_row['pct_change']:+.1f}%)"})

            potential_range_cols = ['매출금액 구간', '매출건수 구간', '유니크 고객 수 구간', '객단가 구간', '취소율 구간']
            range_cols = [col for col in potential_range_cols if col in store_data.columns]
            range_frequency = {col: store_data[col].mode()[0] for col in range_cols if not store_data[col].mode().empty}
        except Exception as e:
            self.context['error'] = f"데이터 압축 중 오류: {e}"; return

        self.context['exploration'] = {
            'meta': meta, 'avg_metrics': avg_metrics, 'trend': trend,
            'recent_3months': recent_3_detailed, 'significant_events': significant_events, 'range_frequency': range_frequency
        }
        console_logger.info("2-2단계: 데이터 압축 완료.")
    
    def _fetch_external_data(self, dong_name: Optional[str], biz_category: Optional[str]) -> Optional[Dict[str, Any]]:
        self._last_external_data = None
        if not dong_name or not biz_category or not self.json_data:
            console_logger.warning(f"⚠️ 외부 데이터 검색 건너뛰기: 정보 부족 또는 JSON 데이터 없음")
            return None
        try: last_biz_word = biz_category.split('>')[-1].strip()
        except Exception: return None
        if not last_biz_word: return None
        processed_dong_name = dong_name.replace('제', '').replace('.', '').strip() if dong_name else ""
        console_logger.info(f"🔍 외부 데이터 검색 (행정동: '{dong_name}', 업종키워드: '{last_biz_word}')...")
        for item in self.json_data:
            json_dong_raw = item.get('INPUT_ADUN_NM')
            json_dong_processed = json_dong_raw.replace('제', '').replace('.', '').strip() if json_dong_raw else ""
            json_biz_name = item.get('INPUT_BZC_NM')
            if json_dong_processed and json_biz_name and processed_dong_name == json_dong_processed and last_biz_word == json_biz_name:
                console_logger.info(f"✅ 외부 데이터 찾음: {json_dong_raw}, {json_biz_name}")
                self._last_external_data = item; return item
        console_logger.warning(f"⚠️ 외부 데이터 찾을 수 없음.")
        return None

    def _define_problem(self, exploration_result: Dict[str, Any], transformation_result: Dict[str, Any], external_data: Optional[Dict[str, Any]]):
        console_logger.info("3단계: 문제 정의 시작...")
        external_data_prompt_part = "외부 환경 데이터: 해당 지역/업종 정보 없음"
        if external_data:
            relevant_keys = ["ADUN_FULL_NM", "SMMB_BZC_SCLS_CD_NM", "BSENV_NIDX", "BSENV_DGNS_CTT", "PRSPT_NIDX", "PRSPT_DGNS_CTT", "CPITS_NIDX", "CPITS_DGNS_CTT"]
            filtered_data = {k: external_data.get(k) for k in relevant_keys if external_data.get(k) is not None}
            if filtered_data: external_data_prompt_part = f"외부 환경 데이터:\n{json.dumps(filtered_data, indent=2, ensure_ascii=False)}"
            else: external_data_prompt_part = "외부 환경 데이터: 관련 정보 추출 실패"
        user_prompt = f"🔹 문맥(Context):\n- 최초 요청: {json.dumps(transformation_result, indent=2, ensure_ascii=False)}\n- 압축된 내부 성과 데이터: {json.dumps(exploration_result, indent=2, ensure_ascii=False)}\n- {external_data_prompt_part}"
        full_prompt = f"{self._DEFINE_PROBLEM_SYSTEM_PROMPT}\n\n{user_prompt}"
        response = self._generate_content_sync(full_prompt, response_schema=self._DEFINE_PROBLEM_SCHEMA, generation_config={"max_output_tokens": 8000})
        self.context['problem_definition'] = response
        console_logger.info("3단계: 문제 정의 완료.")

    def _generate_strategy_and_report(self) -> str:
        console_logger.info("4단계: 최종 리포트 생성 시작...")
        problem_def = self.context.get('problem_definition', {})
        exploration_res = self.context.get('exploration', {})
        transformation_res = self.context.get('transformation')
        if not transformation_res or "error" in transformation_res: return f"# 보고서 실패: 1단계 오류"
        if not exploration_res or "error" in self.context: return f"# 보고서 실패: 2단계 오류"
        if not problem_def or "error" in problem_def: return f"# 보고서 실패: 3단계 오류"

        solution_direction = transformation_res.get('solution_direction', '구체적인 전략 제안 및 실행 계획 제시')

        external_summary = "정보 없음"
        if self._last_external_data:
             ext = self._last_external_data; parts = []
             if ext.get('BSENV_DGNS_CTT'): parts.append(f"영업환경: {ext['BSENV_DGNS_CTT']} (지수: {ext.get('BSENV_NIDX', 'N/A')})")
             if ext.get('PRSPT_DGNS_CTT'): parts.append(f"잠재고객: {ext['PRSPT_DGNS_CTT']} (지수: {ext.get('PRSPT_NIDX', 'N/A')})")
             if ext.get('CPITS_DGNS_CTT'):
                 idx = ext.get('CPITS_NIDX', 'N/A'); idx_str = '정보없음' if idx == 99 else str(idx)
                 parts.append(f"경쟁강도: {ext['CPITS_DGNS_CTT']} (지수: {idx_str})")
             if parts: external_summary = ", ".join(parts)
             

        final_context = {
            "1_초기_요청사항": transformation_res, "2_데이터기반_문제정의": problem_def,
            "3_주요_데이터_인사이트": {
                "가맹점_정보": exploration_res.get('meta'), "성과_트렌드": exploration_res.get('trend'),
                "최근_3개월_동향": exploration_res.get('recent_3months'), "외부_환경_요약": external_summary
            }
        }
        user_prompt = f"🔹 보고서 작성을 위한 종합 컨텍스트:\n{json.dumps(final_context, indent=2, ensure_ascii=False)}"
        full_prompt = f"{self._GENERATE_STRATEGY_REPORT_SYSTEM_PROMPT}\n\n{user_prompt}"
        final_response = self._generate_content_sync(full_prompt, response_schema=self._GENERATE_STRATEGY_REPORT_SCHEMA, generation_config={"max_output_tokens": 16384, "temperature": 0.3})
        if "error" in final_response: return f"# 보고서 실패: LLM 오류 {final_response['error']}"
        console_logger.info("4단계: 최종 리포트 생성 완료.")
        return final_response.get("report", "# 보고서 실패: 내용 없음")
    

# ---------------------------
# Streamlit UI
# ---------------------------
system_prompt = (
    "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, "
    "분석 결과를 바탕으로 적절한 마케팅 방법과 채널, 마케팅 메시지를 추천합니다. "
    "결과는 짧고 간결하게, 분석 결과에는 가능한 표를 사용하여 알아보기 쉽게 설명해주세요."
)
greeting = "가맹점 명을 입력하거나, 고민을 말씀해주세요. AI비밀상담사가 도와드리겠습니다!"

def ensure_state():
    ss = st.session_state
    if "messages" not in ss: ss.messages = [AIMessage(content=greeting)]
    if "pipeline" not in ss:
        ss.pipeline = {"running": False, "step": "idle", "user_query": "", "trans": None,
                    "merchant_id": None, "exploration": None, "pd_out": None, "report_md": None, "timeseries_data": None}
    ss.setdefault("awaiting_candidate", False)
    ss.setdefault("candidates", [])

ensure_state()

def clear_chat_history():
    st.session_state.messages = [AIMessage(content=greeting)]
    keys_to_clear = ["pipeline", "awaiting_candidate", "candidates", "log_entries"]
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]
    ensure_state() # 상태 재초기화

# 후속 질문 처리   
def handle_followup(user_question: str) -> str:
    P = st.session_state.pipeline
    context_for_followup = {
        "previous_report": P.get("report_md"),
        "problem_definition": P.get("pd_out"),
        "data_summary": P.get("exploration"),
    }
    prompt = (
        "당신은 분석 컨설턴트입니다. 아래 컨텍스트를 바탕으로 한국어로 간결하게 답하세요.\n"
        "숫자는 표/불릿으로 명확히, 모르는 정보는 추측하지 말고 부족하다고 밝혀주세요.\n\n"
        f"### 컨텍스트\n{json.dumps(context_for_followup, ensure_ascii=False, indent=2)}\n\n"
        f"### 사용자의 질문\n{user_question}\n\n"
        "### 답변"
    )
    try:
        response = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt)
        return response.text
    except Exception as e: return f"답변 생성 중 오류가 발생했습니다: {e}"

# 후보 표시 포맷
def _id_of(cand: dict):
    return cand.get("가맹점구분번호")

def _fmt_cand_with_id(c: dict) -> str:
    id_val = _id_of(c) or "ID?"
    name = c.get('가맹점명', '?')
    biz_cat = c.get('업종_분류', c.get('업종', '?')) 
    area = c.get('상권', '?')
    dong = c.get('행정동', '?')
    return f"[{id_val}] {name} ({biz_cat} / {area} / {dong})"

def _set_step(step: str): st.session_state.pipeline["step"] = step

# =========================
# 5) 사이드바
# =========================
with st.sidebar:
    logo = ASSETS / "shc_ci_basic_00.png"
    if logo.exists():
        st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest • AI DATA 활용</p>", unsafe_allow_html=True)
    st.button("Clear Chat History", on_click=clear_chat_history)

    if st.session_state.pipeline.get("running"):
        if st.button("분석 중단", type="secondary"): st.session_state.pipeline["cancel_requested"] = True
        
# --- 히스토리 렌더 ---
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role): st.write(msg.content)

# --- KPI 차트 생성 함수 ---
def create_kpi_charts(kpi_list: list, timeseries_df: Optional[pd.DataFrame]) -> list:
    """
    AI가 선정한 KPI 리스트와 시계열 데이터프레임을 받아 Plotly 그래프를 생성합니다.
    PLOTTING_NUMERIC_COLS에 포함된 변수만 그래프로 그립니다. (매칭 로직 개선)
    """
    charts = []
    if timeseries_df is None or timeseries_df.empty:
        console_logger.warning("📈 KPI 차트 생성 건너뛰기: 시계열 데이터 없음")
        return charts 

    valid_kpi_columns = []
    for col in PLOTTING_NUMERIC_COLS: 
        if col in timeseries_df.columns: 
            for kpi_item in kpi_list: 
                if col in str(kpi_item): 
                    valid_kpi_columns.append(col)
                    break 

    # 디버깅용 로그 추가
    console_logger.info(f"📈 AI KPI 목록: {kpi_list}")
    console_logger.info(f"📈 그래프 대상 컬럼 필터링 결과: {valid_kpi_columns}")

    plotted_cols_count = 0
    for kpi_col in sorted(set(valid_kpi_columns)):
        if plotted_cols_count >= 4: break
        try:
            # ✅ [수정] 그래프 그리기 전에 이상치(-999999 등)를 np.nan으로 대체
            df_cleaned = timeseries_df.copy()
            df_cleaned[kpi_col] = pd.to_numeric(df_cleaned[kpi_col], errors='coerce') # 숫자로 변환
            df_cleaned[kpi_col] = df_cleaned[kpi_col].replace([-999999.9, -999999], np.nan) # 이상치 NaN 처리

            # NaN 아닌 값이 하나라도 있을 때만 그래프 생성
            if df_cleaned[kpi_col].notna().any():
                fig = px.line(
                    df_cleaned, # 정제된 데이터 사용
                    x='기준년월',
                    y=kpi_col,
                    title=f'<b>{kpi_col} 추이</b>',
                    markers=True,
                    labels={'기준년월': '월', kpi_col: '값'}
                )
                fig.update_layout(
                    title_font_size=16, title_x=0.5,
                    xaxis_title_font_size=12, yaxis_title_font_size=12,
                    legend_title_text=''
                )
                charts.append(fig)
                plotted_cols_count += 1
            else:
                 console_logger.warning(f"📈 '{kpi_col}' 컬럼에 유효한 데이터가 없어 그래프 생성 건너뜀")

        except Exception as e:
            console_logger.error(f"📈 Plotly 그래프 생성 오류 ({kpi_col}): {e}")

    return charts

# --- 전체 파이프라인 ---
# ✅ 데이터 로딩 및 Agent 초기화 (앱 실행 시 한 번만)
@st.cache_resource
def load_data_and_init_agent():
    try:
        csv_path = "./data/labeling_with_geo.csv"
        json_path = "./data/api_results.json"  
        
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
        
        sync_model = genai.GenerativeModel('gemini-2.5-flash') 
        
        agent = InteractiveParallelAgent(sync_model, csv_path, json_content)
        
        if agent.df is None or agent.df.empty:
            st.error(f"CSV 데이터 로딩 실패. 파일 확인: {csv_path}"); st.stop()
        console_logger.info("에이전트 초기화 완료")
        return agent
    
    except FileNotFoundError as e:
        st.error(f"필수 파일({e.filename}) 없음. 앱 종료."); st.stop()
    except Exception as e:
        st.error(f"초기화 오류: {e}"); st.stop()

agent = load_data_and_init_agent()

def perform_analysis_sync(agent_instance):
    """
    비동기(async) 로직을 모두 제거하고 순차적(sync)으로 데이터 분석을 수행하는 함수입니다.
    """
    store_info = agent_instance.context.get('target_store_info')
    if not store_info:
        raise RuntimeError("Target store information is missing.")

    # 1. 내부 데이터 압축 (동기)
    agent_instance._compress_store_data()

    # _compress_store_data 실행 후 오류 확인
    if "error" in agent_instance.context and "exploration" not in agent_instance.context:
        # 데이터 압축 단계에서 발생한 오류를 명확히 전달합니다.
        raise RuntimeError(f"Internal data compression failed: {agent_instance.context['error']}")

    # 2. 외부 데이터 가져오기 (동기)
    fetched_external_data = agent_instance._fetch_external_data(store_info.get('행정동'), store_info.get('업종_분류'))
    
    # 3. 데이터 기반 문제 정의 (동기)
    agent_instance._define_problem(agent_instance.context['exploration'], agent_instance.context['transformation'], fetched_external_data)

def run_full_pipeline_resumable(user_text: str | None = None):
    P = st.session_state.pipeline
    logger = StreamlitLogHandler()
    
    if not P.get("running"):
        if user_text is None: return
        P.update({"running": True, "step": "transform", "user_query": user_text})
        agent.context = {}

    if P["step"] == "transform":
        if not st.session_state.get("log_entries"): logger.log("🤔 1단계: 사용자 요청 분석 중...")
        agent._transform(P["user_query"])
        trans = agent.context.get("transformation", {})
        if "error" in trans: user_err("요청 분석 실패", details=trans); P["running"] = False; _set_step("idle"); return
        P["trans"] = trans
        logger.log("✅ 1단계: 사용자 요청 분석 완료", details=trans, expander_label="사용자 요청 내역 보기")
        _set_step("search"); st.rerun()

    if P["step"] == "search":
        if len(st.session_state.log_entries) < 2: logger.log(f"🔍 2단계: '{P['trans']['target']}' 가맹점 정보 검색 중...")
        agent._find_and_clarify_store()
        if "error" in agent.context: user_err(agent.context["error"]); P["running"] = False; _set_step("idle"); return
        if "clarification_needed" in agent.context:
            st.session_state.candidates = agent.context["clarification_needed"]
            st.session_state.awaiting_candidate = True
            logger.log("⚠️ 2단계: 후보 가맹점 발견", details={"candidates": agent.context["clarification_needed"]}, expander_label="후보 가맹점 목록 보기")
            return
        store_info = agent.context.get("target_store_info", {})
        logger.log(f"✅ 2단계: '{store_info.get('가맹점명')}' 정보 확인 완료", details=store_info, expander_label="확인된 가맹점 정보 보기")
        _set_step("analysis"); st.rerun()

    if P["step"] == "analysis":
        try:
            perform_analysis_sync(agent)
        except RuntimeError as e: 
             user_err(f"데이터 분석 중 오류 발생: {e}", details=agent.context)
             P["running"] = False; _set_step("idle"); return
        except Exception as e: 
             user_err(f"예상치 못한 분석 오류: {e}", details=agent.context)
             P["running"] = False; _set_step("idle"); return

        # 결과 저장 및 로그 기록
        P["exploration"] = agent.context.get("exploration")
        P["timeseries_data"] = agent.context.get('target_store_timeseries')
        logger.log("✔️ 3-1: 내부/외부 데이터 분석 완료", details=P["exploration"], expander_label="데이터 요약 결과 보기")
        
        P["pd_out"] = agent.context.get("problem_definition")
        if not P["pd_out"] or "error" in P["pd_out"]: # None 체크 추가
            user_err("AI가 문제점을 정의하는 데 실패했습니다.", details=P["pd_out"]); P["running"] = False; _set_step("idle"); return
        logger.log("✔️ 3-2: 핵심 문제점 정의 완료", details=P["pd_out"], expander_label="정의된 문제점 보기")

        report_md = agent._generate_strategy_and_report()
        if report_md.startswith("# 보고서 실패"):
             user_err("최종 보고서를 생성하는 데 실패했습니다.", details={"message": report_md})
        P["report_md"] = report_md
        logger.log("✔️ 3-3: 맞춤형 마케팅 전략 생성 완료")
        
        _set_step("done"); st.rerun()

    if P["step"] == "done":
        P["running"] = False
        _set_step("idle")
        st.rerun()

# --- 입력 및 제어 로직 ---

# 1. Logger 객체 생성
logger = StreamlitLogHandler()
user_query = st.chat_input("가맹점 이름이나 고민을 입력하세요.")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"): st.write(user_query)
    if st.session_state.pipeline.get("report_md"):
        with st.spinner("답변 생성 중..."): reply = handle_followup(user_query)
        st.session_state.messages.append(AIMessage(content=reply)); st.rerun()
    else:
        st.session_state.log_entries = []
        st.session_state.pipeline.update({"report_md": None, "exploration": None, "pd_out": None, "trans": None, "timeseries_data": None}) # timeseries_data 초기화 추가
        run_full_pipeline_resumable(user_text=user_query)

P = st.session_state.pipeline

if P.get("running"):
    with st.spinner("⏳ AI 에이전트가 분석 중입니다..."):
        st.write("---"); st.subheader("🤖 AI 에이전트 분석 과정"); log_container = st.container(border=True)
        logger.render(log_container)
        if not st.session_state.awaiting_candidate: run_full_pipeline_resumable()
else:
    if st.session_state.get("log_entries"):
        st.write("---"); st.subheader("🤖 AI 에이전트 분석 과정"); log_container = st.container(border=True)
        logger.render(log_container)

    if P.get("report_md") and not P.get("report_md").startswith("# 보고서 실패"):
        # ✅ [디버깅 추가] 데이터 존재 여부 확인
        pd_out_exists = P.get("pd_out") is not None
        ts_data_exists = P.get("timeseries_data") is not None and not P.get("timeseries_data").empty

        if pd_out_exists and ts_data_exists:
            kpi_list = P["pd_out"].get("kpis", [])
            if kpi_list:
                st.subheader("📊 주요 KPI 시계열 추이")
                charts = create_kpi_charts(kpi_list, P["timeseries_data"])
                if charts:
                    for chart in charts:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("선정된 KPI에 대한 시계열 그래프를 생성할 수 없습니다.") 
        # 최종 보고서 표시
        st.subheader("📘 최종 보고서 (상세)"); st.markdown(P["report_md"], unsafe_allow_html=True)

    elif P.get("report_md"): # 보고서 생성 실패 메시지 표시
         st.error(f"보고서 생성에 실패했습니다: {P['report_md']}")


if st.session_state.awaiting_candidate and st.session_state.candidates:
    st.info("이름이 유사한 가맹점이 여러 개 발견되었습니다. 아래에서 하나를 선택해주세요.")
    cands = st.session_state.candidates
    choice_idx = st.radio("분석할 가맹점을 선택하세요:", options=range(len(cands)), format_func=lambda i: _fmt_cand_with_id(cands[i]), label_visibility="collapsed")
    if st.button("선택 확정", type="primary"):
        agent.context['target_store_info'] = cands[choice_idx]
        if 'clarification_needed' in agent.context: del agent.context['clarification_needed']
        st.session_state.awaiting_candidate = False
        _set_step("analysis")
        st.rerun()
    st.stop()

if P.get("running") and P.get("cancel_requested"):
    P.update({"running": False, "step": "idle", "cancel_requested": False}); st.info("⏹️ 분석을 중지했습니다.")
    st.rerun()