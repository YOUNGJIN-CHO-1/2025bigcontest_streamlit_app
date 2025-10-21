# streamlit_app_V1.py
# streamlit_app_v1.py
import os
import sys
import json
import time
import shutil
import asyncio
import streamlit as st
import google.generativeai as genai

from typing import Any, Dict
from pathlib import Path
from PIL import Image

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ---------------------------
# 환경설정: Google API Key
# ---------------------------
# GOOGLE_API_KEY = (
#     st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets
#     else os.environ.get("GOOGLE_API_KEY")
# )
# if not GOOGLE_API_KEY:
#     st.warning("⚠️ GOOGLE_API_KEY가 설정되지 않았습니다. st.secrets 또는 환경변수를 설정하세요.")
# genai.configure(api_key=GOOGLE_API_KEY)
GOOGLE_API_KEY = "AIzaSyB8R3nurDOohfAvKXSgBUVRkoliXtfnTKo"
genai.configure(api_key=GOOGLE_API_KEY)

# ---------------------------
# LLM (Gemini) 헬퍼
# ---------------------------
def call_gemini_json(system_prompt: str, user_payload: str) -> Dict[str, Any]:
    model = genai.GenerativeModel("gemini-2.5-flash")
    time.sleep(0.2)
    resp = model.generate_content(
        f"{system_prompt}\n\n{user_payload}",
        generation_config={"response_mime_type": "application/json"},
    )
    try:
        return json.loads(resp.text)
    except Exception as e:
        return {"error": f"JSON 파싱 실패: {e}", "raw": resp.text}

# ---------------------------
# 에이전트(추론/리포팅) 로직
# ---------------------------
class InteractiveParallelAgent:
    # 1) 변환
    _TRANSFORM_SYSTEM_PROMPT = """
당신은 사용자의 자연어 요청을 구조화된 데이터로 변환하는 전문가입니다.
사용자의 요청을 분석하여 다음 네 가지 JSON 항목으로 요약하세요.
- "target": 사용자가 전략을 요청한 가맹점명과 업종은 무엇인가요? (가맹점 이름. 예: '성우**')
- "challenge": 사용자가 직면한 어려움이나 해결해야 할 문제는 무엇인가요?
- "objective": 목표 달성을 위한 구체적인 목적은 무엇인가요?
- "solution_direction": 제안될 수 있는 해결책의 방향은 무엇인가요?
"""

    # 3) 문제 정의
    _DEFINE_PROBLEM_SYSTEM_PROMPT = """
당신은 데이터 분석가입니다. 제공된 가맹점의 시계열 데이터와 최초 사용자 요청을 기반으로, 문제 상황과 핵심 성과 지표(KPI)를 구체적으로 정의하세요.
특히, 시간의 흐름에 따른 데이터의 변화와 추세를 명확하게 파악하고, 이를 문제 정의에 반영하세요.
결과 JSON:
- "problem_statement": 시계열 변화에 기반한 문제 정의
- "kpis": 측정 가능한 핵심 지표 목록(문자열 배열)
"""

    # 4) 전략 제안
    _PROPOSE_STRATEGY_SYSTEM_PROMPT = """
당신은 비즈니스 전략 컨설턴트입니다.
문제 정의를 기반으로 구체적이고 실행 가능한 전략을 제안하세요.
시계열에서 감지된 추세를 반전/강화하는 방향으로 설계하세요.
결과 JSON:
- "problem_definition": 핵심 문제 요약
- "proposed_strategy": 실행 방안 목록
- "strategic_rationale": 왜 이것이 효과적인지 (데이터 추세와 연결)
"""

    # 5) 보고서 작성
    _GENERATE_REPORT_SYSTEM_PROMPT = """
당신은 전문 보고서 작성자입니다.
지금까지의 컨텍스트를 종합하여, 서론(문제 배경) - 본론(시계열 분석/문제정의/전략) - 결론(기대효과) 구조의 마크다운 보고서를 생성하세요.
최종 출력은 {"report": "<markdown>"} JSON 한 개 키만 포함해야 합니다.
"""

    def __init__(self):
        self.context: Dict[str, Any] = {}

    def transform(self, initial_input: str):
        user = f"🔹 사용자 입력: {initial_input}"
        self.context["transformation"] = call_gemini_json(
            self._TRANSFORM_SYSTEM_PROMPT, user
        )
        return self.context["transformation"]

    def define_problem(self, exploration: Dict[str, Any], transformation: Dict[str, Any]):
        user = "🔹 문맥:\n" + json.dumps(
            {"최초요청": transformation, "시계열": exploration}, ensure_ascii=False, indent=2
        )
        self.context["problem_definition"] = call_gemini_json(
            self._DEFINE_PROBLEM_SYSTEM_PROMPT, user
        )
        return self.context["problem_definition"]

    def propose_strategy(self, problem_definition: Dict[str, Any]):
        user = "🔹 문맥:\n" + json.dumps(
            {"문제정의": problem_definition}, ensure_ascii=False, indent=2
        )
        self.context["strategy"] = call_gemini_json(
            self._PROPOSE_STRATEGY_SYSTEM_PROMPT, user
        )
        return self.context["strategy"]

    def generate_report(self) -> str:
        user = "🔹 전체 컨텍스트:\n" + json.dumps(
            self.context, ensure_ascii=False, indent=2
        )
        out = call_gemini_json(self._GENERATE_REPORT_SYSTEM_PROMPT, user)
        return out.get("report", "보고서 생성에 실패했습니다.")

# ---------------------------
# MCP 호출 유틸 (필요 시마다 연결)
# ---------------------------
async def _mcp_call(tool_name: str, args: dict) -> dict:
    server_path = (Path(__file__).parent / "mcp_server_v2.py").resolve()
    if not server_path.exists():
        return {"error": f"MCP 서버 파일을 찾을 수 없습니다: {server_path}"}

    uv_path = shutil.which("uv")
    if uv_path:
        command = uv_path
        cmd_args = ["run", str(server_path)]
    else:
        command = sys.executable
        cmd_args = ["-u", str(server_path)]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    server_params = StdioServerParameters(command=command, args=cmd_args, env=env)

    try:
        async with asyncio.timeout(20):
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    tool_map = {t.name: t for t in tools}
                    if tool_name not in tool_map:
                        return {"error": f"툴 '{tool_name}'을(를) 찾을 수 없습니다.",
                                "available": list(tool_map.keys())}
                    result = await tool_map[tool_name].ainvoke(args)
                    if isinstance(result, str):
                        try:
                            return json.loads(result)
                        except Exception:
                            return {"raw": result}
                    return result
    except Exception as e:
        return {"error": f"MCP 호출 중 예외: {e.__class__.__name__}: {e}"}

def mcp_call(tool_name: str, args: dict) -> dict:
    try:
        return asyncio.run(_mcp_call(tool_name, args))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_mcp_call(tool_name, args))
        finally:
            loop.close()

# ---------------------------
# Streamlit UI
# ---------------------------
ASSETS = Path("assets")
st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사", layout="wide")
st.title("신한카드 소상공인 비밀상담소 🔑")

@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)

system_prompt = "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, 분석 결과를 바탕으로 적절한 마케팅 방법과 채널, 마케팅 메시지를 추천합니다. 결과는 짧고 간결하게, 분석 결과에는 가능한 표를 사용하여 알아보기 쉽게 설명해주세요."
greeting = "마케팅이 필요한 가맹점을 알려주세요  \n(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"

def clear_chat_history():
    st.session_state.messages = [{"role":"system","content":system_prompt},{"role":"assistant","content":greeting}]

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

# 사이드바
with st.sidebar:
    if (ASSETS / "shc_ci_basic_00.png").exists():
        st.image(load_image("shc_ci_basic_00.png"), use_container_width=True)
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    st.button('Clear Chat History', on_click=clear_chat_history, use_container_width=True)

# 세션 상태
if "stage" not in st.session_state:
    st.session_state.stage = "idle"  # idle → need_clarification → analysis → done
if "agent" not in st.session_state:
    st.session_state.agent = InteractiveParallelAgent()
if "exploration" not in st.session_state:
    st.session_state.exploration = None
if "transformation" not in st.session_state:
    st.session_state.transformation = None
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# 사용자 입력
user_query = st.chat_input("가맹점 이름이나 고민을 입력하세요 (예: 유유*, 본죽, 동대*)")
if user_query:
    st.session_state.user_query = user_query

    # 1) 사용자 의도 파악 (타겟 가맹점명 뽑기)
    with st.spinner("의도 파악 중..."):
        trans = st.session_state.agent.transform(user_query)
    st.json(trans, expanded=False)
    target = (trans or {}).get("target") or user_query

    # 2) 이름으로 후보 검색 (MCP)
    with st.spinner("가맹점 검색(MCP)..."):
        resp = mcp_call("search_merchant", {"store_name": target})

    st.write("🔎 MCP raw response:", resp)
    if "error" in resp:
        st.error(resp["error"])
        st.stop()

    status = resp.get("status")
    if status == "error":
        st.error(resp.get("message", "검색 실패"))
        st.session_state.stage = "idle"

    elif status == "single_candidate":
        cand = resp["candidate"]
        merchant_id = _id_of(cand)
        with st.spinner("시계열 조회(MCP by ID)..."):
            ts_resp = mcp_call("get_store_timeseries", {"merchant_id": str(merchant_id)})
        st.write("📈 timeseries resp:", ts_resp)
        if ts_resp.get("status") == "success":
            data = ts_resp.get("data", {})
            st.session_state.exploration = {
                "store_identity": data.get("store_identity"),
                "time_series_analysis_data": data.get("time_series_data"),
            }
            st.session_state.transformation = trans
            st.session_state.stage = "analysis"
            st.success("가맹점 확정 및 시계열 로드 완료.")
        else:
            st.error(ts_resp.get("message", "시계열 조회 실패"))
            st.session_state.stage = "idle"

    elif status == "clarification_needed":
        st.session_state.candidates = resp.get("candidates", [])
        st.session_state.stage = "need_clarification"
        st.info("이름이 유사한 가맹점이 여러 개입니다. 하나를 선택하세요.")

# 3) 후보 선택 단계
if st.session_state.stage == "need_clarification":
    cands = st.session_state.candidates
    if not cands:
        st.error("후보가 없습니다.")
        st.session_state.stage = "idle"
    else:
        idx = st.radio(
            "가맹점을 선택하세요",
            options=list(range(len(cands))),
            format_func=lambda i: _fmt_cand_with_id(cands[i]),
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("선택 확정"):
                sel = cands[idx]
                merchant_id = _id_of(sel)
                with st.spinner("시계열 조회(MCP by ID)..."):
                    resp2 = mcp_call("get_store_timeseries", {"merchant_id": str(merchant_id)})
                st.write("📈 timeseries resp:", resp2)
                if resp2.get("status") == "success":
                    data = resp2.get("data", {})
                    st.session_state.exploration = {
                        "store_identity": data.get("store_identity"),
                        "time_series_analysis_data": data.get("time_series_data"),
                    }
                    if st.session_state.transformation is None:
                        st.session_state.transformation = {"target": sel.get("가맹점명")}
                    st.session_state.stage = "analysis"
                    st.success("가맹점 확정 및 시계열 로드 완료.")
                else:
                    st.error(resp2.get("message", "시계열 조회 실패"))
                    st.session_state.stage = "idle"
        with col2:
            if st.button("선택 취소"):
                st.session_state.stage = "idle"
                st.session_state.candidates = []
                st.session_state.exploration = None

# 4) 분석/전략/리포트 단계
if st.session_state.stage == "analysis":
    agent = st.session_state.agent
    exploration = st.session_state.exploration
    transformation = st.session_state.transformation

    if not exploration or not transformation:
        st.error("분석에 필요한 컨텍스트가 누락되었습니다.")
        st.session_state.stage = "idle"
    else:
        st.subheader("① 가맹점 정보")
        st.json(exploration.get("store_identity"), expanded=False)

        ts = exploration.get("time_series_analysis_data") or []
        st.subheader("② 시계열 전체(2023~2024)")
        st.dataframe(ts)  # 전체를 먼저 보여줌

        # 최근 N개월만 요약해서 보는 옵션
        if len(ts) > 12:
            n = st.slider("최근 N개월만 보기", 6, min(36, len(ts)), 12)
            st.subheader(f"②-1 최근 {n}개월")
            st.dataframe(ts[-n:])

        with st.spinner("③ 문제 정의 생성(Gemini)..."):
            pd_out = agent.define_problem(exploration, transformation)
        st.subheader("③ 문제 정의")
        st.json(pd_out, expanded=False)

        with st.spinner("④ 전략 제안 생성(Gemini)..."):
            st_out = agent.propose_strategy(pd_out)
        st.subheader("④ 전략 제안")
        st.json(st_out, expanded=False)

        with st.spinner("⑤ 최종 보고서 생성(Gemini)..."):
            report_md = agent.generate_report()
        st.subheader("⑤ 최종 보고서")
        st.markdown(report_md)

        st.session_state.stage = "done"