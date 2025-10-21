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
import re, difflib

from typing import Any, Dict, List
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
- "challenge": 사용자가 직면한 어려움이나 해결해야 할 핵심 문제는 무엇인가요?
- "objective": 목표 달성을 위한 구체적인 목표은 무엇인가요?
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
    
    # ⬇️ InteractiveParallelAgent 안에 유틸 하나 추가
    def _compact_context(self, full_ctx: Dict[str, Any], max_rows: int = 12) -> Dict[str, Any]:
        """LLM에 넘길 컨텍스트를 슬림화: 최근 N개월만 포함"""
        slim = {}
        # 1) 변환/문제정의/전략만 유지
        for k in ("transformation", "problem_definition", "strategy"):
            if k in full_ctx:
                slim[k] = full_ctx[k]
        # 2) 시계열은 최근 N개월만
        exp = full_ctx.get("exploration") or {}
        ts = (exp.get("time_series_analysis_data") or [])[-max_rows:]
        slim["exploration_summary"] = {
            "store_identity": exp.get("store_identity"),
            "time_series_tail": ts,
            "tail_note": f"최근 {min(max_rows, len(ts))}개월만 포함",
        }
        return slim

    def generate_report(self) -> str:
        user = "🔹 전체 컨텍스트:\n" + json.dumps(
            self.context, ensure_ascii=False, indent=2
        )
        out = call_gemini_json(self._GENERATE_REPORT_SYSTEM_PROMPT, user)
        return out.get("report", "보고서 생성에 실패했습니다.")

    # ⬇️ generate_report() 교체
    def generate_report(self) -> str:
        # self.context에 exploration을 미리 넣어두었다는 전제(이미 run_full_pipeline에서 그렇게 사용 중)
        compact = self._compact_context(self.context, max_rows=12)

        user = "🔹 전체 컨텍스트(슬림):\n" + json.dumps(compact, ensure_ascii=False, indent=2)
        out = call_gemini_json(self._GENERATE_REPORT_SYSTEM_PROMPT, user)

        # 1차: 정상 키 사용
        if isinstance(out, dict) and out.get("report"):
            return out["report"]

        # 2차: 응답 복구(모델이 report 키를 놓친 경우)
        raw = out.get("raw") if isinstance(out, dict) else None
        if raw:
            # report 키만 강제 래핑해서 사용
            return raw if raw.strip().startswith("#") else f"# 요약 보고서\n\n{raw}"

        return "보고서 생성에 실패했습니다."

# ---------------------------
# MCP 호출 유틸 (필요 시마다 연결)
# ---------------------------
async def _mcp_call(tool_name: str, args: dict) -> dict:
    server_path = (Path(__file__).parent / "mcp_server_v3.py").resolve()
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

system_prompt = "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, 분석 결과를 바탕으로 적절한 마케팅 방법과 채널, 마케팅 메시지를 추천합니다. 결과는 짧고 간결하게, 분석 결과에는 가능한 표를 사용하여 알아보기 쉽게 설명해주세요."
greeting = "마케팅이 필요한 가맹점을 알려주세요."

@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)

def clear_chat_history():
    # 채팅/상태 전부 초기화
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

# 세션 상태
# --- session state ---
if "messages" not in st.session_state:
    st.session_state.messages: List[Any] = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting),
    ]
if "agent" not in st.session_state:
    st.session_state.agent = InteractiveParallelAgent()
if "exploration" not in st.session_state:
    st.session_state.exploration = None
if "transformation" not in st.session_state:
    st.session_state.transformation = None
if "candidates" not in st.session_state:
    st.session_state.candidates: List[dict] = []
if "awaiting_candidate" not in st.session_state:
    st.session_state.awaiting_candidate = False   # 후보 선택 단계인지
if "selected_merchant_id" not in st.session_state:
    st.session_state.selected_merchant_id = None
if "last_chart_b64" not in st.session_state:
    st.session_state.last_chart_b64 = None

with st.sidebar:
    if (ASSETS / "shc_ci_basic_00.png").exists():
        st.image(load_image("shc_ci_basic_00.png"), width="stretch")
    st.markdown("<p style='text-align: center;'>2025 Big Contest • AI DATA 활용</p>", unsafe_allow_html=True)
    st.button("Clear Chat History", on_click=clear_chat_history)  # 버튼 추가

    # 최근 자동 생성된 차트
    if st.session_state.last_chart_b64:
        st.markdown("#### 최근 KPI 차트")
        st.image(base64.b64decode(st.session_state.last_chart_b64), width="stretch")

def render_history():
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
        elif isinstance(msg, SystemMessage):
            # 시스템 메시지는 화면엔 안 보이게 하려면 주석
            pass

render_history()

def _normalize_kpi_name(name: str) -> str:
    """괄호 뒤 설명 제거, 공백 정리, 대표 치환 등."""
    if not name:
        return ""
    s = str(name).strip()
    # 뒤쪽 괄호 설명 제거: "KPI명 (설명)" → "KPI명"
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    # 흔한 동의어/오타 치환
    synonyms = {
        "동일 업종 매출금액 비율": "동일 업종 내 매출 금액 비율",
        "동일 업종 매출 금액 비율": "동일 업종 내 매출 금액 비율",
        "동일 업종 매출건수 비율": "동일 업종 내 매출 건수 비율",
        "동일 업종 매출 건수 비율": "동일 업종 내 매출 건수 비율",
        "동일 상권 매출 순위 비율": "동일 상권 내 매출 순위 비율",
        "유니크 고객 수 구간": "유니크 고객 수 구간",  # 실제 컬럼 있으면 서버 available_kpis에 나옵니다
        "취소율 구간": "취소율 구간",
    }
    s = synonyms.get(s, s)
    # 불필요한 이중 공백 제거
    s = re.sub(r"\s+", " ", s)
    return s

def _map_kpis_to_available(requested: list[str], available: list[str]) -> list[str]:
    """
    요청 KPI 리스트를 available_kpis로 매핑.
    - 1차: 정규화 후 완전일치
    - 2차: 부분 포함(양방향)
    - 3차: 유사도 매칭(difflib)
    """
    if not requested:
        return []
    req_norm = [_normalize_kpi_name(k) for k in requested if k]
    avail_norm = [re.sub(r"\s+", " ", str(a).strip()) for a in available]

    chosen = []
    def pick(candidate: str):
        if candidate and candidate not in chosen:
            chosen.append(candidate)

    # 1) 완전 일치
    for r in req_norm:
        for a in avail_norm:
            if r == a:
                pick(a)

    # 2) 부분 포함 (r in a 또는 a in r)
    for r in req_norm:
        if any(r == c for c in chosen):
            continue
        hits = [a for a in avail_norm if r in a or a in r]
        if hits:
            pick(hits[0])

    # 3) 유사도 기반
    for r in req_norm:
        if any(r == c or r in c or c in r for c in chosen):
            continue
        near = difflib.get_close_matches(r, avail_norm, n=1, cutoff=0.6)
        if near:
            pick(near[0])

    return chosen

def render_timeseries(exploration: dict, merchant_id: str | None = None):
    """시계열 표 전체 + 최근 N개월 요약을 보여주는 공용 섹션."""
    ts = exploration.get("time_series_analysis_data") or []
    if not ts:
        st.info("시계열 데이터가 없습니다.")
        return

    df = pd.DataFrame(ts)

    # 기준년월 컬럼이 문자열/타입 혼재일 수 있으니 안전 정렬
    if "기준년월" in df.columns:
        try:
            df["기준년월"] = pd.to_datetime(df["기준년월"])
        except Exception:
            pass
        df = df.sort_values("기준년월")

    st.subheader("② 시계열 전체(2023~2024)")
    st.dataframe(df, width="stretch")  # 최신 streamlit 권장 파라미터

    # 최근 N개월만 요약
    if len(df) > 12:
        slider_key = f"ts_slider_{merchant_id or 'na'}"
        n = st.slider("최근 N개월만 보기", min_value=6, max_value=min(36, len(df)), value=12, key=slider_key)
        st.subheader(f"②-1 최근 {n}개월")
        st.dataframe(df.tail(n), width="stretch")


# --- 한 번에 전체 파이프라인 실행 (후보가 1개 또는 이미 선택된 경우) ---
def run_full_pipeline(user_text: str, merchant_id: str | None = None):
    agent = st.session_state.agent

    # 1) 변환
    trans = agent.transform(user_text)
    st.session_state.transformation = trans

    # 2) 후보 검색(merchant_id 미지정이면)
    if merchant_id is None:
        target = (trans or {}).get("target") or user_text
        resp = mcp_call("search_merchant", {"store_name": target})
        st.write("🔎 MCP search_merchant:", resp)

        if "error" in resp:
            ai_out = f"가맹점 검색 중 오류: {resp['error']}"
            st.session_state.messages.append(AIMessage(content=ai_out))
            with st.chat_message("assistant"):
                st.error(ai_out)
            return

        elif resp.get("status") == "clarification_needed":
            st.session_state.candidates = resp.get("candidates", [])
            st.session_state.awaiting_candidate = True
            # 채팅 리스트를 추가로 쓰지 않음 (중복 방지)
            return


        elif resp.get("status") == "single_candidate":
            merchant_id = resp["candidate"]["가맹점구분번호"]
        else:
            st.session_state.messages.append(AIMessage(content="검색 실패 또는 알 수 없는 상태입니다."))
            with st.chat_message("assistant"):
                st.error("검색 실패 또는 알 수 없는 상태입니다.")
            return

    # 3) 시계열 조회
    ts_resp = mcp_call("get_store_timeseries", {"merchant_id": str(merchant_id)})
    st.write("📈 get_store_timeseries:", ts_resp)
    if ts_resp.get("status") != "success":
        ai_out = ts_resp.get("message", "시계열 조회 실패")
        st.session_state.messages.append(AIMessage(content=ai_out))
        with st.chat_message("assistant"):
            st.error(ai_out)
        return

    data = ts_resp["data"]
    exploration = {
        "store_identity": data.get("store_identity"),
        "time_series_analysis_data": data.get("time_series_data"),
    }
    st.session_state.exploration = exploration
    # ⬇️ 에이전트 컨텍스트에도 저장 (generate_report가 _compact_context에서 읽습니다)
    st.session_state.agent.context["exploration"] = exploration
    st.session_state.selected_merchant_id = merchant_id
    render_timeseries(exploration, merchant_id=merchant_id)

    # 4) 문제 정의
    pd_out = agent.define_problem(exploration, st.session_state.transformation)
    # 5) 전략 제안
    st_out = agent.propose_strategy(pd_out)

    # 6) KPI 자동 차트 생성 (버튼 없이 자동)
    kpis_from_llm: List[str] = []
    if isinstance(pd_out, dict):
        kpis_from_llm = pd_out.get("kpis") or []
    chart_b64 = None
    used_kpis_shown = None

    if merchant_id and kpis_from_llm:
        # 1차: 그대로 시도
        chart_resp = mcp_call(
            "render_kpi_chart",
            {
                "merchant_id": str(merchant_id),
                "kpi_keys": kpis_from_llm,
                "title": f"KPI Trend – {exploration['store_identity'].get('name')}",
                "normalize_0_1": False,
            },
        )
        st.write("🖼️ render_kpi_chart (1st):", chart_resp)

        # 실패 시: available_kpis 받아서 매핑 후 2차 시도
        if chart_resp.get("status") != "success":
            avail = chart_resp.get("available_kpis", []) or []
            mapped = _map_kpis_to_available(kpis_from_llm, avail)
            # 너무 많으면 3~5개 정도로 제한(가독성)
            mapped = mapped[:5]
            if mapped:
                chart_resp = mcp_call(
                    "render_kpi_chart",
                    {
                        "merchant_id": str(merchant_id),
                        "kpi_keys": mapped,
                        "title": f"KPI Trend – {exploration['store_identity'].get('name')}",
                        "normalize_0_1": False,
                    },
                )
                st.write("🖼️ render_kpi_chart (2nd, mapped):", chart_resp)

        if chart_resp.get("status") == "success":
            chart_b64 = (chart_resp.get("image") or {}).get("base64")
            used_kpis_shown = chart_resp.get("used_kpis", [])
            st.session_state.last_chart_b64 = chart_b64

    # 7) 최종 보고서
    report_md = agent.generate_report()

    if not report_md or "보고서 생성에 실패" in report_md:
        si = exploration.get("store_identity", {})
        fallback = [
            f"# {si.get('name','가맹점')} 컨설팅 요약",
            "## 문제 정의",
            "```json",
            json.dumps(pd_out, ensure_ascii=False, indent=2),
            "```",
            "## 전략 제안",
            "```json",
            json.dumps(st_out, ensure_ascii=False, indent=2),
            "```",
            "## 기대 효과",
            "- 매출/방문 지표 개선 기대",
            "- 신규 유입 및 재방문 확대",
            "- 고객층 세분화 기반 메시지 정교화",
        ]
        report_md = "\n".join(fallback)

    # --- 응답을 "채팅 메시지"로 구성해 히스토리에 남김 ---
    parts = []
    si = exploration["store_identity"] or {}
    parts.append(f"**가맹점:** {si.get('name')} / {si.get('industry')} / {si.get('commercial_area')}")
    parts.append("### 1) 문제 정의")
    parts.append("```json\n" + json.dumps(pd_out, ensure_ascii=False, indent=2) + "\n```")
    parts.append("### 2) 전략 제안")
    parts.append("```json\n" + json.dumps(st_out, ensure_ascii=False, indent=2) + "\n```")
    if kpis_from_llm:
        parts.append("**자동 선택된 KPI:** " + ", ".join(kpis_from_llm))
    parts.append("### 3) 최종 보고서 (요약 보기)")
    # 보고서는 길 수 있으니 일부만
    parts.append(report_md[:1800] + ("..." if len(report_md) > 1800 else ""))
    ai_reply = "\n\n".join(parts)

    st.session_state.messages.append(AIMessage(content=ai_reply))
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
        if chart_b64:
            st.image(base64.b64decode(chart_b64), width="stretch")
            st.caption("자동 생성 KPI 추세 차트")
            if used_kpis_shown:
                st.caption("사용된 KPI: " + ", ".join(map(str, used_kpis_shown)))

# --- 입력창 ---
user_query = st.chat_input("가맹점 이름이나 고민을 입력하세요.")
if user_query:
    st.session_state.user_query = user_query
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # 이미 후보 선택 단계면 새 검색은 건너뛰고, 아래 후보 UI가 렌더되도록 둡니다.
    if not (st.session_state.awaiting_candidate and st.session_state.candidates):
        # 일반 검색 → 전체 파이프라인 실행 (필요하면 이 안에서 awaiting_candidate=True로 전환됨)
        run_full_pipeline(user_query, merchant_id=None)

# --- 후보 선택 UI (여기 한 곳에서만 출력) ---
if st.session_state.awaiting_candidate and st.session_state.candidates:
    st.info("이름이 유사한 가맹점이 여러 개입니다. 아래에서 하나를 선택하세요.")

    cands = st.session_state.candidates
    idx = st.radio(
        "가맹점을 선택하세요",
        options=list(range(len(cands))),
        format_func=lambda i: _fmt_cand_with_id(cands[i]),
        key="cand_radio",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("선택 확정", key="cand_confirm"):
            sel = cands[idx]
            picked_id = str(sel.get("가맹점구분번호"))
            st.session_state.awaiting_candidate = False
            # 선택 즉시 전체 파이프라인 재개 (시계열 표 → 문제정의/전략/KPI/보고서)
            run_full_pipeline(st.session_state.user_query or sel.get("가맹점명", ""), merchant_id=picked_id)

    with col2:
        if st.button("선택 취소", key="cand_cancel"):
            st.session_state.awaiting_candidate = False
            st.session_state.candidates = []
            st.session_state.exploration = None