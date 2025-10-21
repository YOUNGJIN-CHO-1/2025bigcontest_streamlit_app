import streamlit as st
import asyncio

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from PIL import Image
from pathlib import Path

# 환경변수
ASSETS = Path("assets")
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_API_KEY = "AIzaSyB8R3nurDOohfAvKXSgBUVRkoliXtfnTKo"  # Replace with your actual API key

system_prompt = "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, 분석 결과를 바탕으로 적절한 마케팅 방법과 채널, 마케팅 메시지를 추천합니다. 결과는 짧고 간결하게, 분석 결과에는 가능한 표를 사용하여 알아보기 쉽게 설명해주세요."
greeting = "마케팅이 필요한 가맹점을 알려주세요  \n(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"

# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사")

def clear_chat_history():
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])  # 비율 조정 가능
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더
st.title("신한카드 소상공인 비밀상담소 🔑")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 #비밀상담사")
st.image(load_image("KMWL.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

# 초기 메시지 화면 표시 (히스토리 렌더 루프)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content.replace("<br>", "  \n"))

# LLM 모델 선택
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # 최신 Gemini 2.5 Flash 모델
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

# MCP 서버 파라미터(환경에 맞게 명령 수정)
server_params = StdioServerParameters(
    command="uv",
    args=["run","mcp_server_baseline.py"],
    env=None
)

# 사용자 입력 처리
async def process_user_input():
    """사용자 입력을 처리하는 async 함수"""
    async with stdio_client(server_params) as (read, write):
        # 스트림으로 ClientSession을 만들고
        async with ClientSession(read, write) as session:
            # 세션을 initialize 한다
            await session.initialize()

            # MCP 툴 로드
            tools = await load_mcp_tools(session)

            # 에이전트 생성
            agent = create_react_agent(llm, tools)

            # 에이전트에 전체 대화 히스토리 전달
            agent_response = await agent.ainvoke({"messages": st.session_state.messages})
            
            # AI 응답을 대화 히스토리에 추가
            ai_message = agent_response["messages"][-1]  # 마지막 메시지가 AI 응답

            return ai_message.content
            

# 사용자 입력 창
if query := st.chat_input("가맹점 이름을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=query))
    render_chat_message("user", query)

    with st.spinner("Thinking..."):
        try:
            # 사용자 입력 처리
            reply = asyncio.run(process_user_input())
            st.session_state.messages.append(AIMessage(content=reply))
            render_chat_message("assistant", reply)
        except* Exception as eg:
            # 오류 처리
            for i, exc in enumerate(eg.exceptions, 1):
                error_msg = f"오류가 발생했습니다 #{i}: {exc!r}"
                st.session_state.messages.append(AIMessage(content=error_msg))
                render_chat_message("assistant", error_msg)
