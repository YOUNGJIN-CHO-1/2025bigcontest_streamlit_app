# mcp_server_v4.py
import pandas as pd
from fastmcp.server import FastMCP
from typing import Dict, Any, List, Optional
from pathlib import Path
import re

# =========================
# 전역 데이터프레임
# =========================
DF: Optional[pd.DataFrame] = None

# =========================
# MCP 서버 초기화
# =========================
mcp = FastMCP(
    "MerchantSearchServerV4",
    instructions="""
    신한카드 가맹점/시계열 데이터를 조회하는 MCP 서버입니다.

    ## 제공 툴
    1) search_merchant(store_name: str)
        - '*' 제거 후 부분 일치(대소문자 무시)로 가맹점을 검색합니다.
        - 결과는 3가지 상태 중 하나입니다:
            - {"status":"clarification_needed","candidates":[...]} : 후보 여러 개
            - {"status":"single_candidate","candidate":{...}}      : 후보 1개
            - {"status":"error","message":"..."}                   : 0개 또는 오류
        - 각 후보에는 반드시 "가맹점구분번호"가 포함됩니다. (향후 ID 기반 시계열 조회용)

    2) get_store_timeseries(merchant_id: str)
        - 가맹점구분번호(고유 ID)로 해당 가맹점의 전체 시계열(2023~2024)을 반환합니다.
        - 성공 시: {"status":"success","data":{store_identity, time_series_data}}
        - 실패 시: {"status":"error","message":"..."}

    ## 데이터 컬럼 예시
    - 가맹점명, 업종, 상권, 기준년월, 가맹점주소
    - 매출금액, 매출건수, 객단가, 배달매출금액 비율
    - 남성/여성 연령대 비중, 재방문/신규/거주/직장/유동 고객 비율
    - 동일 업종/상권 내 매출 순위 비율
    """
)

def _load_df(file_path: str | None = None):
    """CSV 로드 + 컬럼 정리 + 기준년월 변환(YYYYMM→datetime)"""
    global DF
    try:
        # ✅ 요구사항 반영: 'labeling_no_preprocessing.csv' 파일을 읽도록 경로 수정
        file_path = Path(file_path) if file_path else (Path(__file__).parent / "data" / "labeling_no_preprocessing.csv")
        DF = pd.read_csv(file_path)
        DF.columns = DF.columns.str.strip()

        # ---- 날짜 변환: YYYYMM → datetime(월초), 5자리 케이스까지 zfill(6) 보정 ----
        if "기준년월" in DF.columns:
            DF["기준년월"] = (
                DF["기준년월"]
                .astype(str)
                .str.replace(r"\D", "", regex=True)  # 숫자만 남기기
                .str.zfill(6)                        # '20231' → '202301'
            )
            DF["기준년월"] = pd.to_datetime(DF["기준년월"], format="%Y%m", errors="coerce")

        # ---- 고유 ID 컬럼 확정 ----
        if "가맹점구분번호" not in DF.columns:
            raise ValueError("데이터에 '가맹점구분번호' 컬럼이 없습니다.")

        print(f"✅ 데이터 로드 성공: {file_path} (rows={len(DF)})")

    except FileNotFoundError:
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        DF = pd.DataFrame()

# 서버 시작 시 데이터 로드
_load_df()

def _serialize_records(df: pd.DataFrame) -> list[dict]:
    """NaN/NaT → None 치환(직렬화 안전)"""
    return df.where(pd.notnull(df), None).to_dict(orient="records")

# =========================
# MCP TOOLS
# =========================
@mcp.tool()
def search_merchant(store_name: str) -> Dict[str, Any]:
    """
    '*' 제거 후 부분 일치(대소문자 무시)로 가맹점을 검색.
    - 후보에는 반드시 '가맹점구분번호' 포함 (이후 ID 기반 조회를 위해)
    """
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."
    if DF.empty:
        return {"status": "error", "message": "서버에 데이터가 로드되지 않았습니다."}

    search_term = (store_name or "").replace("*", "")
    hits = DF[DF["가맹점명"].astype(str).str.contains(
        re.escape(search_term), case=False, na=False, regex=True
    )]

    if hits.empty:
        return {"status":"error","message":f"'{search_term}'로 검색된 가맹점이 없습니다."}

    # 후보: (가맹점구분번호, 가맹점명, 업종, 상권, 주소)
    cand_cols = ["가맹점구분번호","가맹점명","업종","상권","가맹점주소"]
    cand_cols = [c for c in cand_cols if c in DF.columns]
    cands = hits[cand_cols].drop_duplicates().reset_index(drop=True)
    candidates = _serialize_records(cands)

    if len(candidates) == 1:
        return {"status":"single_candidate","candidate":candidates[0]}

    return {"status":"clarification_needed","candidates":candidates}

@mcp.tool()
def get_store_timeseries(merchant_id: str) -> Dict[str, Any]:
    """가맹점구분번호로 전체 시계열(2023~2024)을 반환"""
    assert DF is not None
    if DF.empty:
        return {"status":"error","message":"서버에 데이터가 로드되지 않았습니다."}

    ts = DF[DF["가맹점구분번호"].astype(str) == str(merchant_id)].copy()
    if ts.empty:
        return {"status":"error","message":"해당 가맹점의 시계열 데이터가 없습니다."}

    if "기준년월" in ts.columns:
        ts = ts.sort_values("기준년월")

    first = ts.iloc[0]
    payload = {
        "store_identity": {
            "id": first.get("가맹점구분번호"),
            "name": first.get("가맹점명"),
            "address": first.get("가맹점주소"),
            "industry": first.get("업종"),
            "commercial_area": first.get("상권"),
        },
        "time_series_data": _serialize_records(ts),
    }
    return {"status":"success","data":payload}


if __name__ == "__main__":
    mcp.run()
