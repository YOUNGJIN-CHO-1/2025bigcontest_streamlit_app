import pandas as pd
from fastmcp.server import FastMCP
from typing import Dict, Any, Optional
from pathlib import Path

# 전역 데이터 저장
DF: Optional[pd.DataFrame] = None

# MCP 서버 초기화
mcp = FastMCP(
    "MerchantSearchServer",
    instructions="""
    신한카드 가맹점/시계열 데이터를 조회하는 MCP 서버입니다.

    ## 제공 툴
    1) search_merchant(store_name: str)
       - 부분 일치(대소문자 무시)로 가맹점을 검색합니다.
       - 결과는 3가지 상태 중 하나입니다:
         - {"status":"success","data":{...}} : 정확히 하나의 가맹점이 식별되어 시계열까지 반환
         - {"status":"clarification_needed","candidates":[...]} : 후보 여러 개
         - {"status":"error","message":"..."} : 0개 또는 오류

    2) get_store_timeseries(name: str, industry: str, area: str)
       - 특정 가맹점(이름/업종/상권으로 식별)의 시계열 데이터를 반환합니다.
       - 반환 형식은 search_merchant 성공 케이스와 동일합니다.

    ## 데이터 컬럼 예시
    - 가맹점명, 업종, 상권, 기준년월, 가맹점주소
    - 매출금액 구간, 매출건수 구간, 객단가 구간, 배달매출금액 비율
    - 남성/여성 연령대 비중, 재방문/신규/거주/직장/유동 고객 비율
    - 동일 업종/상권 내 매출 순위 비율
    """
)

def _load_df(file_path: str | None = None):
    global DF
    try:
        if file_path is None:
            file_path = Path(__file__).parent / "data" / "data.csv"
        else:
            file_path = Path(file_path)
        DF = pd.read_csv(file_path)
        DF.columns = DF.columns.str.strip()
        if "기준년월" in DF.columns:
            DF["기준년월"] = pd.to_datetime(DF["기준년월"], errors="coerce")
        print(f"✅ 데이터 로드 성공: {file_path} (rows={len(DF)})")
    except FileNotFoundError:
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        DF = pd.DataFrame()

# 서버 시작 시 데이터 로드
_load_df()  # ← 인자 없이, 파일은 스크립트/../data/data.csv 로 찾음


def _serialize_records(df: pd.DataFrame) -> list[dict]:
    """NaN/NaT를 None으로 치환해 JSON 직렬화 호환 형태로 변환."""
    cleaned = df.where(pd.notnull(df), None)
    return cleaned.to_dict(orient="records")

@mcp.tool()
def search_merchant(store_name: str) -> Dict[str, Any]:
    """
    부분 일치(대소문자 무시)로 가맹점을 검색합니다.
    - 성공: {"status":"success","data":{...}}
    - 후보: {"status":"clarification_needed","candidates":[...]}
    - 오류: {"status":"error","message":"..."}
    """
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."
    if DF.empty:
        return {"status": "error", "message": "서버에 데이터가 로드되지 않았습니다."}

    # '*' 제거 후 부분 일치(대소문자 무시)
    search_term = (store_name or "").replace("*", "")
    candidates_df = DF[DF["가맹점명"].astype(str).str.contains(search_term, case=False, na=False)]

    # 고유 후보 (가맹점명, 업종, 상권)
    if not {"가맹점명", "업종", "상권"}.issubset(candidates_df.columns):
        return {"status":"error","message":"필수 컬럼(가맹점명/업종/상권)이 없습니다."}

    unique_candidates = (
        candidates_df[["가맹점명", "업종", "상권"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    if len(unique_candidates) == 0:
        return {"status": "error", "message": f"'{search_term}'(을)를 포함하는 가맹점을 찾을 수 없습니다."}

    if len(unique_candidates) > 1:
        return {"status": "clarification_needed", "candidates": unique_candidates}

    # 정확히 하나라면 시계열까지 반환
    sel = unique_candidates[0]
    mask = (
        (DF["가맹점명"] == sel["가맹점명"])
        & (DF["업종"] == sel["업종"])
        & (DF["상권"] == sel["상권"])
    )
    ts = DF[mask].copy()
    if "기준년월" in ts.columns:
        ts = ts.sort_values(by="기준년월")

    if ts.empty:
        return {"status": "error", "message": "가맹점은 찾았으나 시계열 데이터가 없습니다."}

    first = ts.iloc[0]
    payload = {
        "store_identity": {
            "name": first.get("가맹점명"),
            "address": first.get("가맹점주소"),
            "industry": first.get("업종"),
            "commercial_area": first.get("상권"),
        },
        "time_series_data": _serialize_records(ts),
    }
    return {"status": "success", "data": payload}

@mcp.tool()
def get_store_timeseries(name: str, industry: str, area: str) -> Dict[str, Any]:
    """
    (가맹점명/업종/상권)으로 특정 가맹점의 시계열을 반환합니다.
    성공 시: {"status":"success","data":{...}} / 실패 시: {"status":"error","message":"..."}
    """
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."
    if DF.empty:
        return {"status": "error", "message": "서버에 데이터가 로드되지 않았습니다."}

    mask = (
        (DF["가맹점명"] == name)
        & (DF["업종"] == industry)
        & (DF["상권"] == area)
    )
    ts = DF[mask].copy()
    if "기준년월" in ts.columns:
        ts = ts.sort_values(by="기준년월")

    if ts.empty:
        return {"status": "error", "message": "해당 조건에 맞는 시계열 데이터가 없습니다."}

    first = ts.iloc[0]
    payload = {
        "store_identity": {
            "name": first.get("가맹점명"),
            "address": first.get("가맹점주소"),
            "industry": first.get("업종"),
            "commercial_area": first.get("상권"),
        },
        "time_series_data": _serialize_records(ts),
    }
    return {"status": "success", "data": payload}

if __name__ == "__main__":
    mcp.run()