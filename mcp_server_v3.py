import pandas as pd
from fastmcp.server import FastMCP
from typing import Dict, Any, List, Optional
from pathlib import Path
import io, base64, re, math

# --- matplotlib (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 전역 데이터 저장
DF: Optional[pd.DataFrame] = None

# MCP 서버 초기화
mcp = FastMCP(
    "MerchantSearchServer",
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

    3) render_kpi_chart(merchant_id: str, kpi_keys: List[str], title: str|None = None, normalize_0_1: bool = False)
        - ID로 시계열을 조회하고, 지정한 KPI 컬럼들을 선그래프로 그려 PNG(base64)로 반환합니다.
        - KPI 문자열(예: '34.5%', '0.345', '34.5')을 숫자로 변환 시도합니다.
        - 하나도 유효한 KPI가 없으면 사용 가능한 KPI 후보 목록을 함께 안내합니다.

    ## 데이터 컬럼 예시
    - 가맹점명, 업종, 상권, 기준년월, 가맹점주소
    - 매출금액 구간, 매출건수 구간, 객단가 구간, 배달매출금액 비율
    - 남성/여성 연령대 비중, 재방문/신규/거주/직장/유동 고객 비율
    - 동일 업종/상권 내 매출 순위 비율
    """
)

def _load_df(file_path: str | None = None):
    """CSV 로드 + 컬럼 정리 + 기준년월 변환(YYYYMM→datetime)"""
    global DF
    try:
        file_path = Path(file_path) if file_path else (Path(__file__).parent / "data" / "data.csv")
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
_load_df()  # ← 인자 없이, 파일은 스크립트/../data/data.csv 로 찾음


def _serialize_records(df: pd.DataFrame) -> list[dict]:
    """NaN/NaT → None 치환(직렬화 안전)"""
    return df.where(pd.notnull(df), None).to_dict(orient="records")

# -------- 공통 유틸: KPI 이름 후보 필터링/수치 변환 --------

_PERCENT_HINT = re.compile(r"%")
_NUMBER = re.compile(r"-?\d+(\.\d+)?")

def _to_number(val) -> Optional[float]:
    """문자/숫자 혼재 값을 float로 변환. '34.5%', '0.345', '34.5' 등 처리."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    # 퍼센트 기호가 있으면 퍼센트로 간주
    if _PERCENT_HINT.search(s):
        m = _NUMBER.search(s.replace(",", ""))
        if m:
            return float(m.group(0))  # 이미 % 값(0~100)로 본다
        return None
    # 일반 숫자 추출
    m = _NUMBER.search(s.replace(",", ""))
    if not m:
        return None
    num = float(m.group(0))
    return num

def _numeric_series_from_column(series: pd.Series, normalize_0_1: bool = False) -> pd.Series:
    """칼럼에서 수치 시리즈 생성. 퍼센트/소수/정수 혼재 허용."""
    vals = series.map(_to_number)
    # 만약 대부분이 0~1 범위면 퍼센트 스케일로 바꿔도 됨(옵션)
    if normalize_0_1:
        return vals
    return vals

def _available_kpi_candidates(df: pd.DataFrame) -> List[str]:
    """
    데이터셋에서 '비율', '비중', '순위', '객단가', '매출금액', '매출건수' 등
    수치로 변환이 가능해 보이는 컬럼 후보를 휴리스틱으로 제안.
    """
    name_like = ("비율", "비중", "순위", "객단가", "금액", "건수")
    cols = []
    for c in df.columns:
        if c == "기준년월":
            continue
        if any(tok in str(c) for tok in name_like):
            cols.append(c)
    # 중복 제거/정렬
    return sorted(set(cols))

# ----------------- MCP TOOLS -----------------

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

@mcp.tool()
def render_kpi_chart(
    merchant_id: str,
    kpi_keys: List[str],
    title: Optional[str] = None,
    normalize_0_1: bool = False
) -> Dict[str, Any]:
    """
    지정한 KPI 컬럼들로 라인차트를 그려 PNG(base64)로 반환합니다.

    파라미터:
      - merchant_id: 가맹점구분번호
      - kpi_keys: KPI 컬럼명 목록
      - title: (선택) 차트 제목
      - normalize_0_1: 대부분 0~1 범위일 때 스케일 유지용(기본 False)

    응답:
      - 성공: {"status":"success","image":{"format":"png","base64":"..."}, "used_kpis":[...]}
      - 실패: {"status":"error","message":"...", "available_kpis":[...]}  # 후보 제공
    """
    # --- 한글 폰트 설정 ---
    plt.rcParams["font.family"] = "Malgun Gothic"   # Windows
    plt.rcParams["axes.unicode_minus"] = False      # 마이너스 부호 깨짐 방지

    assert DF is not None
    if DF.empty:
        return {"status":"error","message":"서버에 데이터가 로드되지 않았습니다."}

    ts = DF[DF["가맹점구분번호"].astype(str) == str(merchant_id)].copy()
    if ts.empty:
        return {"status":"error","message":"해당 가맹점의 시계열 데이터가 없습니다."}

    if "기준년월" not in ts.columns:
        return {"status":"error","message":"'기준년월' 컬럼이 없어 시계열을 그릴 수 없습니다."}

    ts = ts.sort_values("기준년월")
    x = ts["기준년월"]

    # 유효 KPI만 필터링 & 수치화
    valid = []
    numeric_series = {}
    for k in (kpi_keys or [])[:3]:
        if k not in ts.columns:
            continue
        s = _numeric_series_from_column(ts[k], normalize_0_1=normalize_0_1)
        if s.notna().sum() >= 1:
            valid.append(k)
            numeric_series[k] = s

    if not valid:
        return {
            "status":"error",
            "message":"유효한 KPI가 없습니다. 존재하는 수치 KPI 중에서 선택하세요.",
            "available_kpis": _available_kpi_candidates(ts)
        }

    # --- 그리기 ---
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi = 120)
    for k in valid:
        ax.plot(x, numeric_series[k], label=k)  # 색상 자동 할당

    ax.set_xlabel("기준년월")
    ax.set_ylabel("값")
    ax.set_title(title or f"KPI Trend (merchant_id={merchant_id})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "status":"success",
        "image": {"format":"png", "base64": img_b64},
        "used_kpis": valid
    }

if __name__ == "__main__":
    mcp.run()