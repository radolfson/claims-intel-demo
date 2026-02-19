import os
from datetime import date
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Snowflake (key-pair auth)
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Claims Intelligence – Daily Summary", layout="wide")


# ============================================================
# SETTINGS
# ============================================================
def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read Streamlit Community Cloud secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def get_data_source() -> str:
    return str(get_setting("DATA_SOURCE", "snowflake")).lower().strip()


SOURCE = get_data_source()

# Snowflake objects
SF_DAILY_VIEW = get_setting("SF_DAILY_VIEW", "NARS.PROCESSED.V_FEATURE_DAILY_STATUS")
SF_DETAIL_VIEW = get_setting("SF_DETAIL_VIEW", "NARS.PROCESSED.V_TRANSPORTATION_CLAIMS_DETAIL")

DETAIL_DAYS_BACK = int(get_setting("DETAIL_DAYS_BACK", "180"))

# Optional local CSV fallback (only if you ever intentionally use csv)
DATA_PATH = os.getenv("DEMO_DATA_PATH", "demo_features_latest.csv")


# ============================================================
# UI HELPERS
# ============================================================
def fmt_money(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "$0"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}${x/1_000_000_000:,.2f}B"
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:,.1f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:,.1f}K"
    return f"{sign}${x:,.0f}"


def to_date_safe(v) -> Optional[date]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Force expected column names/types for the daily KPI dataset."""
    if df is None or df.empty:
        return df

    df = df.rename(columns={c: c.upper() for c in df.columns})

    required = ["REPORTED_DATE", "TOTAL_CLAIMS", "OPEN_CLAIMS", "PAID", "OUTSTANDING", "INCURRED"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Daily view missing expected columns: {missing}. Found: {list(df.columns)}. "
            f"Check SF_DAILY_VIEW."
        )

    df["REPORTED_DATE"] = pd.to_datetime(df["REPORTED_DATE"], errors="coerce").dt.date

    df["TOTAL_CLAIMS"] = pd.to_numeric(df["TOTAL_CLAIMS"], errors="coerce").fillna(0).astype(int)
    df["OPEN_CLAIMS"] = pd.to_numeric(df["OPEN_CLAIMS"], errors="coerce").fillna(0).astype(int)

    for c in ["PAID", "OUTSTANDING", "INCURRED"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "METRIC_NOTE" not in df.columns:
        df["METRIC_NOTE"] = None

    return df.sort_values("REPORTED_DATE")


def normalize_detail(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.rename(columns={c: c.upper() for c in df.columns})

    # Normalize candidate date cols if present
    for dcol in ["REPORTED_DATE", "REPORT_DATE", "LOSS_DATE", "CLOSED_DATE"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date

    # Normalize open flag columns if present
    # Snowflake view has IS_OPEN/IS_CLOSED as NUMBER based on your screenshot.
    for col in ["IS_OPEN", "IS_CLOSED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================
def _load_private_key_der() -> bytes:
    # Your secrets example used "private_key_pem".
    pkey = get_setting("SNOWFLAKE_PRIVATE_KEY") or get_setting("private_key_pem")
    if not pkey:
        raise ValueError("Missing SNOWFLAKE_PRIVATE_KEY (or private_key_pem) in secrets/env.")

    if "BEGIN" not in pkey:
        # If user stored base64, decode it
        import base64
        pkey = base64.b64decode(pkey).decode("utf-8")

    key = serialization.load_pem_private_key(
        pkey.encode("utf-8"),
        password=None,
        backend=default_backend(),
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def sf_connect():
    account = get_setting("SNOWFLAKE_ACCOUNT")
    user = get_setting("SNOWFLAKE_USER")
    warehouse = get_setting("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database = get_setting("SNOWFLAKE_DATABASE", "NARS")
    schema = get_setting("SNOWFLAKE_SCHEMA", "PROCESSED")
    role = get_setting("SNOWFLAKE_ROLE")

    if not account or not user:
        raise ValueError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in secrets/env.")

    pkb = _load_private_key_der()

    kwargs = dict(
        account=account,
        user=user,
        private_key=pkb,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )
    if role:
        kwargs["role"] = role

    return snowflake.connector.connect(**kwargs)


@st.cache_data(show_spinner=False, ttl=900)
def sf_read(sql: str) -> pd.DataFrame:
    with sf_connect() as conn:
        return pd.read_sql(sql, conn)


def _parse_fqn(obj_name: str, default_db: str, default_schema: str) -> Tuple[str, str, str]:
    parts = [p.strip().strip('"') for p in obj_name.split(".") if p.strip()]
    if len(parts) == 1:
        return default_db, default_schema, parts[0]
    if len(parts) == 2:
        return default_db, parts[0], parts[1]
    return parts[0], parts[1], parts[2]


@st.cache_data(show_spinner=False, ttl=900)
def _columns_for_object(obj_name: str, default_db: str, default_schema: str) -> List[str]:
    db, schema, name = _parse_fqn(obj_name, default_db, default_schema)
    sql = f"""
    SELECT UPPER(column_name) AS column_name
    FROM {db}.INFORMATION_SCHEMA.COLUMNS
    WHERE UPPER(table_schema) = UPPER('{schema}')
      AND UPPER(table_name)   = UPPER('{name}')
    ORDER BY ordinal_position
    """
    df = sf_read(sql)
    return df["COLUMN_NAME"].tolist() if not df.empty else []


def pick_detail_date_column(obj_name: str, default_db: str, default_schema: str) -> str:
    cols = set(_columns_for_object(obj_name, default_db, default_schema))
    for c in ["REPORTED_DATE", "REPORT_DATE", "LOSS_DATE", "REPORT_DATE_DAY"]:
        if c in cols:
            return c
    raise ValueError(f"Could not find a usable date column in {obj_name}. Found: {sorted(cols)}")


def load_snowflake() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[date]]]:
    # Daily KPI
    daily_sql = f"""
    SELECT
      REPORTED_DATE,
      TOTAL_CLAIMS,
      OPEN_CLAIMS,
      PAID,
      OUTSTANDING,
      INCURRED,
      METRIC_NOTE
    FROM {SF_DAILY_VIEW}
    ORDER BY REPORTED_DATE;
    """
    daily = normalize_daily(sf_read(daily_sql))

    if daily.empty:
        return daily, pd.DataFrame(), {"daily_max": None, "detail_max": None, "detail_used_asof": None}

    daily_max = to_date_safe(daily["REPORTED_DATE"].max())

    # Detail
    default_db = get_setting("SNOWFLAKE_DATABASE", "NARS")
    default_schema = get_setting("SNOWFLAKE_SCHEMA", "PROCESSED")
    detail_date_col = pick_detail_date_column(SF_DETAIL_VIEW, default_db, default_schema)

    # Determine freshness of detail dataset
    detail_max_sql = f"SELECT MAX({detail_date_col}) AS MAX_D FROM {SF_DETAIL_VIEW};"
    detail_max_df = sf_read(detail_max_sql)
    detail_max = to_date_safe(detail_max_df.iloc[0]["MAX_D"]) if not detail_max_df.empty else None

    # If detail is stale, use its max date and warn loudly (politely, but still)
    detail_used_asof = daily_max
    if detail_max and daily_max and detail_max < daily_max:
        detail_used_asof = detail_max

    detail_sql = f"""
    SELECT *
    FROM {SF_DETAIL_VIEW}
    WHERE {detail_date_col} >= DATEADD('day', -{DETAIL_DAYS_BACK}, '{detail_used_asof}')
    ;
    """
    detail = normalize_detail(sf_read(detail_sql))

    meta = {"daily_max": daily_max, "detail_max": detail_max, "detail_used_asof": detail_used_asof}
    return daily, detail, meta


def load_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[date]]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "REPORTED_DATE" in df.columns:
        df["REPORTED_DATE"] = pd.to_datetime(df["REPORTED_DATE"], errors="coerce").dt.date
    meta = {"daily_max": to_date_safe(df["REPORTED_DATE"].max()) if "REPORTED_DATE" in df.columns else None,
            "detail_max": None,
            "detail_used_asof": None}
    return df, pd.DataFrame(), meta


# ============================================================
# LOAD
# ============================================================
if SOURCE == "csv":
    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find {DATA_PATH}. Put the CSV next to app.py (or set DEMO_DATA_PATH).")
        st.stop()
    daily_df, detail_df, meta = load_csv(DATA_PATH)
else:
    try:
        daily_df, detail_df, meta = load_snowflake()
    except Exception as e:
        st.error("Data load failed.")
        st.exception(e)
        st.stop()

st.sidebar.markdown(f"**Data source:** `{SOURCE.upper()}`")
st.sidebar.markdown(f"**Daily view:** `{SF_DAILY_VIEW}`")
st.sidebar.markdown(f"**Detail view:** `{SF_DETAIL_VIEW}`")

# Debug block
with st.sidebar.expander("Debug", expanded=False):
    st.write("Daily rows:", len(daily_df))
    st.write("Detail rows:", len(detail_df))
    st.write("Daily max:", meta.get("daily_max"))
    st.write("Detail max:", meta.get("detail_max"))
    st.write("Detail used as-of:", meta.get("detail_used_asof"))
    st.dataframe(daily_df.tail(10), use_container_width=True)

# ============================================================
# HEADER
# ============================================================
LOGO_PATH = os.getenv("NARS_LOGO_PATH", "narslogo.jpg")
left, right = st.columns([1, 5], vertical_alignment="center")
with left:
    if LOGO_PATH and os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=220)
with right:
    st.title("Claims Intelligence – Daily Summary")
    st.write("Client-facing daily KPI rollup sourced from Snowflake.")

# ============================================================
# MAIN LAYOUT
# ============================================================
if daily_df is None or daily_df.empty:
    st.error("Daily KPI dataset is empty.")
    st.stop()

daily_asof = meta.get("daily_max")
detail_max = meta.get("detail_max")
detail_asof = meta.get("detail_used_asof")

if detail_max and daily_asof and detail_max < daily_asof:
    st.warning(
        f"Detail dataset is stale. Daily KPIs are current through **{daily_asof}**, "
        f"but detail only goes through **{detail_max}**. "
        f"Detail sections below are being shown as-of **{detail_asof}**."
    )

# Latest KPI row from daily dataset
latest_row = daily_df[daily_df["REPORTED_DATE"] == daily_asof].iloc[0]

open_claims = int(latest_row["OPEN_CLAIMS"])
total_claims = int(latest_row["TOTAL_CLAIMS"])
closed_claims = int(total_claims - open_claims)
paid = float(latest_row["PAID"])
outstanding = float(latest_row["OUTSTANDING"])
incurred = float(latest_row["INCURRED"])
metric_note = latest_row.get("METRIC_NOTE", None)

main_col, filter_col = st.columns([4, 1], gap="large")

# ============================================================
# FILTERS (DETAIL)
# ============================================================
scoped_detail = detail_df.copy()
filter_summary = "None (All data)"

with filter_col:
    st.markdown("### Filters")

    if SOURCE != "snowflake" or scoped_detail.empty:
        st.info("Detail filters require claim-level data. (Detail dataset is empty or DATA_SOURCE != snowflake.)")
    else:
        parts = []

        # Adjuster
        adj_col = "ADJUSTER_ID" if "ADJUSTER_ID" in scoped_detail.columns else None
        if adj_col:
            adjs = ["All"] + sorted([a for a in scoped_detail[adj_col].dropna().unique().tolist()])
            sel_adj = st.selectbox("Adjuster", adjs, index=0)
            if sel_adj != "All":
                scoped_detail = scoped_detail[scoped_detail[adj_col] == sel_adj]
                parts.append(f"{adj_col}={sel_adj}")

        # Open-only toggle
        if "IS_OPEN" in scoped_detail.columns:
            open_only = st.toggle("Open claims only", value=True)
            if open_only:
                scoped_detail = scoped_detail[scoped_detail["IS_OPEN"] == 1]
                parts.append("OpenOnly")

        filter_summary = ", ".join(parts) if parts else "None (All data)"

# ============================================================
# MAIN CONTENT
# ============================================================
with main_col:
    st.caption(f"As-of (Daily KPI): {daily_asof}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Claims", f"{open_claims:,}")
    k2.metric("Total Incurred", fmt_money(incurred))
    k3.metric("Paid", fmt_money(paid))
    k4.metric("Outstanding", fmt_money(outstanding))
    k5.metric("Closed Claims", f"{closed_claims:,}")

    if metric_note and str(metric_note).strip().lower() not in ["none", "nan"]:
        st.info(str(metric_note))

    st.caption(f"In scope: {filter_summary}")

    # Trend
    st.subheader("Trend")
    trend = daily_df.copy()
    trend["REPORTED_DATE"] = pd.to_datetime(trend["REPORTED_DATE"])

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(trend.set_index("REPORTED_DATE")["OPEN_CLAIMS"])
    with c2:
        st.line_chart(trend.set_index("REPORTED_DATE")["INCURRED"])

    # Detail
    st.subheader(f"Open claims (detail) as-of: {detail_asof}")
    if SOURCE == "snowflake" and not scoped_detail.empty:
        preferred_cols = [
            "CLAIM_ID",
            "CLAIM_NUMBER",
            "CLIENT_CODE",
            "CLAIM_TYPE_CODE",
            "POLICY_OID",
            "POLICY_LOCATION_OID",
            "ADJUSTER_ID",
            "REPORTED_DATE",
            "CLOSED_DATE",
            "CLAIM_STATUS_CODE",
            "IS_OPEN",
            "IS_CLOSED",
        ]
        show_cols = [c for c in preferred_cols if c in scoped_detail.columns]
        if not show_cols:
            show_cols = scoped_detail.columns.tolist()[:12]

        sort_cols = [c for c in ["REPORTED_DATE", "CLAIM_ID"] if c in scoped_detail.columns]
        if sort_cols:
            scoped_detail = scoped_detail.sort_values(sort_cols, ascending=False)

        st.dataframe(scoped_detail[show_cols].head(500), use_container_width=True)

        if "ADJUSTER_ID" in scoped_detail.columns:
            st.subheader("Open claims by adjuster")
            by_adj = (
                scoped_detail.groupby("ADJUSTER_ID", dropna=True)
                .size()
                .reset_index(name="Open Claims")
                .sort_values("Open Claims", ascending=False)
                .head(20)
            )
            st.bar_chart(by_adj.set_index("ADJUSTER_ID")["Open Claims"])
    else:
        st.info("Detail view not available. Set SF_DETAIL_VIEW to a claim-level table/view you can query.")
