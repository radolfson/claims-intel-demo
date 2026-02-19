import os
from datetime import date
from typing import Optional, Tuple, List

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

# CSV fallback (only if DATA_SOURCE=csv)
DATA_PATH = os.getenv("DEMO_DATA_PATH", "demo_features_latest.csv")

# Snowflake objects
SF_DAILY_VIEW = get_setting("SF_DAILY_VIEW", "NARS.PROCESSED.V_FEATURE_DAILY_STATUS")
SF_DETAIL_VIEW = get_setting("SF_DETAIL_VIEW", "NARS.PROCESSED.V_TRANSPORTATION_CLAIMS_DETAIL")
DETAIL_DAYS_BACK = int(get_setting("DETAIL_DAYS_BACK", "180"))

# Connection defaults
SNOWFLAKE_ACCOUNT = get_setting("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = get_setting("SNOWFLAKE_USER")
SNOWFLAKE_WAREHOUSE = get_setting("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE = get_setting("SNOWFLAKE_DATABASE", "NARS")
SNOWFLAKE_SCHEMA = get_setting("SNOWFLAKE_SCHEMA", "PROCESSED")
SNOWFLAKE_ROLE = get_setting("SNOWFLAKE_ROLE")


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
    """Force expected column names for the daily KPI dataset."""
    if df is None or df.empty:
        return df

    df = df.rename(columns={c: c.upper() for c in df.columns})

    required = ["REPORTED_DATE", "TOTAL_CLAIMS", "OPEN_CLAIMS", "PAID", "OUTSTANDING", "INCURRED"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Daily view missing columns: {missing}. Found: {list(df.columns)}. "
            f"SF_DAILY_VIEW currently = {SF_DAILY_VIEW}"
        )

    df["REPORTED_DATE"] = pd.to_datetime(df["REPORTED_DATE"], errors="coerce").dt.date

    for c in ["TOTAL_CLAIMS", "OPEN_CLAIMS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["PAID", "OUTSTANDING", "INCURRED"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "METRIC_NOTE" not in df.columns:
        df["METRIC_NOTE"] = None

    return df.sort_values("REPORTED_DATE")


def normalize_detail(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.rename(columns={c: c.upper() for c in df.columns})
    for dcol in ["REPORTED_DATE", "REPORT_DATE", "LOSS_DATE", "CLOSED_DATE"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    return df


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================
def _load_private_key_der() -> bytes:
    # In your secrets you showed it as private_key_pem; handle both names:
    pkey = get_setting("SNOWFLAKE_PRIVATE_KEY") or get_setting("private_key_pem")
    if not pkey:
        raise ValueError("Missing SNOWFLAKE_PRIVATE_KEY (or private_key_pem) in secrets/env.")

    # raw PEM text, or base64 of PEM
    if "BEGIN" not in pkey:
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
    if not SNOWFLAKE_ACCOUNT or not SNOWFLAKE_USER:
        raise ValueError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in secrets/env.")

    pkb = _load_private_key_der()

    kwargs = dict(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        private_key=pkb,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )
    if SNOWFLAKE_ROLE:
        kwargs["role"] = SNOWFLAKE_ROLE

    return snowflake.connector.connect(**kwargs)


@st.cache_data(show_spinner=False, ttl=900)
def sf_read(sql: str) -> pd.DataFrame:
    """Cached query runner (new connection each call)."""
    with sf_connect() as conn:
        return pd.read_sql(sql, conn)


def _parse_fqn(obj_name: str, default_db: str, default_schema: str) -> Tuple[str, str, str]:
    """Return (db, schema, name) for NAME, SCHEMA.NAME, DB.SCHEMA.NAME."""
    parts = [p.strip().strip('"') for p in obj_name.split(".") if p.strip()]
    if len(parts) == 1:
        return default_db, default_schema, parts[0]
    if len(parts) == 2:
        return default_db, parts[0], parts[1]
    return parts[0], parts[1], parts[2]


def _get_columns_for_object(obj_name: str) -> List[str]:
    """Non-cached metadata read (kept simple and robust)."""
    db, schema, name = _parse_fqn(obj_name, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA)
    sql = f"""
        SELECT UPPER(column_name) AS column_name
        FROM {db}.INFORMATION_SCHEMA.COLUMNS
        WHERE UPPER(table_schema) = UPPER('{schema}')
          AND UPPER(table_name)   = UPPER('{name}')
        ORDER BY ordinal_position
    """
    df = sf_read(sql)
    return df["COLUMN_NAME"].tolist() if not df.empty else []


def _pick_detail_date_column(obj_name: str) -> str:
    cols = set(_get_columns_for_object(obj_name))

    # Prefer exactly what your detail view actually has (from your screenshot)
    preferred = [
        "REPORTED_DATE",   # you DO have this
        "REPORT_DATE",     # you do NOT, but we’ll tolerate if present elsewhere
        "LOSS_DATE",
        "CLOSED_DATE",
        "DATE_DAY",
        "REPORT_DATE_DAY",
    ]
    for c in preferred:
        if c in cols:
            return c

    # Last resort: find anything with DATE in the name
    date_like = [c for c in cols if "DATE" in c]
    if date_like:
        return sorted(date_like)[0]

    raise ValueError(f"Could not find a usable date column in {obj_name}. Columns: {sorted(cols)}")


def load_snowflake() -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Detail: dynamically choose the date column so we don't explode on REPORT_DATE again
    detail_date_col = _pick_detail_date_column(SF_DETAIL_VIEW)

    detail_sql = f"""
        SELECT *
        FROM {SF_DETAIL_VIEW}
        WHERE {detail_date_col} >= DATEADD('day', -{DETAIL_DAYS_BACK}, CURRENT_DATE())
    """

    daily = normalize_daily(sf_read(daily_sql))
    detail = normalize_detail(sf_read(detail_sql))
    return daily, detail


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "ASOF_DATE" in df.columns:
        df["ASOF_DATE"] = pd.to_datetime(df["ASOF_DATE"], errors="coerce").dt.date
    return df


# ============================================================
# LOAD DATA
# ============================================================
if SOURCE == "csv":
    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find {DATA_PATH}. Put the CSV next to app.py (or set DEMO_DATA_PATH).")
        st.stop()
    daily_df = load_csv(DATA_PATH)
    detail_df = pd.DataFrame()
else:
    try:
        daily_df, detail_df = load_snowflake()
    except Exception as e:
        st.error("Data load failed.")
        st.exception(e)
        st.stop()


# ============================================================
# SIDEBAR DEBUG
# ============================================================
st.sidebar.markdown(f"**Data source:** `{SOURCE.upper()}`")
st.sidebar.markdown(f"**Daily view:** `{SF_DAILY_VIEW}`")
st.sidebar.markdown(f"**Detail view:** `{SF_DETAIL_VIEW}`")

with st.sidebar.expander("Debug", expanded=False):
    st.write("Daily rows:", len(daily_df))
    if not daily_df.empty and "REPORTED_DATE" in daily_df.columns:
        st.write("Min REPORTED_DATE:", daily_df["REPORTED_DATE"].min())
        st.write("Max REPORTED_DATE:", daily_df["REPORTED_DATE"].max())
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
# LAYOUT
# ============================================================
main_col, filter_col = st.columns([4, 1], gap="large")


# ============================================================
# AS-OF + KPIs
# ============================================================
if daily_df.empty:
    st.error("Daily KPI dataset is empty.")
    st.stop()

max_date = to_date_safe(daily_df["REPORTED_DATE"].max())
latest_row = daily_df[daily_df["REPORTED_DATE"] == max_date].iloc[0]

open_claims = int(latest_row["OPEN_CLAIMS"])
total_claims = int(latest_row["TOTAL_CLAIMS"])
closed_claims = int(total_claims - open_claims)
paid = float(latest_row["PAID"]) if pd.notna(latest_row["PAID"]) else 0.0
outstanding = float(latest_row["OUTSTANDING"]) if pd.notna(latest_row["OUTSTANDING"]) else 0.0
incurred = float(latest_row["INCURRED"]) if pd.notna(latest_row["INCURRED"]) else 0.0
metric_note = latest_row.get("METRIC_NOTE", None)


# ============================================================
# FILTERS (DETAIL)
# ============================================================
scoped_detail = detail_df.copy()
filter_summary = "None (All data)"

with filter_col:
    st.markdown("### Filters")

    if SOURCE != "snowflake" or detail_df.empty:
        st.info("Detail filters need claim-level data. (Detail view is empty or DATA_SOURCE != snowflake.)")
    else:
        def has(col: str) -> bool:
            return col in scoped_detail.columns

        parts = []

        # Ensure date is not null if available
        if has("REPORTED_DATE"):
            scoped_detail = scoped_detail[scoped_detail["REPORTED_DATE"].notna()]

        # State-ish
        state_col = None
        for c in ["LOSS_STATE", "STATE", "POLICY_STATE", "POLICY_LOCATION_STATE"]:
            if has(c):
                state_col = c
                break
        if state_col:
            states = ["All"] + sorted([s for s in scoped_detail[state_col].dropna().unique().tolist()])
            sel_state = st.selectbox("State", states, index=0)
            if sel_state != "All":
                scoped_detail = scoped_detail[scoped_detail[state_col] == sel_state]
                parts.append(f"{state_col}={sel_state}")

        # Adjuster
        adj_col = "ADJUSTER_ID" if has("ADJUSTER_ID") else ("ADJUSTER" if has("ADJUSTER") else None)
        if adj_col:
            adjs = ["All"] + sorted([a for a in scoped_detail[adj_col].dropna().unique().tolist()])
            sel_adj = st.selectbox("Adjuster", adjs, index=0)
            if sel_adj != "All":
                scoped_detail = scoped_detail[scoped_detail[adj_col] == sel_adj]
                parts.append(f"{adj_col}={sel_adj}")

        # Open-only
        if has("IS_OPEN"):
            open_only = st.toggle("Open claims only", value=True)
            if open_only:
                scoped_detail = scoped_detail[scoped_detail["IS_OPEN"].fillna(0).astype(int) == 1]
                parts.append("OpenOnly")

        filter_summary = ", ".join(parts) if parts else "None (All data)"


# ============================================================
# MAIN CONTENT
# ============================================================
with main_col:
    st.caption(f"As-of: {max_date}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Claims", f"{open_claims:,}")
    k2.metric("Total Incurred", fmt_money(incurred))
    k3.metric("Paid", fmt_money(paid))
    k4.metric("Outstanding", fmt_money(outstanding))
    k5.metric("Closed Claims", f"{closed_claims:,}")

    if metric_note:
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

    # Detail table
    st.subheader("Open claims (detail)")
    if SOURCE == "snowflake" and not scoped_detail.empty:
        preferred_cols = [
            "CLAIM_ID",
            "CLAIM_NUMBER",
            "CLIENT_CODE",
            "CLAIM_TYPE_CODE",
            "POLICY_ID",
            "POLICY_LOCATION_OID",
            "ADJUSTER_ID",
            "REPORTED_DATE",
            "LOSS_DATE",
            "CLOSED_DATE",
            "CLAIM_STATUS_CODE",
            "IS_OPEN",
            "IS_CLOSED",
            "IS_LITIGATION",
        ]
        show_cols = [c for c in preferred_cols if c in scoped_detail.columns]
        if not show_cols:
            show_cols = scoped_detail.columns.tolist()[:12]

        sort_cols = [c for c in ["REPORTED_DATE", "CLAIM_ID"] if c in scoped_detail.columns]
        df_show = scoped_detail.sort_values(sort_cols, ascending=False) if sort_cols else scoped_detail

        st.dataframe(df_show[show_cols].head(500), use_container_width=True)

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
