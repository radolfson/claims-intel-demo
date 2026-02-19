import os
from datetime import date
from typing import Any, Dict, Optional, Tuple

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

# If you ever fall back to CSV, it is explicit.
DATA_PATH = os.getenv("DEMO_DATA_PATH", "demo_features_latest.csv")

# Snowflake objects (keep these together so you can change them in one place)
SF_DAILY_VIEW = os.getenv("SF_DAILY_VIEW", "NARS.PROCESSED.V_FEATURE_DAILY_STATUS")
# For tonight, we tolerate STAGING for claim-list/detail. Replace with a PROCESSED contract view later.
SF_DETAIL_VIEW = os.getenv("SF_DETAIL_VIEW", "NARS.STAGING.STG_TRANSPORTATION_CLAIMS")

# How much detail data to pull for filters/tables (keeps the app fast)
DETAIL_DAYS_BACK = int(os.getenv("DETAIL_DAYS_BACK", "180"))

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

    # Snowflake returns uppercase by default
    cols = {c.upper(): c for c in df.columns}

    # Required (from your DESC screenshot)
    required = ["REPORTED_DATE", "TOTAL_CLAIMS", "OPEN_CLAIMS", "PAID", "OUTSTANDING", "INCURRED"]
    for r in required:
        if r not in cols:
            raise ValueError(
                f"Daily view missing expected column {r}. Found: {list(df.columns)}. "
                f"Point SF_DAILY_VIEW to the right object."
            )

    # Standardize to upper-case column labels
    df = df.rename(columns={cols[k]: k for k in cols})

    # Ensure date type
    df["REPORTED_DATE"] = pd.to_datetime(df["REPORTED_DATE"]).dt.date

    # Numeric
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
    # Try common date columns
    for dcol in ["REPORTED_DATE", "REPORT_DATE", "LOSS_DATE", "CLOSED_DATE"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.date
    return df


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

def _load_private_key() -> bytes:
    pkey = get_setting("SNOWFLAKE_PRIVATE_KEY")
    if not pkey:
        raise ValueError("Missing SNOWFLAKE_PRIVATE_KEY in secrets/env.")

    # Supports: raw PEM text, or base64 of PEM text
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
    account = get_setting("SNOWFLAKE_ACCOUNT")
    user = get_setting("SNOWFLAKE_USER")
    warehouse = get_setting("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database = get_setting("SNOWFLAKE_DATABASE", "NARS")
    schema = get_setting("SNOWFLAKE_SCHEMA", "PROCESSED")
    role = get_setting("SNOWFLAKE_ROLE")

    if not account or not user:
        raise ValueError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in secrets/env.")

    pkb = _load_private_key()

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


@st.cache_data(show_spinner=True, ttl=900)
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

    detail_sql = f"""
    SELECT *
    FROM {SF_DETAIL_VIEW}
    WHERE COALESCE(REPORTED_DATE, REPORT_DATE) >= DATEADD('day', -{DETAIL_DAYS_BACK}, CURRENT_DATE())
    ;
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
# LOAD
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

st.sidebar.markdown(f"**Data source:** `{SOURCE.upper()}`")

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
# AS-OF + KPI ROW
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
# FILTERS (SNOWFLAKE DETAIL)
# ============================================================
scoped_detail = detail_df.copy()
filter_summary = "None (All data)"

with filter_col:
    st.markdown("### Filters")

    if SOURCE != "snowflake" or detail_df.empty:
        st.info("Detail filters need claim-level data. (SF_DETAIL_VIEW is empty or DATA_SOURCE != snowflake.)")
    else:
        # Try to offer filters when columns exist
        def opt(col: str):
            return col in scoped_detail.columns

        parts = []

        # Restrict detail to last N days and non-null report date
        if opt("REPORTED_DATE"):
            scoped_detail = scoped_detail[scoped_detail["REPORTED_DATE"].notna()]

        # State-like
        state_col = None
        for c in ["LOSS_STATE", "STATE", "POLICY_STATE", "POLICY_LOCATION_STATE"]:
            if opt(c):
                state_col = c
                break
        if state_col:
            states = ["All"] + sorted([s for s in scoped_detail[state_col].dropna().unique().tolist()])
            sel_state = st.selectbox("State", states, index=0)
            if sel_state != "All":
                scoped_detail = scoped_detail[scoped_detail[state_col] == sel_state]
                parts.append(f"{state_col}={sel_state}")

        # Adjuster
        adj_col = "ADJUSTER_ID" if opt("ADJUSTER_ID") else ("ADJUSTER" if opt("ADJUSTER") else None)
        if adj_col:
            adjs = ["All"] + sorted([a for a in scoped_detail[adj_col].dropna().unique().tolist()])
            sel_adj = st.selectbox("Adjuster", adjs, index=0)
            if sel_adj != "All":
                scoped_detail = scoped_detail[scoped_detail[adj_col] == sel_adj]
                parts.append(f"{adj_col}={sel_adj}")

        # Open-only toggle
        is_open_col = "IS_OPEN" if opt("IS_OPEN") else None
        if is_open_col:
            open_only = st.toggle("Open claims only", value=True)
            if open_only:
                scoped_detail = scoped_detail[scoped_detail[is_open_col].fillna(0).astype(int) == 1]
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

    # Trend (daily)
    st.subheader("Trend")
    trend = daily_df.copy()
    trend["REPORTED_DATE"] = pd.to_datetime(trend["REPORTED_DATE"])

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(trend.set_index("REPORTED_DATE")["OPEN_CLAIMS"])
    with c2:
        st.line_chart(trend.set_index("REPORTED_DATE")["INCURRED"])

    # Detail
    st.subheader("Open claims (detail)")
    if SOURCE == "snowflake" and not scoped_detail.empty:
        # Choose a reasonable set of columns to show if they exist
        preferred = [
            "CLAIM_ID",
            "CLAIM_NUMBER",
            "CLIENT_CODE",
            "CLAIM_TYPE_CODE",
            "POLICY_OID",
            "ADJUSTER_ID",
            "REPORTED_DATE",
            "LOSS_DATE",
            "CLOSED_DATE",
            "CLAIM_STATUS",
            "IS_OPEN",
        ]
        show_cols = [c for c in preferred if c in scoped_detail.columns]
        if not show_cols:
            show_cols = scoped_detail.columns.tolist()[:12]

        st.dataframe(
            scoped_detail.sort_values(
                [c for c in ["REPORTED_DATE", "CLAIM_ID"] if c in scoped_detail.columns],
                ascending=False,
            )[show_cols].head(500),
            use_container_width=True,
        )

        # Counts by adjuster (if available)
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
