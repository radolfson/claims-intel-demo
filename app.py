import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# APP CONFIG + THEME (NARS BRANDING)
# ============================================================
st.set_page_config(page_title="Claims Intelligence – Daily Summary (Demo)", layout="wide")

NARS_BLUE = "#01426A"
st.markdown(
    f"""
<style>
header[data-testid="stHeader"] {{
  background-color: {NARS_BLUE} !important;
}}
h1,h2,h3,h4,h5,h6 {{
  color: {NARS_BLUE} !important;
}}
[data-testid="stMetricValue"] {{
  color: {NARS_BLUE} !important;
  font-weight: 700 !important;
}}
.block-container {{
  max-width: 1400px;
}}

.sticky-filter {{
  position: sticky;
  top: 80px;
  z-index: 10;
  background: white;
  padding: 0.85rem;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 12px;
}}

.card {{
  background: white;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 14px 10px 14px;
}}
.small-muted {{
  color: #666;
  font-size: 12px;
}}
hr {{
  border: none;
  border-top: 1px solid rgba(0,0,0,0.08);
  margin: 12px 0;
}}
code {{
  font-size: 0.9em;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SETTINGS HELPERS (Secrets -> Env)
# ============================================================
def get_setting(key: str, default: Any = None) -> Any:
    """Prefer Streamlit secrets, fall back to env vars."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


def get_data_source() -> str:
    src = get_setting("DATA_SOURCE", "csv")
    return str(src).lower().strip()


SOURCE = get_data_source()
DATA_PATH = os.getenv("DEMO_DATA_PATH", "demo_features_latest.csv")
LOGO_PATH = os.getenv("NARS_LOGO_PATH", "narslogo.jpg")


# ============================================================
# FORMAT HELPERS
# ============================================================
def fmt_money(x: float) -> str:
    return f"${float(x):,.0f}"


def fmt_money_short(x: float) -> str:
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"${x/1_000_000_000:,.1f}B"
    if ax >= 1_000_000:
        return f"${x/1_000_000:,.1f}M"
    if ax >= 1_000:
        return f"${x/1_000:,.1f}K"
    return f"${x:,.0f}"


def pct_change(new: float, old: float) -> Optional[float]:
    if old is None or old == 0:
        return None
    return (new - old) / old


def fmt_delta(new: float, old: Optional[float], is_money: bool = False) -> str:
    if old is None:
        return "no prior comparison available"
    if old == 0 and new == 0:
        return "flat at 0"
    if old == 0 and new != 0:
        return f"new activity to {fmt_money(new) if is_money else f'{int(new):,}'}"

    p = pct_change(new, old)
    direction = "up" if new > old else "down" if new < old else "flat"
    if p is None:
        return f"{direction} to {fmt_money(new) if is_money else f'{int(new):,}'}"
    return f"{direction} {abs(p)*100:,.1f}% to {fmt_money(new) if is_money else f'{int(new):,}'}"


# ============================================================
# SNOWFLAKE CONFIG + AUTH (KEYPAIR FIRST)
# ============================================================
@dataclass
class SnowflakeCfg:
    account: str
    user: str
    role: str = ""
    warehouse: str = ""
    database: str = ""
    schema: str = ""

    # Auth options:
    password: str = ""  # Avoid for Streamlit Cloud if MFA is on
    private_key_pem: str = ""  # PEM text (-----BEGIN PRIVATE KEY-----)
    private_key_b64: str = ""  # base64 of PEM text
    private_key_passphrase: str = ""  # optional passphrase if key is encrypted


def _read_snowflake_cfg() -> SnowflakeCfg:
    try:
        cfg = st.secrets["snowflake"]
    except Exception as e:
        raise RuntimeError(
            "Missing [snowflake] config in Streamlit secrets. Add it under App → Settings → Secrets."
        ) from e

    def s(k: str) -> str:
        v = cfg.get(k, "")
        return "" if v is None else str(v)

    return SnowflakeCfg(
        account=s("account"),
        user=s("user"),
        role=s("role"),
        warehouse=s("warehouse"),
        database=s("database"),
        schema=s("schema"),
        password=s("password"),
        private_key_pem=s("private_key_pem"),
        private_key_b64=s("private_key_b64"),
        private_key_passphrase=s("private_key_passphrase"),
    )


def _load_private_key_bytes(cfg: SnowflakeCfg) -> Optional[bytes]:
    """
    Returns PEM bytes for the private key if provided.
    Supports:
      - cfg.private_key_pem (raw PEM text)
      - cfg.private_key_b64 (base64 of PEM text)
    """
    if cfg.private_key_pem.strip():
        return cfg.private_key_pem.strip().encode("utf-8")

    if cfg.private_key_b64.strip():
        try:
            return base64.b64decode(cfg.private_key_b64.strip())
        except Exception as e:
            raise RuntimeError("private_key_b64 is not valid base64.") from e

    return None


def _connect_snowflake_keypair(cfg: SnowflakeCfg):
    """
    Keypair auth: avoids MFA. This is what you want for Streamlit Cloud.
    Requires snowflake-connector-python + cryptography.
    """
    import snowflake.connector
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    pem_bytes = _load_private_key_bytes(cfg)
    if not pem_bytes:
        raise RuntimeError(
            "Snowflake keypair auth selected but no private key found. "
            "Set snowflake.private_key_pem OR snowflake.private_key_b64 in Streamlit secrets."
        )

    passphrase = cfg.private_key_passphrase.strip().encode("utf-8") if cfg.private_key_passphrase.strip() else None

    try:
        pkey = load_pem_private_key(pem_bytes, password=passphrase)
        private_key_der = pkey.private_bytes(
            encoding=__import__("cryptography.hazmat.primitives.serialization").hazmat.primitives.serialization.Encoding.DER,
            format=__import__("cryptography.hazmat.primitives.serialization").hazmat.primitives.serialization.PrivateFormat.PKCS8,
            encryption_algorithm=__import__("cryptography.hazmat.primitives.serialization").hazmat.primitives.serialization.NoEncryption(),
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load private key. The PEM may be wrong, encrypted without passphrase, or corrupted."
        ) from e

    return snowflake.connector.connect(
        account=cfg.account,
        user=cfg.user,
        private_key=private_key_der,
        role=cfg.role or None,
        warehouse=cfg.warehouse or None,
        database=cfg.database or None,
        schema=cfg.schema or None,
    )


def _connect_snowflake_password(cfg: SnowflakeCfg):
    """
    Password auth: will fail if the user is enforced to use MFA/TOTP for this interface.
    Useful locally sometimes; risky for Streamlit Cloud.
    """
    import snowflake.connector

    if not cfg.password.strip():
        raise RuntimeError("Snowflake password auth selected but snowflake.password is empty.")

    return snowflake.connector.connect(
        account=cfg.account,
        user=cfg.user,
        password=cfg.password,
        role=cfg.role or None,
        warehouse=cfg.warehouse or None,
        database=cfg.database or None,
        schema=cfg.schema or None,
    )


def _connect_snowflake(cfg: SnowflakeCfg):
    """
    Prefer keypair if present; fall back to password if not.
    """
    if _load_private_key_bytes(cfg):
        return _connect_snowflake_keypair(cfg)
    return _connect_snowflake_password(cfg)


# ============================================================
# DATA LOADERS
# ============================================================
def _coerce_csv_schema(df: pd.DataFrame) -> pd.DataFrame:
    if "ACCIDENT_YEAR" in df.columns:
        df["ACCIDENT_YEAR"] = pd.to_numeric(df["ACCIDENT_YEAR"], errors="coerce").fillna(0).astype(int)

    for flag_col in ["OPEN_FLAG", "HIGH_SEVERITY_FLAG"]:
        if flag_col in df.columns:
            df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)

    for money_col in ["PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT"]:
        if money_col in df.columns:
            df[money_col] = pd.to_numeric(df[money_col], errors="coerce").fillna(0.0)

    if "ASOF_DATE" in df.columns:
        df["ASOF_DATE"] = pd.to_datetime(df["ASOF_DATE"], errors="coerce").dt.date

    # enforce incurred consistency
    if "PAID_AMT" in df.columns and "OUTSTANDING_AMT" in df.columns:
        df["INCURRED_AMT"] = (df["PAID_AMT"].fillna(0.0) + df["OUTSTANDING_AMT"].fillna(0.0)).round(2)

    return df


def load_data_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _coerce_csv_schema(df)


def _coerce_snowflake_kpi_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize to app fields
    rename_map = {
        "REPORTED_DATE": "ASOF_DATE",
        "TOTAL_CLAIMS": "TOTAL_CT",
        "OPEN_CLAIMS": "OPEN_CT",
        "CLOSED_CLAIMS": "CLOSED_CT",
        "PAID": "PAID_AMT",
        "OUTSTANDING": "OUTSTANDING_AMT",
        "INCURRED": "INCURRED_AMT",
    }
    for src, tgt in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: tgt})

    if "ASOF_DATE" in df.columns:
        df["ASOF_DATE"] = pd.to_datetime(df["ASOF_DATE"], errors="coerce").dt.date

    for col in ["TOTAL_CT", "OPEN_CT", "CLOSED_CT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def load_data_snowflake() -> pd.DataFrame:
    cfg = _read_snowflake_cfg()
    conn = _connect_snowflake(cfg)

    sql = """
        SELECT
            REPORTED_DATE,
            TOTAL_CLAIMS,
            OPEN_CLAIMS,
            CLOSED_CLAIMS,
            PAID,
            OUTSTANDING,
            INCURRED,
            METRIC_NOTE
        FROM NARS.PROCESSED.V_TRANSPORTATION_DAILY
        ORDER BY REPORTED_DATE
    """

    try:
        df = pd.read_sql(sql, conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return _coerce_snowflake_kpi_schema(df)


@st.cache_data(show_spinner=False, ttl=900)
def load_data(source: str, csv_path: str) -> pd.DataFrame:
    if source == "snowflake":
        return load_data_snowflake()
    return load_data_csv(csv_path)


# ============================================================
# METRIC COMPUTATION
# ============================================================
def latest_and_prior_by_asof(df_scoped: pd.DataFrame):
    if df_scoped is None or len(df_scoped) == 0 or "ASOF_DATE" not in df_scoped.columns:
        return df_scoped, None, None, None

    dates = sorted([d for d in df_scoped["ASOF_DATE"].dropna().unique().tolist() if d is not None])
    if not dates:
        return df_scoped, None, None, None

    latest = dates[-1]
    prior = dates[-2] if len(dates) >= 2 else None

    latest_df = df_scoped[df_scoped["ASOF_DATE"] == latest].copy()
    prior_df = df_scoped[df_scoped["ASOF_DATE"] == prior].copy() if prior else None
    return latest_df, prior_df, latest, prior


def snapshot_metrics(df_in: pd.DataFrame, source: str) -> Dict[str, Any]:
    if df_in is None or len(df_in) == 0:
        return {
            "open_ct": 0,
            "total_ct": 0,
            "closed_ct": 0,
            "incurred": 0.0,
            "paid": 0.0,
            "outstanding": 0.0,
            "high_sev_ct": 0,
            "sev_share": 0.0,
        }

    if source == "snowflake":
        row = df_in.iloc[0]
        open_ct = int(row.get("OPEN_CT", 0))
        total_ct = int(row.get("TOTAL_CT", 0))
        closed_ct = int(row.get("CLOSED_CT", 0))
        incurred = float(row.get("INCURRED_AMT", 0.0))
        paid = float(row.get("PAID_AMT", 0.0))
        outstanding = float(row.get("OUTSTANDING_AMT", 0.0))

        return {
            "open_ct": open_ct,
            "total_ct": total_ct,
            "closed_ct": closed_ct,
            "incurred": incurred,
            "paid": paid,
            "outstanding": outstanding,
            "high_sev_ct": 0,
            "sev_share": 0.0,
        }

    # CSV feature-level mode
    open_ct = int(df_in["OPEN_FLAG"].sum()) if "OPEN_FLAG" in df_in.columns else 0
    total_ct = int(len(df_in))
    closed_ct = total_ct - open_ct

    incurred = float(df_in["INCURRED_AMT"].sum()) if "INCURRED_AMT" in df_in.columns else 0.0
    paid = float(df_in["PAID_AMT"].sum()) if "PAID_AMT" in df_in.columns else 0.0
    outstanding = float(df_in["OUTSTANDING_AMT"].sum()) if "OUTSTANDING_AMT" in df_in.columns else 0.0
    high_sev_ct = int(df_in["HIGH_SEVERITY_FLAG"].sum()) if "HIGH_SEVERITY_FLAG" in df_in.columns else 0

    sev_share = (high_sev_ct / open_ct) if open_ct else 0.0

    return {
        "open_ct": open_ct,
        "total_ct": total_ct,
        "closed_ct": closed_ct,
        "incurred": incurred,
        "paid": paid,
        "outstanding": outstanding,
        "high_sev_ct": high_sev_ct,
        "sev_share": sev_share,
    }


def build_incurred_stratification(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or len(df_in) == 0 or "INCURRED_AMT" not in df_in.columns:
        return pd.DataFrame(columns=["Incurred Range", "Feature Count"])

    bin_edges = [0, 10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000, float("inf")]
    labels = ["$0–$10K", "$10K–$25K", "$25K–$50K", "$50K–$100K", "$100K–$250K", "$250K–$1M", "$1M+"]

    cut = pd.cut(df_in["INCURRED_AMT"], bins=bin_edges, labels=labels, include_lowest=True, right=True)
    sev = cut.value_counts(dropna=False).reindex(labels, fill_value=0).reset_index()
    sev.columns = ["Incurred Range", "Feature Count"]
    return sev


# ============================================================
# LOAD DATA (WITH SAFETY + VERY LOUD FAILURES)
# ============================================================
if SOURCE not in ("csv", "snowflake"):
    st.error(f"Invalid DATA_SOURCE={SOURCE!r}. Use 'csv' or 'snowflake'.")
    st.stop()

if SOURCE == "csv":
    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find {DATA_PATH}. Put the CSV next to app.py (or set DEMO_DATA_PATH).")
        st.stop()

try:
    df = load_data(SOURCE, DATA_PATH)
except ModuleNotFoundError as e:
    # This is the classic Streamlit Cloud "you forgot requirements.txt" failure.
    st.error(
        "Missing Python package in Streamlit environment.\n\n"
        f"Details: `{e}`\n\n"
        "Fix: add required packages to `requirements.txt` (see notes below)."
    )
    st.stop()
except Exception as e:
    st.error(
        "Data load failed.\n\n"
        f"Source: `{SOURCE}`\n"
        f"Error: `{type(e).__name__}: {e}`\n\n"
        "If SOURCE is snowflake, this is almost always one of:\n"
        "- using password auth while MFA/TOTP is enforced\n"
        "- missing private key in secrets\n"
        "- wrong account/user/role/warehouse\n"
        "- network policy restriction\n"
    )
    st.stop()

# Always show the active source so you can't accidentally misrepresent it
st.sidebar.markdown(f"**Data source:** `{SOURCE.upper()}`")


with st.sidebar.expander("Debug (useful until this stops hurting)"):
    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    if "ASOF_DATE" in df.columns and len(df):
        st.write("Min ASOF_DATE:", df["ASOF_DATE"].min())
        st.write("Max ASOF_DATE:", df["ASOF_DATE"].max())
    st.dataframe(df.head(10), use_container_width=True)


# ============================================================
# HEADER (LOGO + TITLE)
# ============================================================
header_left, header_right = st.columns([1, 5], vertical_alignment="center")
with header_left:
    if LOGO_PATH and os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=220)

with header_right:
    st.title("Claims Intelligence – Daily Summary (Demo)")
    st.write("Representative dataset demonstrating NARS metric logic and the client delivery experience.")


# ============================================================
# LAYOUT: MAIN + FILTER PANEL
# ============================================================
main_col, filter_col = st.columns([4, 1], gap="large")


# ============================================================
# FILTERS (CSV ONLY)
# ============================================================
filter_summary = "None (All data)"
f = df.copy()

if SOURCE == "csv":
    states = ["All"] + sorted(f["LOSS_STATE"].dropna().unique().tolist()) if "LOSS_STATE" in f.columns else ["All"]
    years = ["All"] + sorted(f["ACCIDENT_YEAR"].dropna().unique().tolist()) if "ACCIDENT_YEAR" in f.columns else ["All"]
    adjusters = ["All"] + sorted(f["ADJUSTER_ID"].dropna().unique().tolist()) if "ADJUSTER_ID" in f.columns else ["All"]
    coverages = ["All"] + sorted(f["COVERAGE_CODE"].dropna().unique().tolist()) if "COVERAGE_CODE" in f.columns else ["All"]

    with filter_col:
        st.markdown('<div class="sticky-filter">', unsafe_allow_html=True)
        st.markdown("### Filters")

        sel_state = st.selectbox("State", states, index=0)
        sel_year = st.selectbox("Accident Year", years, index=0)
        sel_adjuster = st.selectbox("Adjuster", adjusters, index=0)
        sel_cov = st.selectbox("Coverage Type", coverages, index=0)

        st.markdown(
            '<div class="small-muted">Filters apply to the dashboard (CSV mode only).</div>',
            unsafe_allow_html=True,
        )

        if st.button("Reset filters", use_container_width=True):
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters
    filter_parts = []
    if sel_state != "All" and "LOSS_STATE" in f.columns:
        filter_parts.append(f"State={sel_state}")
        f = f[f["LOSS_STATE"] == sel_state]

    if sel_year != "All" and "ACCIDENT_YEAR" in f.columns:
        filter_parts.append(f"AccidentYear={sel_year}")
        f = f[f["ACCIDENT_YEAR"] == int(sel_year)]

    if sel_adjuster != "All" and "ADJUSTER_ID" in f.columns:
        filter_parts.append(f"Adjuster={sel_adjuster}")
        f = f[f["ADJUSTER_ID"] == sel_adjuster]

    if sel_cov != "All" and "COVERAGE_CODE" in f.columns:
        filter_parts.append(f"CoverageType={sel_cov}")
        f = f[f["COVERAGE_CODE"] == sel_cov]

    filter_summary = ", ".join(filter_parts) if filter_parts else "None (All data)"
else:
    with filter_col:
        st.markdown('<div class="sticky-filter">', unsafe_allow_html=True)
        st.markdown("### Filters")
        st.info("Snowflake KPI mode: row-level filters are disabled (daily KPI view).")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SNAPSHOT METRICS
# ============================================================
f_latest, f_prior, asof, prior_asof = latest_and_prior_by_asof(f)

cur = snapshot_metrics(f_latest, source=SOURCE)
open_ct = cur["open_ct"]
total_ct = cur["total_ct"]
closed_ct = cur["closed_ct"]
incurred = cur["incurred"]
paid = cur["paid"]
outstanding = cur["outstanding"]
high_sev_ct = cur["high_sev_ct"]
sev_share = cur["sev_share"]

prior = snapshot_metrics(f_prior, source=SOURCE) if f_prior is not None and len(f_prior) else None


# ============================================================
# MAIN CONTENT
# ============================================================
with main_col:
    st.caption(f"As-of: {asof if asof else 'N/A'}")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Claims" if SOURCE == "snowflake" else "Open Features", f"{open_ct:,}")
    k2.metric("Total Incurred", fmt_money_short(incurred))
    k3.metric("Paid", fmt_money_short(paid))
    k4.metric("Outstanding", fmt_money_short(outstanding))
    k5.metric("High Severity Features" if SOURCE == "csv" else "High Severity", f"{high_sev_ct:,}")

    st.caption(
        f"Full totals: Total Incurred {fmt_money(incurred)} | Paid {fmt_money(paid)} | Outstanding {fmt_money(outstanding)}"
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Today’s Headlines")
    st.write(f"**In scope:** {filter_summary}")

    if prior:
        open_vs_prior = fmt_delta(open_ct, prior["open_ct"], is_money=False)
        inc_vs_prior = fmt_delta(incurred, prior["incurred"], is_money=True)
        prior_label = f"{prior_asof}" if prior_asof else "prior as-of"
    else:
        open_vs_prior = "no prior as-of available"
        inc_vs_prior = "no prior as-of available"
        prior_label = "prior as-of"

    st.write(f"**As-of:** {asof if asof else 'N/A'}  |  **Comparison:** {prior_label}")
    st.write(f"• **Open:** {open_ct:,} ({open_vs_prior})")
    st.write(f"• **Total Incurred:** {fmt_money(incurred)} ({inc_vs_prior})")

    if SOURCE == "csv":
        hs_vs_prior = fmt_delta(high_sev_ct, prior["high_sev_ct"], is_money=False) if prior else "no prior as-of available"
        st.write(f"• **High Severity:** {high_sev_ct:,} ({hs_vs_prior})  |  Share of open: **{sev_share*100:,.1f}%**")

    st.markdown("</div>", unsafe_allow_html=True)

    # Visuals (only meaningful in CSV feature-level mode)
    if SOURCE == "csv":
        tab_open, tab_incurred, tab_sev = st.tabs(["Open Features", "Total Incurred", "Severity Stratification"])

        with tab_open:
            if "ACCIDENT_YEAR" in f_latest.columns:
                by_year = (
                    f_latest.groupby("ACCIDENT_YEAR", dropna=True)
                    .size()
                    .reset_index(name="Feature Count")
                    .sort_values("ACCIDENT_YEAR")
                )
                st.bar_chart(by_year.set_index("ACCIDENT_YEAR")["Feature Count"])
            else:
                st.info("ACCIDENT_YEAR not available in this dataset for the Open Features trend.")

        with tab_incurred:
            if "ACCIDENT_YEAR" in f_latest.columns and "INCURRED_AMT" in f_latest.columns:
                inc_by_year = (
                    f_latest.groupby("ACCIDENT_YEAR", dropna=True)["INCURRED_AMT"]
                    .sum()
                    .reset_index()
                    .sort_values("ACCIDENT_YEAR")
                )
                st.bar_chart(inc_by_year.set_index("ACCIDENT_YEAR")["INCURRED_AMT"])
                st.caption("Chart shows Total Incurred summed by Accident Year (feature-level).")
            else:
                st.info("ACCIDENT_YEAR / INCURRED_AMT not available for Total Incurred trend.")

        with tab_sev:
            sev_tbl = build_incurred_stratification(f_latest)
            if len(sev_tbl):
                st.dataframe(sev_tbl, use_container_width=True, hide_index=True)
            else:
                st.info("No incurred data available for severity stratification in the current filter set.")
    else:
        st.info("Snowflake KPI mode: visual breakdowns are hidden (you’re using a daily KPI view).")
