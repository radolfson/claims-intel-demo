import os
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# EMAIL VIA OUTLOOK (WINDOWS DESKTOP)
# (Will NOT work on Streamlit Cloud. Kept for local demo only.)
# ============================================================
def send_demo_email_outlook(to_email: str, subject: str, html_body: str) -> tuple[bool, str]:
    """Send HTML email via local Outlook desktop (Windows)."""
    try:
        import pythoncom
        pythoncom.CoInitialize()

        import win32com.client
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.To = to_email
        mail.Subject = subject
        mail.HTMLBody = html_body
        mail.Send()
        return True, "Sent via Outlook."
    except Exception as e:
        return False, f"Outlook send failed: {e}"
    finally:
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass


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
</style>
""",
    unsafe_allow_html=True,
)


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


def pct_change(new: float, old: float) -> float | None:
    if old is None or old == 0:
        return None
    return (new - old) / old


def fmt_delta(new: float, old: float, is_money: bool = False) -> str:
    """Format change as 'up/down X% to Y' with guardrails."""
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
# DATA LOAD + HELPERS (CSV feature-level OR Snowflake KPI-level)
# ============================================================
DATA_PATH = os.getenv("DEMO_DATA_PATH", "demo_features_latest.csv")


def _get_data_source() -> str:
    # Streamlit Cloud prefers secrets; local can use env var
    try:
        return str(st.secrets.get("DATA_SOURCE", os.getenv("DATA_SOURCE", "csv"))).lower().strip()
    except Exception:
        return str(os.getenv("DATA_SOURCE", "csv")).lower().strip()


def _load_data_from_snowflake() -> pd.DataFrame:
    """
    Snowflake loader for Streamlit Community Cloud.
    Expects Streamlit secrets:
      [snowflake] account/user/password/role/warehouse/database/schema
    """
    import snowflake.connector

    cfg = st.secrets["snowflake"]
    conn = snowflake.connector.connect(
        account=str(cfg["account"]),
        user=str(cfg["user"]),
        password=str(cfg["password"]),
        role=str(cfg.get("role", "")),
        warehouse=str(cfg.get("warehouse", "")),
        database=str(cfg.get("database", "")),
        schema=str(cfg.get("schema", "")),
    )

    # Transportation KPI daily view you validated
    sql = """
        SELECT *
        FROM NARS.PROCESSED.V_TRANSPORTATION_DAILY
        ORDER BY REPORTED_DATE
    """

    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    return df


def _coerce_app_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize columns for downstream logic.
    - Snowflake KPI mode: maps KPI columns to app-friendly fields
    - CSV mode: keeps row-level feature fields and enforces incurred = paid + outstanding
    """
    if df is None or len(df) == 0:
        return df

    if source == "snowflake":
        rename_map = {
            "REPORTED_DATE": "ASOF_DATE",
            "OPEN_CLAIMS": "OPEN_CT",
            "TOTAL_CLAIMS": "TOTAL_CT",
            "CLOSED_CLAIMS": "CLOSED_CT",
            "PAID": "PAID_AMT",
            "OUTSTANDING": "OUTSTANDING_AMT",
            "INCURRED": "INCURRED_AMT",
            "METRIC_NOTE": "METRIC_NOTE",
        }
        for src, tgt in rename_map.items():
            if src in df.columns:
                df = df.rename(columns={src: tgt})

        if "ASOF_DATE" in df.columns:
            df["ASOF_DATE"] = pd.to_datetime(df["ASOF_DATE"], errors="coerce").dt.date

        for col in ["PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in ["OPEN_CT", "TOTAL_CT", "CLOSED_CT"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        return df

    # CSV mode
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

    # Enforce incurred consistency if inputs exist
    if "PAID_AMT" in df.columns and "OUTSTANDING_AMT" in df.columns:
        df["INCURRED_AMT"] = (df["PAID_AMT"].fillna(0.0) + df["OUTSTANDING_AMT"].fillna(0.0)).round(2)

    return df


@st.cache_data(show_spinner=False, ttl=900)
def load_data(path: str) -> pd.DataFrame:
    """
    Returns a dataframe normalized for the app.
    - CSV mode: row-level features dataset (demo_features_latest.csv)
    - Snowflake mode: KPI-per-day transportation dataset (V_TRANSPORTATION_DAILY)
    """
    source = _get_data_source()
    if source == "snowflake":
        df = _load_data_from_snowflake()
    else:
        df = pd.read_csv(path)

    df = _coerce_app_schema(df, source=source)
    return df


def latest_and_prior_by_asof(df_scoped: pd.DataFrame):
    """Returns (latest_df, prior_df_or_none, latest_date, prior_date)."""
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


def snapshot_metrics(df_in: pd.DataFrame, source: str) -> dict:
    """
    CSV mode: sums flags/amounts across rows.
    Snowflake KPI mode: reads the single KPI row for that date.
    """
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
    """Return a clean, labeled incurred stratification table."""
    if df_in is None or len(df_in) == 0 or "INCURRED_AMT" not in df_in.columns:
        return pd.DataFrame(columns=["Incurred Range", "Feature Count"])

    bin_edges = [0, 10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000, float("inf")]
    labels = ["$0–$10K", "$10K–$25K", "$25K–$50K", "$50K–$100K", "$100K–$250K", "$250K–$1M", "$1M+"]

    cut = pd.cut(
        df_in["INCURRED_AMT"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    sev = cut.value_counts(dropna=False).reindex(labels, fill_value=0).reset_index()
    sev.columns = ["Incurred Range", "Feature Count"]
    return sev


# ============================================================
# LOAD DATA
# ============================================================
SOURCE = _get_data_source()

if SOURCE != "snowflake":
    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find {DATA_PATH}. Put the CSV next to app.py (or set DEMO_DATA_PATH).")
        st.stop()

df = load_data(DATA_PATH)

# Sidebar badge so you cannot accidentally lie tomorrow
st.sidebar.caption(f"Data source: **{SOURCE.upper()}**")

# Build filter options only when row-level columns exist (CSV mode)
states = ["All"] + sorted(df["LOSS_STATE"].dropna().unique().tolist()) if (SOURCE != "snowflake" and "LOSS_STATE" in df.columns) else ["All"]
years = ["All"] + sorted(df["ACCIDENT_YEAR"].dropna().unique().tolist()) if (SOURCE != "snowflake" and "ACCIDENT_YEAR" in df.columns) else ["All"]
adjusters = ["All"] + sorted(df["ADJUSTER_ID"].dropna().unique().tolist()) if (SOURCE != "snowflake" and "ADJUSTER_ID" in df.columns) else ["All"]
coverages = ["All"] + sorted(df["COVERAGE_CODE"].dropna().unique().tolist()) if (SOURCE != "snowflake" and "COVERAGE_CODE" in df.columns) else ["All"]


# ============================================================
# HEADER + LAYOUT + FILTER PANEL
# ============================================================
LOGO_PATH = os.getenv("NARS_LOGO_PATH", "narslogo.jpg")

header_left, header_right = st.columns([1, 5], vertical_alignment="center")
with header_left:
    if LOGO_PATH and os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=220)

with header_right:
    st.title("Claims Intelligence – Daily Summary (Demo)")
    st.write(
        "Representative dataset demonstrating NARS feature-level metric logic "
        "and the client delivery experience (daily email + dashboard)."
    )

main_col, filter_col = st.columns([4, 1], gap="large")

with filter_col:
    st.markdown('<div class="sticky-filter">', unsafe_allow_html=True)
    st.markdown("### Filters")

    if SOURCE == "snowflake":
        st.info("Snowflake KPI mode: row-level filters are disabled for this view.")
        sel_state = "All"
        sel_year = "All"
        sel_adjuster = "All"
        sel_cov = "All"
    else:
        sel_state = st.selectbox("State", states, index=0)
        sel_year = st.selectbox("Accident Year", years, index=0)
        sel_adjuster = st.selectbox("Adjuster", adjusters, index=0)
        sel_cov = st.selectbox("Coverage Type", coverages, index=0)

        st.markdown(
            '<div class="small-muted">Filters apply to both the dashboard and the emailed snapshot.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Reset filters", use_container_width=True):
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FILTER LOGIC + KPI COMPUTATIONS
# ============================================================
f = df.copy()

if SOURCE == "snowflake":
    filter_summary = "Snowflake KPI mode (row-level filters disabled)"
else:
    filter_parts = []
    if sel_state != "All":
        filter_parts.append(f"State={sel_state}")
    if sel_year != "All":
        filter_parts.append(f"AccidentYear={sel_year}")
    if sel_adjuster != "All":
        filter_parts.append(f"Adjuster={sel_adjuster}")
    if sel_cov != "All":
        filter_parts.append(f"CoverageType={sel_cov}")
    filter_summary = ", ".join(filter_parts) if filter_parts else "None (All data)"

    if sel_state != "All" and "LOSS_STATE" in f.columns:
        f = f[f["LOSS_STATE"] == sel_state]
    if sel_year != "All" and "ACCIDENT_YEAR" in f.columns:
        f = f[f["ACCIDENT_YEAR"] == int(sel_year)]
    if sel_adjuster != "All" and "ADJUSTER_ID" in f.columns:
        f = f[f["ADJUSTER_ID"] == sel_adjuster]
    if sel_cov != "All" and "COVERAGE_CODE" in f.columns:
        f = f[f["COVERAGE_CODE"] == sel_cov]

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

# Session memory for “since last email” deltas (EMAIL only)
if "last_sent_metrics" not in st.session_state:
    st.session_state["last_sent_metrics"] = None

last = st.session_state["last_sent_metrics"]
open_vs_last = fmt_delta(open_ct, last["open_ct"], is_money=False) if last else "no prior email sent this session"
inc_vs_last = fmt_delta(incurred, last["incurred"], is_money=True) if last else "no prior email sent this session"
hs_vs_last = fmt_delta(high_sev_ct, last["high_sev_ct"], is_money=False) if (last and SOURCE != "snowflake") else "no prior email sent this session"


# ============================================================
# ASK NARS (DETERMINISTIC PROTOTYPE)
# (Only meaningful in CSV row-level mode.)
# ============================================================
def answer_ask_nars(question: str) -> dict:
    q = (question or "").strip().lower()

    if SOURCE == "snowflake":
        return {"text": "Ask NARS is disabled in Snowflake KPI mode (no feature-level rows in this dataset)."}

    if len(f_latest) == 0:
        return {"text": "No features match the current filters. Adjust filters to ask questions."}

    def money(v: float) -> str:
        return fmt_money(float(v))

    base = f_latest
    top = base.sort_values("INCURRED_AMT", ascending=False).head(10) if "INCURRED_AMT" in base.columns else base.head(10)

    top_cols = [
        c
        for c in ["FEATURE_KEY", "CLAIM_NBR", "LOSS_STATE", "ACCIDENT_YEAR", "COVERAGE_CODE", "ADJUSTER_ID", "INCURRED_AMT"]
        if c in top.columns
    ]

    if not q:
        return {"text": "Try: 'top 10 severe', 'total incurred', 'open features', 'state with highest incurred'."}

    if "top" in q and ("severe" in q or "severity" in q or "incurred" in q or "exposure" in q):
        return {"text": f"Top exposure drivers in scope (sorted by incurred). Features in scope: {total_ct:,}.", "table": top.loc[:, top_cols]}

    if "incurred" in q or "exposure" in q:
        return {"text": f"Total Incurred in scope is {money(incurred)} across {total_ct:,} features."}

    if "outstanding" in q:
        return {"text": f"Outstanding in scope is {money(outstanding)}."}

    if "paid" in q:
        return {"text": f"Paid in scope is {money(paid)}."}

    if "open" in q:
        return {"text": f"Open Features: {open_ct:,} (out of {total_ct:,} features in scope)."}

    if "high" in q and ("severity" in q or "severe" in q):
        pct = (high_sev_ct / open_ct * 100) if open_ct else 0.0
        return {"text": f"High Severity Features: {high_sev_ct:,}. That’s {pct:,.1f}% of open features in scope."}

    if "state" in q and ("most" in q or "highest" in q):
        if "LOSS_STATE" not in base.columns or "INCURRED_AMT" not in base.columns:
            return {"text": "State/incurred fields not available in this dataset."}
        g = base.groupby("LOSS_STATE", dropna=True)["INCURRED_AMT"].sum().sort_values(ascending=False)
        if len(g) == 0:
            return {"text": "No state data available in the current filter set."}
        return {"text": f"Highest incurred state in scope is {g.index[0]} at {money(g.iloc[0])}."}

    return {"text": "Try: 'top 10 severe', 'total incurred', 'open features', 'state with highest incurred'."}


def render_answer(ans: dict):
    st.write(ans.get("text", ""))
    if "table" in ans:
        tbl = ans["table"].copy()
        for col in ["PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT"]:
            if col in tbl.columns:
                tbl[col] = tbl[col].map(lambda v: f"{float(v):,.2f}")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ============================================================
# MAIN CONTENT
# ============================================================
with main_col:
    st.caption(f"As-of: {asof if asof else 'N/A'}")

    if len(f_latest) == 0:
        st.warning("No rows match the current scope. Adjust filters (CSV mode) or check data currency (Snowflake mode).")

    # Ask NARS (only if CSV)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Ask NARS (prototype)")
    st.caption("Deterministic responses computed from the filtered dataset. Not production AI.")

    if SOURCE == "snowflake":
        st.info("Ask NARS is disabled in Snowflake KPI mode.")
    else:
        quick1, quick2, quick3, quick4 = st.columns(4)
        if quick1.button("Top 10 severe", use_container_width=True):
            render_answer(answer_ask_nars("top 10 severe"))
        if quick2.button("Total incurred", use_container_width=True):
            render_answer(answer_ask_nars("total incurred"))
        if quick3.button("Open features", use_container_width=True):
            render_answer(answer_ask_nars("open features"))
        if quick4.button("State w/ highest incurred", use_container_width=True):
            render_answer(answer_ask_nars("state w/ highest incurred"))

        with st.form("ask_nars_form", clear_on_submit=False):
            q = st.text_input("Ask NARS", placeholder="e.g., top 10 severe, total incurred, open features")
            submitted = st.form_submit_button("Ask")

        if submitted:
            render_answer(answer_ask_nars(q))

    st.markdown("</div>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Claims", f"{open_ct:,}")
    k2.metric("Total Incurred", fmt_money_short(incurred))
    k3.metric("Paid", fmt_money_short(paid))
    k4.metric("Outstanding", fmt_money_short(outstanding))
    k5.metric("High Severity", f"{high_sev_ct:,}" if SOURCE != "snowflake" else "N/A")

    st.caption(f"Full totals: Total Incurred {fmt_money(incurred)} | Paid {fmt_money(paid)} | Outstanding {fmt_money(outstanding)}")
    st.write("")

    # Headlines
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Today’s Headlines")

    st.write(f"**In scope:** {filter_summary}")

    if prior:
        open_vs_prior = fmt_delta(open_ct, prior["open_ct"], is_money=False)
        inc_vs_prior = fmt_delta(incurred, prior["incurred"], is_money=True)
        if SOURCE != "snowflake":
            hs_vs_prior = fmt_delta(high_sev_ct, prior["high_sev_ct"], is_money=False)
        else:
            hs_vs_prior = "N/A"
        prior_label = f"{prior_asof}" if prior_asof else "prior as-of"
    else:
        open_vs_prior = "no prior as-of available"
        inc_vs_prior = "no prior as-of available"
        hs_vs_prior = "no prior as-of available"
        prior_label = "prior as-of"

    st.write(f"**As-of:** {asof if asof else 'N/A'}  |  **Comparison:** {prior_label}")
    st.write(f"• **Open:** {open_ct:,} ({open_vs_prior})")
    st.write(f"• **Total Incurred:** {fmt_money(incurred)} ({inc_vs_prior})")
    if SOURCE != "snowflake":
        st.write(f"• **High Severity:** {high_sev_ct:,} ({hs_vs_prior})  |  Share of open: **{sev_share*100:,.1f}%**")
    else:
        st.write("• **High Severity:** N/A (KPI mode dataset)")

    # Tabs (row-level charts only make sense for CSV)
    tab_open, tab_incurred, tab_sev, tab_drivers = st.tabs(["Open Features", "Total Incurred", "High Severity", "Top Exposure Drivers"])

    with tab_open:
        if SOURCE == "snowflake":
            st.info("Trend charts are disabled in KPI mode (this dataset is already aggregated).")
        elif "ACCIDENT_YEAR" in f_latest.columns:
            by_year = f_latest.groupby("ACCIDENT_YEAR", dropna=True).size().reset_index(name="Feature Count").sort_values("ACCIDENT_YEAR")
            st.bar_chart(by_year.set_index("ACCIDENT_YEAR")["Feature Count"])
        else:
            st.info("ACCIDENT_YEAR not available in this dataset for the Open Features trend.")

    with tab_incurred:
        if SOURCE == "snowflake":
            st.info("Trend charts are disabled in KPI mode (this dataset is already aggregated).")
        elif "ACCIDENT_YEAR" in f_latest.columns and "INCURRED_AMT" in f_latest.columns:
            inc_by_year = f_latest.groupby("ACCIDENT_YEAR", dropna=True)["INCURRED_AMT"].sum().reset_index().sort_values("ACCIDENT_YEAR")
            st.bar_chart(inc_by_year.set_index("ACCIDENT_YEAR")["INCURRED_AMT"])
            st.caption("Chart shows Total Incurred summed by Accident Year (feature-level).")
        else:
            st.info("ACCIDENT_YEAR / INCURRED_AMT not available for Total Incurred trend.")

    with tab_sev:
        if SOURCE == "snowflake":
            st.info("Severity stratification is disabled in KPI mode (no feature-level rows).")
        else:
            sev_tbl = build_incurred_stratification(f_latest)
            if len(sev_tbl):
                st.dataframe(sev_tbl, use_container_width=True, hide_index=True)
                st.caption("YTD Feature Stratification by Incurred Range (feature-level).")
            else:
                st.info("No incurred data available for severity stratification in the current filter set.")

    with tab_drivers:
        if SOURCE == "snowflake":
            st.info("Exposure drivers are disabled in KPI mode (no feature-level rows).")
        else:
            if "INCURRED_AMT" in f_latest.columns:
                cols = ["FEATURE_KEY", "CLAIM_NBR", "LOSS_STATE", "ACCIDENT_YEAR", "COVERAGE_CODE", "ADJUSTER_ID", "PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT", "OPEN_FLAG"]
                cols = [c for c in cols if c in f_latest.columns]
                drivers = f_latest.sort_values("INCURRED_AMT", ascending=False).loc[:, cols].head(15).copy()
                for col in ["PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT"]:
                    if col in drivers.columns:
                        drivers[col] = drivers[col].map(lambda v: f"{float(v):,.2f}")
                st.dataframe(drivers, use_container_width=True, hide_index=True)
                st.caption("Top exposure drivers = highest incurred features in scope.")
            else:
                st.info("INCURRED_AMT not available to compute exposure drivers.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ============================================================
    # Email demo (SAFE / OPTIONAL)
    # ============================================================
    SHOW_EMAIL_DEMO = False  # keep False for tomorrow's Accredited demo

    if SHOW_EMAIL_DEMO:
        st.subheader("Daily Email Demo")
        st.caption("Sends a summary email via Outlook (local demo). Preview below shows exactly what is sent.")

        _client = "Client"
        _asof = asof or "N/A"

        _filter_summary = str(filter_summary or "None")
        _total_ct = int(total_ct or 0)
        _open_ct = int(open_ct or 0)
        _high_sev_ct = int(high_sev_ct or 0)
        _incurred = float(incurred or 0.0)
        _paid = float(paid or 0.0)
        _outstanding = float(outstanding or 0.0)

        _open_vs_last = str(open_vs_last or "N/A")
        _inc_vs_last = str(inc_vs_last or "N/A")
        _hs_vs_last = str(hs_vs_last or "N/A")

        to_email = st.text_input("Send to", value=os.getenv("DEMO_EMAIL_TO", ""))
        dashboard_link = st.text_input("Dashboard link", value=os.getenv("DEMO_DASHBOARD_LINK", "https://nars-demo.streamlit.app"))

        summary_html = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.4;">
          <h2 style="margin:0 0 8px 0;">{_client} Daily Claims Snapshot</h2>

          <div style="color:#555; margin-bottom:10px;">
            As-of: {_asof}
          </div>

          <div style="padding:10px 12px; background:#f6f7f9; border-radius:8px; margin-bottom:12px;">
            <div style="font-size:14px; margin-bottom:6px;"><b>Today’s headlines</b></div>
            <div style="margin:4px 0;">• Open Features {_open_vs_last} (since last email)</div>
            <div style="margin:4px 0;">• Total Incurred {_inc_vs_last} (since last email)</div>
            <div style="margin:4px 0;">• High Severity Features {_hs_vs_last} (since last email)</div>
          </div>

          <div style="font-size:12px; color:#666; margin-bottom:12px;">
            <b>Filters applied:</b> {_filter_summary}<br/>
            <b>Features included (latest as-of):</b> {_total_ct:,}
          </div>

          <ul>
            <li><b>Open Features:</b> {_open_ct:,}</li>
            <li><b>Total Incurred:</b> {fmt_money(_incurred)}</li>
            <li><b>Paid:</b> {fmt_money(_paid)}</li>
            <li><b>Outstanding:</b> {fmt_money(_outstanding)}</li>
            <li><b>High Severity Features:</b> {_high_sev_ct:,}</li>
          </ul>

          <p style="margin-top:12px;">
            <a href="{dashboard_link}">Open dashboard</a>
          </p>

          <hr/>
          <div style="font-size:12px; color:#777;">
            Demo note: This snapshot demonstrates the NARS delivery experience (email + dashboard).
          </div>
        </div>
        """

        with st.expander("Preview email HTML"):
            st.markdown(summary_html, unsafe_allow_html=True)

        # Guard: Streamlit Cloud won't have Outlook libs
        can_send = callable(locals().get("send_demo_email_outlook", None))
        if not can_send:
            st.info("Email sending is disabled in this environment (demo UI only).")

        if st.button("Send me the demo email", type="primary", disabled=not can_send):
            if not to_email.strip():
                st.error("Enter an email address in 'Send to'.")
            else:
                st.session_state["last_sent_metrics"] = {
                    "sent_at": datetime.now().isoformat(timespec="seconds"),
                    "filter_summary": _filter_summary,
                    "open_ct": _open_ct,
                    "incurred": _incurred,
                    "paid": _paid,
                    "outstanding": _outstanding,
                    "high_sev_ct": _high_sev_ct,
                }
                ok, msg = send_demo_email_outlook(to_email=to_email.strip(), subject=f"{_client} Daily Claims Snapshot (Demo)", html_body=summary_html)
                if ok:
                    st.success("Email sent.")
                else:
                    st.error(msg)
