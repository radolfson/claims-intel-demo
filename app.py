import os
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# EMAIL VIA OUTLOOK (WINDOWS DESKTOP)
# ============================================================
def send_demo_email_outlook(to_email: str, subject: str, html_body: str) -> tuple[bool, str]:
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
# APP CONFIG + THEME
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
  padding: 14px;
}}
.small-muted {{
  color: #666;
  font-size: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOAD
# ============================================================
DATA_PATH = os.getenv("DEMO_DATA_PATH", "demo_features_latest.csv")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "ACCIDENT_YEAR" in df.columns:
        df["ACCIDENT_YEAR"] = pd.to_numeric(df["ACCIDENT_YEAR"], errors="coerce").fillna(0).astype(int)

    for c in ["OPEN_FLAG", "HIGH_SEVERITY_FLAG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["PAID_AMT", "OUTSTANDING_AMT", "INCURRED_AMT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "ASOF_DATE" in df.columns:
        df["ASOF_DATE"] = pd.to_datetime(df["ASOF_DATE"], errors="coerce").dt.date

    if "PAID_AMT" in df.columns and "OUTSTANDING_AMT" in df.columns:
        df["INCURRED_AMT"] = df["PAID_AMT"] + df["OUTSTANDING_AMT"]

    return df


df = load_data(DATA_PATH)

states = ["All"] + sorted(df["LOSS_STATE"].dropna().unique().tolist())
years = ["All"] + sorted(df["ACCIDENT_YEAR"].dropna().unique().tolist())
adjusters = ["All"] + sorted(df["ADJUSTER_ID"].dropna().unique().tolist())
coverages = ["All"] + sorted(df["COVERAGE_CODE"].dropna().unique().tolist())

# ============================================================
# HEADER
# ============================================================
st.title("Claims Intelligence – Daily Summary (Demo)")
st.write(
    "Representative dataset demonstrating feature-level claims intelligence, "
    "daily monitoring, and executive-ready delivery."
)

# ============================================================
# FILTERS (CLIENT REMOVED)
# ============================================================
main_col, filter_col = st.columns([4, 1], gap="large")

with filter_col:
    st.markdown('<div class="sticky-filter">', unsafe_allow_html=True)
    st.markdown("### Filters")

    sel_state = st.selectbox("State", states)
    sel_year = st.selectbox("Accident Year", years)
    sel_adjuster = st.selectbox("Adjuster", adjusters)
    sel_cov = st.selectbox("Coverage Type", coverages)

    if st.button("Reset filters", use_container_width=True):
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FILTER APPLICATION
# ============================================================
f = df.copy()

if sel_state != "All":
    f = f[f["LOSS_STATE"] == sel_state]
if sel_year != "All":
    f = f[f["ACCIDENT_YEAR"] == int(sel_year)]
if sel_adjuster != "All":
    f = f[f["ADJUSTER_ID"] == sel_adjuster]
if sel_cov != "All":
    f = f[f["COVERAGE_CODE"] == sel_cov]

# ============================================================
# METRICS
# ============================================================
latest_date = max(f["ASOF_DATE"])
latest = f[f["ASOF_DATE"] == latest_date]

open_ct = int(latest["OPEN_FLAG"].sum())
total_ct = len(latest)
incurred = latest["INCURRED_AMT"].sum()
paid = latest["PAID_AMT"].sum()
outstanding = latest["OUTSTANDING_AMT"].sum()
high_sev = int(latest["HIGH_SEVERITY_FLAG"].sum())

# ============================================================
# MAIN CONTENT
# ============================================================
with main_col:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Open Features", f"{open_ct:,}")
    k2.metric("Total Incurred", f"${incurred:,.0f}")
    k3.metric("Paid", f"${paid:,.0f}")
    k4.metric("Outstanding", f"${outstanding:,.0f}")
    k5.metric("High Severity Features", f"{high_sev:,}")

    st.caption(f"As-of: {latest_date} | Feature-level demo dataset")

    st.markdown("### Top Exposure Drivers")
    top = latest.sort_values("INCURRED_AMT", ascending=False).head(15)
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("### Daily Email Demo")
    st.caption("Preview of automated daily executive snapshot.")

    to_email = st.text_input("Send to")
    dashboard_link = st.text_input("Dashboard link", value="")

    summary_html = f"""
    <h2>Daily Claims Snapshot</h2>
    <p>As-of: {latest_date}</p>
    <ul>
      <li>Open Features: {open_ct:,}</li>
      <li>Total Incurred: ${incurred:,.0f}</li>
      <li>Paid: ${paid:,.0f}</li>
      <li>Outstanding: ${outstanding:,.0f}</li>
      <li>High Severity Features: {high_sev:,}</li>
    </ul>
    <a href="{dashboard_link}">Open dashboard</a>
    """

    if st.button("Send demo email"):
        send_demo_email_outlook(
            to_email=to_email,
            subject="Daily Claims Snapshot (Demo)",
            html_body=summary_html,
        )
