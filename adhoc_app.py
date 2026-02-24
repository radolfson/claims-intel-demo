# adhoc_app.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cover Whale Ad Hoc Delivery Preview",
    page_icon="📄",
    layout="wide",
)

# -----------------------------
# Styling (match dashboard vibe)
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1400px; }
.cw-card {
  background: #FFFFFF;
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.cw-label {
  font-size: 12px;
  color: rgba(49, 51, 63, 0.70);
  margin-bottom: 2px;
}
.cw-value {
  font-size: 22px;
  font-weight: 650;
  letter-spacing: 0.2px;
}
.cw-h2 {
  font-size: 22px;
  font-weight: 750;
  margin: 0.2rem 0 0.6rem 0;
}
.cw-muted {
  color: rgba(49, 51, 63, 0.65);
  font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def _money(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _fmt_money(v):
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "—"


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def load_claims_table() -> pd.DataFrame:
    """
    Prefer: demo_features_latest.csv (same repo) -> transform to a claim-like output.
    Fallback: generate synthetic claim list so the demo always works.
    """
    path = "demo_features_latest.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)

        # Normalize dates if present
        for c in ["report_date", "loss_date", "open_date", "close_date", "reserve_open_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        # Try to map to a claim-list output (claim-level)
        # We'll build a standard output schema even if some fields are missing.
        out = pd.DataFrame()
        out["Client"] = df.get("client_name", df.get("client", "Cover Whale"))
        out["Policy Number"] = df.get("policy_number", df.get("policy_id", df.get("policy", "—")))
        out["Claim Number"] = df.get("claim_number", df.get("claim_id", df.get("claim", "—")))
        out["Feature Key"] = df.get("feature_key", df.get("feature_id", "—"))
        out["State"] = df.get("state", "—")
        out["Coverage"] = df.get("coverage_type", df.get("coverage_code", df.get("coverage", "—")))
        out["Loss Date"] = df.get("loss_date", df.get("accident_date", pd.NaT))
        out["Open Date"] = df.get("reserve_open_date", df.get("open_date", pd.NaT))
        out["Status"] = df.get("feature_status", df.get("claim_status", "—"))
        out["Adjuster"] = df.get("adjuster", df.get("adjuster_name", "—"))
        out["Defense Firm"] = df.get("defense_firm", df.get("law_firm", "—"))
        out["Denial Reason"] = df.get("denial_reason", "—")

        # Financials
        out["Paid"] = df.get("paid_amount", df.get("paid", 0)).apply(_money)
        out["Outstanding"] = df.get("outstanding_amount", df.get("outstanding", 0)).apply(_money)
        out["Incurred"] = df.get("incurred_amount", df.get("total_incurred", out["Paid"] + out["Outstanding"])).apply(_money)

        # If you want the delivered output to feel like a “loss run”, keep it clean:
        out = out.replace({np.nan: "—"})
        return out

    # -----------------------------
    # Fallback synthetic data
    # -----------------------------
    rng = np.random.default_rng(42)
    n = 600  # "big list" feel
    today = pd.Timestamp.today().normalize()

    clients = ["FLORIDA HOSPITAL SELF INSURANCE FUND", "LAUNCH ENVIRONMENTAL", "COVER WHALE (DEMO)"]
    coverages = ["WC-IND", "AUTO", "GL", "EXCESS"]
    states = ["FL", "TX", "CA", "IL", "GA", "NC", "NJ"]
    denial_reasons = ["None", "Late Notice", "Coverage Exclusion", "Policy Lapse", "MCS90"]

    loss_dates = pd.to_datetime(today - pd.to_timedelta(rng.integers(30, 365 * 5, size=n), unit="D"))
    open_dates = loss_dates + pd.to_timedelta(rng.integers(0, 30, size=n), unit="D")

    paid = rng.gamma(shape=2.0, scale=15000, size=n)
    paid[rng.random(n) < 0.35] = 0  # many claims have no paid yet
    outstanding = rng.gamma(shape=2.0, scale=12000, size=n)
    incurred = paid + outstanding

    df = pd.DataFrame(
        {
            "Client": rng.choice(clients, size=n, p=[0.55, 0.25, 0.20]),
            "Policy Number": [f"POL-{rng.integers(100000, 999999)}" for _ in range(n)],
            "Claim Number": [f"CLM-{rng.integers(100000, 999999)}" for _ in range(n)],
            "Feature Key": [f"F-{rng.integers(2021, 2026)}-{rng.integers(10000, 99999)}" for _ in range(n)],
            "State": rng.choice(states, size=n),
            "Coverage": rng.choice(coverages, size=n),
            "Loss Date": loss_dates,
            "Open Date": open_dates,
            "Status": rng.choice(["OPEN", "CLOSED"], size=n, p=[0.72, 0.28]),
            "Adjuster": rng.choice(["All Adjusters", "J. Smith", "A. Patel", "M. Chen", "R. Garcia"], size=n),
            "Defense Firm": rng.choice(["All Firms", "Smith & Cole", "Parker LLP", "Hart & Gray", "None"], size=n),
            "Denial Reason": rng.choice(denial_reasons, size=n, p=[0.65, 0.10, 0.10, 0.10, 0.05]),
            "Paid": paid,
            "Outstanding": outstanding,
            "Incurred": incurred,
        }
    )
    return df


def card(col, label, value):
    col.markdown(
        f"""
        <div class="cw-card">
          <div class="cw-label">{label}</div>
          <div class="cw-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Masthead
# -----------------------------
mast = st.columns([0.18, 0.82], vertical_alignment="center")
with mast[0]:
    logo_path = "narslogo.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=210)
with mast[1]:
    st.markdown(
        """
        <div style="line-height:1.05;">
          <div style="font-size:30px; font-weight:800; letter-spacing:0.2px;">
            Cover Whale Ad Hoc Delivery Preview
          </div>
          <div class="cw-muted" style="margin-top:6px;">
            Client-facing preview of what is delivered for ad hoc loss run requests (data-only output).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# -----------------------------
# Load data (no uploader)
# -----------------------------
df = load_claims_table()

# Normalize date types
for c in ["Loss Date", "Open Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# -----------------------------
# Filters (simple + professional)
# -----------------------------
st.markdown("<div class='cw-h2'>Request Context</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([0.30, 0.25, 0.25, 0.20], gap="medium")

client_vals = sorted(df["Client"].dropna().astype(str).unique().tolist()) if "Client" in df.columns else []
policy_vals = sorted(df["Policy Number"].dropna().astype(str).unique().tolist())[:500] if "Policy Number" in df.columns else []

with c1:
    client_sel = st.selectbox("Client", ["All Clients"] + client_vals, index=0)
with c2:
    policy_sel = st.selectbox("Policy (optional)", ["All Policies"] + policy_vals, index=0)
with c3:
    out_type = st.selectbox("Output Type", ["Data Only (Excel)", "PDF", "CSV"], index=0)
with c4:
    priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)

# Date range filter
min_date = df["Loss Date"].min() if "Loss Date" in df.columns else pd.NaT
max_date = df["Loss Date"].max() if "Loss Date" in df.columns else pd.NaT
if pd.isna(min_date) or pd.isna(max_date):
    min_date = pd.Timestamp("2021-01-01")
    max_date = pd.Timestamp.today().normalize()

dcol1, dcol2 = st.columns([0.5, 0.5], gap="medium")
with dcol1:
    start_date = st.date_input("Loss Date Start", value=min_date.date())
with dcol2:
    end_date = st.date_input("Loss Date End", value=max_date.date())

# Apply filters
dff = df.copy()
if client_sel != "All Clients":
    dff = dff[dff["Client"] == client_sel]
if policy_sel != "All Policies":
    dff = dff[dff["Policy Number"] == policy_sel]

if "Loss Date" in dff.columns:
    dff = dff[
        (pd.to_datetime(dff["Loss Date"], errors="coerce") >= pd.Timestamp(start_date))
        & (pd.to_datetime(dff["Loss Date"], errors="coerce") <= pd.Timestamp(end_date))
    ]

# -----------------------------
# Summary cards
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Delivery Summary</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
card(k1, "Rows Delivered", f"{len(dff):,}")
card(k2, "Total Paid", _fmt_money(dff["Paid"].sum()) if "Paid" in dff.columns else "—")
card(k3, "Total Outstanding", _fmt_money(dff["Outstanding"].sum()) if "Outstanding" in dff.columns else "—")
card(k4, "Total Incurred", _fmt_money(dff["Incurred"].sum()) if "Incurred" in dff.columns else "—")
card(k5, "Output", out_type)

st.markdown(
    f"<div class='cw-muted'>This preview simulates the delivered file for an ad hoc request. "
    f"It is formatted to match the look-and-feel of the executive dashboard.</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Big list of claims (like the PDF vibe)
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Loss Run Detail (Claim List)</div>", unsafe_allow_html=True)

# Sort like a typical loss run output
sort_cols = [c for c in ["Loss Date", "Open Date", "Incurred"] if c in dff.columns]
if sort_cols:
    dff = dff.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

# Display-friendly formatting
display_df = dff.copy()

for c in ["Paid", "Outstanding", "Incurred"]:
    if c in display_df.columns:
        display_df[c] = display_df[c].apply(_fmt_money)

for c in ["Loss Date", "Open Date"]:
    if c in display_df.columns:
        display_df[c] = pd.to_datetime(display_df[c], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")

# Keep columns tight, like a real loss run export
preferred_cols = [
    "Client",
    "Policy Number",
    "Claim Number",
    "Feature Key",
    "State",
    "Coverage",
    "Loss Date",
    "Open Date",
    "Status",
    "Paid",
    "Outstanding",
    "Incurred",
    "Adjuster",
    "Defense Firm",
    "Denial Reason",
]
cols = [c for c in preferred_cols if c in display_df.columns]
st.dataframe(display_df[cols], use_container_width=True, height=650, hide_index=True)

# -----------------------------
# Download package (what client gets)
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Download</div>", unsafe_allow_html=True)

# Raw export should not have formatted money strings
export_df = dff.copy()
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

d1, d2 = st.columns([0.35, 0.65], gap="medium")
with d1:
    st.download_button(
        label="Download Delivered File (CSV)",
        data=csv_bytes,
        file_name="loss_run_delivery.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d2:
    st.markdown(
        "<div class='cw-muted'>For the demo: this represents the data-only delivery file. "
        "In production you’d attach Excel/PDF, but the preview and schema are what matter here.</div>",
        unsafe_allow_html=True,
    )