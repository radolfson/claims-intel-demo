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

/* Give the top area more vertical breathing room so logos never look clipped */
.cw-mast { padding: 0.25rem 0 0.75rem 0; }

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


def _safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


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


def load_claims_table() -> pd.DataFrame:
    """Load claim-level rows.

    Prefer demo_features_latest.csv (same repo as dashboard).
    If the CSV is missing OR if we can't derive usable dates, fall back to synthetic rows.
    """

    csv_path = "demo_features_latest.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Parse candidate date columns we might need
        for c in ["report_date", "loss_date", "accident_date", "open_date", "close_date", "reserve_open_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        out = pd.DataFrame()
        out["Client"] = df.get("client_name", df.get("client", "COVER WHALE (DEMO)"))
        out["Policy Number"] = df.get("policy_number", df.get("policy_id", df.get("policy", "—")))
        out["Claim Number"] = df.get("claim_number", df.get("claim_id", df.get("claim", "—")))
        out["Feature Key"] = df.get("feature_key", df.get("feature_id", "—"))
        out["State"] = df.get("state", "—")
        out["Coverage"] = df.get("coverage_type", df.get("coverage_code", df.get("coverage", "—")))

        # Try to derive Loss Date / Open Date in a resilient way.
        # Some demo extracts won't have loss_date; use accident_date; else report_date.
        loss = df.get("loss_date", None)
        if loss is None:
            loss = df.get("accident_date", None)
        if loss is None:
            loss = df.get("report_date", None)

        opn = df.get("reserve_open_date", None)
        if opn is None:
            opn = df.get("open_date", None)
        if opn is None:
            opn = df.get("report_date", None)

        out["Loss Date"] = pd.to_datetime(loss, errors="coerce")
        out["Open Date"] = pd.to_datetime(opn, errors="coerce")

        out["Status"] = df.get("feature_status", df.get("claim_status", "—"))
        out["Adjuster"] = df.get("adjuster", df.get("adjuster_name", "—"))
        out["Defense Firm"] = df.get("defense_firm", df.get("law_firm", "—"))
        out["Denial Reason"] = df.get("denial_reason", "—")

        out["Paid"] = df.get("paid_amount", df.get("paid", 0)).apply(_money)
        out["Outstanding"] = df.get("outstanding_amount", df.get("outstanding", 0)).apply(_money)

        if "incurred_amount" in df.columns:
            out["Incurred"] = df["incurred_amount"].apply(_money)
        elif "total_incurred" in df.columns:
            out["Incurred"] = df["total_incurred"].apply(_money)
        else:
            out["Incurred"] = out["Paid"] + out["Outstanding"]

        # If dates are completely missing/NaT, create plausible ones so the demo isn't empty.
        if out["Loss Date"].isna().all() or out["Open Date"].isna().all():
            # Use report_date if present, otherwise synthesize
            base = pd.to_datetime(df.get("report_date", pd.Timestamp.today()), errors="coerce")
            if isinstance(base, pd.Series) and base.notna().any():
                base = base.fillna(base.max())
            else:
                base = pd.Series([pd.Timestamp.today()] * len(df))

            rng = np.random.default_rng(42)
            # Backdate loss 0-5 years, open 0-30 days after loss
            loss_dates = pd.to_datetime(base) - pd.to_timedelta(rng.integers(30, 365 * 5, size=len(out)), unit="D")
            open_dates = loss_dates + pd.to_timedelta(rng.integers(0, 30, size=len(out)), unit="D")
            out["Loss Date"] = loss_dates
            out["Open Date"] = open_dates

        out = out.replace({np.nan: "—"})
        return out

    # -----------------------------
    # Fallback synthetic data
    # -----------------------------
    rng = np.random.default_rng(42)
    n = 750  # big list vibe
    today = pd.Timestamp.today().normalize()

    clients = [
        "FLORIDA HOSPITAL SELF INSURANCE FUND",
        "LAUNCH ENVIRONMENTAL",
        "COVER WHALE (DEMO)",
    ]
    coverages = ["WC-IND", "AUTO", "GL", "EXCESS"]
    states = ["FL", "TX", "CA", "IL", "GA", "NC", "NJ"]
    denial_reasons = ["None", "Late Notice", "Coverage Exclusion", "Policy Lapse", "MCS90"]

    loss_dates = pd.to_datetime(today - pd.to_timedelta(rng.integers(30, 365 * 5, size=n), unit="D"))
    open_dates = loss_dates + pd.to_timedelta(rng.integers(0, 30, size=n), unit="D")

    paid = rng.gamma(shape=2.0, scale=15000, size=n)
    paid[rng.random(n) < 0.35] = 0
    outstanding = rng.gamma(shape=2.0, scale=12000, size=n)
    incurred = paid + outstanding

    return pd.DataFrame(
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
            "Adjuster": rng.choice(["J. Smith", "A. Patel", "M. Chen", "R. Garcia"], size=n),
            "Defense Firm": rng.choice(["Smith & Cole", "Parker LLP", "Hart & Gray", "None"], size=n),
            "Denial Reason": rng.choice(denial_reasons, size=n, p=[0.65, 0.10, 0.10, 0.10, 0.05]),
            "Paid": paid,
            "Outstanding": outstanding,
            "Incurred": incurred,
        }
    )


# -----------------------------
# Masthead (fix clipped logo)
# -----------------------------
st.markdown('<div class="cw-mast">', unsafe_allow_html=True)

mast = st.columns([0.20, 0.80], vertical_alignment="center")
with mast[0]:
    logo_path = "narslogo.jpg"
    if os.path.exists(logo_path):
        # Use a slightly smaller width and add a little top padding
        st.markdown("<div style='padding-top:6px;'></div>", unsafe_allow_html=True)
        st.image(logo_path, width=230)

with mast[1]:
    st.markdown(
        """
        <div style="line-height:1.06; padding-top:6px;">
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

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Data (always present)
# -----------------------------
df = load_claims_table()

# Ensure dates are datetime
for c in ["Loss Date", "Open Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# -----------------------------
# Delivery Summary (keep it, kill Request Context)
# -----------------------------
st.markdown("<div class='cw-h2'>Delivery Summary</div>", unsafe_allow_html=True)

rows = len(df)
paid_total = df["Paid"].sum() if "Paid" in df.columns else 0
out_total = df["Outstanding"].sum() if "Outstanding" in df.columns else 0
inc_total = df["Incurred"].sum() if "Incurred" in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
card(k1, "Rows Delivered", f"{rows:,}")
card(k2, "Total Paid", _fmt_money(paid_total))
card(k3, "Total Outstanding", _fmt_money(out_total))
card(k4, "Total Incurred", _fmt_money(inc_total))
card(k5, "Output", "Data Only (Excel)")

st.markdown(
    "<div class='cw-muted'>This preview simulates the delivered file for an ad hoc request. "
    "It is intentionally styled to match the executive dashboard.</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Big list of claims (like the PDF)
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Loss Run Detail (Claim List)</div>", unsafe_allow_html=True)

# Sort like a typical loss run: newest losses first, then highest incurred
sort_by = [c for c in ["Loss Date", "Incurred"] if c in df.columns]
if sort_by:
    df = df.sort_values(by=sort_by, ascending=[False] * len(sort_by))

# Display formatting
show = df.copy()
for c in ["Paid", "Outstanding", "Incurred"]:
    if c in show.columns:
        show[c] = show[c].apply(_fmt_money)

for c in ["Loss Date", "Open Date"]:
    if c in show.columns:
        show[c] = pd.to_datetime(show[c], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")

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
cols = [c for c in preferred_cols if c in show.columns]

st.dataframe(show[cols], use_container_width=True, height=700, hide_index=True)

# -----------------------------
# Download (what client receives)
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Download</div>", unsafe_allow_html=True)

export_df = df.copy()

# Convert dates back to ISO for export (no NaT strings)
for c in ["Loss Date", "Open Date"]:
    if c in export_df.columns:
        export_df[c] = pd.to_datetime(export_df[c], errors="coerce").dt.strftime("%Y-%m-%d")

csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Delivered File (CSV)",
    data=csv_bytes,
    file_name="loss_run_delivery.csv",
    mime="text/csv",
    use_container_width=True,
)
