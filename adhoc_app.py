import os
import base64
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cover Whale Ad Hoc Delivery Preview",
    page_icon="📄",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1400px; }
.cw-mast { padding: 0.25rem 0 0.75rem 0; }

.cw-card {
  background: #FFFFFF;
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.cw-label { font-size: 12px; color: rgba(49, 51, 63, 0.70); }
.cw-value { font-size: 22px; font-weight: 650; }
.cw-h2 { font-size: 22px; font-weight: 750; margin: 0.2rem 0 0.6rem 0; }
.cw-muted { color: rgba(49, 51, 63, 0.65); font-size: 13px; }

/* Fake hyperlink styling */
.fake-link {
    color: #1f77b4;
    font-size: 14px;
    text-decoration: underline;
    cursor: pointer;
}
.fake-link:hover { opacity: 0.75; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def fmt_money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "$0.00"

def card(col, label: str, value: str):
    col.markdown(
        f"""
        <div class="cw-card">
          <div class="cw-label">{label}</div>
          <div class="cw-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_logo_html(path: str, width: int = 260) -> str:
    """Render logo via base64 so we can control spacing and avoid clipping."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"""
    <div style="padding-top:12px; padding-bottom:10px;">
        <img src="data:image/jpeg;base64,{b64}" style="width:{width}px; height:auto; display:block;">
    </div>
    """

# -----------------------------
# Demo Data
# -----------------------------
def generate_demo_loss_run(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    today = pd.Timestamp.today().normalize()

    loss_dates = today - pd.to_timedelta(rng.integers(30, 365 * 5, size=n), unit="D")
    open_dates = loss_dates + pd.to_timedelta(rng.integers(0, 30, size=n), unit="D")

    paid = rng.gamma(2.0, 15000, n)
    outstanding = rng.gamma(2.0, 12000, n)
    incurred = paid + outstanding

    clients = [
        "FLORIDA HOSPITAL SELF INSURANCE FUND",
        "LAUNCH ENVIRONMENTAL",
        "COVER WHALE (DEMO)",
    ]
    coverages = ["WC-IND", "AUTO", "GL", "EXCESS"]
    states = ["FL", "TX", "CA", "IL", "GA", "NC", "NJ"]

    df = pd.DataFrame(
        {
            "Client": rng.choice(clients, size=n, p=[0.45, 0.20, 0.35]),
            "Policy Number": [f"POL-{rng.integers(100000,999999)}" for _ in range(n)],
            "Claim Number": [f"CLM-{rng.integers(100000,999999)}" for _ in range(n)],
            "Feature Key": [
                f"F-{rng.integers(2021,2026)}-{rng.integers(10000,99999)}" for _ in range(n)
            ],
            "State": rng.choice(states, size=n),
            "Coverage": rng.choice(coverages, size=n),
            "Loss Date": loss_dates,
            "Open Date": open_dates,
            "Status": rng.choice(["OPEN", "CLOSED"], size=n, p=[0.70, 0.30]),
            "Paid": paid,
            "Outstanding": outstanding,
            "Incurred": incurred,
            "Adjuster": rng.choice(["J. Smith", "A. Patel", "M. Chen", "R. Garcia"], size=n),
            "Defense Firm": rng.choice(["Smith & Cole", "Parker LLP", "Hart & Gray", "None"], size=n),
            "Denial Reason": rng.choice(["None", "Late Notice", "Coverage Exclusion", "Policy Lapse"], size=n),
        }
    )
    return df

df = generate_demo_loss_run(900)

# -----------------------------
# Masthead
# -----------------------------
st.markdown('<div class="cw-mast">', unsafe_allow_html=True)
mast = st.columns([0.22, 0.78], vertical_alignment="center")

with mast[0]:
    logo_path = "narslogo.jpg"
    if os.path.exists(logo_path):
        st.markdown(get_logo_html(logo_path, width=260), unsafe_allow_html=True)

with mast[1]:
    st.markdown(
        """
        <div style="font-size:30px; font-weight:800; line-height:1.1;">
            Cover Whale Ad Hoc Delivery Preview
        </div>
        <div class="cw-muted" style="margin-top:4px;">
            Client-facing preview of what is delivered for ad hoc loss run requests.
        </div>
        <div style="margin-top:8px;">
            <span class="fake-link">View Original Request (ServiceNow Ticket #INC-2026-00421)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# -----------------------------
# Delivery Summary
# -----------------------------
st.markdown("<div class='cw-h2'>Delivery Summary</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
card(k1, "Rows Delivered", f"{len(df):,}")
card(k2, "Total Paid", fmt_money(df["Paid"].sum()))
card(k3, "Total Outstanding", fmt_money(df["Outstanding"].sum()))
card(k4, "Total Incurred", fmt_money(df["Incurred"].sum()))
card(k5, "Output", "Data Only (Excel)")

st.markdown(
    "<div class='cw-muted'>This preview simulates the delivered file for an ad hoc request.</div>",
    unsafe_allow_html=True,
)

# -----------------------------
# Claim List
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Loss Run Detail (Claim List)</div>", unsafe_allow_html=True)

display_df = df.copy()
display_df["Loss Date"] = pd.to_datetime(display_df["Loss Date"]).dt.strftime("%Y-%m-%d")
display_df["Open Date"] = pd.to_datetime(display_df["Open Date"]).dt.strftime("%Y-%m-%d")

for c in ["Paid", "Outstanding", "Incurred"]:
    display_df[c] = display_df[c].apply(fmt_money)

st.dataframe(display_df, use_container_width=True, height=700, hide_index=True)

# -----------------------------
# Download
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Download</div>", unsafe_allow_html=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Delivered File (CSV)",
    data=csv_bytes,
    file_name="loss_run_delivery.csv",
    mime="text/csv",
    use_container_width=True,
)