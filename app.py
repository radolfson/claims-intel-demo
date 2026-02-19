# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st


# -----------------------------
# Demo config (no clients/ folder required)
# -----------------------------
DEFAULT_CLIENT = {
    "client_slug": "coverwhale",
    "display_name": "Cover Whale (Demo)",
    "client_code": "DEMO",
    "line_code": None,
    "severity_threshold": 250_000,
    "disclaimer": "Demo environment. Data current through latest available demo extract.",
    "enabled_sections": {"email_preview": True},
    "default_filters": {},
}


# -----------------------------
# Data loader (no lib/ required)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_demo_features(*, client_code: str | None = None, line_code: str | None = None) -> pd.DataFrame:
    """
    Loads demo data from a local CSV committed to the repo.
    Keeps the same signature as the old lib.data.load_demo_features() so your main() barely changes.
    """
    path = "demo_features_latest.csv"
    df = pd.read_csv(path)

    # Normalize date columns if present
    for col in ("ASOF_DATE", "REPORTED_DATE", "LOSS_DATE", "CLOSED_DATE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Optional filtering if those fields exist in the CSV
    if client_code and "CLIENT_CODE" in df.columns:
        df = df[df["CLIENT_CODE"].astype(str) == str(client_code)]

    if line_code and "LINE_CODE" in df.columns:
        df = df[df["LINE_CODE"].astype(str) == str(line_code)]

    return df


# -----------------------------
# Minimal UI helpers (no lib.ui required)
# -----------------------------
def sidebar_filters(df: pd.DataFrame, defaults:_

                            )
                            st.success(f"Sent to {to_email}.")
                        except Exception as e:
                            st.error(f"Email send failed: {e}")

            with cols[1]:
                st.caption("This is demo-grade: sends to you only. Scheduling + client delivery comes later.")
            with cols[2]:
                st.caption("Tip: set [app].base_url in secrets so the email link opens the deployed app.")

    st.divider()

    # KPI row
    render_kpis(dff)

    st.divider()

    # Severity distribution
    if cfg["enabled_sections"].get("severity_distribution", True) and "INCURRED_AMT" in dff.columns:
        st.subheader("Where the Risk Lives")
        sev = severity_buckets(dff)
        st.bar_chart(sev.set_index("SEV_BUCKET"))

    # High severity table
    if cfg["enabled_sections"].get("high_severity_table", True) and "INCURRED_AMT" in dff.columns:
        st.subheader("High Severity Features")
        thresh = cfg["severity_threshold"]
        top = dff[dff["INCURRED_AMT"] >= thresh].copy().sort_values("INCURRED_AMT", ascending=False)
        cols = ["CLAIM_NBR", "COVERAGE_DESC", "LOSS_STATE", "INCURRED_AMT", "PAID_AMT", "OUTSTANDING_AMT"]
        cols = [c for c in cols if c in top.columns]
        st.dataframe(top[cols].head(50), use_container_width=True)

    # Placeholder for planned denial/MCS-90 context
    if cfg["enabled_sections"].get("denial_context", False):
        st.info("Denial / Coverage Contested / MCS-90 context planned for late-February (post-demo).")


if __name__ == "__main__":
    main()

