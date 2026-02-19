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
def sidebar_filters(df: pd.DataFrame, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    f: dict = {}

    st.sidebar.header("Filters")

    # Date range
    if "REPORTED_DATE" in df.columns and df["REPORTED_DATE"].notna().any():
        min_d = df["REPORTED_DATE"].min()
        max_d = df["REPORTED_DATE"].max()
        start, end = st.sidebar.date_input(
            "Reported date range",
            value=(defaults.get("reported_start", min_d), defaults.get("reported_end", max_d)),
            min_value=min_d,
            max_value=max_d,
        )
        f["reported_start"] = start
        f["reported_end"] = end

    # Open / Closed
    if "IS_OPEN" in df.columns:
        f["open_only"] = st.sidebar.checkbox("Open claims only", value=defaults.get("open_only", False))

    # Claim status
    if "CLAIM_STATUS_CODE" in df.columns:
        statuses = sorted([x for x in df["CLAIM_STATUS_CODE"].dropna().unique()])
        chosen = st.sidebar.multiselect("Claim Status", statuses, default=defaults.get("claim_status", statuses))
        f["claim_status"] = chosen

    # Client code (if present)
    if "CLIENT_CODE" in df.columns:
        clients = sorted([x for x in df["CLIENT_CODE"].dropna().unique()])
        chosen = st.sidebar.multiselect("Client", clients, default=defaults.get("clients", clients))
        f["clients"] = chosen

    return f


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()

    # Date range
    if "REPORTED_DATE" in out.columns and f.get("reported_start") and f.get("reported_end"):
        out = out[
            (out["REPORTED_DATE"] >= f["reported_start"]) &
            (out["REPORTED_DATE"] <= f["reported_end"])
        ]

    # Open only
    if f.get("open_only") and "IS_OPEN" in out.columns:
        out = out[out["IS_OPEN"] == 1]

    # Status filter
    if "CLAIM_STATUS_CODE" in out.columns and isinstance(f.get("claim_status"), list) and f["claim_status"]:
        out = out[out["CLAIM_STATUS_CODE"].isin(f["claim_status"])]

    # Client filter
    if "CLIENT_CODE" in out.columns and isinstance(f.get("clients"), list) and f["clients"]:
        out = out[out["CLIENT_CODE"].isin(f["clients"])]

    return out


def render_kpis(dff: pd.DataFrame, cfg: dict) -> None:
    # Safe KPI calculations
    total = len(dff)

    incurred = None
    if "INCURRED_AMT" in dff.columns:
        incurred = float(pd.to_numeric(dff["INCURRED_AMT"], errors="coerce").fillna(0).sum())

    open_count = None
    if "IS_OPEN" in dff.columns:
        open_count = int((pd.to_numeric(dff["IS_OPEN"], errors="coerce").fillna(0) == 1).sum())

    sev_thresh = cfg.get("severity_threshold")
    high_sev = None
    if sev_thresh is not None and "INCURRED_AMT" in dff.columns:
        amt = pd.to_numeric(dff["INCURRED_AMT"], errors="coerce").fillna(0)
        high_sev = int((amt >= float(sev_thresh)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", f"{total:,}")
    c2.metric("Open", f"{open_count:,}" if open_count is not None else "—")
    c3.metric("Total Incurred", fmt_currency(incurred))
    c4.metric(f"High Sev (≥ {cfg.get('severity_threshold', 0):,})", f"{high_sev:,}" if high_sev is not None else "—")


# -----------------------------
# Email stub (no lib.emailer required)
# -----------------------------
def send_demo_email(*, to_email: str, subject: str, html_body: str) -> None:
    """
    Demo-safe behavior: show the email content and optionally send via SMTP only if secrets exist.
    This prevents the app from crashing if email isn't configured.
    """
    st.info("Email sending is disabled in this minimal demo build unless SMTP secrets are configured.")
    st.code(f"TO: {to_email}\nSUBJECT: {subject}", language="text")
    st.markdown(html_body, unsafe_allow_html=True)


# -----------------------------
# Your existing helper functions can stay
# -----------------------------
def severity_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 10_000, 50_000, 250_000, 1_000_000_000]
    labels = ["< $10k", "$10k–$50k", "$50k–$250k", "≥ $250k"]
    tmp = df.copy()
    tmp["SEV_BUCKET"] = pd.cut(tmp["INCURRED_AMT"].fillna(0), bins=bins, labels=labels)
    out = tmp.groupby("SEV_BUCKET", dropna=False).size().reset_index(name="Feature Count")
    return out


def fmt_currency(value) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return f"${float(value):,.0f}"
    except Exception:
        return "—"


def get_dashboard_url(cfg: dict) -> str:
    base_url = st.secrets.get("app", {}).get("base_url")
    if base_url:
        return f"{base_url}?client={cfg['client_slug']}"
    return f"/?client={cfg['client_slug']}"

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

