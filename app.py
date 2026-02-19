# app.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


# =========================================
# Config (single-file demo)
# =========================================
@dataclass(frozen=True)
class DemoClientConfig:
    client_slug: str = "coverwhale"
    display_name: str = "Cover Whale (Demo)"
    client_code: str = "DEMO"
    line_code: Optional[str] = None
    severity_threshold: float = 250_000
    disclaimer: str = "Demo environment. Data current through latest available demo extract."
    enabled_sections: Dict[str, bool] = None
    default_filters: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "client_slug": self.client_slug,
            "display_name": self.display_name,
            "client_code": self.client_code,
            "line_code": self.line_code,
            "severity_threshold": self.severity_threshold,
            "disclaimer": self.disclaimer,
            "enabled_sections": self.enabled_sections or {
                "email_preview": True,
                "severity_distribution": True,
                "high_severity_table": True,
                "denial_context": False,
            },
            "default_filters": self.default_filters or {},
        }


DEFAULT_CFG = DemoClientConfig().as_dict()


# =========================================
# Data loading
# =========================================
@st.cache_data(show_spinner=False)
def load_demo_features(*, client_code: str | None = None, line_code: str | None = None) -> pd.DataFrame:
    """
    Loads demo data from repo CSV: demo_features_latest.csv
    Keeps signature similar to your old lib.data.load_demo_features().
    """
    path = "demo_features_latest.csv"
    df = pd.read_csv(path)

    # Normalize date-ish columns if present
    for col in ("ASOF_DATE", "REPORTED_DATE", "LOSS_DATE", "CLOSED_DATE", "DATE_CREATED", "DATE_MODIFIED"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Normalize numeric-ish columns if present
    for col in ("INCURRED_AMT", "PAID_AMT", "OUTSTANDING_AMT"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional filters (only if columns exist)
    if client_code and "CLIENT_CODE" in df.columns:
        df = df[df["CLIENT_CODE"].astype(str) == str(client_code)]

    if line_code and "LINE_CODE" in df.columns:
        df = df[df["LINE_CODE"].astype(str) == str(line_code)]

    return df


# =========================================
# Formatting helpers
# =========================================
def fmt_currency(value) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return f"${float(value):,.0f}"
    except Exception:
        return "—"


def get_dashboard_url(cfg: dict) -> str:
    """
    If [app].base_url exists in secrets, generate real link.
    Otherwise, return a relative link (still useful for demo copy).
    """
    base_url = st.secrets.get("app", {}).get("base_url")
    if base_url:
        return f"{base_url}?client={cfg['client_slug']}"
    return f"/?client={cfg['client_slug']}"


# =========================================
# UI: filters + apply
# =========================================
def sidebar_filters(df: pd.DataFrame, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    f: dict = {}

    st.sidebar.header("Filters")

    # Reported date range
    if "REPORTED_DATE" in df.columns and df["REPORTED_DATE"].notna().any():
        min_d = df["REPORTED_DATE"].min()
        max_d = df["REPORTED_DATE"].max()
        start_default = defaults.get("reported_start", min_d)
        end_default = defaults.get("reported_end", max_d)

        start, end = st.sidebar.date_input(
            "Reported date range",
            value=(start_default, end_default),
            min_value=min_d,
            max_value=max_d,
        )
        f["reported_start"] = start
        f["reported_end"] = end

    # Open only
    if "IS_OPEN" in df.columns:
        f["open_only"] = st.sidebar.checkbox("Open only", value=defaults.get("open_only", False))

    # Claim status
    if "CLAIM_STATUS_CODE" in df.columns:
        statuses = sorted([x for x in df["CLAIM_STATUS_CODE"].dropna().unique()])
        default_statuses = defaults.get("claim_status", statuses)
        chosen = st.sidebar.multiselect("Claim Status", statuses, default=default_statuses)
        f["claim_status"] = chosen

    # Loss state
    if "LOSS_STATE" in df.columns:
        states = sorted([x for x in df["LOSS_STATE"].dropna().unique()])
        default_states = defaults.get("loss_state", states)
        chosen = st.sidebar.multiselect("Loss State", states, default=default_states)
        f["loss_state"] = chosen

    # Coverage description
    if "COVERAGE_DESC" in df.columns:
        covs = sorted([x for x in df["COVERAGE_DESC"].dropna().unique()])
        default_covs = defaults.get("coverage_desc", covs)
        chosen = st.sidebar.multiselect("Coverage", covs, default=default_covs)
        f["coverage_desc"] = chosen

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
        out = out[pd.to_numeric(out["IS_OPEN"], errors="coerce").fillna(0) == 1]

    # Claim status
    if "CLAIM_STATUS_CODE" in out.columns and isinstance(f.get("claim_status"), list) and f["claim_status"]:
        out = out[out["CLAIM_STATUS_CODE"].isin(f["claim_status"])]

    # Loss state
    if "LOSS_STATE" in out.columns and isinstance(f.get("loss_state"), list) and f["loss_state"]:
        out = out[out["LOSS_STATE"].isin(f["loss_state"])]

    # Coverage
    if "COVERAGE_DESC" in out.columns and isinstance(f.get("coverage_desc"), list) and f["coverage_desc"]:
        out = out[out["COVERAGE_DESC"].isin(f["coverage_desc"])]

    return out


# =========================================
# KPIs + charts
# =========================================
def render_kpis(dff: pd.DataFrame, cfg: dict) -> None:
    total = len(dff)

    open_count = None
    if "IS_OPEN" in dff.columns:
        open_count = int((pd.to_numeric(dff["IS_OPEN"], errors="coerce").fillna(0) == 1).sum())

    total_incurred = None
    if "INCURRED_AMT" in dff.columns:
        total_incurred = float(pd.to_numeric(dff["INCURRED_AMT"], errors="coerce").fillna(0).sum())

    sev_thresh = cfg.get("severity_threshold", 250_000)
    high_sev_count = None
    if "INCURRED_AMT" in dff.columns:
        amt = pd.to_numeric(dff["INCURRED_AMT"], errors="coerce").fillna(0)
        high_sev_count = int((amt >= float(sev_thresh)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", f"{total:,}")
    c2.metric("Open", f"{open_count:,}" if open_count is not None else "—")
    c3.metric("Total Incurred", fmt_currency(total_incurred))
    c4.metric(f"High Sev (≥ {int(sev_thresh):,})", f"{high_sev_count:,}" if high_sev_count is not None else "—")


def severity_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if "INCURRED_AMT" not in df.columns:
        return pd.DataFrame({"SEV_BUCKET": [], "Feature Count": []})

    bins = [-1, 10_000, 50_000, 250_000, 1_000_000_000]
    labels = ["< $10k", "$10k–$50k", "$50k–$250k", "≥ $250k"]

    tmp = df.copy()
    tmp["INCURRED_AMT"] = pd.to_numeric(tmp["INCURRED_AMT"], errors="coerce").fillna(0)
    tmp["SEV_BUCKET"] = pd.cut(tmp["INCURRED_AMT"], bins=bins, labels=labels)
    out = tmp.groupby("SEV_BUCKET", dropna=False).size().reset_index(name="Feature Count")
    return out


# =========================================
# Email: safe demo stub (won't crash)
# =========================================
def send_demo_email(*, to_email: str, subject: str, html_body: str) -> None:
    """
    Demo-safe: doesn't actually send unless you later wire SMTP.
    Tonight: show preview + confirm click.
    """
    st.info("Demo mode: email sending is stubbed (preview only).")
    st.code(f"TO: {to_email}\nSUBJECT: {subject}", language="text")
    st.markdown(html_body, unsafe_allow_html=True)


# =========================================
# Main
# =========================================
def main() -> None:
    st.set_page_config(page_title="NARS Claims Demo", layout="wide")

    # Minimal "client" selection via query param, but same config structure
    client_slug = st.query_params.get("client", DEFAULT_CFG["client_slug"])
    cfg = dict(DEFAULT_CFG)
    cfg["client_slug"] = client_slug

    st.title(f"{cfg['display_name']} — Claims Portfolio Snapshot")
    st.caption(cfg.get("disclaimer", ""))

    # Load demo data
    df = load_demo_features(client_code=cfg.get("client_code"), line_code=cfg.get("line_code"))

    if df.empty:
        st.error("Demo dataset loaded, but returned 0 rows. Check demo_features_latest.csv contents.")
        st.stop()

    # Filters
    f = sidebar_filters(df, cfg.get("default_filters", {}))
    dff = apply_filters(df, f)

    # As-of banner (simple + defensible)
    as_of = None
    if "ASOF_DATE" in dff.columns and dff["ASOF_DATE"].notna().any():
        as_of = max(dff["ASOF_DATE"])
    elif "REPORTED_DATE" in dff.columns and dff["REPORTED_DATE"].notna().any():
        as_of = max(dff["REPORTED_DATE"])

    banner = f"Data current through: {as_of}" if as_of else "Data current through: Latest Available"
    st.markdown(f"**{banner}**")

    dashboard_url = get_dashboard_url(cfg)

    # Email preview
    if cfg.get("enabled_sections", {}).get("email_preview", True):
        st.subheader("📧 Daily Email Preview")
        with st.container(border=True):
            subject = f"{cfg['display_name']} — Daily Claims Summary"

            total_features = len(dff)
            total_incurred = dff["INCURRED_AMT"].sum(skipna=True) if "INCURRED_AMT" in dff.columns else None

            sev_thresh = cfg.get("severity_threshold")
            high_sev_count = None
            if sev_thresh is not None and "INCURRED_AMT" in dff.columns:
                amt = pd.to_numeric(dff["INCURRED_AMT"], errors="coerce").fillna(0)
                high_sev_count = int((amt >= float(sev_thresh)).sum())

            st.markdown(f"**Subject:** {subject}")
            st.markdown(f"**As of:** {as_of if as_of else 'Latest Available'}")
            st.markdown(
                "\n".join(
                    [
                        f"- Total features in view: **{total_features:,}**",
                        f"- Total incurred exposure: **{fmt_currency(total_incurred)}**",
                        (
                            f"- High severity features (≥ ${int(sev_thresh):,}): **{high_sev_count:,}**"
                            if high_sev_count is not None
                            else "- High severity features: **—**"
                        ),
                        f"- Link: **{dashboard_url}**",
                    ]
                )
            )

            html = f"""
            <h3>Daily Claims Summary — {cfg['display_name']}</h3>
            <p><b>As of:</b> {as_of if as_of else "Latest Available"}</p>
            <ul>
              <li>Total features in view: <b>{total_features:,}</b></li>
              <li>Total incurred exposure: <b>{fmt_currency(total_incurred)}</b></li>
              <li>High severity features (≥ ${int(sev_thresh) if sev_thresh else 0:,}): <b>{f"{high_sev_count:,}" if high_sev_count is not None else "—"}</b></li>
            </ul>
            <p><a href="{dashboard_url}">Open dashboard</a></p>
            """

            cols = st.columns([1, 2, 3])
            with cols[0]:
                if st.button("Send me the demo email", use_container_width=True):
                    to_email = st.secrets.get("email", {}).get("test_recipient")
                    if not to_email:
                        st.error("Missing st.secrets[email][test_recipient]. Add it to Streamlit secrets.")
                    else:
                        send_demo_email(to_email=to_email, subject=subject, html_body=html)
                        st.success(f"Preview generated for {to_email}.")

            with cols[1]:
                st.caption("Demo-grade: preview only. Real delivery comes later.")
            with cols[2]:
                st.caption("Tip: set [app].base_url in secrets so the link opens the deployed app.")

    st.divider()

    # KPIs
    render_kpis(dff, cfg)

    st.divider()

    # Severity distribution
    if cfg.get("enabled_sections", {}).get("severity_distribution", True) and "INCURRED_AMT" in dff.columns:
        st.subheader("Where the Risk Lives")
        sev = severity_buckets(dff)
        if not sev.empty:
            st.bar_chart(sev.set_index("SEV_BUCKET"))
        else:
            st.caption("No severity data available.")

    # High severity table
    if cfg.get("enabled_sections", {}).get("high_severity_table", True) and "INCURRED_AMT" in dff.columns:
        st.subheader("High Severity Features")
        thresh = float(cfg.get("severity_threshold", 250_000))
        top = dff[pd.to_numeric(dff["INCURRED_AMT"], errors="coerce").fillna(0) >= thresh].copy()
        top = top.sort_values("INCURRED_AMT", ascending=False)

        preferred_cols = [
            "CLAIM_NUMBER",
            "CLAIM_NBR",
            "COVERAGE_DESC",
            "LOSS_STATE",
            "REPORTED_DATE",
            "INCURRED_AMT",
            "PAID_AMT",
            "OUTSTANDING_AMT",
        ]
        cols = [c for c in preferred_cols if c in top.columns]
        if cols:
            st.dataframe(top[cols].head(50), use_container_width=True)
        else:
            st.dataframe(top.head(50), use_container_width=True)

    # Placeholder section
    if cfg.get("enabled_sections", {}).get("denial_context", False):
        st.info("Denial / Coverage Contested / MCS-90 context planned post-demo.")


if __name__ == "__main__":
    main()
