# app.py
from __future__ import annotations

import importlib
import pandas as pd
import streamlit as st

from lib.data import load_demo_features
from lib.ui import sidebar_filters, apply_filters, render_kpis
from lib.emailer import send_demo_email


def load_client_config(slug: str) -> dict:
    mod = importlib.import_module(f"clients.{slug}")
    return mod.CLIENT


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
    """
    Prefer a real deployed URL if provided, otherwise return a relative link.
    Add this to .streamlit/secrets.toml to make links real:
      [app]
      base_url = "https://YOUR_STREAMLIT_HOST/app"
    """
    base_url = st.secrets.get("app", {}).get("base_url")
    if base_url:
        return f"{base_url}?client={cfg['client_slug']}"
    return f"/app?client={cfg['client_slug']}"


def main():
    st.set_page_config(page_title="NARS Claims Demo", layout="wide")

    client_slug = st.query_params.get("client", "coverwhale")
    cfg = load_client_config(client_slug)

    st.title(f"{cfg['display_name']} — Claims Portfolio Snapshot")
    st.caption(cfg.get("disclaimer", ""))

    df = load_demo_features(client_code=cfg["client_code"], line_code=cfg.get("line_code"))

    # Filters
    f = sidebar_filters(df, cfg.get("default_filters", {}))
    dff = apply_filters(df, f)

    dashboard_url = get_dashboard_url(cfg)

    # Email preview + send (experience requirement)
    if cfg["enabled_sections"].get("email_preview", True):
        st.subheader("📧 Daily Email Preview")
        with st.container(border=True):
            subject = f"{cfg['display_name']} — Daily Claims Summary"
            as_of = dff["ASOF_DATE"].max() if "ASOF_DATE" in dff.columns and len(dff) else "Latest Available"

            total_features = len(dff)
            total_incurred = dff["INCURRED_AMT"].sum(skipna=True) if "INCURRED_AMT" in dff.columns else None
            high_sev_count = (
                int((dff["INCURRED_AMT"] >= cfg["severity_threshold"]).sum())
                if "INCURRED_AMT" in dff.columns and "severity_threshold" in cfg
                else None
            )

            # Visible preview (what you'll narrate)
            st.markdown(f"**Subject:** {subject}")
            st.markdown(f"**As of:** {as_of}")
            st.markdown(
                "\n".join([
                    f"- Total features in view: **{total_features:,}**",
                    f"- Total incurred exposure: **{fmt_currency(total_incurred)}**",
                    (
                        f"- High severity features (≥ ${cfg['severity_threshold']:,}): **{high_sev_count:,}**"
                        if high_sev_count is not None else
                        "- High severity features: **—**"
                    ),
                    f"- Link: **{dashboard_url}**",
                ])
            )

            # Actual email body (HTML)
            html = f"""
            <h3>Daily Claims Summary — {cfg['display_name']}</h3>
            <p><b>As of:</b> {as_of}</p>
            <ul>
              <li>Total features in view: <b>{total_features:,}</b></li>
              <li>Total incurred exposure: <b>{fmt_currency(total_incurred)}</b></li>
              <li>High severity features (≥ ${cfg['severity_threshold']:,}): <b>{f"{high_sev_count:,}" if high_sev_count is not None else "—"}</b></li>
            </ul>
            <p><a href="{dashboard_url}">Open dashboard</a></p>
            """

            # Send button (real email to you)
            cols = st.columns([1, 2, 3])
            with cols[0]:
                if st.button("Send me the demo email", use_container_width=True):
                    to_email = st.secrets.get("email", {}).get("test_recipient")
                    if not to_email:
                        st.error("Missing st.secrets[email][test_recipient]. Add it to .streamlit/secrets.toml.")
                    else:
                        try:
                            send_demo_email(
                                to_email=to_email,
                                subject=subject,
                                html_body=html
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

