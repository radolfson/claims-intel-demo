from __future__ import annotations

import os
import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import altair as alt
import pandas as pd
import requests
import streamlit as st


# ============================================================
# Config
# ============================================================
DEFAULT_SEVERITY_THRESHOLD = 250_000
STATUS_OPEN_SET = {"OPEN", "PENDING", "REOPEN"}

COVERAGE_CODE_TO_COVERAGE = {
    "AUTO-L": "BODILY INJURY",
    "AUTO-PD": "PROPERTY DAMAGE",
    "GL-BI": "BODILY INJURY",
    "GL-PD": "PROPERTY DAMAGE",
    "WC-IND": "INDEMNITY",
    "CARGO": "CARGO",
}

DASHBOARD_URL = str(st.secrets.get("DASHBOARD_URL", "https://nars-demo.streamlit.app")).strip()


@dataclass
class AppConfig:
    data_source: str
    data_file: str


def get_config() -> AppConfig:
    data_source = str(st.secrets.get("DATA_SOURCE", "csv")).strip().lower()
    data_file = str(st.secrets.get("DATA_FILE", "demo_features_latest.csv"))

    data_source = os.getenv("DATA_SOURCE", data_source).strip().lower()
    data_file = os.getenv("DATA_FILE", data_file)

    return AppConfig(data_source=data_source, data_file=data_file)


# ============================================================
# Formatting helpers
# ============================================================
def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def fmt_money_compact(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"

    sign = "-" if v < 0 else ""
    v = abs(v)

    if v >= 1_000_000_000:
        return f"{sign}${v/1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{sign}${v/1_000_000:.1f}M"
    if v >= 100_000:
        return f"{sign}${v/1_000:.1f}K"
    return f"{sign}${v:,.0f}"


def safe_list(series: pd.Series) -> list:
    return sorted([x for x in series.dropna().unique()])


def month_floor(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("M").dt.to_timestamp()


def pct_change(curr: float, prev: float) -> Optional[float]:
    if prev is None or prev == 0:
        return None
    return (curr - prev) / prev * 100.0


def direction_word(delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return "flat"
    if delta_pct > 0.5:
        return "up"
    if delta_pct < -0.5:
        return "down"
    return "flat"


def strength_word(delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return "steady"
    a = abs(delta_pct)
    if a >= 10:
        return "sharply"
    if a >= 4:
        return "notably"
    if a >= 1:
        return "slightly"
    return "roughly"


# ============================================================
# Data normalization
# ============================================================
REQUIRED_COLUMNS = [
    "client",
    "feature_key",
    "claim_number",
    "claimant_id",
    "coverage_code",
    "coverage_type",
    "feature_status",
    "state",
    "accident_year",
    "paid_amount",
    "outstanding_amount",
    "incurred_amount",
    "report_date",
    "feature_created_date",
    "adjuster",
    "line_of_business",
    "cause_of_loss",
    "is_litigated",
    "vendor_name",
    "defense_firm",
    "denial_reason",
]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    for c in REQUIRED_COLUMNS:
        if c not in d.columns:
            d[c] = None

    for col in ("report_date", "feature_created_date"):
        d[col] = pd.to_datetime(d[col], errors="coerce")

    for col in ("paid_amount", "outstanding_amount", "incurred_amount"):
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    code = d["coverage_code"].astype("string").fillna("")
    mapped = code.map(COVERAGE_CODE_TO_COVERAGE).astype("string")
    d["coverage_type"] = mapped.where(mapped.notna() & (mapped != ""), d["coverage_code"]).astype("string")

    if "is_open_inventory" not in d.columns:
        d["is_open_inventory"] = d["feature_status"].isin(list(STATUS_OPEN_SET)).astype(int)
    else:
        d["is_open_inventory"] = pd.to_numeric(d["is_open_inventory"], errors="coerce").fillna(0).astype(int)

    d["is_litigated"] = pd.to_numeric(d["is_litigated"], errors="coerce").fillna(0).astype(int)

    if d["feature_created_date"].notna().any():
        d["trend_month"] = month_floor(d["feature_created_date"])
    elif d["report_date"].notna().any():
        d["trend_month"] = month_floor(d["report_date"])
    else:
        d["trend_month"] = pd.NaT

    for col in (
        "client",
        "state",
        "coverage_type",
        "feature_status",
        "adjuster",
        "line_of_business",
        "cause_of_loss",
        "vendor_name",
        "defense_firm",
        "denial_reason",
    ):
        d[col] = d[col].astype("string")

    d["accident_year"] = pd.to_numeric(d["accident_year"], errors="coerce")

    return d


def add_synthetic_denial_reason(df: pd.DataFrame) -> pd.DataFrame:
    reasons = [
        "MCS90",
        "Coverage Exclusion",
        "Late Notice",
        "Policy Lapse",
        "Fraud Suspected",
        "No Liability",
    ]

    d = df.copy()
    status = d["feature_status"].fillna("").astype(str).str.upper()
    denied = status.eq("DENIED")

    d["denial_reason"] = d.get("denial_reason", pd.Series([None] * len(d)))
    d["denial_reason"] = d["denial_reason"].astype("string")

    def pick_reason(fk) -> str:
        s = str(fk) if fk is not None else ""
        h = 0
        for ch in s:
            h = (h * 31 + ord(ch)) % 10_000
        return reasons[h % len(reasons)]

    if denied.any():
        d.loc[denied, "denial_reason"] = d.loc[denied, "feature_key"].apply(pick_reason).astype("string")

    d["denial_reason"] = d["denial_reason"].replace({"": pd.NA})
    return d


@st.cache_data(show_spinner=False)
def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_df(df)


def load_data(cfg: AppConfig) -> Tuple[pd.DataFrame, str]:
    df = load_from_csv(cfg.data_file)
    return df, "csv"


# ============================================================
# Business rules (light)
# ============================================================
def standardize_financials(df: pd.DataFrame, strict_incurred_open_only: bool) -> pd.DataFrame:
    d = df.copy()
    status = d["feature_status"].fillna("UNKNOWN").astype(str)

    denied_mask = status.str.upper().eq("DENIED")
    d.loc[denied_mask, ["paid_amount", "outstanding_amount", "incurred_amount"]] = 0.0
    d.loc[denied_mask, "is_open_inventory"] = 0

    closed_mask = status.str.upper().eq("CLOSED")
    d.loc[closed_mask, "outstanding_amount"] = 0.0
    d.loc[closed_mask, "incurred_amount"] = d.loc[closed_mask, "paid_amount"]
    d.loc[closed_mask, "is_open_inventory"] = 0

    sane_mask = ~(denied_mask | closed_mask)
    recomputed = d.loc[sane_mask, "paid_amount"] + d.loc[sane_mask, "outstanding_amount"]
    d.loc[sane_mask, "incurred_amount"] = d.loc[sane_mask, ["incurred_amount"]].max(axis=1)
    d.loc[sane_mask, "incurred_amount"] = d.loc[sane_mask, "incurred_amount"].where(
        d.loc[sane_mask, "incurred_amount"] >= recomputed, recomputed
    )

    if strict_incurred_open_only:
        open_mask = status.str.upper().isin(list(STATUS_OPEN_SET))
        d.loc[~open_mask, ["incurred_amount", "paid_amount", "outstanding_amount"]] = 0.0

    return d


def synthesize_monthly_history_to_start(df: pd.DataFrame, start: str = "2021-01-01", seed: int = 7) -> pd.DataFrame:
    import numpy as np

    d = df.copy()
    d = d.dropna(subset=["trend_month"]).copy()
    if d.empty:
        return df

    start_dt = pd.to_datetime(start)
    if d["trend_month"].min() <= start_dt:
        return df

    cur_month = d["trend_month"].max()
    months = pd.date_range(start=start_dt, end=cur_month, freq="MS")

    rng = np.random.default_rng(seed)
    frames = []
    for m in months:
        f = d.copy()
        f["trend_month"] = m

        drift = 1.0 + rng.normal(0.0, 0.06, size=len(f))
        drift = np.clip(drift, 0.80, 1.25)

        f["paid_amount"] = (f["paid_amount"] * drift).round(2)
        f["outstanding_amount"] = (
            f["outstanding_amount"] * (1.0 + rng.normal(0.0, 0.05, size=len(f)))
        ).clip(lower=0).round(2)
        f["incurred_amount"] = (f["paid_amount"] + f["outstanding_amount"]).round(2)

        frames.append(f)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Filters
# ============================================================
def init_filter_state() -> None:
    if st.session_state.get("_filters_initialized"):
        return
    st.session_state["_filters_initialized"] = True

    st.session_state.setdefault("f_state", "All States")
    st.session_state.setdefault("f_acc_year", "All Years")
    st.session_state.setdefault("f_coverage", "All Coverages")
    st.session_state.setdefault("f_adjuster", "All Adjusters")
    st.session_state.setdefault("f_lob", "All Lines")
    st.session_state.setdefault("f_status", "All Statuses")
    st.session_state.setdefault("f_cause", "All Causes")
    st.session_state.setdefault("f_litigated", "All")
    st.session_state.setdefault("f_vendor", "All Vendors")
    st.session_state.setdefault("f_defense", "All Firms")
    st.session_state.setdefault("f_denial_reason", "All Reasons")
    st.session_state.setdefault("f_sev_thresh", DEFAULT_SEVERITY_THRESHOLD)


def reset_filters() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("f_") or k == "_filters_initialized":
            del st.session_state[k]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    if st.session_state["f_state"] != "All States":
        dff = dff[dff["state"] == st.session_state["f_state"]]
    if st.session_state["f_acc_year"] != "All Years":
        dff = dff[dff["accident_year"] == float(st.session_state["f_acc_year"])]
    if st.session_state["f_coverage"] != "All Coverages":
        dff = dff[dff["coverage_type"] == st.session_state["f_coverage"]]
    if st.session_state["f_adjuster"] != "All Adjusters":
        dff = dff[dff["adjuster"] == st.session_state["f_adjuster"]]
    if st.session_state["f_lob"] != "All Lines":
        dff = dff[dff["line_of_business"] == st.session_state["f_lob"]]
    if st.session_state["f_status"] != "All Statuses":
        dff = dff[dff["feature_status"] == st.session_state["f_status"]]
    if st.session_state["f_cause"] != "All Causes":
        dff = dff[dff["cause_of_loss"] == st.session_state["f_cause"]]
    if st.session_state["f_litigated"] != "All":
        want = 1 if st.session_state["f_litigated"] == "Litigated" else 0
        dff = dff[dff["is_litigated"] == want]
    if st.session_state["f_vendor"] != "All Vendors":
        dff = dff[dff["vendor_name"] == st.session_state["f_vendor"]]
    if st.session_state["f_defense"] != "All Firms":
        dff = dff[dff["defense_firm"] == st.session_state["f_defense"]]
    if st.session_state.get("f_denial_reason", "All Reasons") != "All Reasons":
        dff = dff[dff["denial_reason"] == st.session_state["f_denial_reason"]]

    return dff


# ============================================================
# KPI + Trends helpers
# ============================================================
def calc_kpis(dff: pd.DataFrame, sev_thresh: float) -> dict:
    open_features = int((dff["is_open_inventory"] == 1).sum())
    total_incurred = float(dff["incurred_amount"].sum())
    paid = float(dff["paid_amount"].sum())
    outstanding = float(dff["outstanding_amount"].sum())
    high_sev = int((dff["incurred_amount"] >= sev_thresh).sum())
    total_features = int(len(dff))
    reserve_ratio = (outstanding / total_incurred * 100.0) if total_incurred else 0.0

    return dict(
        open_features=open_features,
        total_features=total_features,
        total_incurred=total_incurred,
        paid=paid,
        outstanding=outstanding,
        high_sev=high_sev,
        reserve_ratio=reserve_ratio,
    )


def monthly_rollup(dff: pd.DataFrame, sev_thresh: float) -> pd.DataFrame:
    m = dff.dropna(subset=["trend_month"]).copy()
    if m.empty:
        return pd.DataFrame(columns=["trend_month"])

    m["is_hs"] = (m["incurred_amount"] >= sev_thresh).astype(int)
    m["is_closed"] = m["feature_status"].fillna("").astype(str).str.upper().eq("CLOSED").astype(int)

    roll = (
        m.groupby("trend_month", as_index=False)
        .agg(
            total_features=("feature_key", "count"),
            open_features=("is_open_inventory", "sum"),
            closed_features=("is_closed", "sum"),
            total_incurred=("incurred_amount", "sum"),
            paid=("paid_amount", "sum"),
            outstanding=("outstanding_amount", "sum"),
            high_sev=("is_hs", "sum"),
        )
        .sort_values("trend_month")
    )
    roll["reserve_ratio"] = roll.apply(
        lambda r: (r["outstanding"] / r["total_incurred"] * 100.0) if r["total_incurred"] else 0.0,
        axis=1,
    )
    return roll


def last_two_months(roll: pd.DataFrame, metric: str) -> Tuple[Optional[float], Optional[float]]:
    if roll.empty or metric not in roll.columns:
        return None, None
    if len(roll) == 1:
        return float(roll.iloc[-1][metric]), None
    return float(roll.iloc[-1][metric]), float(roll.iloc[-2][metric])


# ============================================================
# Headlines / Narrative
# ============================================================
def build_headline_story(dff: pd.DataFrame, sev_thresh: float) -> list[str]:
    k = calc_kpis(dff, sev_thresh)
    roll = monthly_rollup(dff, sev_thresh)

    open_curr, open_prev = last_two_months(roll, "open_features")
    inc_curr, inc_prev = last_two_months(roll, "total_incurred")
    rr_curr, rr_prev = last_two_months(roll, "reserve_ratio")

    open_delta = pct_change(open_curr or 0, open_prev or 0) if open_prev is not None else None
    inc_delta = pct_change(inc_curr or 0, inc_prev or 0) if inc_prev is not None else None
    rr_delta = pct_change(rr_curr or 0, rr_prev or 0) if rr_prev is not None else None

    open_dir = direction_word(open_delta)
    open_strength = strength_word(open_delta)
    inc_dir = direction_word(inc_delta)
    inc_strength = strength_word(inc_delta)
    rr_dir = direction_word(rr_delta)

    story_1 = (
        f"Open inventory is {open_strength} {open_dir}. If this persists, it can signal slower claim resolution, "
        f"fresh reporting inflow, or operational capacity constraints (adjuster staffing, intake surges, or litigation drag)."
    )

    story_2 = (
        f"Total incurred is {inc_strength} {inc_dir}. When incurred rises faster than open count, severity is usually the driver; "
        f"when it rises with open count, volume is the story. Either way, it’s an early budget signal."
    )

    story_3 = (
        f"Reserve posture is {rr_dir} with outstanding at {fmt_money_compact(k['outstanding'])} "
        f"({k['reserve_ratio']:.1f}% of incurred). A rising reserve ratio often reflects cautious case reserving or developing claim complexity; "
        f"a falling ratio can reflect settlements closing out or paid catching up."
    )

    story_4 = (
        f"High severity exposure is {fmt_int(k['high_sev'])} features at ≥ {fmt_money_compact(sev_thresh)}. "
        f"Watch for clustering by coverage and state."
    )

    return [story_1, story_2, story_3, story_4]


# ============================================================
# Ask NARS (light demo)
# ============================================================
def answer_question(dff: pd.DataFrame, q: str, sev_thresh: float) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "Ask something like: 'open features', 'paid', 'outstanding', 'high severity'."

    if "open" in ql and "features" in ql:
        return f"Open inventory features: {fmt_int((dff['is_open_inventory'] == 1).sum())}."

    if "paid" in ql:
        return f"Total paid: {fmt_money_compact(dff['paid_amount'].sum())}."

    if "outstanding" in ql or "reserve" in ql:
        return f"Total outstanding: {fmt_money_compact(dff['outstanding_amount'].sum())}."

    if "high severity" in ql or "severe" in ql:
        hs = int((dff["incurred_amount"] >= sev_thresh).sum())
        return f"High severity features (≥ {fmt_money_compact(sev_thresh)}): {hs:,}."

    return "Try: 'open features', 'paid', 'outstanding', 'high severity'."


# ============================================================
# Email (SendGrid)
# ============================================================
def _require_sendgrid_secrets() -> Tuple[str, str, str]:
    api_key = str(st.secrets.get("SENDGRID_API_KEY", "")).strip()
    from_email = str(st.secrets.get("EMAIL_FROM", "")).strip()
    from_name = str(st.secrets.get("EMAIL_FROM_NAME", "Cover Whale Daily")).strip()

    if not api_key or not from_email:
        raise ValueError("Missing SENDGRID_API_KEY and/or EMAIL_FROM in Streamlit secrets.")
    return api_key, from_email, from_name


def build_email_html(headlines: List[str], as_of: str, dashboard_url: str) -> str:
    # Simple, “corporate-safe” HTML. No external CSS. Works in Outlook.
    headline_cards = ""
    for h in headlines:
        headline_cards += f"""
        <tr>
          <td style="padding:10px 0;">
            <table width="100%" cellpadding="0" cellspacing="0" style="border-left:5px solid #1F4E79;background:#F3F6FA;border-radius:10px;">
              <tr>
                <td style="padding:14px 14px 14px 14px;font-family:Arial,Helvetica,sans-serif;color:#102A43;font-size:14px;line-height:20px;">
                  {h}
                </td>
              </tr>
            </table>
          </td>
        </tr>
        """

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#ffffff;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background:#ffffff;">
          <tr>
            <td align="center" style="padding:24px 14px;">
              <table width="640" cellpadding="0" cellspacing="0" style="border:1px solid #E6EAF0;border-radius:14px;overflow:hidden;">
                <tr>
                  <td style="padding:18px 18px 8px 18px;background:#ffffff;">
                    <div style="font-family:Arial,Helvetica,sans-serif;color:#102A43;font-size:22px;font-weight:700;line-height:28px;">
                      Cover Whale Daily, Powered by NARS
                    </div>
                    <div style="font-family:Arial,Helvetica,sans-serif;color:#5A6B7B;font-size:13px;line-height:18px;margin-top:4px;">
                      As of: <b>{as_of}</b>
                    </div>
                  </td>
                </tr>

                <tr>
                  <td style="padding:10px 18px 0 18px;">
                    <div style="font-family:Arial,Helvetica,sans-serif;color:#102A43;font-size:16px;font-weight:700;">
                      Today’s Headlines
                    </div>
                  </td>
                </tr>

                <tr>
                  <td style="padding:6px 18px 0 18px;">
                    <table width="100%" cellpadding="0" cellspacing="0">
                      {headline_cards}
                    </table>
                  </td>
                </tr>

                <tr>
                  <td style="padding:10px 18px 18px 18px;">
                    <table cellpadding="0" cellspacing="0">
                      <tr>
                        <td align="center" bgcolor="#1F4E79" style="border-radius:12px;">
                          <a href="{dashboard_url}"
                             style="display:inline-block;padding:12px 16px;font-family:Arial,Helvetica,sans-serif;color:#ffffff;text-decoration:none;font-weight:700;font-size:14px;">
                             Open Dashboard
                          </a>
                        </td>
                      </tr>
                    </table>

                    <div style="font-family:Arial,Helvetica,sans-serif;color:#8A97A6;font-size:12px;line-height:16px;margin-top:10px;">
                      If the button doesn’t work, copy/paste this link: {dashboard_url}
                    </div>
                  </td>
                </tr>

              </table>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """
    return html


def build_email_text(headlines: List[str], as_of: str, dashboard_url: str) -> str:
    return (
        "Cover Whale Daily, Powered by NARS\n"
        f"As of: {as_of}\n\n"
        "Today's Headlines:\n"
        + "\n".join([f"- {h}" for h in headlines])
        + "\n\nDashboard:\n"
        + dashboard_url
        + "\n"
    )


def send_email_sendgrid(to_email: str, subject: str, html: str, text: str) -> None:
    api_key, from_email, from_name = _require_sendgrid_secrets()

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email, "name": from_name},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": text},
            {"type": "text/html", "value": html},
        ],
    }

    r = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )

    # SendGrid returns 202 on success
    if r.status_code != 202:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text}")


# ============================================================
# UI Sections
# ============================================================
def render_kpi_row(dff: pd.DataFrame, sev_thresh: float) -> None:
    k = calc_kpis(dff, sev_thresh)
    st.markdown("### Key Metrics (2021–Present)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Features", fmt_int(k["open_features"]))
    c2.metric("Total Incurred", fmt_money_compact(k["total_incurred"]))
    c3.metric("Paid", fmt_money_compact(k["paid"]))
    c4.metric("Outstanding", fmt_money_compact(k["outstanding"]))
    c5.metric("High Severity Features", fmt_int(k["high_sev"]))


def render_trend_section(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Trends (vs prior month)")
    roll = monthly_rollup(dff, sev_thresh).sort_values("trend_month")
    roll = roll[roll["trend_month"] >= pd.Timestamp("2025-01-01")]

    if roll.empty or roll["trend_month"].nunique() < 2:
        st.caption("Not enough monthly history to show trends since 2025 (need 2+ months).")
        return

    metrics = [
        ("total_incurred", "Total Incurred", True),
        ("paid", "Paid", True),
        ("outstanding", "Outstanding", True),
        ("high_sev", "High Severity Features", False),
    ]

    stat_cols = st.columns(4)
    for i, (col, label, is_money) in enumerate(metrics):
        curr, prev = last_two_months(roll, col)
        delta = pct_change(curr or 0, prev or 0) if prev is not None else None
        display_val = fmt_money_compact(curr) if is_money else fmt_int(curr)
        display_delta = None if delta is None else f"{delta:+.1f}%"
        stat_cols[i].metric(label, display_val, display_delta, delta_color="off")

    st.divider()

    def line(metric_col: str, title: str, height: int = 360) -> None:
        vals = pd.to_numeric(roll[metric_col], errors="coerce").fillna(0.0)
        vmin = float(vals.min())
        vmax = float(vals.max())

        if vmin == vmax:
            pad = 1.0 if vmax == 0 else abs(vmax) * 0.05
        else:
            pad = (vmax - vmin) * 0.15

        domain = [vmin - pad, vmax + pad]

        st.altair_chart(
            alt.Chart(roll)
            .mark_line(point=True)
            .encode(
                x=alt.X("trend_month:T", title="Month"),
                y=alt.Y(f"{metric_col}:Q", title=None, scale=alt.Scale(domain=domain, zero=False)),
                tooltip=["trend_month:T", alt.Tooltip(f"{metric_col}:Q", format=",.2f")],
            )
            .properties(title=title, height=height),
            use_container_width=True,
        )

    r1 = st.columns(2)
    with r1[0]:
        line("total_incurred", "Total Incurred ($)")
    with r1[1]:
        line("paid", "Paid ($)")

    r2 = st.columns(2)
    with r2[0]:
        line("outstanding", "Outstanding ($)")
    with r2[1]:
        line("high_sev", "High Severity Features")


def render_email_sender(headlines: List[str], as_of: str) -> None:
    st.markdown("### Email this summary")
    st.caption("Sends a real email (SendGrid). No draft. No excuses.")

    c1, c2 = st.columns([4, 1])
    to_email = c1.text_input("Recipient email address", label_visibility="collapsed", placeholder="name@company.com")

    subject = f"Cover Whale Daily – {as_of}"
    html = build_email_html(headlines=headlines, as_of=as_of, dashboard_url=DASHBOARD_URL)
    text = build_email_text(headlines=headlines, as_of=as_of, dashboard_url=DASHBOARD_URL)

    can_send = bool(to_email and "@" in to_email)

    if c2.button("Send email", disabled=not can_send, use_container_width=True):
        try:
            with st.spinner("Sending..."):
                send_email_sendgrid(to_email.strip(), subject, html, text)
            st.success(f"Sent to {to_email.strip()}")
        except Exception as e:
            st.error(f"Email failed: {e}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    st.set_page_config(page_title="Cover Whale Daily, Powered by NARS", layout="wide", initial_sidebar_state="expanded")

    # Layout CSS: fixed left headlines + fixed right sidebar
    st.markdown(
        """
        <style>
          :root{
            --leftPanelWidth: 22.5rem;
            --rightPanelWidth: 20.5rem;
            --panelTop: 7.8rem;
          }

          .block-container {
            max-width: 1750px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 4.4rem;
            padding-left: calc(var(--leftPanelWidth) + 1.75rem);
            padding-right: calc(var(--rightPanelWidth) + 1.25rem);
          }

          section[data-testid="stSidebar"]{
            left: auto !important;
            right: 0.75rem !important;
            width: var(--rightPanelWidth) !important;
            min-width: var(--rightPanelWidth) !important;
            max-width: var(--rightPanelWidth) !important;
            border-left: 1px solid rgba(16,42,67,0.10);
            background: #F7FAFC;
            box-shadow: 0 2px 10px rgba(16,42,67,0.06);
            border-radius: 12px;
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
          }
          section[data-testid="stSidebar"] > div{
            padding: 0.75rem 0.65rem 1.0rem 0.65rem;
            height: 100vh;
            overflow-y: auto;
            overscroll-behavior: contain;
          }

          section[data-testid="stSidebar"]{
            position: fixed !important;
            top: 0 !important;
            bottom: 0 !important;
            right: 0 !important;
            left: auto !important;
            transform: none !important;
            z-index: 100 !important;
          }

          [data-testid="collapsedControl"]{ display: none !important; }
          button[data-testid="stSidebarCollapseButton"]{ display: none !important; }
          button[title="Close sidebar"]{ display: none !important; }
          button[title="Open sidebar"]{ display: none !important; }
          a[title="Open sidebar"], a[title="Close sidebar"]{ display: none !important; }
          [data-testid="stSidebarCollapsedControl"]{ display:none !important; }
          [data-testid="stSidebarNav"]{ display:none !important; }

          #left-headlines-panel{
            position: fixed;
            top: var(--panelTop);
            left: 1.25rem;
            width: var(--leftPanelWidth);
            max-height: calc(100vh - var(--panelTop) - 1rem);
            overflow: hidden;
            background: transparent;
            z-index: 10;
          }

          .headline-title{
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0.25rem 0 0.85rem 0;
            color: #102A43;
          }

          .headline-box{
            background: #F3F6FA;
            border-left: 6px solid #1F4E79;
            padding: 1.05rem 1.0rem;
            margin: 0.85rem 0;
            border-radius: 12px;
            font-size: 1.08rem;
            line-height: 1.55;
            color: #102A43;
            box-shadow: 0 1px 0 rgba(16,42,67,0.06);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = get_config()
    init_filter_state()

    df, src = load_data(cfg)
    if df.empty:
        st.error("Dataset loaded but returned 0 rows.")
        st.stop()

    df_std = standardize_financials(df, strict_incurred_open_only=False)
    df_std = synthesize_monthly_history_to_start(df_std, start="2021-01-01", seed=7)
    df_std = add_synthetic_denial_reason(df_std)

    # Filters (right)
    with st.sidebar:
        st.markdown("## Filters")

        states = ["All States"] + safe_list(df_std["state"])
        years = ["All Years"] + [str(int(y)) for y in safe_list(df_std["accident_year"]) if pd.notna(y)]
        covs = ["All Coverages"] + safe_list(df_std["coverage_type"])
        adjs = ["All Adjusters"] + safe_list(df_std["adjuster"])
        lobs = ["All Lines"] + safe_list(df_std["line_of_business"])
        statuses = ["All Statuses"] + safe_list(df_std["feature_status"])
        causes = ["All Causes"] + safe_list(df_std["cause_of_loss"])
        vendors = ["All Vendors"] + safe_list(df_std["vendor_name"])
        firms = ["All Firms"] + safe_list(df_std["defense_firm"])
        denials = ["All Reasons"] + [x for x in safe_list(df_std["denial_reason"]) if x not in (None, "", "nan")]

        st.selectbox("State", states, key="f_state")
        st.selectbox("Accident Year", years, key="f_acc_year")
        st.selectbox("Coverage", covs, key="f_coverage")
        st.selectbox("Line of Business", lobs, key="f_lob")
        st.selectbox("Adjuster", adjs, key="f_adjuster")
        st.selectbox("Feature Status", statuses, key="f_status")
        st.selectbox("Cause of Loss", causes, key="f_cause")
        st.selectbox("Litigation", ["All", "Litigated", "Not Litigated"], key="f_litigated")
        st.selectbox("Vendor", vendors, key="f_vendor")
        st.selectbox("Defense Firm", firms, key="f_defense")
        st.selectbox("Denial Reason", denials, key="f_denial_reason")

        st.number_input(
            "High severity threshold",
            min_value=50_000,
            max_value=2_000_000,
            step=25_000,
            key="f_sev_thresh",
        )

        st.button("Reset Filters", on_click=reset_filters)
        st.caption(f"Data source: **{src}**")

    dff = apply_filters(df_std)
    sev_thresh = float(st.session_state["f_sev_thresh"])

    # Left headlines panel
    bullets = build_headline_story(dff, sev_thresh)
    headline_html = "<div id='left-headlines-panel'>"
    headline_html += "<div class='headline-title'>Today’s Headlines</div>"
    for b in bullets:
        headline_html += f"<div class='headline-box'>{b}</div>"
    headline_html += "</div>"
    st.markdown(headline_html, unsafe_allow_html=True)

    # Masthead
    mast = st.columns([0.22, 0.78], vertical_alignment="center")
    with mast[0]:
        logo_path = Path(__file__).resolve().parent / "narslogo.jpg"
        if logo_path.exists():
            st.image(str(logo_path), width=280)

    with mast[1]:
        st.markdown(
            """
            <div style="line-height:1.05; padding-top:6px;">
              <div style="font-size:30px; font-weight:650; letter-spacing:0.25px;">
                Cover Whale Daily, Powered by NARS
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        as_of = "Latest"
        if dff["report_date"].notna().any():
            as_of = str(dff["report_date"].max().date())

        st.markdown(
            f"""<div style='font-size:16px; margin-top:6px;'><b>As of:</b> {as_of}</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)

    st.divider()

    # Ask NARS
    st.markdown("### Ask NARS (Prototype)")
    qcols = st.columns([5, 1])
    q = qcols[0].text_input("Ask a question...", label_visibility="collapsed")
    if qcols[1].button("Ask"):
        st.session_state["_ask_answer"] = answer_question(dff, q, sev_thresh)
    if st.session_state.get("_ask_answer"):
        st.write(st.session_state["_ask_answer"])

    st.divider()

    # Key Metrics
    render_kpi_row(dff, sev_thresh)

    st.divider()

    # Trends
    render_trend_section(dff, sev_thresh)

    # Email sender (new behavior)
    st.divider()
    render_email_sender(headlines=bullets, as_of=as_of)


if __name__ == "__main__":
    main()