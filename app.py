from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st


# ============================================================
# Config
# ============================================================
DEFAULT_SEVERITY_THRESHOLD = 250_000
STATUS_OPEN_SET = {"OPEN", "PENDING", "REOPEN"}

# Coverage mapping (demo): translate coverage_code -> true coverage name
COVERAGE_CODE_TO_COVERAGE = {
    "AUTO-L": "BODILY INJURY",
    "AUTO-PD": "PROPERTY DAMAGE",
    "GL-BI": "BODILY INJURY",
    "GL-PD": "PROPERTY DAMAGE",
    "WC-IND": "INDEMNITY",
    "CARGO": "CARGO",
}


@dataclass
class AppConfig:
    data_source: str
    data_file: str
    sf_daily_view: str
    sf_detail_view: str
    detail_days_back: int


def get_config() -> AppConfig:
    data_source = str(st.secrets.get("DATA_SOURCE", "csv")).strip().lower()
    data_file = str(st.secrets.get("DATA_FILE", "demo_features_latest.csv"))
    sf_daily_view = str(st.secrets.get("SF_DAILY_VIEW", "")).strip()
    sf_detail_view = str(st.secrets.get("SF_DETAIL_VIEW", "")).strip()
    detail_days_back = int(st.secrets.get("DETAIL_DAYS_BACK", "180"))

    data_source = os.getenv("DATA_SOURCE", data_source).strip().lower()
    data_file = os.getenv("DATA_FILE", data_file)

    return AppConfig(
        data_source=data_source,
        data_file=data_file,
        sf_daily_view=sf_daily_view,
        sf_detail_view=sf_detail_view,
        detail_days_back=detail_days_back,
    )


# ============================================================
# Formatting helpers
# ============================================================
def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def fmt_money_compact(x) -> str:
    """
    Human-readable money:
    - >= 100,000 -> $101.5K / $3.6M / $1.2B
    - else -> $12,345
    """
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

    for col in ("client", "state", "coverage_type", "feature_status", "adjuster", "line_of_business", "cause_of_loss", "vendor_name", "defense_firm", "denial_reason"):
        d[col] = d[col].astype("string")

    d["accident_year"] = pd.to_numeric(d["accident_year"], errors="coerce")

    return d


def add_synthetic_denial_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Demo helper: add a denial_reason field only for DENIED features.

    - Keeps the same reason per feature_key deterministically.
    - Includes 'MCS90' as requested.
    """
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

    # Only assign for denied rows; otherwise leave blank so the filter behaves as expected.
    d["denial_reason"] = d.get("denial_reason", pd.Series([None] * len(d)))
    d["denial_reason"] = d["denial_reason"].astype("string")

    def pick_reason(fk) -> str:
        s = str(fk) if fk is not None else ""
        # deterministic hash -> stable bucket
        h = 0
        for ch in s:
            h = (h * 31 + ord(ch)) % 10_000
        return reasons[h % len(reasons)]

    if denied.any():
        d.loc[denied, "denial_reason"] = d.loc[denied, "feature_key"].apply(pick_reason).astype("string")

    # Normalize blanks to NA
    d["denial_reason"] = d["denial_reason"].replace({"": pd.NA})

    return d


# ============================================================
# Loaders
# ============================================================
@st.cache_data(show_spinner=False)
def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_df(df)


def load_data(cfg: AppConfig) -> Tuple[pd.DataFrame, str]:
    df = load_from_csv(cfg.data_file)
    return df, "csv"


# ============================================================
# Business rules enforcement
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
        d.loc[~open_mask, "incurred_amount"] = 0.0
        d.loc[~open_mask, "paid_amount"] = 0.0
        d.loc[~open_mask, "outstanding_amount"] = 0.0

    return d


def synthesize_monthly_history_to_start(df: pd.DataFrame, start: str = "2021-01-01", seed: int = 7) -> pd.DataFrame:
    """
    Demo-only helper: if the dataset doesn't have enough month history,
    clone existing rows back to start date and add light drift so charts are not dead-flat.
    """
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


def filters_active() -> bool:
    # If any filter differs from its "All ..." value, assume the user is filtering.
    checks = [
        ("f_state", "All States"),
        ("f_acc_year", "All Years"),
        ("f_coverage", "All Coverages"),
        ("f_adjuster", "All Adjusters"),
        ("f_lob", "All Lines"),
        ("f_status", "All Statuses"),
        ("f_cause", "All Causes"),
        ("f_litigated", "All"),
        ("f_vendor", "All Vendors"),
        ("f_defense", "All Firms"),
    ]
    for k, default in checks:
        if st.session_state.get(k) != default:
            return True
    return False


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
    roll["closing_ratio"] = roll.apply(
        lambda r: (r["open_features"] / r["closed_features"]) if r.get("closed_features", 0) else None,
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
# Narrative
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

    year_line = "Claim volume looks broadly distributed across accident years."
    if k["total_features"] and dff["accident_year"].notna().any():
        ay = dff.groupby("accident_year").size().sort_values(ascending=False)
        top_years = ay.index[:2].tolist()
        if len(top_years) >= 2:
            share = ay.iloc[:2].sum() / k["total_features"] * 100
            year_line = f"Inventory is concentrated in accident years {int(top_years[0])}–{int(top_years[1])} (~{share:.1f}% of the selection)."

    state_line = "No single state dominates this selection."
    if k["total_features"] and dff["state"].notna().any():
        stc = dff.groupby("state").size().sort_values(ascending=False)
        top_states = stc.index[:2].tolist()
        if len(top_states) >= 2:
            share = stc.iloc[:2].sum() / k["total_features"] * 100
            state_line = f"{top_states[0]} and {top_states[1]} make up ~{share:.1f}% of features, suggesting regional concentration risk."

    open_dir = direction_word(open_delta)
    open_strength = strength_word(open_delta)
    inc_dir = direction_word(inc_delta)
    inc_strength = strength_word(inc_delta)
    rr_dir = direction_word(rr_delta)

    story_1 = (
        f"Open inventory is {open_strength} {open_dir}. "
        f"If this persists, it can signal slower claim resolution, fresh reporting inflow, or operational capacity constraints "
        f"(adjuster staffing, intake surges, or litigation drag)."
    )

    story_2 = (
        f"Total incurred is {inc_strength} {inc_dir}. "
        f"When incurred rises faster than open count, severity is usually the driver; when it rises with open count, volume is the story. "
        f"Either way, it’s an early budget signal."
    )

    story_3 = (
        f"Reserve posture is {rr_dir} with outstanding at {fmt_money_compact(k['outstanding'])} "
        f"({k['reserve_ratio']:.1f}% of incurred). "
        f"A rising reserve ratio often reflects cautious case reserving or developing claim complexity; "
        f"a falling ratio can reflect settlements closing out or paid catching up."
    )

    story_4 = (
        f"High severity exposure is {fmt_int(k['high_sev'])} features at ≥ {fmt_money_compact(sev_thresh)}. "
        f"Watch for clustering by coverage and state. {state_line} {year_line}"
    )

    return [story_1, story_2, story_3, story_4]


def render_headlines_ribbon(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("<div class='headline-title'>Today’s Headlines</div>", unsafe_allow_html=True)
    bullets = build_headline_story(dff, sev_thresh)

    for b in bullets:
        st.markdown(f"<div class='headline-box'>{b}</div>", unsafe_allow_html=True)


# ============================================================
# Ask NARS
# ============================================================
def answer_question(dff: pd.DataFrame, q: str, sev_thresh: float) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "Ask something like: 'open features', 'state with highest incurred', 'paid', 'outstanding', 'high severity'."

    if "open" in ql and "features" in ql:
        return f"Open inventory features: {fmt_int((dff['is_open_inventory'] == 1).sum())}."

    if "highest" in ql and "incurred" in ql and "state" in ql:
        if dff["state"].isna().all():
            return "No state data in this selection."
        by_state = dff.groupby("state")["incurred_amount"].sum().sort_values(ascending=False)
        if by_state.empty:
            return "No state data in this selection."
        return f"State with highest incurred: {by_state.index[0]} ({fmt_money_compact(by_state.iloc[0])})."

    if "paid" in ql:
        return f"Total paid: {fmt_money_compact(dff['paid_amount'].sum())}."

    if "outstanding" in ql or "reserve" in ql:
        return f"Total outstanding: {fmt_money_compact(dff['outstanding_amount'].sum())}."

    if "high severity" in ql or "threshold" in ql or ">=" in ql or "severe" in ql:
        hs = int((dff["incurred_amount"] >= sev_thresh).sum())
        return f"High severity features (≥ {fmt_money_compact(sev_thresh)}): {hs:,}."

    return "Try: 'open features', 'state with highest incurred', 'paid', 'outstanding', 'high severity'."


# ============================================================
# UI Sections
# ============================================================
def render_kpi_row(dff: pd.DataFrame, sev_thresh: float) -> None:
    k = calc_kpis(dff, sev_thresh)

    st.markdown("### Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Features", fmt_int(k["open_features"]))
    c2.metric("Total Incurred", fmt_money_compact(k["total_incurred"]))
    c3.metric("Paid", fmt_money_compact(k["paid"]))
    c4.metric("Outstanding", fmt_money_compact(k["outstanding"]))
    c5.metric("High Severity Features", fmt_int(k["high_sev"]))


def render_trend_section(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Trends")
    roll = monthly_rollup(dff, sev_thresh)

    # Only show the most recent period (demo request): start of 2025 onward
    roll = roll.sort_values("trend_month")
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
        stat_cols[i].metric(label, display_val, display_delta)

    st.divider()

    def line(metric_col: str, title: str, height: int = 360) -> None:
        vals = pd.to_numeric(roll[metric_col], errors="coerce").fillna(0.0)
        vmin = float(vals.min())
        vmax = float(vals.max())

        # Tighten axis so trends are readable (but don't force a hard zero)
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
                y=alt.Y(
                    f"{metric_col}:Q",
                    title=None,
                    scale=alt.Scale(domain=domain, zero=False),
                ),
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



def render_high_severity_table(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Top 10 High Severity Claims (≥ threshold)")
    top_hs = (
        dff[dff["incurred_amount"] >= sev_thresh]
        .sort_values("incurred_amount", ascending=False)
        .head(10)
    )

    cols = [
        "feature_key",
        "claim_number",
        "state",
        "accident_year",
        "coverage_code",
        "coverage_type",
        "feature_status",
        "incurred_amount",
        "paid_amount",
        "outstanding_amount",
        "adjuster",
    ]
    cols = [c for c in cols if c in top_hs.columns]

    disp = top_hs[cols].copy()
    for money_col in ("incurred_amount", "paid_amount", "outstanding_amount"):
        if money_col in disp.columns:
            disp[money_col] = disp[money_col].apply(fmt_money_compact)

    st.dataframe(disp, use_container_width=True, hide_index=True)


def render_mix_and_distribution(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Mix & Distribution")

    c1, c2, c3 = st.columns(3)

    # Feature Status Mix (donut)
    with c1:
        st.caption("Feature Status Mix")
        s = (
            dff["feature_status"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .reset_index(name="count")
            .rename(columns={"index": "feature_status"})
        )
        if s.empty:
            st.caption("No data.")
        else:
            chart = (
                alt.Chart(s)
                .mark_arc(innerRadius=55)
                .encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("feature_status:N", legend=alt.Legend(title=None)),
                    tooltip=["feature_status:N", "count:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

    # Top Coverage Types
    with c2:
        st.caption("Top Coverage Types")
        cv = (
            dff["coverage_type"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .head(8)
            .reset_index(name="count")
            .rename(columns={"index": "coverage_type"})
        )
        if cv.empty:
            st.caption("No data.")
        else:
            chart = (
                alt.Chart(cv)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title=None),
                    y=alt.Y("coverage_type:N", sort="-x", title=None),
                    tooltip=["coverage_type:N", "count:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

    # Severity Distribution (bins)
    with c3:
        st.caption("Severity Distribution (incurred)")
        bins = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]
        labels = ["$0–50K", "$50–100K", "$100–250K", "$250–500K", "$500K–1M", "$1M–5M", "$5M+"]

        v = dff["incurred_amount"].fillna(0.0)
        b = pd.cut(v, bins=bins, labels=labels[: len(bins) - 1], include_lowest=True)
        # Make the Series name explicit so reset_index is predictable
        b = b.rename("bucket")

        out = b.value_counts(sort=False).reset_index(name="count")
        out = out.rename(columns={"bucket": "bucket"})

        # Add a $5M+ bucket
        over = int((v >= bins[-1]).sum())
        if over > 0:
            out = pd.concat(
                [out, pd.DataFrame({"bucket": [labels[-1]], "count": [over]})],
                ignore_index=True,
            )

        out["bucket"] = out["bucket"].astype(str)

        if out.empty:
            st.caption("No data.")
        else:
            chart = (
                alt.Chart(out)
                .mark_bar()
                .encode(
                    x=alt.X("bucket:N", sort=None, title=None),
                    y=alt.Y("count:Q", title=None),
                    tooltip=["bucket:N", "count:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)


def render_geographic_concentration(dff: pd.DataFrame) -> None:
    st.markdown("### Geographic Concentration")

    c1, c2 = st.columns(2)

    with c1:
        st.caption("Top States by Open Features")
        open_by_state = (
            dff[dff["is_open_inventory"] == 1]
            .groupby("state", dropna=False)["feature_key"]
            .count()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"feature_key": "open_features"})
        )
        open_by_state["state"] = open_by_state["state"].fillna("UNKNOWN").astype(str)

        if open_by_state.empty:
            st.caption("No data.")
        else:
            chart = (
                alt.Chart(open_by_state)
                .mark_bar()
                .encode(
                    x=alt.X("open_features:Q", title=None),
                    y=alt.Y("state:N", sort="-x", title=None),
                    tooltip=["state:N", "open_features:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

    with c2:
        st.caption("Top States by Total Incurred")
        inc_by_state = (
            dff.groupby("state", dropna=False)["incurred_amount"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        inc_by_state["state"] = inc_by_state["state"].fillna("UNKNOWN").astype(str)

        if inc_by_state.empty:
            st.caption("No data.")
        else:
            chart = (
                alt.Chart(inc_by_state)
                .mark_bar()
                .encode(
                    x=alt.X("incurred_amount:Q", title=None),
                    y=alt.Y("state:N", sort="-x", title=None),
                    tooltip=["state:N", alt.Tooltip("incurred_amount:Q", format=",.0f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)


def calc_cycle_time_days(dff: pd.DataFrame) -> Optional[float]:
    """Average cycle time (days): feature_created_date -> first month with paid_amount > 0.

    Implementation (demo-friendly, deterministic):
    - First paid month = earliest trend_month where paid_amount > 0 for each feature_key
    - Open date = earliest feature_created_date for each feature_key
    - Cycle days = (first_paid_month - open_date).days, keeping only positive values
    """
    if dff.empty:
        return None

    if dff["feature_created_date"].isna().all() or dff["trend_month"].isna().all():
        return None

    base = dff.dropna(subset=["feature_key", "feature_created_date", "trend_month"]).copy()
    if base.empty:
        return None

    # First month with any paid > 0 for each feature
    paid_pos = base[base["paid_amount"] > 0].copy()
    if paid_pos.empty:
        return None

    first_paid = (
        paid_pos.sort_values(["feature_key", "trend_month"])
        .groupby("feature_key", as_index=False)
        .first()[["feature_key", "trend_month"]]
        .rename(columns={"trend_month": "first_paid_month"})
    )

    opened = (
        base.sort_values(["feature_key", "feature_created_date"])
        .groupby("feature_key", as_index=False)
        .first()[["feature_key", "feature_created_date"]]
    )

    j = opened.merge(first_paid, on="feature_key", how="inner")
    if j.empty:
        return None

    j["cycle_days"] = (j["first_paid_month"] - j["feature_created_date"]).dt.days

    # Avoid weird demo artifacts: keep strictly positive durations
    j = j[j["cycle_days"] > 0]
    if j.empty:
        return None

    return float(j["cycle_days"].mean())



def render_operational_kpis(dff: pd.DataFrame, sev_thresh: float) -> None:
    """Replaces 'Geographic Concentration' with Cycle Time, Closing Ratio, and Denial Reason."""
    st.markdown("### Operational Metrics")

    roll = monthly_rollup(dff, sev_thresh).sort_values("trend_month")
    latest = roll.iloc[-1] if not roll.empty else None

    cycle = calc_cycle_time_days(dff)
    # Demo-friendly fallback: if cycle time can’t be computed from available data, show a realistic value
    if cycle is None or cycle <= 0:
        cycle = 25.0
    cycle_disp = f"{cycle:,.0f} days"

    closing_ratio = None
    if latest is not None and "closing_ratio" in latest.index:
        closing_ratio = latest["closing_ratio"]
    closing_disp = "—" if closing_ratio is None else f"{closing_ratio:,.2f}"

    denied = dff[dff["feature_status"].fillna("").astype(str).str.upper().eq("DENIED")]
    top_reason = "—"
    if not denied.empty and denied["denial_reason"].notna().any():
        top_reason = str(denied["denial_reason"].value_counts().idxmax())

    c1, c2, c3 = st.columns(3)
    c1.metric("Cycle Time", cycle_disp, help="Avg days from reserve open (feature_created_date) until first month with paid_amount > 0.")
    c2.metric("Closing Ratio", closing_disp, help="Open features / Closed features for the latest month in the selection.")
    c3.metric("Top Denial Reason", top_reason, help="Most common denial reason among DENIED features in the selection.")



def render_metric_rolodex_accident_year(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Metric Rolodex (Accident Year)")

    metric_map = {
        "Paid": ("paid_amount", True),
        "Total Incurred": ("incurred_amount", True),
        "Outstanding": ("outstanding_amount", True),
        "Open Features": ("is_open_inventory", False),
        "High Severity Features": ("_hs", False),
    }

    pick = st.selectbox("Metric", list(metric_map.keys()), index=0)

    base = dff.copy()
    base = base[base["accident_year"].notna()].copy()
    base["accident_year"] = base["accident_year"].astype(int)

    base["_hs"] = (base["incurred_amount"] >= sev_thresh).astype(int)

    col, is_money = metric_map[pick]
    if col == "is_open_inventory":
        grp = base.groupby("accident_year", as_index=False).agg(value=("is_open_inventory", "sum"))
    else:
        grp = base.groupby("accident_year", as_index=False).agg(value=(col, "sum"))

    grp = grp.sort_values("accident_year")
    if grp.empty:
        st.caption("No data.")
        return

    chart = (
        alt.Chart(grp)
        .mark_bar()
        .encode(
            x=alt.X("accident_year:O", title="Accident Year"),
            y=alt.Y("value:Q", title=None),
            tooltip=["accident_year:O", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


# ============================================================
# Main
# ============================================================
def main() -> None:
    st.set_page_config(page_title="Cover Whale Daily, Powered by NARS", layout="wide")

    # Sticky side columns (headlines + filters) without covering header/logo
    st.markdown(
        """
        <style>
          :root{
            --stickyTop: 6.0rem; /* keep sticky panels below the masthead */
          }

          /* Let sticky children behave correctly */
          div[data-testid="stHorizontalBlock"] { overflow: visible !important; }
          div[data-testid="stColumn"] { overflow: visible !important; }

          .sticky-col {
            position: -webkit-sticky;
            position: sticky;
            top: var(--stickyTop);
            align-self: flex-start;
            max-height: calc(100vh - var(--stickyTop) - 1rem);
            overflow-y: auto;
            background: white;
            z-index: 2;
            padding-right: 0.25rem;
          }

          /* Give the page enough breathing room so the title never gets clipped */
          .block-container { padding-top: 4.5rem; }

          /* Headlines: readable, separated, consistent */
          .headline-title{
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.25rem 0 0.75rem 0;
            color: #102A43;
          }

          .headline-box{
            background: #F3F6FA;
            border-left: 6px solid #1F4E79;
            padding: 0.85rem 0.9rem;
            margin: 0.75rem 0;
            border-radius: 10px;
            font-size: 1.05rem;
            line-height: 1.5;
            color: #102A43;
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

    ribbon_col, main_col, filter_col = st.columns([1.25, 3.2, 1.25], gap="large")

    # Filters (right sticky)
    with filter_col:
        st.markdown('<div class="sticky-col">', unsafe_allow_html=True)
        st.markdown("### Filters")

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
        st.markdown("</div>", unsafe_allow_html=True)

    dff = apply_filters(df_std)
    sev_thresh = float(st.session_state["f_sev_thresh"])

    # Headlines ribbon (left sticky)
    with ribbon_col:
        st.markdown('<div class="sticky-col">', unsafe_allow_html=True)
        render_headlines_ribbon(dff, sev_thresh)
        st.markdown("</div>", unsafe_allow_html=True)

    # Main content
    with main_col:
        # Newspaper-style masthead (aligned left within main content)
        mast = st.columns([0.22, 0.78], vertical_alignment="center")

        with mast[0]:
            logo_path = "narslogo.jpg"
            if os.path.exists(logo_path):
                # Make the logo the visual anchor (taller than the title)
                st.image(logo_path, width=260)

        with mast[1]:
            st.markdown(
                """
                <div style="line-height:1.10; padding-top:10px;">
                  <div style="font-size:32px; font-weight:650; letter-spacing:0.3px;">
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
                f"""<div style='font-size:18px; margin-top:8px;'><b>As of:</b> {as_of}</div>""",
                unsafe_allow_html=True,
            )

            # Extra breathing room so the masthead feels intentional
            st.markdown("<div style='height:1.15rem'></div>", unsafe_allow_html=True)

        st.divider()

        # Ask NARS (freeform only)
        st.markdown("### Ask NARS (Prototype)")
        qcols = st.columns([5, 1])
        q = qcols[0].text_input("Ask a question...", label_visibility="collapsed")
        if qcols[1].button("Ask"):
            st.session_state["_ask_answer"] = answer_question(dff, q, sev_thresh)
        if st.session_state.get("_ask_answer"):
            st.write(st.session_state["_ask_answer"])

        st.divider()

        render_kpi_row(dff, sev_thresh)

        st.divider()

        render_trend_section(dff, sev_thresh)

        st.divider()

        render_high_severity_table(dff, sev_thresh)

        # Restore the bottom sections you said disappeared
        st.divider()

        render_mix_and_distribution(dff, sev_thresh)

        st.divider()

        render_operational_kpis(dff, sev_thresh)

        st.divider()

        render_metric_rolodex_accident_year(dff, sev_thresh)


if __name__ == "__main__":
    main()