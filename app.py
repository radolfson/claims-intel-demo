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

    for col in ("client", "state", "coverage_type", "feature_status", "adjuster", "line_of_business", "cause_of_loss"):
        d[col] = d[col].astype("string")

    d["accident_year"] = pd.to_numeric(d["accident_year"], errors="coerce")

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
    roll = (
        m.groupby("trend_month", as_index=False)
        .agg(
            total_features=("feature_key", "count"),
            open_features=("is_open_inventory", "sum"),
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
    st.markdown("### Today’s Headlines")
    bullets = build_headline_story(dff, sev_thresh)
    st.info(bullets[0])
    st.success(bullets[1])
    st.warning(bullets[2])
    st.info(bullets[3])


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
    if roll.empty or roll["trend_month"].nunique() < 2:
        st.caption("Not enough monthly history to show trends (need 2+ months).")
        return

    roll = roll.sort_values("trend_month")

    metrics = [
        ("open_features", "Open Features", False),
        ("total_incurred", "Total Incurred", True),
        ("paid", "Paid", True),
        ("outstanding", "Outstanding", True),
        ("high_sev", "High Severity Features", False),
    ]

    stat_cols = st.columns(5)
    for idx, (col, label, is_money) in enumerate(metrics):
        curr, prev = last_two_months(roll, col)
        delta = pct_change(curr or 0, prev or 0) if prev is not None else None
        display_val = fmt_money_compact(curr) if is_money else fmt_int(curr)
        display_delta = None if delta is None else f"{delta:+.1f}%"
        stat_cols[idx].metric(label, display_val, display_delta)

    st.divider()

    def line(metric_col: str, title: str) -> None:
        st.altair_chart(
            alt.Chart(roll)
            .mark_line(point=True)
            .encode(
                x=alt.X("trend_month:T", title="Month"),
                y=alt.Y(f"{metric_col}:Q", title=None),
                tooltip=["trend_month:T", alt.Tooltip(f"{metric_col}:Q")],
            )
            .properties(title=title, height=220),
            use_container_width=True,
        )

    r1 = st.columns(3)
    with r1[0]:
        line("open_features", "Open Features")
    with r1[1]:
        line("total_incurred", "Total Incurred ($)")
    with r1[2]:
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
            .reset_index()
            .rename(columns={"index": "feature_status", "feature_status": "count"})
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
                .properties(height=240)
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
            .reset_index()
            .rename(columns={"index": "coverage_type", "coverage_type": "count"})
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
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)

    # Severity Distribution (bins)
    with c3:
        st.caption("Severity Distribution (incurred)")
        bins = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]
        labels = ["$0–50K", "$50–100K", "$100–250K", "$250–500K", "$500K–1M", "$1M–5M", "$5M+"]

        v = dff["incurred_amount"].fillna(0.0)
        # pd.cut requires len(labels) == len(bins) if include_lowest etc. We'll handle last bin separately.
        b = pd.cut(v, bins=bins, labels=labels[: len(bins) - 1], include_lowest=True)
        out = b.value_counts().sort_index().reset_index()
        out.columns = ["bucket", "count"]

        # Add a $5M+ bucket
        over = int((v >= bins[-1]).sum())
        if over > 0:
            out = pd.concat(
                [out, pd.DataFrame({"bucket": [labels[-1]], "count": [over]})],
                ignore_index=True,
            )

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
                .properties(height=240)
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
# -----------------------------
# Email delivery (SendGrid)
# -----------------------------
import html

def _build_email_html(headlines: list[str], dashboard_url: str, as_of: str) -> str:
    """Simple, eye-catching HTML email (no external assets, works in Outlook/Gmail)."""
    safe_url = html.escape(dashboard_url, quote=True)
    safe_as_of = html.escape(as_of)

    cards = []
    for h in headlines:
        h_safe = html.escape(h)
        cards.append(f"""
        <tr>
          <td style="padding:10px 0;">
            <div style="border:1px solid #e5e7eb; border-left:6px solid #1f4e79; border-radius:10px; padding:14px 16px; font-family:Arial,Helvetica,sans-serif; font-size:14px; color:#111827; line-height:1.4;">
              {h_safe}
            </div>
          </td>
        </tr>
        """)

    cards_html = "\n".join(cards) if cards else "<tr><td style='padding:10px 0; font-family:Arial,Helvetica,sans-serif; color:#6b7280;'>No headlines available.</td></tr>"

    return f"""<!doctype html>
<html>
  <body style="margin:0; padding:0; background:#ffffff;">
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#ffffff;">
      <tr>
        <td align="center" style="padding:24px 16px;">
          <table role="presentation" cellpadding="0" cellspacing="0" width="680" style="max-width:680px; width:100%;">
            <tr>
              <td style="font-family:Arial,Helvetica,sans-serif;">
                <div style="font-size:18px; font-weight:700; color:#111827;">Cover Whale Daily, Powered by NARS</div>
                <div style="font-size:12px; color:#6b7280; margin-top:4px;">As of: {safe_as_of}</div>
              </td>
            </tr>

            <tr><td style="height:14px;"></td></tr>

            <tr>
              <td>
                <div style="font-family:Arial,Helvetica,sans-serif; font-size:14px; color:#374151;">
                  Here are today’s headlines:
                </div>
              </td>
            </tr>

            <tr><td style="height:8px;"></td></tr>

            <tr>
              <td>
                <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                  {cards_html}
                </table>
              </td>
            </tr>

            <tr><td style="height:18px;"></td></tr>

            <tr>
              <td align="center">
                <a href="{safe_url}"
                   style="display:inline-block; background:#1f4e79; color:#ffffff; text-decoration:none; font-family:Arial,Helvetica,sans-serif;
                          font-size:14px; font-weight:700; padding:12px 18px; border-radius:10px;">
                  Open Dashboard
                </a>
                <div style="font-family:Arial,Helvetica,sans-serif; font-size:12px; color:#6b7280; margin-top:10px;">
                  Or copy/paste: <span style="color:#1f4e79;">{safe_url}</span>
                </div>
              </td>
            </tr>

            <tr><td style="height:10px;"></td></tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def _send_email_sendgrid(to_email: str, subject: str, html_content: str) -> tuple[bool, str]:
    """Send an email via SendGrid. Requires secrets: SENDGRID_API_KEY and SENDGRID_FROM."""
    api_key = None
    from_email = None

    # Streamlit secrets first, then env vars as fallback
    try:
        api_key = st.secrets.get("SENDGRID_API_KEY")
        from_email = st.secrets.get("SENDGRID_FROM")
    except Exception:
        api_key = None
        from_email = None

    api_key = api_key or os.getenv("SENDGRID_API_KEY")
    from_email = from_email or os.getenv("SENDGRID_FROM")

    if not api_key or not from_email:
        return (False, "Missing SENDGRID_API_KEY / SENDGRID_FROM (set in Streamlit Secrets).")

    try:
        import requests  # Streamlit Cloud usually has it; if not, add to requirements.txt
    except Exception as e:
        return (False, f"Missing dependency 'requests': {e}")

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_content}],
    }

    try:
        resp = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        if resp.status_code in (200, 202):
            return (True, "Sent.")
        # 401 is the classic “your key is wrong/expired/revoked”
        if resp.status_code == 401:
            return (False, "SendGrid 401 Unauthorized: API key invalid/expired/revoked (create a new key + update Streamlit Secrets).")
        return (False, f"SendGrid error {resp.status_code}: {resp.text[:500]}")
    except Exception as e:
        return (False, f"SendGrid request failed: {e}")


def render_email_section(dff: pd.DataFrame, headlines: list[str]) -> None:
    st.markdown("### Email this summary")

    # Determine as-of date and dashboard URL
    as_of = "Latest"
    if "report_date" in dff.columns and dff["report_date"].notna().any():
        try:
            as_of = str(pd.to_datetime(dff["report_date"]).max().date())
        except Exception:
            as_of = "Latest"

    dashboard_url = None
    try:
        dashboard_url = st.secrets.get("DASHBOARD_URL")
    except Exception:
        dashboard_url = None
    dashboard_url = dashboard_url or os.getenv("DASHBOARD_URL") or "https://nars-demo.streamlit.app"

    to_email = st.text_input("Recipient email", placeholder="name@company.com", label_visibility="collapsed", key="email_to")

    c1, c2 = st.columns([1, 3])
    with c1:
        send = st.button("Send email", use_container_width=True)
    with c2:
        st.caption("Requires SendGrid secrets in Streamlit Cloud. If it fails with 401, your API key is wrong. Humans, always with the wrong keys.")

    if send:
        if not to_email or "@" not in to_email:
            st.error("Enter a valid recipient email address.")
            return

        subject = f"Cover Whale Daily (As of {as_of})"
        html_body = _build_email_html(headlines=headlines, dashboard_url=dashboard_url, as_of=as_of)

        ok, msg = _send_email_sendgrid(to_email=to_email, subject=subject, html_content=html_body)
        if ok:
            st.success("Email sent.")
        else:
            st.error(f"Email failed: {msg}")

def main() -> None:
    st.set_page_config(page_title="Claims Intelligence – Daily Summary", layout="wide")

    # Sticky side columns (headlines + filters) without covering header/logo
    st.markdown(
        """
        <style>
          :root{
            --stickyTop: 0.75rem;
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

          .block-container { padding-top: 1.0rem; }
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
        mast = st.columns([0.16, 0.84], vertical_alignment="center")
        with mast[0]:
            logo_path = "narslogo.jpg"
            if os.path.exists(logo_path):
                st.image(logo_path, width=150)
        with mast[1]:
            st.markdown(
                """
                <div style="line-height:1.05">
                  <div style="font-size:34px; font-weight:700;">Claims Intelligence – Daily Summary</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            as_of = "Latest"
            if dff["report_date"].notna().any():
                as_of = str(dff["report_date"].max().date())
            st.markdown(f"**As of:** {as_of}")

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

        render_geographic_concentration(dff)

        st.divider()

        render_metric_rolodex_accident_year(dff, sev_thresh)



        # Email delivery
        render_email_section(dff, build_headline_story(dff, sev_thresh))
if __name__ == "__main__":
    main()