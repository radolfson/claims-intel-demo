# app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt


# ============================================================
# Config
# ============================================================
DEFAULT_SEVERITY_THRESHOLD = 250_000
STATUS_OPEN_SET = {"OPEN", "PENDING", "REOPEN"}


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
def fmt_currency(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def safe_list(series: pd.Series) -> list:
    return sorted([x for x in series.dropna().unique()])


def month_floor(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("M").dt.to_timestamp()


def pct_change(curr: float, prev: float) -> Optional[float]:
    if prev is None or prev == 0:
        return None
    return (curr - prev) / prev * 100.0


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


@st.cache_data(show_spinner=False)
def load_from_snowflake(detail_view: str) -> pd.DataFrame:
    try:
        import snowflake.connector  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Snowflake connector not installed. Add snowflake-connector-python to requirements.txt."
        ) from e

    account = st.secrets.get("SNOWFLAKE_ACCOUNT")
    user = st.secrets.get("SNOWFLAKE_USER")
    warehouse = st.secrets.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database = st.secrets.get("SNOWFLAKE_DATABASE", "NARS")
    schema = st.secrets.get("SNOWFLAKE_SCHEMA", "PROCESSED")

    private_key_pem = str(st.secrets.get("SNOWFLAKE_PRIVATE_KEY", "")).strip()
    password = st.secrets.get("SNOWFLAKE_PASSWORD", None)

    if not account or not user:
        raise RuntimeError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in Streamlit secrets.")

    connect_kwargs = dict(
        account=account,
        user=user,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )

    if private_key_pem:
        try:
            from cryptography.hazmat.primitives import serialization  # type: ignore
            from cryptography.hazmat.backends import default_backend  # type: ignore

            pkey = serialization.load_pem_private_key(
                private_key_pem.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )
            connect_kwargs["private_key"] = pkey
        except Exception as e:
            raise RuntimeError("Failed to parse SNOWFLAKE_PRIVATE_KEY (PEM).") from e
    elif password:
        connect_kwargs["password"] = password
    else:
        raise RuntimeError("Provide SNOWFLAKE_PRIVATE_KEY or SNOWFLAKE_PASSWORD in secrets.")

    sql = f"SELECT * FROM {detail_view}"

    ctx = snowflake.connector.connect(**connect_kwargs)
    try:
        cur = ctx.cursor()
        try:
            cur.execute(sql)
            df = cur.fetch_pandas_all()
        finally:
            cur.close()
    finally:
        ctx.close()

    return normalize_df(df)


def load_data(cfg: AppConfig) -> Tuple[pd.DataFrame, str]:
    if cfg.data_source == "snowflake":
        if not cfg.sf_detail_view:
            raise RuntimeError("DATA_SOURCE=snowflake but SF_DETAIL_VIEW is not set in secrets.")
        df = load_from_snowflake(cfg.sf_detail_view)
        return df, "snowflake"
    df = load_from_csv(cfg.data_file)
    return df, "csv"


# ============================================================
# Business rules enforcement (fast, demo-safe, defensible)
# ============================================================
def standardize_financials(df: pd.DataFrame, strict_incurred_open_only: bool) -> pd.DataFrame:
    """
    Enforce demo rules so KPIs/tables are consistent even if source data is messy.
    Rules:
      - DENIED: paid/outstanding/incurred = 0
      - CLOSED: outstanding = 0, incurred = paid
      - Otherwise: incurred = max(incurred, paid + outstanding) (keeps things sane)
      - Optional strict: if status not in OPEN/PENDING/REOPEN then incurred = 0
    """
    d = df.copy()
    status = d["feature_status"].fillna("UNKNOWN").astype(str)

    # Denied => zero everything
    denied_mask = status.str.upper().eq("DENIED")
    d.loc[denied_mask, ["paid_amount", "outstanding_amount", "incurred_amount"]] = 0.0
    d.loc[denied_mask, "is_open_inventory"] = 0

    # Closed => outstanding 0, incurred = paid
    closed_mask = status.str.upper().eq("CLOSED")
    d.loc[closed_mask, "outstanding_amount"] = 0.0
    d.loc[closed_mask, "incurred_amount"] = d.loc[closed_mask, "paid_amount"]
    d.loc[closed_mask, "is_open_inventory"] = 0

    # Recompute incurred sanity for remaining
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


def synthesize_monthly_history(df: pd.DataFrame, months_back: int = 9, seed: int = 7) -> pd.DataFrame:
    """
    Demo-only helper: if you only have 1 month, we create a plausible multi-month history
    by duplicating features across prior months with small randomized drift.
    This does NOT change feature-level totals in the "current month" slice; it only
    enables trend charts to render.
    """
    import numpy as np

    d = df.copy()
    d = d.dropna(subset=["trend_month"]).copy()
    if d.empty:
        return df

    cur_month = d["trend_month"].max()
    months = pd.date_range(end=cur_month, periods=months_back + 1, freq="MS")

    rng = np.random.default_rng(seed)

    frames = []
    for m in months:
        f = d.copy()
        f["trend_month"] = m

        # Apply gentle drift so charts look real but not insane
        drift = 1.0 + rng.normal(0.0, 0.06, size=len(f))  # ~6% std dev
        drift = np.clip(drift, 0.80, 1.25)

        f["paid_amount"] = (f["paid_amount"] * drift).round(2)
        f["outstanding_amount"] = (f["outstanding_amount"] * (1.0 + rng.normal(0.0, 0.05, size=len(f)))).clip(lower=0).round(2)
        f["incurred_amount"] = (f["paid_amount"] + f["outstanding_amount"]).round(2)

        frames.append(f)

    out = pd.concat(frames, ignore_index=True)
    return out


# ============================================================
# Filters
# ============================================================
def init_filter_state() -> None:
    if st.session_state.get("_filters_initialized"):
        return
    st.session_state["_filters_initialized"] = True

    st.session_state["f_client"] = "All Clients"
    st.session_state["f_state"] = "All States"
    st.session_state["f_acc_year"] = "All Years"
    st.session_state["f_coverage"] = "All Coverage"
    st.session_state["f_adjuster"] = "All Adjusters"
    st.session_state["f_lob"] = "All Lines"
    st.session_state["f_status"] = "All Statuses"
    st.session_state["f_cause"] = "All Causes"
    st.session_state["f_litigated"] = "All"
    st.session_state["f_vendor"] = "All Vendors"
    st.session_state["f_defense"] = "All Firms"

    st.session_state["f_open_only"] = False
    st.session_state["f_sev_thresh"] = DEFAULT_SEVERITY_THRESHOLD

    # New: rule toggles
    st.session_state["f_strict_incurred"] = False
    st.session_state["f_demo_synth_trends"] = True


def reset_filters() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("f_") or k == "_filters_initialized":
            del st.session_state[k]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    if st.session_state["f_client"] != "All Clients":
        dff = dff[dff["client"] == st.session_state["f_client"]]

    if st.session_state["f_state"] != "All States":
        dff = dff[dff["state"] == st.session_state["f_state"]]

    if st.session_state["f_acc_year"] != "All Years":
        dff = dff[dff["accident_year"] == float(st.session_state["f_acc_year"])]

    if st.session_state["f_coverage"] != "All Coverage":
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

    if st.session_state["f_open_only"]:
        dff = dff[dff["is_open_inventory"] == 1]

    return dff


# ============================================================
# KPI + MoM helpers
# ============================================================
def calc_kpis(dff: pd.DataFrame, sev_thresh: float) -> dict:
    open_features = int((dff["is_open_inventory"] == 1).sum())
    total_incurred = float(dff["incurred_amount"].sum())
    paid = float(dff["paid_amount"].sum())
    outstanding = float(dff["outstanding_amount"].sum())
    high_sev = int((dff["incurred_amount"] >= sev_thresh).sum())
    total_features = int(len(dff))
    return dict(
        open_features=open_features,
        total_features=total_features,
        total_incurred=total_incurred,
        paid=paid,
        outstanding=outstanding,
        high_sev=high_sev,
    )


def monthly_rollup(dff: pd.DataFrame, sev_thresh: float) -> pd.DataFrame:
    m = dff.dropna(subset=["trend_month"]).copy()
    if m.empty:
        return pd.DataFrame(columns=["trend_month", "open_features", "total_incurred", "paid", "outstanding", "high_sev", "total_features"])

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
    return roll


def last_two_months(roll: pd.DataFrame, metric: str) -> Tuple[Optional[float], Optional[float]]:
    if roll.empty or metric not in roll.columns:
        return None, None
    if len(roll) == 1:
        return float(roll.iloc[-1][metric]), None
    return float(roll.iloc[-1][metric]), float(roll.iloc[-2][metric])


# ============================================================
# Charts (Altair)
# ============================================================
def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, title="Month"),
            y=alt.Y(y, title=None),
            tooltip=[x, y],
        )
        .properties(title=title, height=220)
    )


def donut_chart(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta(field=value, type="quantitative"),
            color=alt.Color(field=category, type="nominal"),
            tooltip=[category, value],
        )
        .properties(title=title, height=260)
    )


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False) -> alt.Chart:
    if horizontal:
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y(x, sort="-x", title=None),
                x=alt.X(y, title=None),
                tooltip=[x, y],
            )
            .properties(title=title, height=260)
        )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, sort="-y", title=None),
            y=alt.Y(y, title=None),
            tooltip=[x, y],
        )
        .properties(title=title, height=260)
    )


# ============================================================
# Narrative (safe)
# ============================================================
def render_headlines(dff: pd.DataFrame, sev_thresh: float) -> None:
    k = calc_kpis(dff, sev_thresh)
    open_ct = k["open_features"]
    total_ct = k["total_features"]
    open_pct = (open_ct / total_ct * 100) if total_ct else 0.0

    hs_ct = k["high_sev"]
    hs_pct = (hs_ct / open_ct * 100) if open_ct else 0.0

    total_incurred = k["total_incurred"]
    total_out = k["outstanding"]
    out_pct = (total_out / total_incurred * 100) if total_incurred else 0.0

    year_msg = f"Open features total {open_ct:,} (~{open_pct:.1f}% of filtered inventory)."
    if total_ct and dff["accident_year"].notna().any():
        ay = dff.groupby("accident_year").size().sort_values(ascending=False)
        top_years = ay.index[:2].tolist()
        if len(top_years) >= 2:
            top_share = ay.iloc[:2].sum() / total_ct * 100
            year_msg = f"Open features total {open_ct:,}, concentrated in accident years {int(top_years[0])}–{int(top_years[1])} (~{top_share:.1f}% of inventory)."
        elif len(top_years) == 1:
            year_msg = f"Open features total {open_ct:,}, concentrated in accident year {int(top_years[0])} (filtered selection)."

    state_msg = "Geographic concentration not available for current filters."
    if total_ct and dff["state"].notna().any():
        stc = dff.groupby("state").size().sort_values(ascending=False)
        top_states = stc.index[:2].tolist()
        if len(top_states) >= 2:
            st_share = stc.iloc[:2].sum() / total_ct * 100
            state_msg = f"{top_states[0]} and {top_states[1]} account for ~{st_share:.1f}% of feature count."
        elif len(top_states) == 1:
            state_msg = f"{top_states[0]} accounts for 100% of feature count (filtered selection)."

    st.markdown("### Today’s Headlines")
    st.info(year_msg)
    st.success(f"High severity features represent {hs_pct:.1f}% of open inventory ({hs_ct:,} features at ≥ {fmt_currency(sev_thresh)}).")
    st.warning(f"Total incurred stands at {fmt_currency(total_incurred)} with {fmt_currency(total_out)} outstanding ({out_pct:.1f}% case reserves).")
    st.info(state_msg)


# ============================================================
# Ask NARS (prototype deterministic) + Quick buttons
# ============================================================
def answer_question(dff: pd.DataFrame, q: str, sev_thresh: float) -> str:
    ql = (q or "").lower().strip()
    if not ql:
        return "Try: 'top 10 severe', 'open features', 'state with highest incurred', 'paid', 'outstanding'."

    if "top" in ql and ("severe" in ql or "severity" in ql):
        return "Use the Top 10 Severe table below (it’s already sorted)."

    if "open" in ql and "features" in ql:
        return f"Open inventory features: {fmt_int((dff['is_open_inventory'] == 1).sum())}."

    if "highest" in ql and "incurred" in ql and "state" in ql:
        if dff["state"].isna().all():
            return "No state data in this selection."
        by_state = dff.groupby("state")["incurred_amount"].sum().sort_values(ascending=False)
        if by_state.empty:
            return "No state data in this selection."
        return f"State with highest incurred: {by_state.index[0]} ({fmt_currency(by_state.iloc[0])})."

    if "paid" in ql:
        return f"Total paid: {fmt_currency(dff['paid_amount'].sum())}."

    if "outstanding" in ql or "reserve" in ql:
        return f"Total outstanding: {fmt_currency(dff['outstanding_amount'].sum())}."

    if "high severity" in ql or "threshold" in ql or ">=" in ql:
        hs = int((dff["incurred_amount"] >= sev_thresh).sum())
        return f"High severity features (≥ {fmt_currency(sev_thresh)}): {hs:,}."

    return "Try: 'open features', 'state with highest incurred', 'paid', 'outstanding'."


# ============================================================
# UI Sections
# ============================================================
def render_kpi_row(dff: pd.DataFrame, sev_thresh: float) -> None:
    k = calc_kpis(dff, sev_thresh)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Features", fmt_int(k["open_features"]))
    c2.metric("Total Incurred", fmt_currency(k["total_incurred"]))
    c3.metric("Paid", fmt_currency(k["paid"]))
    c4.metric("Outstanding", fmt_currency(k["outstanding"]))
    c5.metric("High Severity Features", fmt_int(k["high_sev"]))


def render_trend_section(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Trends (Month over Month)")

    roll = monthly_rollup(dff, sev_thresh)

    if roll.empty or roll["trend_month"].nunique() < 2:
        st.caption("Not enough monthly history to show MoM trends (need 2+ months).")
        return

    roll = roll.sort_values("trend_month").tail(12)

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
        stat_cols[idx].metric(
            label,
            fmt_currency(curr) if is_money else fmt_int(curr),
            None if delta is None else f"{delta:+.1f}%",
        )

    st.divider()

    r1 = st.columns(3)
    r2 = st.columns(2)
    charts = [
        ("open_features", "Open Features"),
        ("total_incurred", "Total Incurred ($)"),
        ("paid", "Paid ($)"),
        ("outstanding", "Outstanding ($)"),
        ("high_sev", "High Severity Features"),
    ]
    containers = [r1[0], r1[1], r1[2], r2[0], r2[1]]

    for (metric_col, title), container in zip(charts, containers):
        with container:
            st.altair_chart(
                line_chart(roll, "trend_month:T", f"{metric_col}:Q", title),
                use_container_width=True,
            )


def render_mix_distribution(dff: pd.DataFrame) -> None:
    st.markdown("### Mix & Distribution")

    left, mid, right = st.columns([1.2, 1.0, 1.2], gap="large")

    with left:
        status_counts = dff["feature_status"].fillna("UNKNOWN").value_counts().reset_index()
        status_counts.columns = ["feature_status", "count"]
        st.altair_chart(donut_chart(status_counts, "feature_status", "count", "Feature Status Mix"), use_container_width=True)

    with mid:
        cov = dff["coverage_type"].fillna("UNKNOWN").value_counts().head(10).reset_index()
        cov.columns = ["coverage_type", "count"]
        st.altair_chart(bar_chart(cov, "coverage_type", "count", "Top Coverage Types"), use_container_width=True)

    with right:
        bins = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 10_000_000_000]
        labels = ["$0–50K", "$50–100K", "$100–250K", "$250–500K", "$500K–1M", "$1M+"]
        tmp = dff.copy()
        tmp["sev_bucket"] = pd.cut(tmp["incurred_amount"], bins=bins, labels=labels, include_lowest=True)
        hist = tmp["sev_bucket"].value_counts().reindex(labels, fill_value=0).reset_index()
        hist.columns = ["sev_bucket", "count"]
        st.altair_chart(bar_chart(hist, "sev_bucket", "count", "Severity Distribution"), use_container_width=True)


def render_geo(dff: pd.DataFrame) -> None:
    st.markdown("### Geographic Concentration")

    a, b = st.columns(2, gap="large")

    with a:
        tmp = dff.groupby("state", dropna=False)["is_open_inventory"].sum().sort_values(ascending=False).head(10).reset_index()
        tmp.columns = ["state", "open_features"]
        st.altair_chart(bar_chart(tmp, "state", "open_features", "Top States by Open Features", horizontal=True), use_container_width=True)

    with b:
        tmp = dff.groupby("state", dropna=False)["incurred_amount"].sum().sort_values(ascending=False).head(10).reset_index()
        tmp.columns = ["state", "total_incurred"]
        st.altair_chart(bar_chart(tmp, "state", "total_incurred", "Top States by Total Incurred", horizontal=True), use_container_width=True)


def render_rolodex(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Metric Rolodex (Accident Year)")

    metric = st.selectbox(
        "Metric",
        ["Open Features", "Total Features", "Total Incurred", "Paid", "Outstanding", "High Severity Features"],
        index=0,
        label_visibility="collapsed",
    )

    if dff["accident_year"].isna().all():
        st.caption("No accident year data.")
        return

    tmp = dff.copy()
    if metric == "Open Features":
        tmp = tmp[tmp["is_open_inventory"] == 1]
        chart_df = tmp.groupby("accident_year").size().reset_index(name="value")
    elif metric == "Total Features":
        chart_df = tmp.groupby("accident_year").size().reset_index(name="value")
    elif metric == "Total Incurred":
        chart_df = tmp.groupby("accident_year")["incurred_amount"].sum().reset_index(name="value")
    elif metric == "Paid":
        chart_df = tmp.groupby("accident_year")["paid_amount"].sum().reset_index(name="value")
    elif metric == "Outstanding":
        chart_df = tmp.groupby("accident_year")["outstanding_amount"].sum().reset_index(name="value")
    else:
        tmp["is_hs"] = (tmp["incurred_amount"] >= sev_thresh).astype(int)
        chart_df = tmp.groupby("accident_year")["is_hs"].sum().reset_index(name="value")

    chart_df = chart_df.sort_values("accident_year")
    st.altair_chart(
        alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("accident_year:O", title="Accident Year"),
            y=alt.Y("value:Q", title=None),
            tooltip=["accident_year", "value"],
        ).properties(height=260),
        use_container_width=True,
    )


def render_tables(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### Tables")

    left, right = st.columns([1.0, 1.4], gap="large")

    with left:
        st.markdown("#### Feature Status Summary")
        total = len(dff)
        open_ct = int(dff["feature_status"].isin(list(STATUS_OPEN_SET)).sum())
        closed_ct = int((dff["feature_status"] == "CLOSED").sum())
        pending_ct = int((dff["feature_status"] == "PENDING").sum())
        denied_ct = int((dff["feature_status"] == "DENIED").sum())

        st.write(f"**Total Features:** {total:,}")
        st.write(f"**Open Features:** {open_ct:,}")
        st.write(f"**Pending Features:** {pending_ct:,}")
        st.write(f"**Closed Features:** {closed_ct:,}")
        st.write(f"**Denied Features:** {denied_ct:,}")

        st.divider()
        st.markdown("#### Top Open Inventory (by incurred)")
        top_open = dff[dff["is_open_inventory"] == 1].sort_values("incurred_amount", ascending=False).head(20)
        cols = ["feature_key", "claim_number", "state", "accident_year", "coverage_type", "feature_status", "incurred_amount", "paid_amount", "outstanding_amount", "adjuster"]
        cols = [c for c in cols if c in top_open.columns]
        st.dataframe(top_open[cols], use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### High Severity Features (≥ threshold)")
        top_hs = dff[dff["incurred_amount"] >= sev_thresh].sort_values("incurred_amount", ascending=False).head(50)
        cols = ["feature_key", "claim_number", "state", "accident_year", "coverage_code", "coverage_type", "feature_status", "incurred_amount", "paid_amount", "outstanding_amount", "adjuster"]
        cols = [c for c in cols if c in top_hs.columns]
        st.dataframe(top_hs[cols], use_container_width=True, hide_index=True)


def render_top10_severe(dff: pd.DataFrame) -> None:
    st.markdown("#### Top 10 Severe (by incurred)")
    top = dff.sort_values("incurred_amount", ascending=False).head(10)
    cols = ["feature_key", "claim_number", "state", "accident_year", "coverage_type", "feature_status", "incurred_amount", "paid_amount", "outstanding_amount"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)


# ============================================================
# Main
# ============================================================
def main() -> None:
    st.set_page_config(page_title="Claims Intelligence – Daily Summary", layout="wide")
    cfg = get_config()
    init_filter_state()

    try:
        df, src = load_data(cfg)
    except Exception as e:
        st.error("Failed to load data.")
        st.exception(e)
        st.stop()

    if df.empty:
        st.error("Dataset loaded but returned 0 rows.")
        st.stop()

    # Layout: main + right filter panel
    main_col, filter_col = st.columns([3.2, 1.2], gap="large")

    with filter_col:
        st.markdown("### Filters")

        # Options
        clients = ["All Clients"] + safe_list(df["client"])
        states = ["All States"] + safe_list(df["state"])
        years = ["All Years"] + [str(int(y)) for y in safe_list(df["accident_year"]) if pd.notna(y)]
        covs = ["All Coverage"] + safe_list(df["coverage_type"])
        adjs = ["All Adjusters"] + safe_list(df["adjuster"])
        lobs = ["All Lines"] + safe_list(df["line_of_business"])
        statuses = ["All Statuses"] + safe_list(df["feature_status"])
        causes = ["All Causes"] + safe_list(df["cause_of_loss"])
        vendors = ["All Vendors"] + safe_list(df["vendor_name"])
        firms = ["All Firms"] + safe_list(df["defense_firm"])

        st.selectbox("Client", clients, key="f_client")
        st.selectbox("State", states, key="f_state")
        st.selectbox("Accident Year", years, key="f_acc_year")
        st.selectbox("Coverage Type", covs, key="f_coverage")
        st.selectbox("Adjuster", adjs, key="f_adjuster")
        st.selectbox("Line of Business", lobs, key="f_lob")
        st.selectbox("Feature Status", statuses, key="f_status")
        st.selectbox("Cause of Loss", causes, key="f_cause")
        st.selectbox("Litigation", ["All", "Litigated", "Not Litigated"], key="f_litigated")
        st.selectbox("Vendor", vendors, key="f_vendor")
        st.selectbox("Defense Firm", firms, key="f_defense")

        st.checkbox("Open inventory only", key="f_open_only")
        st.number_input(
            "High severity threshold",
            min_value=50_000,
            max_value=2_000_000,
            step=25_000,
            value=int(st.session_state["f_sev_thresh"]),
            key="f_sev_thresh",
        )

        st.divider()
        st.markdown("### Rules (demo controls)")
        st.checkbox("Strict: non-open statuses have $0 incurred", key="f_strict_incurred")
        st.checkbox("Demo: synthesize multi-month trends if needed", key="f_demo_synth_trends")

        st.button("Reset Filters", on_click=reset_filters)
        st.caption(f"Data source: **{src}**")

    with main_col:
        st.title("Claims Intelligence – Daily Summary")
        st.caption("Demo environment. Data generated for presentation purposes.")

        sev_thresh = float(st.session_state["f_sev_thresh"])

        # Apply financial rules early so EVERYTHING uses consistent definitions
        df_std = standardize_financials(df, strict_incurred_open_only=bool(st.session_state["f_strict_incurred"]))

        # If trend history is thin, synthesize (demo only)
        trend_months = df_std.dropna(subset=["trend_month"])["trend_month"].nunique()
        if bool(st.session_state["f_demo_synth_trends"]) and trend_months < 2:
            df_std = synthesize_monthly_history(df_std, months_back=9, seed=7)

        # Now filter
        dff = apply_filters(df_std)

        # As-of
        as_of = "Latest"
        if dff["report_date"].notna().any():
            as_of = str(dff["report_date"].max().date())
        st.markdown(f"**As of:** {as_of}")

        st.divider()
        render_kpi_row(dff, sev_thresh)

        st.divider()
        st.markdown("### Ask NARS (Prototype)")
        st.caption("Deterministic responses computed from the filtered dataset.")

        # Quick buttons
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Top 10 Severe"):
            st.session_state["_ask_answer"] = "Use the Top 10 Severe table below (already sorted)."
        if b2.button("Open Features"):
            st.session_state["_ask_answer"] = f"Open inventory features: {fmt_int((dff['is_open_inventory'] == 1).sum())}."
        if b3.button("Total Incurred"):
            st.session_state["_ask_answer"] = f"Total incurred: {fmt_currency(dff['incurred_amount'].sum())}."
        if b4.button("State w/ Highest Incurred"):
            if dff["state"].isna().all():
                st.session_state["_ask_answer"] = "No state data in this selection."
            else:
                by_state = dff.groupby("state")["incurred_amount"].sum().sort_values(ascending=False)
                st.session_state["_ask_answer"] = f"State with highest incurred: {by_state.index[0]} ({fmt_currency(by_state.iloc[0])})."

        qcols = st.columns([5, 1])
        q = qcols[0].text_input("Ask a question...", label_visibility="collapsed")
        if qcols[1].button("Ask"):
            st.session_state["_ask_answer"] = answer_question(dff, q, sev_thresh)

        if st.session_state.get("_ask_answer"):
            st.write(st.session_state["_ask_answer"])

        # Always show Top 10 Severe table (matches the “executive needs” vibe)
        render_top10_severe(dff)

        st.divider()
        render_headlines(dff, sev_thresh)

        st.divider()
        render_trend_section(dff, sev_thresh)

        st.divider()
        render_mix_distribution(dff)

        st.divider()
        render_geo(dff)

        st.divider()
        render_rolodex(dff, sev_thresh)

        st.divider()
        render_tables(dff, sev_thresh)


if __name__ == "__main__":
    main()
