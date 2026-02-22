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


# Default behavior (no UI checkboxes anymore)
DEFAULT_STRICT_INCURRED_OPEN_ONLY = False
DEFAULT_SYNTH_TRENDS_IF_NEEDED = True


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

    # Allow env override locally
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
# Styling (make it feel closer to the “executive BI” aesthetic)
# ============================================================
def apply_css() -> None:
    st.markdown(
        """
        <style>
          /* tighten page */
          .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1400px; }
          h1, h2, h3 { letter-spacing: -0.02em; }
          /* soften default look */
          .stMetric { background: #ffffff; border: 1px solid #eee; border-radius: 10px; padding: 12px; }
          /* nicer section spacing */
          .section-title { margin-top: 1.25rem; margin-bottom: 0.25rem; font-weight: 700; font-size: 1.15rem; }
          /* reduce giant empty space in tables */
          [data-testid="stDataFrame"] { border: 1px solid #eee; border-radius: 10px; }
          /* sidebar-like filter panel */
          .filter-card {
            background: #ffffff;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 12px 12px 4px 12px;
          }
          /* make captions calmer */
          .stCaption { color: #667085; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def enable_altair_theme() -> None:
    # A clean, PowerBI-ish theme: subtle grid, readable labels, consistent typography.
    def _theme():
        return {
            "config": {
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelFont": "Inter, Segoe UI, Arial",
                    "titleFont": "Inter, Segoe UI, Arial",
                    "labelColor": "#344054",
                    "titleColor": "#344054",
                    "gridColor": "#EEF2F6",
                    "tickColor": "#EEF2F6",
                    "domainColor": "#D0D5DD",
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
                "legend": {
                    "labelFont": "Inter, Segoe UI, Arial",
                    "titleFont": "Inter, Segoe UI, Arial",
                    "labelColor": "#344054",
                    "titleColor": "#344054",
                    "labelFontSize": 12,
                    "titleFontSize": 12,
                },
                "title": {
                    "font": "Inter, Segoe UI, Arial",
                    "color": "#101828",
                    "fontSize": 14,
                    "anchor": "start",
                    "offset": 8,
                },
                "mark": {"tooltip": True},
            }
        }

    alt.themes.register("nars_exec", _theme)
    alt.themes.enable("nars_exec")


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
    "legal_incurred_amount",
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

    for col in ("paid_amount", "outstanding_amount", "incurred_amount", "legal_incurred_amount"):
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    if "is_open_inventory" not in d.columns:
        d["is_open_inventory"] = d["feature_status"].astype("string").str.upper().isin(list(STATUS_OPEN_SET)).astype(int)
    else:
        d["is_open_inventory"] = pd.to_numeric(d["is_open_inventory"], errors="coerce").fillna(0).astype(int)

    d["is_litigated"] = pd.to_numeric(d["is_litigated"], errors="coerce").fillna(0).astype(int)
    d["accident_year"] = pd.to_numeric(d["accident_year"], errors="coerce")

    # trend_month
    if d["feature_created_date"].notna().any():
        d["trend_month"] = month_floor(d["feature_created_date"])
    elif d["report_date"].notna().any():
        d["trend_month"] = month_floor(d["report_date"])
    else:
        d["trend_month"] = pd.NaT

    # normalize string cols for filters
    for col in ("client", "state", "coverage_type", "feature_status", "adjuster", "line_of_business", "cause_of_loss", "vendor_name", "defense_firm"):
        d[col] = d[col].astype("string")

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
        raise RuntimeError("Missing snowflake-connector-python in requirements.txt.") from e

    account = st.secrets.get("SNOWFLAKE_ACCOUNT")
    user = st.secrets.get("SNOWFLAKE_USER")
    warehouse = st.secrets.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    database = st.secrets.get("SNOWFLAKE_DATABASE", "NARS")
    schema = st.secrets.get("SNOWFLAKE_SCHEMA", "PROCESSED")

    private_key_pem = str(st.secrets.get("SNOWFLAKE_PRIVATE_KEY", "")).strip()
    password = st.secrets.get("SNOWFLAKE_PASSWORD", None)

    if not account or not user:
        raise RuntimeError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in secrets.")

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
            raise RuntimeError("DATA_SOURCE=snowflake but SF_DETAIL_VIEW not set.")
        return load_from_snowflake(cfg.sf_detail_view), "snowflake"
    return load_from_csv(cfg.data_file), "csv"


# ============================================================
# Business rules enforcement (silent defaults, no UI)
# ============================================================
def standardize_financials(df: pd.DataFrame, strict_incurred_open_only: bool) -> pd.DataFrame:
    """
    Rules:
      - DENIED: paid/outstanding/incurred = 0
      - CLOSED: outstanding = 0; incurred = paid
      - Otherwise: incurred >= paid + outstanding (sanity)
      - Optional strict: if not OPEN/PENDING/REOPEN => all financials 0
    """
    d = df.copy()
    status = d["feature_status"].fillna("UNKNOWN").astype(str).str.upper()

    denied = status.eq("DENIED")
    closed = status.eq("CLOSED")

    d.loc[denied, ["paid_amount", "outstanding_amount", "incurred_amount", "legal_incurred_amount"]] = 0.0
    d.loc[denied, "is_open_inventory"] = 0

    d.loc[closed, "outstanding_amount"] = 0.0
    d.loc[closed, "incurred_amount"] = d.loc[closed, "paid_amount"]
    d.loc[closed, "is_open_inventory"] = 0

    sane = ~(denied | closed)
    recomputed = d.loc[sane, "paid_amount"] + d.loc[sane, "outstanding_amount"]
    d.loc[sane, "incurred_amount"] = d.loc[sane, "incurred_amount"].where(d.loc[sane, "incurred_amount"] >= recomputed, recomputed)

    if strict_incurred_open_only:
        open_mask = status.isin(list(STATUS_OPEN_SET))
        d.loc[~open_mask, ["paid_amount", "outstanding_amount", "incurred_amount", "legal_incurred_amount"]] = 0.0
        d.loc[~open_mask, "is_open_inventory"] = 0

    return d


def synthesize_monthly_history(df: pd.DataFrame, months_back: int = 9, seed: int = 7) -> pd.DataFrame:
    """
    Demo-only: if only one month exists, fabricate a back-history so MoM charts render.
    """
    import numpy as np

    d = df.dropna(subset=["trend_month"]).copy()
    if d.empty:
        return df

    cur_month = d["trend_month"].max()
    months = pd.date_range(end=cur_month, periods=months_back + 1, freq="MS")

    rng = np.random.default_rng(seed)
    frames = []

    for m in months:
        f = d.copy()
        f["trend_month"] = m

        drift = 1.0 + rng.normal(0.0, 0.06, size=len(f))
        drift = np.clip(drift, 0.80, 1.25)

        f["paid_amount"] = (f["paid_amount"] * drift).round(2)
        f["outstanding_amount"] = (f["outstanding_amount"] * (1.0 + rng.normal(0.0, 0.05, size=len(f)))).clip(lower=0).round(2)
        f["incurred_amount"] = (f["paid_amount"] + f["outstanding_amount"]).round(2)
        f["legal_incurred_amount"] = (f["legal_incurred_amount"] * (1.0 + rng.normal(0.0, 0.06, size=len(f)))).clip(lower=0).round(2)

        frames.append(f)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Filter state
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
# KPI + rollups
# ============================================================
def calc_kpis(dff: pd.DataFrame, sev_thresh: float) -> dict:
    open_features = int((dff["is_open_inventory"] == 1).sum())
    total_features = int(len(dff))
    total_incurred = float(dff["incurred_amount"].sum())
    paid = float(dff["paid_amount"].sum())
    outstanding = float(dff["outstanding_amount"].sum())
    high_sev = int((dff["incurred_amount"] >= sev_thresh).sum())
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
            legal_incurred=("legal_incurred_amount", "sum"),
            litigated=("is_litigated", "sum"),
        )
        .sort_values("trend_month")
    )

    # rolling avg for volume
    roll["features_roll3"] = roll["total_features"].rolling(3, min_periods=1).mean()
    return roll


def last_two(roll: pd.DataFrame, col: str) -> Tuple[Optional[float], Optional[float]]:
    if roll.empty or col not in roll.columns:
        return None, None
    if len(roll) == 1:
        return float(roll.iloc[-1][col]), None
    return float(roll.iloc[-1][col]), float(roll.iloc[-2][col])


# ============================================================
# Charts (style closer to BI exec dashboards)
# ============================================================
def chart_line(df: pd.DataFrame, x: str, y: str, title: str, y_format: Optional[str] = None, y_title: str = "") -> alt.Chart:
    enc_y = alt.Y(y, title=y_title or None)
    if y_format:
        enc_y = alt.Y(y, title=y_title or None, axis=alt.Axis(format=y_format))
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, title=None),
            y=enc_y,
            tooltip=[x, y],
        )
        .properties(title=title, height=220)
    )


def chart_bar(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False) -> alt.Chart:
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


def chart_stacked_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, title=None),
            y=alt.Y(y, title=None),
            color=alt.Color(color, title=None),
            tooltip=[x, color, y],
        )
        .properties(title=title, height=260)
    )


def chart_donut(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=65)
        .encode(
            theta=alt.Theta(field=value, type="quantitative"),
            color=alt.Color(field=category, type="nominal", title=None),
            tooltip=[category, value],
        )
        .properties(title=title, height=260)
    )


# ============================================================
# Sections
# ============================================================
def render_header(as_of: str) -> None:
    # Logo + Title block like the ask: logo left, as tall as header area above KPI ribbon
    c_logo, c_text = st.columns([0.18, 0.82], vertical_alignment="center")
    with c_logo:
        # If the file is missing, don’t crash the demo
        try:
            st.image("narslogo.jpg", use_container_width=True)
        except Exception:
            st.write("")  # silent
    with c_text:
        st.markdown("## Claims Intelligence – Daily Summary")
        st.caption("Demo environment. Data generated for presentation purposes.")
        st.markdown(f"**As of:** {as_of}")


def render_kpis(dff: pd.DataFrame, sev_thresh: float) -> None:
    k = calc_kpis(dff, sev_thresh)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Features", fmt_int(k["open_features"]))
    c2.metric("Total Incurred", fmt_currency(k["total_incurred"]))
    c3.metric("Paid", fmt_currency(k["paid"]))
    c4.metric("Outstanding", fmt_currency(k["outstanding"]))
    c5.metric("High Severity Features", fmt_int(k["high_sev"]))


def render_ask_nars(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown('<div class="section-title">Ask NARS (Prototype)</div>', unsafe_allow_html=True)
    st.caption("Deterministic responses computed from the filtered dataset.")

    def answer(q: str) -> str:
        ql = (q or "").lower().strip()
        if not ql:
            return "Try: 'top 10 severe', 'open features', 'state with highest incurred', 'paid', 'outstanding'."

        if "top" in ql and ("severe" in ql or "severity" in ql):
            top = dff.sort_values("incurred_amount", ascending=False).head(10)
            if top.empty:
                return "No features in the current selection."
            lines = []
            for _, r in top.iterrows():
                lines.append(f"- {r.get('feature_key','(feature)')} | {r.get('state','')} | {int(r.get('accident_year',0) or 0)} | {fmt_currency(r.get('incurred_amount',0))}")
            return "Top 10 by incurred:\n" + "\n".join(lines)

        if "open" in ql and "features" in ql:
            return f"Open inventory features: {fmt_int((dff['is_open_inventory'] == 1).sum())}."

        if "highest" in ql and "incurred" in ql and "state" in ql:
            if dff["state"].isna().all():
                return "No state data in this selection."
            by_state = dff.groupby("state")["incurred_amount"].sum().sort_values(ascending=False)
            return f"State with highest incurred: {by_state.index[0]} ({fmt_currency(by_state.iloc[0])})."

        if "paid" in ql:
            return f"Total paid: {fmt_currency(dff['paid_amount'].sum())}."

        if "outstanding" in ql or "reserve" in ql:
            return f"Total outstanding: {fmt_currency(dff['outstanding_amount'].sum())}."

        if "high severity" in ql or "threshold" in ql or ">=" in ql:
            hs = int((dff["incurred_amount"] >= sev_thresh).sum())
            return f"High severity features (≥ {fmt_currency(sev_thresh)}): {hs:,}."

        return "Try: 'top 10 severe', 'open features', 'state with highest incurred', 'paid', or 'outstanding'."

    qcols = st.columns([5, 1])
    q = qcols[0].text_input("Ask a question...", label_visibility="collapsed")
    if qcols[1].button("Ask"):
        st.session_state["_ask_answer"] = answer(q)

    if st.session_state.get("_ask_answer"):
        st.write(st.session_state["_ask_answer"])


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

    state_msg = "Geographic concentration not available for current filters."
    if total_ct and dff["state"].notna().any():
        stc = dff.groupby("state").size().sort_values(ascending=False)
        top_states = stc.index[:2].tolist()
        if len(top_states) >= 2:
            st_share = stc.iloc[:2].sum() / total_ct * 100
            state_msg = f"{top_states[0]} and {top_states[1]} account for ~{st_share:.1f}% of feature count."
        elif len(top_states) == 1:
            state_msg = f"{top_states[0]} accounts for 100% of feature count (filtered selection)."

    st.markdown('<div class="section-title">Today’s Headlines</div>', unsafe_allow_html=True)
    st.info(year_msg)
    st.success(f"High severity features represent {hs_pct:.1f}% of open inventory ({hs_ct:,} features at ≥ {fmt_currency(sev_thresh)}).")
    st.warning(f"Total incurred stands at {fmt_currency(total_incurred)} with {fmt_currency(total_out)} outstanding ({out_pct:.1f}% case reserves).")
    st.info(state_msg)


def render_powerbi_style_charts(dff: pd.DataFrame, sev_thresh: float) -> None:
    """
    This section mirrors the chart *types* from the old BRD / PowerBI screenshot:
      - Loss Frequency by Year & LOB (stacked bar)
      - Feature Creation Trends (line + rolling average)
      - Claim Status Summary by Year (clustered bars + closing ratio line)
      - Open Financials by Year (paid/reserve stacked)
      - Litigation overview (donut + legal trend) if data exists
      - Adjuster open counts (horizontal bar)
    """
    st.markdown('<div class="section-title">Trends & Performance</div>', unsafe_allow_html=True)

    # --- Loss Frequency by Year & LOB (stacked bar)
    if dff["accident_year"].notna().any() and dff["line_of_business"].notna().any():
        tmp = (
            dff.groupby(["accident_year", "line_of_business"])
            .size()
            .reset_index(name="feature_count")
            .dropna(subset=["accident_year"])
        )
        tmp["accident_year"] = tmp["accident_year"].astype(int).astype(str)
        st.altair_chart(
            chart_stacked_bar(tmp, "accident_year:N", "feature_count:Q", "line_of_business:N", "Loss Frequency by Year & Line of Business"),
            use_container_width=True,
        )

    # --- Feature Creation Trends (line + rolling avg)
    roll = monthly_rollup(dff, sev_thresh)
    if not roll.empty and roll["trend_month"].nunique() >= 2:
        c1, c2 = st.columns([2.0, 1.0], gap="large")
        with c1:
            base = alt.Chart(roll.tail(12)).encode(x=alt.X("trend_month:T", title=None))
            line_total = base.mark_line(point=True).encode(
                y=alt.Y("total_features:Q", title="Features"),
                tooltip=["trend_month:T", "total_features:Q", "features_roll3:Q"],
            )
            line_avg = base.mark_line(strokeDash=[6, 4]).encode(y="features_roll3:Q")
            st.altair_chart((line_total + line_avg).properties(title="Feature Creation Trends (Monthly + Rolling Avg)", height=240), use_container_width=True)
        with c2:
            st.caption("Latest 12 months (table)")
            tbl = roll.tail(12)[["trend_month", "total_features", "open_features", "high_sev"]].copy()
            tbl["trend_month"] = pd.to_datetime(tbl["trend_month"]).dt.strftime("%Y-%m")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    # --- Claim Status Summary by Year (bars + closing ratio line)
    if dff["accident_year"].notna().any() and dff["feature_status"].notna().any():
        by_y_s = dff.copy()
        by_y_s["yr"] = by_y_s["accident_year"].fillna(0).astype(int).astype(str)

        open_ct = by_y_s[by_y_s["feature_status"].str.upper().isin(list(STATUS_OPEN_SET))].groupby("yr").size().rename("open")
        closed_ct = by_y_s[by_y_s["feature_status"].str.upper().eq("CLOSED")].groupby("yr").size().rename("closed")
        pending_ct = by_y_s[by_y_s["feature_status"].str.upper().eq("PENDING")].groupby("yr").size().rename("pending")

        yr = pd.DataFrame(index=sorted(by_y_s["yr"].unique()))
        yr = yr.join(open_ct, how="left").join(closed_ct, how="left").join(pending_ct, how="left").fillna(0)
        yr = yr.reset_index().rename(columns={"index": "yr"})

        yr["closing_ratio"] = yr["closed"] / (yr["open"] + yr["closed"]).replace(0, pd.NA)

        bars = yr.melt(id_vars=["yr"], value_vars=["open", "closed", "pending"], var_name="status", value_name="count")
        bar = alt.Chart(bars).mark_bar().encode(
            x=alt.X("yr:N", title=None),
            y=alt.Y("count:Q", title="Features"),
            color=alt.Color("status:N", title=None),
            tooltip=["yr:N", "status:N", "count:Q"],
        )

        line = alt.Chart(yr).mark_line(point=True).encode(
            x=alt.X("yr:N", title=None),
            y=alt.Y("closing_ratio:Q", axis=alt.Axis(format="%"), title="Closing Ratio"),
            tooltip=["yr:N", alt.Tooltip("closing_ratio:Q", format=".0%")],
        )

        st.altair_chart(
            alt.layer(bar, line).resolve_scale(y="independent").properties(title="Feature Status Summary (Counts + Closing Ratio)", height=260),
            use_container_width=True,
        )

    # --- Open Financials by Year (paid + outstanding stacked)
    if dff["accident_year"].notna().any():
        open_only = dff[dff["is_open_inventory"] == 1].copy()
        if not open_only.empty:
            fin = (
                open_only.groupby("accident_year", dropna=False)[["paid_amount", "outstanding_amount"]]
                .sum()
                .reset_index()
                .dropna(subset=["accident_year"])
            )
            fin["accident_year"] = fin["accident_year"].astype(int).astype(str)
            fin_m = fin.melt(id_vars=["accident_year"], value_vars=["paid_amount", "outstanding_amount"], var_name="component", value_name="amount")
            st.altair_chart(
                chart_stacked_bar(fin_m, "accident_year:N", "amount:Q", "component:N", "Open Financials by Year (Paid + Outstanding)"),
                use_container_width=True,
            )

    # --- Litigation Overview (if present)
    if dff["is_litigated"].notna().any():
        st.markdown('<div class="section-title">Litigation Overview</div>', unsafe_allow_html=True)
        a, b = st.columns([1.0, 2.0], gap="large")

        with a:
            lit = pd.DataFrame(
                {
                    "litigation": ["Litigated", "Not Litigated"],
                    "count": [int((dff["is_litigated"] == 1).sum()), int((dff["is_litigated"] == 0).sum())],
                }
            )
            st.altair_chart(chart_donut(lit, "litigation", "count", "Litigated Mix"), use_container_width=True)

        with b:
            roll2 = monthly_rollup(dff, sev_thresh)
            if not roll2.empty and roll2["trend_month"].nunique() >= 2:
                st.altair_chart(
                    chart_line(roll2.tail(12), "trend_month:T", "legal_incurred:Q", "Legal Incurred Trend (Monthly)", y_format=",.0f"),
                    use_container_width=True,
                )
            else:
                st.caption("Not enough history for legal trend.")

    # --- Adjuster Open Claim Counts (horizontal bar)
    if dff["adjuster"].notna().any():
        tmp = dff[dff["is_open_inventory"] == 1].groupby("adjuster").size().sort_values(ascending=False).head(12).reset_index(name="open_features")
        if not tmp.empty:
            st.altair_chart(chart_bar(tmp, "adjuster:N", "open_features:Q", "Adjuster Open Claim Counts", horizontal=True), use_container_width=True)


def render_mix_geo_rolodex_tables(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown('<div class="section-title">Mix & Distribution</div>', unsafe_allow_html=True)
    left, mid, right = st.columns([1.2, 1.0, 1.2], gap="large")

    with left:
        status_counts = dff["feature_status"].fillna("UNKNOWN").value_counts().reset_index()
        status_counts.columns = ["feature_status", "count"]
        st.altair_chart(chart_donut(status_counts, "feature_status", "count", "Feature Status Mix"), use_container_width=True)

    with mid:
        cov = dff["coverage_type"].fillna("UNKNOWN").value_counts().head(10).reset_index()
        cov.columns = ["coverage_type", "count"]
        st.altair_chart(chart_bar(cov, "coverage_type:N", "count:Q", "Top Coverage Types"), use_container_width=True)

    with right:
        bins = [0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 10_000_000_000]
        labels = ["$0–50K", "$50–100K", "$100–250K", "$250–500K", "$500K–1M", "$1M+"]
        tmp = dff.copy()
        tmp["sev_bucket"] = pd.cut(tmp["incurred_amount"], bins=bins, labels=labels, include_lowest=True)
        hist = tmp["sev_bucket"].value_counts().reindex(labels, fill_value=0).reset_index()
        hist.columns = ["sev_bucket", "count"]
        st.altair_chart(chart_bar(hist, "sev_bucket:N", "count:Q", "Severity Distribution"), use_container_width=True)

    st.markdown('<div class="section-title">Geographic Concentration</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")

    with a:
        tmp2 = dff.groupby("state", dropna=False)["is_open_inventory"].sum().sort_values(ascending=False).head(12).reset_index()
        tmp2.columns = ["state", "open_features"]
        st.altair_chart(chart_bar(tmp2, "state:N", "open_features:Q", "Top States by Open Features", horizontal=True), use_container_width=True)

    with b:
        tmp3 = dff.groupby("state", dropna=False)["incurred_amount"].sum().sort_values(ascending=False).head(12).reset_index()
        tmp3.columns = ["state", "total_incurred"]
        st.altair_chart(chart_bar(tmp3, "state:N", "total_incurred:Q", "Top States by Total Incurred", horizontal=True), use_container_width=True)

    st.markdown('<div class="section-title">Metric Rolodex (Accident Year)</div>', unsafe_allow_html=True)
    metric = st.selectbox(
        "Metric",
        ["Open Features", "Total Features", "Total Incurred", "Paid", "Outstanding", "High Severity Features"],
        index=0,
        label_visibility="collapsed",
    )

    if dff["accident_year"].isna().all():
        st.caption("No accident year data.")
    else:
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

        chart_df = chart_df.dropna(subset=["accident_year"]).sort_values("accident_year")
        chart_df["accident_year"] = chart_df["accident_year"].astype(int).astype(str)

        st.altair_chart(
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("accident_year:N", title=None),
                y=alt.Y("value:Q", title=None),
                tooltip=["accident_year:N", "value:Q"],
            )
            .properties(height=240),
            use_container_width=True,
        )

    st.markdown('<div class="section-title">Tables</div>', unsafe_allow_html=True)
    left, right = st.columns([1.0, 1.6], gap="large")

    with left:
        st.markdown("#### Feature Status Summary")
        total = len(dff)
        open_ct = int(dff["feature_status"].str.upper().isin(list(STATUS_OPEN_SET)).sum())
        closed_ct = int((dff["feature_status"].str.upper() == "CLOSED").sum())
        pending_ct = int((dff["feature_status"].str.upper() == "PENDING").sum())
        denied_ct = int((dff["feature_status"].str.upper() == "DENIED").sum())

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


# ============================================================
# Main
# ============================================================
def main() -> None:
    st.set_page_config(page_title="Claims Intelligence – Daily Summary", layout="wide")
    apply_css()
    enable_altair_theme()

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

    # Enforce consistent demo definitions (silently)
    df_std = standardize_financials(df, strict_incurred_open_only=DEFAULT_STRICT_INCURRED_OPEN_ONLY)

    # Ensure MoM charts always render for demo if needed
    if DEFAULT_SYNTH_TRENDS_IF_NEEDED:
        months = df_std.dropna(subset=["trend_month"])["trend_month"].nunique()
        if months < 2:
            df_std = synthesize_monthly_history(df_std, months_back=9, seed=7)

    sev_thresh = float(st.session_state["f_sev_thresh"])

    # Layout like your screenshot: main content + right filter panel
    main_col, filter_col = st.columns([3.2, 1.2], gap="large")

    with filter_col:
        st.markdown("### Filters")
        st.markdown('<div class="filter-card">', unsafe_allow_html=True)

        clients = ["All Clients"] + safe_list(df_std["client"])
        states = ["All States"] + safe_list(df_std["state"])
        years = ["All Years"] + [str(int(y)) for y in safe_list(df_std["accident_year"]) if pd.notna(y)]
        covs = ["All Coverage"] + safe_list(df_std["coverage_type"])
        adjs = ["All Adjusters"] + safe_list(df_std["adjuster"])
        lobs = ["All Lines"] + safe_list(df_std["line_of_business"])
        statuses = ["All Statuses"] + safe_list(df_std["feature_status"])
        causes = ["All Causes"] + safe_list(df_std["cause_of_loss"])
        vendors = ["All Vendors"] + safe_list(df_std["vendor_name"])
        firms = ["All Firms"] + safe_list(df_std["defense_firm"])

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

        st.button("Reset Filters", on_click=reset_filters)
        st.caption(f"Data source: **{src}**")

        st.markdown("</div>", unsafe_allow_html=True)

    with main_col:
        dff = apply_filters(df_std)

        as_of = "Latest"
        if dff["report_date"].notna().any():
            as_of = str(dff["report_date"].max().date())

        render_header(as_of)

        st.divider()
        render_kpis(dff, sev_thresh)

        st.divider()
        render_ask_nars(dff, sev_thresh)

        st.divider()
        render_headlines(dff, sev_thresh)

        st.divider()
        render_powerbi_style_charts(dff, sev_thresh)

        st.divider()
        render_mix_geo_rolodex_tables(dff, sev_thresh)


if __name__ == "__main__":
    main()
