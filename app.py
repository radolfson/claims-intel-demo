# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st


# =========================================
# Config
# =========================================
SEVERITY_THRESHOLD_DEFAULT = 250_000


# =========================================
# Data loading
# =========================================
@st.cache_data(show_spinner=False)
def load_features_csv(path: str = "demo_features_latest.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize date columns if present
    for col in ("report_date", "feature_created_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Normalize numeric columns if present
    for col in ("incurred_amount", "paid_amount", "outstanding_amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Normalize flags if present
    if "is_open_inventory" in df.columns:
        df["is_open_inventory"] = pd.to_numeric(df["is_open_inventory"], errors="coerce").fillna(0).astype(int)

    if "is_high_severity" in df.columns:
        df["is_high_severity"] = pd.to_numeric(df["is_high_severity"], errors="coerce").fillna(0).astype(int)

    return df


# =========================================
# Helpers
# =========================================
def fmt_currency(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def safe_unique_sorted(series: pd.Series) -> list:
    return sorted([x for x in series.dropna().unique()])


# =========================================
# Filters
# =========================================
def sidebar_filters(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    f: dict = {}

    # Client
    if "client" in df.columns:
        clients = safe_unique_sorted(df["client"])
        f["client"] = st.sidebar.selectbox("Client", clients, index=0)

    # State
    if "state" in df.columns:
        states = safe_unique_sorted(df["state"])
        f["state"] = st.sidebar.multiselect("State", states, default=states)

    # Accident Year
    if "accident_year" in df.columns:
        years = safe_unique_sorted(df["accident_year"])
        f["accident_year"] = st.sidebar.multiselect("Accident Year", years, default=years)

    # Coverage Type
    if "coverage_type" in df.columns:
        covs = safe_unique_sorted(df["coverage_type"])
        f["coverage_type"] = st.sidebar.multiselect("Coverage Type", covs, default=covs)

    # Adjuster
    if "adjuster" in df.columns:
        adjs = safe_unique_sorted(df["adjuster"])
        f["adjuster"] = st.sidebar.multiselect("Adjuster", adjs, default=adjs)

    # Status (includes PENDING)
    if "feature_status" in df.columns:
        statuses = safe_unique_sorted(df["feature_status"])
        f["feature_status"] = st.sidebar.multiselect("Feature Status", statuses, default=statuses)

    # Open only toggle (computed from is_open_inventory)
    if "is_open_inventory" in df.columns:
        f["open_only"] = st.sidebar.checkbox("Open inventory only (OPEN/PENDING/REOPEN)", value=False)

    # Severity threshold (so you can demo flexibility)
    f["severity_threshold"] = st.sidebar.number_input(
        "High severity threshold",
        min_value=50_000,
        max_value=2_000_000,
        value=SEVERITY_THRESHOLD_DEFAULT,
        step=25_000,
        format="%d",
    )

    return f


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()

    if "client" in out.columns and f.get("client"):
        out = out[out["client"] == f["client"]]

    if "state" in out.columns and isinstance(f.get("state"), list) and f["state"]:
        out = out[out["state"].isin(f["state"])]

    if "accident_year" in out.columns and isinstance(f.get("accident_year"), list) and f["accident_year"]:
        out = out[out["accident_year"].isin(f["accident_year"])]

    if "coverage_type" in out.columns and isinstance(f.get("coverage_type"), list) and f["coverage_type"]:
        out = out[out["coverage_type"].isin(f["coverage_type"])]

    if "adjuster" in out.columns and isinstance(f.get("adjuster"), list) and f["adjuster"]:
        out = out[out["adjuster"].isin(f["adjuster"])]

    if "feature_status" in out.columns and isinstance(f.get("feature_status"), list) and f["feature_status"]:
        out = out[out["feature_status"].isin(f["feature_status"])]

    if f.get("open_only") and "is_open_inventory" in out.columns:
        out = out[out["is_open_inventory"] == 1]

    return out


# =========================================
# KPI + sections
# =========================================
def render_kpis(dff: pd.DataFrame, severity_threshold: float) -> None:
    total_features = int(len(dff))

    open_features = "—"
    if "is_open_inventory" in dff.columns:
        open_features = f"{int((dff['is_open_inventory'] == 1).sum()):,}"

    total_incurred = fmt_currency(dff["incurred_amount"].sum()) if "incurred_amount" in dff.columns else "—"
    total_paid = fmt_currency(dff["paid_amount"].sum()) if "paid_amount" in dff.columns else "—"
    total_outstanding = fmt_currency(dff["outstanding_amount"].sum()) if "outstanding_amount" in dff.columns else "—"

    high_sev = "—"
    if "incurred_amount" in dff.columns:
        high_sev = f"{int((dff['incurred_amount'] >= float(severity_threshold)).sum()):,}"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Features", open_features)
    c2.metric("Total Incurred", total_incurred)
    c3.metric("Paid", total_paid)
    c4.metric("Outstanding", total_outstanding)
    c5.metric("High Severity Features", high_sev)


def render_accident_year_chart(dff: pd.DataFrame) -> None:
    if "accident_year" not in dff.columns:
        st.caption("No accident_year column found.")
        return

    yr = dff.groupby("accident_year").size().reset_index(name="Feature Count")
    yr = yr.sort_values("accident_year")
    st.bar_chart(yr.set_index("accident_year"))


def render_incurred_stratification(dff: pd.DataFrame) -> None:
    if "incurred_amount" not in dff.columns:
        st.caption("No incurred_amount column found.")
        return

    bins = [-1, 50_000, 100_000, 250_000, 500_000, 1_000_000, 10_000_000_000]
    labels = ["$0–$50K", "$50K–$100K", "$100K–$250K", "$250K–$500K", "$500K–$1M", "$1M+"]

    tmp = dff.copy()
    tmp["incurred_amount"] = pd.to_numeric(tmp["incurred_amount"], errors="coerce").fillna(0.0)
    tmp["bucket"] = pd.cut(tmp["incurred_amount"], bins=bins, labels=labels)

    strat = tmp.groupby("bucket", dropna=False).size().reset_index(name="Feature Count")
    st.dataframe(strat, use_container_width=True, hide_index=True)


def render_high_severity_table(dff: pd.DataFrame, severity_threshold: float) -> None:
    if "incurred_amount" not in dff.columns:
        st.caption("No incurred_amount column found.")
        return

    thresh = float(severity_threshold)
    top = dff[pd.to_numeric(dff["incurred_amount"], errors="coerce").fillna(0.0) >= thresh].copy()
    top = top.sort_values("incurred_amount", ascending=False)

    cols_preferred = [
        "feature_key",
        "claim_number",
        "claimant_id",
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
    cols = [c for c in cols_preferred if c in top.columns]
    st.dataframe(top[cols].head(50), use_container_width=True, hide_index=True)


# =========================================
# Main
# =========================================
def main() -> None:
    st.set_page_config(page_title="Claims Intelligence – Daily Summary", layout="wide")
    st.title("Claims Intelligence – Daily Summary")
    st.caption("Demo environment. Data generated for presentation purposes.")

    df = load_features_csv("demo_features_latest.csv")

    if df.empty:
        st.error("demo_features_latest.csv loaded but returned 0 rows. Check the file in the repo root.")
        st.stop()

    f = sidebar_filters(df)
    dff = apply_filters(df, f)

    # As-of banner (uses report_date)
    as_of = None
    if "report_date" in dff.columns and dff["report_date"].notna().any():
        as_of = max(dff["report_date"])
    st.markdown(f"**As of:** {as_of if as_of else 'Latest Available'}")

    st.divider()

    render_kpis(dff, f["severity_threshold"])

    st.divider()

    st.subheader("Metric Rolodex (Accident Year)")
    render_accident_year_chart(dff)

    st.divider()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Incurred Stratification (YTD)")
        render_incurred_stratification(dff)

    with c2:
        st.subheader("High Severity Features")
        render_high_severity_table(dff, f["severity_threshold"])


if __name__ == "__main__":
    main()