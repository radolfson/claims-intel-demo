# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st


# =========================================
# Config
# =========================================
DATA_FILE = "demo_features_latest.csv"
DEFAULT_SEVERITY_THRESHOLD = 250_000

STATUS_OPEN_SET = {"OPEN", "PENDING", "REOPEN"}


# =========================================
# Data loading
# =========================================
@st.cache_data(show_spinner=False)
def load_features_csv(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Dates
    for col in ("report_date", "feature_created_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Numerics
    for col in ("incurred_amount", "paid_amount", "outstanding_amount", "legal_incurred_amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Flags
    if "is_open_inventory" in df.columns:
        df["is_open_inventory"] = pd.to_numeric(df["is_open_inventory"], errors="coerce").fillna(0).astype(int)
    else:
        # derive if missing
        if "feature_status" in df.columns:
            df["is_open_inventory"] = df["feature_status"].isin(list(STATUS_OPEN_SET)).astype(int)
        else:
            df["is_open_inventory"] = 0

    return df


# =========================================
# Helpers
# =========================================
def fmt_currency(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "—"


def safe_list(series: pd.Series) -> list:
    return sorted([x for x in series.dropna().unique()])


def init_filter_state(df: pd.DataFrame) -> None:
    """
    Initialize session_state defaults once.
    """
    if st.session_state.get("_filters_initialized"):
        return

    st.session_state["_filters_initialized"] = True

    st.session_state["f_client"] = "All Clients"
    st.session_state["f_state"] = "All States"
    st.session_state["f_acc_year"] = "All Years"
    st.session_state["f_coverage"] = "All Coverage"
    st.session_state["f_adjuster"] = "All Adjusters"

    # Expanded filters (BRD-style)
    st.session_state["f_lob"] = "All Lines"
    st.session_state["f_status"] = "All Statuses"
    st.session_state["f_cause"] = "All Causes"
    st.session_state["f_litigated"] = "All"
    st.session_state["f_vendor"] = "All Vendors"
    st.session_state["f_defense"] = "All Firms"

    st.session_state["f_open_only"] = False
    st.session_state["f_sev_thresh"] = DEFAULT_SEVERITY_THRESHOLD


def reset_filters() -> None:
    # Wipe the initialized flag and rebuild defaults
    for k in list(st.session_state.keys()):
        if k.startswith("f_") or k == "_filters_initialized":
            del st.session_state[k]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    # Client
    if st.session_state["f_client"] != "All Clients" and "client" in dff.columns:
        dff = dff[dff["client"] == st.session_state["f_client"]]

    # State
    if st.session_state["f_state"] != "All States" and "state" in dff.columns:
        dff = dff[dff["state"] == st.session_state["f_state"]]

    # Accident year
    if st.session_state["f_acc_year"] != "All Years" and "accident_year" in dff.columns:
        dff = dff[dff["accident_year"] == int(st.session_state["f_acc_year"])]

    # Coverage type
    if st.session_state["f_coverage"] != "All Coverage" and "coverage_type" in dff.columns:
        dff = dff[dff["coverage_type"] == st.session_state["f_coverage"]]

    # Adjuster
    if st.session_state["f_adjuster"] != "All Adjusters" and "adjuster" in dff.columns:
        dff = dff[dff["adjuster"] == st.session_state["f_adjuster"]]

    # LOB
    if st.session_state["f_lob"] != "All Lines" and "line_of_business" in dff.columns:
        dff = dff[dff["line_of_business"] == st.session_state["f_lob"]]

    # Status
    if st.session_state["f_status"] != "All Statuses" and "feature_status" in dff.columns:
        dff = dff[dff["feature_status"] == st.session_state["f_status"]]

    # Cause
    if st.session_state["f_cause"] != "All Causes" and "cause_of_loss" in dff.columns:
        dff = dff[dff["cause_of_loss"] == st.session_state["f_cause"]]

    # Litigated
    if st.session_state["f_litigated"] != "All" and "is_litigated" in dff.columns:
        want = 1 if st.session_state["f_litigated"] == "Litigated" else 0
        dff = dff[dff["is_litigated"] == want]

    # Vendor
    if st.session_state["f_vendor"] != "All Vendors" and "vendor_name" in dff.columns:
        dff = dff[dff["vendor_name"] == st.session_state["f_vendor"]]

    # Defense firm
    if st.session_state["f_defense"] != "All Firms" and "defense_firm" in dff.columns:
        dff = dff[dff["defense_firm"] == st.session_state["f_defense"]]

    # Open only toggle
    if st.session_state["f_open_only"] and "is_open_inventory" in dff.columns:
        dff = dff[dff["is_open_inventory"] == 1]

    return dff


# =========================================
# Deterministic "Ask NARS" prototype
# =========================================
def answer_question(dff: pd.DataFrame, q: str, sev_thresh: float) -> str:
    ql = q.lower().strip()

    if not ql:
        return "Ask a question about the filtered portfolio."

    if "top" in ql and ("severe" in ql or "severity" in ql):
        top = dff.sort_values("incurred_amount", ascending=False).head(10)
        if top.empty:
            return "No features in the current filter selection."
        lines = []
        for _, r in top.iterrows():
            lines.append(f"- {r.get('feature_key','(feature)')} | {r.get('state','')} | {r.get('accident_year','')} | {fmt_currency(r.get('incurred_amount',0))}")
        return "Top 10 by incurred:\n" + "\n".join(lines)

    if "open" in ql and "features" in ql:
        open_ct = int((dff["is_open_inventory"] == 1).sum())
        return f"Open inventory features: {open_ct:,}."

    if "highest" in ql and "incurred" in ql and "state" in ql:
        by_state = dff.groupby("state", dropna=False)["incurred_amount"].sum().sort_values(ascending=False)
        if by_state.empty:
            return "No state data in the current filter selection."
        st_top = by_state.index[0]
        return f"State with highest incurred: {st_top} ({fmt_currency(by_state.iloc[0])})."

    if "paid" in ql:
        return f"Total paid in current selection: {fmt_currency(dff['paid_amount'].sum())}."

    if "outstanding" in ql or "reserve" in ql:
        return f"Total outstanding (case reserves) in current selection: {fmt_currency(dff['outstanding_amount'].sum())}."

    if "high severity" in ql or ">= " in ql or "threshold" in ql:
        hs = int((dff["incurred_amount"] >= sev_thresh).sum())
        return f"High severity features (≥ {fmt_currency(sev_thresh)}): {hs:,}."

    return "Try: 'top 10 severe', 'open features', 'state with highest incurred', 'paid', or 'outstanding'."


# =========================================
# Headline generation
# =========================================
def render_headlines(dff: pd.DataFrame, sev_thresh: float) -> None:
    open_ct = int((dff["is_open_inventory"] == 1).sum())
    total_ct = int(len(dff))
    open_pct = (open_ct / total_ct * 100) if total_ct else 0.0

    hs_ct = int((dff["incurred_amount"] >= sev_thresh).sum())
    hs_pct = (hs_ct / open_ct * 100) if open_ct else 0.0

    total_incurred = float(dff["incurred_amount"].sum())
    total_paid = float(dff["paid_amount"].sum())
    total_out = float(dff["outstanding_amount"].sum())
    out_pct = (total_out / total_incurred * 100) if total_incurred else 0.0

    # Accident year concentration
    if "accident_year" in dff.columns and total_ct:
        ay = dff.groupby("accident_year").size().sort_values(ascending=False)
        top_years = ay.index[:2].tolist()
        top_share = ay.iloc[:2].sum() / total_ct * 100
        year_msg = f"Open features total {open_ct:,}, concentrated in accident years {top_years[0]}–{top_years[1]} (~{top_share:.1f}% of inventory)."
    else:
        year_msg = f"Open features total {open_ct:,} (~{open_pct:.1f}% of filtered inventory)."

    # State concentration
    if "state" in dff.columns and total_ct:
        stc = dff.groupby("state").size().sort_values(ascending=False)
        top_states = stc.index[:2].tolist()
        st_share = stc.iloc[:2].sum() / total_ct * 100
        state_msg = f"{top_states[0]} and {top_states[1]} account for ~{st_share:.1f}% of feature count."
    else:
        state_msg = "Geographic concentration not available for current filters."

    st.markdown("### Today’s Headlines")
    st.info(year_msg)
    st.success(f"High severity features represent {hs_pct:.1f}% of open inventory ({hs_ct:,} features at ≥ {fmt_currency(sev_thresh)}).")
    st.warning(f"Total incurred stands at {fmt_currency(total_incurred)} with {fmt_currency(total_out)} outstanding ({out_pct:.1f}% case reserves).")
    st.info(state_msg)


# =========================================
# UI sections
# =========================================
def render_kpis(dff: pd.DataFrame, sev_thresh: float) -> None:
    open_features = int((dff["is_open_inventory"] == 1).sum())
    total_incurred = float(dff["incurred_amount"].sum())
    paid = float(dff["paid_amount"].sum())
    outstanding = float(dff["outstanding_amount"].sum())
    high_sev = int((dff["incurred_amount"] >= sev_thresh).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open Features", f"{open_features:,}")
    c2.metric("Total Incurred", fmt_currency(total_incurred))
    c3.metric("Paid", fmt_currency(paid))
    c4.metric("Outstanding", fmt_currency(outstanding))
    c5.metric("High Severity Features", f"{high_sev:,}")


def render_metric_rolodex(dff: pd.DataFrame) -> None:
    st.markdown("### Metric Rolodex (Accident Year)")
    if "accident_year" not in dff.columns:
        st.caption("No accident_year available.")
        return
    yr = dff.groupby("accident_year").size().reset_index(name="Feature Count").sort_values("accident_year")
    st.bar_chart(yr.set_index("accident_year"))


def render_status_summary(dff: pd.DataFrame) -> None:
    st.markdown("### Feature Status Summary")
    if "feature_status" not in dff.columns:
        st.caption("No feature_status available.")
        return
    s = dff["feature_status"].value_counts()
    total = int(len(dff))
    open_ct = int(dff["feature_status"].isin(list(STATUS_OPEN_SET)).sum())
    closed_ct = int((dff["feature_status"] == "CLOSED").sum())
    pending_ct = int((dff["feature_status"] == "PENDING").sum())

    st.write(f"**Total Features:** {total:,}")
    st.write(f"**Open Features:** {open_ct:,}")
    st.write(f"**Closed Features:** {closed_ct:,}")
    st.write(f"**Pending Features:** {pending_ct:,}")


def render_incurred_stratification(dff: pd.DataFrame) -> None:
    st.markdown("### Incurred Stratification (YTD)")
    bins = [-1, 50_000, 100_000, 250_000, 500_000, 1_000_000, 10_000_000_000]
    labels = ["$0 – $50K", "$50K – $100K", "$100K – $250K", "$250K – $500K", "$500K – $1M", "$1M+"]

    tmp = dff.copy()
    tmp["bucket"] = pd.cut(tmp["incurred_amount"], bins=bins, labels=labels)
    strat = tmp.groupby("bucket", dropna=False).size().reset_index(name="Feature Count")
    st.dataframe(strat, use_container_width=True, hide_index=True)


def render_high_severity_table(dff: pd.DataFrame, sev_thresh: float) -> None:
    st.markdown("### High Severity Features")
    top = dff[dff["incurred_amount"] >= sev_thresh].copy().sort_values("incurred_amount", ascending=False)

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
    ]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols].head(50), use_container_width=True, hide_index=True)


# =========================================
# Main
# =========================================
def main() -> None:
    st.set_page_config(page_title="Claims Intelligence – Daily Summary", layout="wide")

    df = load_features_csv(DATA_FILE)
    if df.empty:
        st.error(f"{DATA_FILE} loaded but returned 0 rows.")
        st.stop()

    init_filter_state(df)

    # Layout like your screenshot: main content + right filter panel
    main_col, filter_col = st.columns([3.2, 1.2], gap="large")

    with main_col:
        st.title("Claims Intelligence – Daily Summary")
        st.caption("Demo environment. Data generated for presentation purposes.")

        dff = apply_filters(df)
        as_of = max(dff["report_date"]) if "report_date" in dff.columns and dff["report_date"].notna().any() else "Latest"
        st.markdown(f"**As of:** {as_of}")

        sev_thresh = float(st.session_state["f_sev_thresh"])

        st.divider()
        render_kpis(dff, sev_thresh)

        # Ask NARS (prototype)
        st.divider()
        st.markdown("### Ask NARS (Prototype)")
        st.caption("Deterministic responses computed from filtered dataset.")
        qcols = st.columns([5, 1])
        q = qcols[0].text_input("Ask a question (e.g., top 10 severe, open features, state with highest incurred)", label_visibility="collapsed")
        if qcols[1].button("Ask"):
            st.write(answer_question(dff, q, sev_thresh))

        # Headlines
        st.divider()
        render_headlines(dff, sev_thresh)

        # Rolodex + lower panels
        st.divider()
        render_metric_rolodex(dff)

        st.divider()
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            render_status_summary(dff)
            st.divider()
            render_incurred_stratification(dff)
        with right:
            render_high_severity_table(dff, sev_thresh)

    with filter_col:
        st.markdown("### Filters")

        # Build option lists with "All"
        clients = ["All Clients"] + (safe_list(df["client"]) if "client" in df.columns else [])
        states = ["All States"] + (safe_list(df["state"]) if "state" in df.columns else [])
        years = ["All Years"] + ([str(y) for y in safe_list(df["accident_year"])] if "accident_year" in df.columns else [])
        covs = ["All Coverage"] + (safe_list(df["coverage_type"]) if "coverage_type" in df.columns else [])
        adjs = ["All Adjusters"] + (safe_list(df["adjuster"]) if "adjuster" in df.columns else [])

        st.selectbox("Client", clients, key="f_client")
        st.selectbox("State", states, key="f_state")
        st.selectbox("Accident Year", years, key="f_acc_year")
        st.selectbox("Coverage Type", covs, key="f_coverage")
        st.selectbox("Adjuster", adjs, key="f_adjuster")

        st.checkbox("Open inventory only", key="f_open_only")
        st.number_input(
            "High severity threshold",
            min_value=50_000,
            max_value=2_000_000,
            step=25_000,
            value=int(st.session_state["f_sev_thresh"]),
            key="f_sev_thresh",
        )

        with st.expander("More filters"):
            lobs = ["All Lines"] + (safe_list(df["line_of_business"]) if "line_of_business" in df.columns else [])
            statuses = ["All Statuses"] + (safe_list(df["feature_status"]) if "feature_status" in df.columns else [])
            causes = ["All Causes"] + (safe_list(df["cause_of_loss"]) if "cause_of_loss" in df.columns else [])
            vendors = ["All Vendors"] + (safe_list(df["vendor_name"]) if "vendor_name" in df.columns else [])
            firms = ["All Firms"] + (safe_list(df["defense_firm"]) if "defense_firm" in df.columns else [])

            st.selectbox("Line of Business", lobs, key="f_lob")
            st.selectbox("Feature Status", statuses, key="f_status")
            st.selectbox("Cause of Loss", causes, key="f_cause")
            st.selectbox("Litigation", ["All", "Litigated", "Not Litigated"], key="f_litigated")
            st.selectbox("Vendor", vendors, key="f_vendor")
            st.selectbox("Defense Firm", firms, key="f_defense")

        st.button("Reset Filters", on_click=reset_filters)

if __name__ == "__main__":
    main()
