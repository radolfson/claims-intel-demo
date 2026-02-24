import io
import os
import uuid
from datetime import datetime, date

import pandas as pd
import streamlit as st


APP_TITLE = "Ad Hoc Data Run — Random Loss Run"


# -----------------------------
# Helpers
# -----------------------------
def _safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _fmt_money(x: float) -> str:
    try:
        return "${:,.0f}".format(float(x))
    except Exception:
        return ""


def load_data() -> tuple[pd.DataFrame, str]:
    """
    Priority:
      1) User-uploaded CSV
      2) Local demo CSV in repo (demo_features_latest.csv)
    """
    uploaded = st.session_state.get("_uploaded_file")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        return df, "uploaded_csv"

    local_path = "demo_features_latest.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        return df, "demo_features_latest.csv"

    return pd.DataFrame(), "none"


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()

    # Common date columns (best-effort)
    for c in ["report_date", "trend_month", "feature_created_date", "loss_date", "accident_date"]:
        if c in dff.columns:
            dff[c] = _safe_dt(dff[c])

    # Common numeric columns (best-effort)
    for c in ["incurred_amount", "paid_amount", "outstanding_amount", "reserve_amount", "total_incurred"]:
        if c in dff.columns:
            dff[c] = pd.to_numeric(dff[c], errors="coerce")

    # Normalize some common categorical columns if present
    for c in ["state", "coverage_type", "feature_status", "line_of_business", "cause_of_loss", "adjuster",
              "vendor_name", "defense_firm", "denial_reason"]:
        if c in dff.columns:
            dff[c] = dff[c].astype(str).replace({"nan": None, "None": None})

    return dff


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    dff = df.copy()

    def _apply(col, val):
        nonlocal dff
        if col in dff.columns and val not in (None, "All", "All States", "All Years", "All Coverages", "All Lines",
                                             "All Adjusters", "All Statuses", "All Causes", "All Vendors", "All Firms",
                                             "All Reasons"):
            dff = dff[dff[col] == val]

    _apply("state", f["state"])
    _apply("accident_year", f["accident_year"])
    _apply("coverage_type", f["coverage_type"])
    _apply("line_of_business", f["line_of_business"])
    _apply("adjuster", f["adjuster"])
    _apply("feature_status", f["feature_status"])
    _apply("cause_of_loss", f["cause_of_loss"])
    _apply("vendor_name", f["vendor_name"])
    _apply("defense_firm", f["defense_firm"])
    _apply("denial_reason", f["denial_reason"])

    # Litigation (best-effort)
    if f["litigation"] != "All":
        lit_col = None
        for cand in ["litigation", "is_litigated", "litigated_flag"]:
            if cand in dff.columns:
                lit_col = cand
                break
        if lit_col is not None:
            if f["litigation"] == "Litigated":
                dff = dff[dff[lit_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "litigated"])]
            elif f["litigation"] == "Not Litigated":
                dff = dff[~dff[lit_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "litigated"])]

    # Severity threshold (uses incurred_amount if present, else total_incurred)
    sev_col = "incurred_amount" if "incurred_amount" in dff.columns else ("total_incurred" if "total_incurred" in dff.columns else None)
    if sev_col is not None:
        dff = dff[pd.to_numeric(dff[sev_col], errors="coerce").fillna(0) >= f["sev_thresh"]]

    return dff


def pick_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "feature_key",
        "claim_number",
        "state",
        "accident_year",
        "coverage_type",
        "feature_status",
        "incurred_amount",
        "paid_amount",
        "outstanding_amount",
        "adjuster",
        "defense_firm",
        "vendor_name",
        "denial_reason",
        "cause_of_loss",
        "trend_month",
        "report_date",
        "feature_created_date",
    ]
    cols = [c for c in preferred if c in df.columns]
    # Add a couple fallbacks if nothing matches
    if not cols:
        cols = df.columns.tolist()[:12]
    return cols


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Run a filtered, exportable loss run on demand. Because humans love buttons.")

    # Upload control
    with st.expander("Data Source", expanded=False):
        up = st.file_uploader("Upload a CSV (optional)", type=["csv"])
        if up is not None:
            st.session_state["_uploaded_file"] = up
            st.success("Using uploaded CSV for this session.")
        st.markdown("- If no upload is provided, the app will look for `demo_features_latest.csv` in the app folder.")

    df, src = load_data()
    if df.empty:
        st.error("No data loaded. Upload a CSV or add `demo_features_latest.csv` next to this app.")
        st.stop()

    df = standardize(df)

    # Sidebar filters
    st.sidebar.header("Filters")

    def _vals(col, label_all):
        if col not in df.columns:
            return [label_all]
        vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) not in ("nan", "None")])
        return [label_all] + vals

    f = {
        "state": st.sidebar.selectbox("State", _vals("state", "All States")),
        "accident_year": st.sidebar.selectbox("Accident Year", _vals("accident_year", "All Years")),
        "coverage_type": st.sidebar.selectbox("Coverage", _vals("coverage_type", "All Coverages")),
        "line_of_business": st.sidebar.selectbox("Line of Business", _vals("line_of_business", "All Lines")),
        "adjuster": st.sidebar.selectbox("Adjuster", _vals("adjuster", "All Adjusters")),
        "feature_status": st.sidebar.selectbox("Feature Status", _vals("feature_status", "All Statuses")),
        "cause_of_loss": st.sidebar.selectbox("Cause of Loss", _vals("cause_of_loss", "All Causes")),
        "litigation": st.sidebar.selectbox("Litigation", ["All", "Litigated", "Not Litigated"]),
        "vendor_name": st.sidebar.selectbox("Vendor", _vals("vendor_name", "All Vendors")),
        "defense_firm": st.sidebar.selectbox("Defense Firm", _vals("defense_firm", "All Firms")),
        "denial_reason": st.sidebar.selectbox("Denial Reason", _vals("denial_reason", "All Reasons")),
        "sev_thresh": st.sidebar.number_input("Severity threshold", min_value=0, max_value=2_000_000, step=25_000, value=250_000),
    }

    st.sidebar.divider()
    n_rows = st.sidebar.selectbox("Rows to return", [25, 50, 100, 250], index=1)
    seed_mode = st.sidebar.selectbox("Randomization", ["Daily (stable)", "New each run"], index=0)

    # Apply filters
    base = apply_filters(df, f)

    # Summary ribbon
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data source", src)
    c2.metric("Rows after filters", f"{len(base):,}")
    sev_col = "incurred_amount" if "incurred_amount" in base.columns else ("total_incurred" if "total_incurred" in base.columns else None)
    if sev_col is not None and not base.empty:
        c3.metric("Avg severity", _fmt_money(base[sev_col].fillna(0).mean()))
        c4.metric("Max severity", _fmt_money(base[sev_col].fillna(0).max()))
    else:
        c3.metric("Avg severity", "—")
        c4.metric("Max severity", "—")

    st.divider()

    # Run button
    run_col1, run_col2, run_col3 = st.columns([1, 2, 2])
    with run_col1:
        run = st.button("Run Loss Run", type="primary", use_container_width=True)

    # Persist last run results
    if run:
        if base.empty:
            st.warning("No rows match the current filters.")
        else:
            # Deterministic seed (daily) vs true random
            if seed_mode == "Daily (stable)":
                seed = int(date.today().strftime("%Y%m%d"))
            else:
                seed = int(uuid.uuid4().int % (2**32 - 1))

            out = base.sample(n=min(n_rows, len(base)), random_state=seed).copy()
            out_cols = pick_columns(out)
            out = out[out_cols]

            run_id = f"LR-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            st.session_state["_last_run"] = {
                "run_id": run_id,
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "rows_requested": int(n_rows),
                "rows_returned": int(len(out)),
                "seed_mode": seed_mode,
                "seed": int(seed),
                "filters": f,
                "data": out,
            }

    last = st.session_state.get("_last_run")
    if last:
        st.subheader("Run Results")
        st.caption(
            f"Run ID: **{last['run_id']}**  •  Created (UTC): **{last['created_utc']}**  •  "
            f"Returned **{last['rows_returned']}** row(s) (requested {last['rows_requested']})"
        )

        # Show filters used (collapsed)
        with st.expander("Selection Criteria (filters used)", expanded=False):
            st.json(last["filters"])

        out_df = last["data"]

        st.dataframe(out_df, use_container_width=True, height=520)

        csv_bytes = df_to_csv_bytes(out_df)
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"{last['run_id']}_loss_run.csv",
            mime="text/csv",
            use_container_width=False,
        )

        st.caption("Tip: switch 'Randomization' to 'New each run' and hit Run again to re-roll.")

    else:
        st.info("Set filters (optional) and click **Run Loss Run** to generate an exportable sample.")


if __name__ == "__main__":
    main()
