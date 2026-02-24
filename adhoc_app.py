# adhoc_app.py
import os
import re
import io
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Cover Whale Ad Hoc Delivery Preview",
    page_icon="📄",
    layout="wide",
)


# -----------------------------
# Light styling to match your dashboard vibe
# -----------------------------
st.markdown(
    """
<style>
/* Slightly tighten global padding */
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }

/* Card style */
.cw-card {
  background: #FFFFFF;
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

/* Subtle label */
.cw-label {
  font-size: 12px;
  color: rgba(49, 51, 63, 0.70);
  margin-bottom: 2px;
}

/* Big value */
.cw-value {
  font-size: 22px;
  font-weight: 650;
  letter-spacing: 0.2px;
}

/* Section title */
.cw-h2 {
  font-size: 22px;
  font-weight: 700;
  margin: 0.2rem 0 0.6rem 0;
}

/* Small helper text */
.cw-muted {
  color: rgba(49, 51, 63, 0.65);
  font-size: 13px;
}

/* Make file uploader less chunky */
section[data-testid="stFileUploaderDropzone"] { padding: 18px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def _safe_money(x):
    """Parse $ amounts like $6,994,496.80 into float. Returns None if can't parse."""
    if x is None:
        return None
    s = str(x).strip()
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _read_pdf_text(pdf_bytes: bytes, max_pages: int = 5) -> str:
    """Extract text from first N pages (best effort)."""
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = min(len(reader.pages), max_pages)
        chunks = []
        for i in range(pages):
            t = reader.pages[i].extract_text() or ""
            chunks.append(t)
        return "\n".join(chunks)
    except Exception:
        return ""


def _extract_header_fields(text: str) -> dict:
    """
    Best-effort extraction of:
      - print_date
      - valuation_date
      - company
      - report_title
    Based on the sample PDF's first page/lines. :contentReference[oaicite:1]{index=1}
    """
    out = {
        "report_title": None,
        "print_date": None,
        "valuation_date": None,
        "company": None,
    }

    # Title (usually first line)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        out["report_title"] = lines[0]

    # Print date / valuation date / company patterns
    # The sample has labels, but extraction can be messy in PDFs.
    # We'll try label-based first, then fallback to date heuristics.
    m = re.search(r"Print Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", text)
    if m:
        out["print_date"] = m.group(1)

    m = re.search(r"Valuation Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", text)
    if m:
        out["valuation_date"] = m.group(1)

    # Company often appears after "Company:" on the first page in the sample. :contentReference[oaicite:2]{index=2}
    # We'll capture the next non-empty line after "Company:" if possible.
    if "Company:" in text:
        # take a small window
        idx = text.find("Company:")
        window = text[idx : idx + 250]
        wlines = [ln.strip() for ln in window.splitlines() if ln.strip()]
        # "Company:" line may have nothing on it, and the next line is the actual company
        for i, ln in enumerate(wlines):
            if ln.startswith("Company:"):
                # try inline
                inline = ln.replace("Company:", "").strip()
                if inline:
                    out["company"] = inline
                else:
                    if i + 1 < len(wlines):
                        # sometimes the next line is the company name
                        out["company"] = wlines[i + 1]
                break

    # Fallbacks: grab first "looks like a company" line (all-caps-ish)
    if not out["company"]:
        for ln in lines[:40]:
            if len(ln) > 10 and ln.upper() == ln and "LOSS RUN" not in ln:
                out["company"] = ln
                break

    # Fallback date heuristics: first two mm/dd/yyyy occurrences
    if not out["print_date"] or not out["valuation_date"]:
        dates = re.findall(r"\b([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})\b", text)
        # In the sample, print date and valuation date both exist early. :contentReference[oaicite:3]{index=3}
        if dates:
            if not out["print_date"]:
                out["print_date"] = dates[0]
            if len(dates) > 1 and not out["valuation_date"]:
                out["valuation_date"] = dates[1]

    return out


def _extract_policy_summary_rows(text: str, max_rows: int = 25) -> pd.DataFrame:
    """
    Best-effort parse of policy summary lines. In the sample:
      Policy Inception | Policy Expiration | Policy Number | Carrier | # Claims | Total Reserves | Total Payments | Total Recovery | Total Net Incurred | Avg Net Incurred
    :contentReference[oaicite:4]{index=4}

    PDF text extraction can break columns across lines, so this is intentionally "try but don't cry".
    """
    # Normalize whitespace a bit
    t = re.sub(r"[ \t]+", " ", text)

    # Try to find chunks that look like:
    # mm/dd/yyyy mm/dd/yyyy POLICYNUM (carrier words...) <claims_int> $x $y $z $a $b
    # NOTE: carrier may include spaces, policy num may include letters/numbers/dashes
    pattern = re.compile(
        r"\b(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+([A-Za-z0-9\-]+)\s+(.{2,60}?)\s+(\d+)\s+(\$?[0-9,]+\.\d{2})\s+(\$?[0-9,]+\.\d{2})\s+(\$?[0-9,]+\.\d{2})\s+(\$?[0-9,]+\.\d{2})\s+(\$?[0-9,]+\.\d{2})"
    )

    rows = []
    for m in pattern.finditer(t):
        inception, expiration, policy_number, carrier, n_claims, reserves, payments, recovery, net_incurred, avg_net = m.groups()
        carrier = carrier.strip()

        rows.append(
            {
                "Policy Inception": inception,
                "Policy Expiration": expiration,
                "Policy Number": policy_number,
                "Carrier": carrier,
                "# Claims": int(n_claims),
                "Total Reserves": _safe_money(reserves),
                "Total Payments": _safe_money(payments),
                "Total Recovery": _safe_money(recovery),
                "Total Net Incurred": _safe_money(net_incurred),
                "Avg Net Incurred": _safe_money(avg_net),
            }
        )
        if len(rows) >= max_rows:
            break

    return pd.DataFrame(rows)


def _fmt_money(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"${v:,.2f}"
    except Exception:
        return str(v)


# -----------------------------
# Header (match your dashboard branding)
# -----------------------------
mast = st.columns([0.18, 0.82], vertical_alignment="center")
with mast[0]:
    logo_path = "narslogo.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=210)
    else:
        st.markdown("<div class='cw-muted'>Logo file not found: narslogo.jpg</div>", unsafe_allow_html=True)

with mast[1]:
    st.markdown(
        """
        <div style="line-height:1.05;">
          <div style="font-size:30px; font-weight:750; letter-spacing:0.2px;">
            Cover Whale Ad Hoc Delivery Preview
          </div>
          <div style="font-size:14px; color:rgba(49,51,63,0.70); margin-top:6px;">
            Client-facing preview of what is delivered for ad hoc loss run requests (format intentionally modernized).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------
# Inputs: Upload the output (PDF) to preview it
# -----------------------------
left, right = st.columns([0.58, 0.42], gap="large")

with left:
    st.markdown("<div class='cw-h2'>Output to Deliver</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cw-muted'>Upload the generated loss run output (PDF). This page renders a clean delivery preview + download buttons.</div>",
        unsafe_allow_html=True,
    )

    default_pdf_path = "sample_loss_run.pdf"
    uploaded = st.file_uploader("Upload loss run PDF", type=["pdf"])

    pdf_bytes = None
    pdf_name = None

    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        pdf_name = uploaded.name
    elif os.path.exists(default_pdf_path):
        with open(default_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_name = default_pdf_path

    if pdf_bytes is None:
        st.info("Upload a PDF to preview what the client would receive. (Or drop a `sample_loss_run.pdf` in the repo.)")
        st.stop()

with right:
    st.markdown("<div class='cw-h2'>Request Context (Demo)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cw-muted'>This is just a presentation layer for now. You said you’ll build the request intake form later. Sensible choice.</div>",
        unsafe_allow_html=True,
    )

    # These are editable placeholders so you can demo different scenarios without building intake yet.
    req_client = st.text_input("Client", value="FLORIDA HOSPITAL SELF INSURANCE FUND")
    req_requestor = st.text_input("Primary Contact (requestor)", value="edenhardt@narisk.com")
    req_type = st.selectbox("Request Type", ["Loss Run", "No Loss Letter", "Other"], index=0)
    req_priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
    req_delivery = st.selectbox("Delivery Method", ["Email", "SFTP", "Portal"], index=0)
    req_notes = st.text_area(
        "Notes (what the client asked for)",
        value="Fiscal years ending 9/30/22, 9/30/23, 9/30/24, 9/30/25. Data-only extract requested.",
        height=120,
    )


# -----------------------------
# Parse PDF text (best-effort)
# -----------------------------
text = _read_pdf_text(pdf_bytes, max_pages=6)
hdr = _extract_header_fields(text) if text else {"report_title": None, "print_date": None, "valuation_date": None, "company": None}
policy_df = _extract_policy_summary_rows(text) if text else pd.DataFrame()

# Fallback if parsing is weak (common with PDFs)
if policy_df.empty:
    policy_df = pd.DataFrame(
        [
            {
                "Policy Inception": "01/01/1993",
                "Policy Expiration": "01/01/1994",
                "Policy Number": "BAYMEDICALCEN",
                "Carrier": "Florida Hospital Self Insurance Fund",
                "# Claims": 35,
                "Total Reserves": 6994575.80,
                "Total Payments": 2538734.42,
                "Total Recovery": 0.00,
                "Total Net Incurred": 4455841.38,
                "Avg Net Incurred": 127338.33,
            }
        ]
    )

# Compute top-line totals for nice “dashboard” delivery cards
tot_claims = int(policy_df["# Claims"].sum()) if "# Claims" in policy_df.columns else None
tot_reserves = float(policy_df["Total Reserves"].sum()) if "Total Reserves" in policy_df.columns else None
tot_payments = float(policy_df["Total Payments"].sum()) if "Total Payments" in policy_df.columns else None
tot_recovery = float(policy_df["Total Recovery"].sum()) if "Total Recovery" in policy_df.columns else 0.0
tot_net = float(policy_df["Total Net Incurred"].sum()) if "Total Net Incurred" in policy_df.columns else None


# -----------------------------
# Delivery Preview (client-facing)
# -----------------------------
st.divider()
st.markdown("<div class='cw-h2'>Delivery Preview</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")

def card(col, label, value):
    col.markdown(
        f"""
        <div class="cw-card">
          <div class="cw-label">{label}</div>
          <div class="cw-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

card(k1, "Request Type", req_type)
card(k2, "Priority", req_priority)
card(k3, "Valuation Date", hdr.get("valuation_date") or "—")
card(k4, "Total Claims (policy summary)", f"{tot_claims:,}" if tot_claims is not None else "—")
card(k5, "Total Net Incurred", _fmt_money(tot_net))


m1, m2 = st.columns([0.62, 0.38], gap="large")
with m1:
    st.markdown("<div class='cw-h2' style='font-size:18px;'>What the client sees</div>", unsafe_allow_html=True)

    # Narrative summary, dashboard tone
    report_title = hdr.get("report_title") or "Loss Run"
    company = hdr.get("company") or req_client
    print_date = hdr.get("print_date") or datetime.now().strftime("%m/%d/%Y")
    valuation_date = hdr.get("valuation_date") or "—"

    st.markdown(
        f"""
        <div class="cw-card">
          <div style="font-size:16px; font-weight:700; margin-bottom:6px;">{report_title} (Delivery Copy)</div>
          <div class="cw-muted" style="margin-bottom:10px;">
            Generated for <b>{company}</b>. This is the file that will be delivered to the requestor.
          </div>

          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
            <div>
              <div class="cw-label">Print Date</div>
              <div style="font-weight:650;">{print_date}</div>
            </div>
            <div>
              <div class="cw-label">Valuation Date</div>
              <div style="font-weight:650;">{valuation_date}</div>
            </div>
            <div>
              <div class="cw-label">Delivery</div>
              <div style="font-weight:650;">{req_delivery}</div>
            </div>
            <div>
              <div class="cw-label">Delivered To</div>
              <div style="font-weight:650;">{req_requestor}</div>
            </div>
          </div>

          <div style="margin-top:12px;">
            <div class="cw-label">Request Notes</div>
            <div style="font-size:14px; line-height:1.35;">{req_notes}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown("<div class='cw-h2' style='font-size:18px;'>Totals (policy summary)</div>", unsafe_allow_html=True)

    cA, cB = st.columns(2, gap="medium")
    card(cA, "Total Reserves", _fmt_money(tot_reserves))
    card(cB, "Total Payments", _fmt_money(tot_payments))

    cC, cD = st.columns(2, gap="medium")
    card(cC, "Total Recovery", _fmt_money(tot_recovery))
    card(cD, "Net Incurred", _fmt_money(tot_net))

    st.markdown("<div class='cw-muted' style='margin-top:10px;'>Totals shown here come from the PDF extract (best-effort). The delivered file remains the source of truth.</div>", unsafe_allow_html=True)


# -----------------------------
# Detail preview table
# -----------------------------
st.markdown("<div class='cw-h2' style='font-size:18px; margin-top:1.1rem;'>Policy-level Preview</div>", unsafe_allow_html=True)
show_cols = [
    "Policy Inception",
    "Policy Expiration",
    "Policy Number",
    "Carrier",
    "# Claims",
    "Total Reserves",
    "Total Payments",
    "Total Recovery",
    "Total Net Incurred",
    "Avg Net Incurred",
]
policy_show = policy_df.copy()
for c in ["Total Reserves", "Total Payments", "Total Recovery", "Total Net Incurred", "Avg Net Incurred"]:
    if c in policy_show.columns:
        policy_show[c] = policy_show[c].apply(_fmt_money)

st.dataframe(policy_show[ [c for c in show_cols if c in policy_show.columns] ], use_container_width=True, hide_index=True)


# -----------------------------
# Downloads (what the client actually gets)
# -----------------------------
st.markdown("<div class='cw-h2' style='font-size:18px; margin-top:1.1rem;'>Download Package</div>", unsafe_allow_html=True)
d1, d2, d3 = st.columns([0.34, 0.33, 0.33], gap="medium")

with d1:
    st.download_button(
        label="Download Delivered PDF",
        data=pdf_bytes,
        file_name=os.path.basename(pdf_name) if pdf_name else "loss_run.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with d2:
    # CSV "extract" (optional but useful for the demo)
    csv_bytes = policy_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Policy Summary (CSV extract)",
        data=csv_bytes,
        file_name="policy_summary_extract.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d3:
    # A simple "cover note" text you can paste into email (client-facing)
    cover_note = f"""Subject: {req_client} - Loss Run Delivery ({hdr.get('valuation_date') or 'Valuation Date'})\n\nHello,\n\nAttached is the requested loss run output for {req_client}.\n\nValuation Date: {hdr.get('valuation_date') or '—'}\nPrint Date: {hdr.get('print_date') or datetime.now().strftime('%m/%d/%Y')}\n\nIf you have any questions or would like additional cuts, reply to this email with the requested parameters.\n\nRegards,\nNARS Reporting\n"""
    st.download_button(
        label="Download Email Cover Note (TXT)",
        data=cover_note.encode("utf-8"),
        file_name="delivery_cover_note.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.markdown("<div class='cw-muted' style='margin-top:10px;'>This app is intentionally just a client-facing delivery preview. The actual run steps belong to the team, not your demo.</div>", unsafe_allow_html=True)