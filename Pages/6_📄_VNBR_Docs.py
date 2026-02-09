"""
VNBR Documents — Vaigai North Bank Road Documents (converted from VNBR Docs.html)
Sub-page under Vaigai NBR (Static).
"""
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="VNBR Documents", layout="wide")

# Theme from VNBR Docs.html
HEADER_BG = "#008AD8"
BORDER_ACCENT = "#e8491d"

st.markdown(
    f"""
    <style>
    .vnbr-docs-header {{ background: {HEADER_BG}; color: white; padding: 0.6rem 1rem; margin: 0 0 1rem 0; border-radius: 4px; border-bottom: 3px solid {BORDER_ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="vnbr-docs-header"><h3>Vaigai North Bank Road Documents</h3></div>', unsafe_allow_html=True)

# Documents table from VNBR Docs.html (Sl.No, Date, Document title, Details, Department)
DOCS = [
    (1, "12.07.2022", "Administrative Sanction GO for Rs 95.94 Crore", "GO (Ms) No. 125, Highways and Minor Ports (HS1) Dept", "HWD", "GO 125 dt 120722.pdf"),
    (2, "20.02.2023", "Letter to SE, WRD Vaigai basin requested joint inspection", "563/AE-1/2022-23/Vaigai NB Road", "WRD", "se to se wrd.pdf"),
    (3, "28.02.2023", "Letter from SE, WRD Vaigai basin recommending construction of Retaining wall along River bund", "DB/D1/78M/C. NOC/2023", "WRD", "SE wrd to SE.pdf"),
    (4, "30.03.2023", "RAS Proposal to CE (H) C&M for obtaining RAS for Rs 176 Crore", "616/AE-1/2022-23/Vaigai NB Road", "HWD", "SE to CE RAS.pdf"),
    (5, "06.04.2023", "Requisition letter to SE, WRD for NOC", "Sa.Na/2022-23/JDO2", "WRD", None),
    (6, "04.03.2024", "Revised Administrative Sanction GO for Rs 176.00 Crore", "GO (Ms) No.32, Highways and Minor Ports (HS1) Dept", "HWD", "GO 32 dt 04.03.24.pdf"),
    (7, "05.03.2024", "Technical Sanction by CE (H) C&M for Rs 176.00 Crore", "4595/Salai/2023", "HWD", "CE TS 05.03.24.pdf"),
    (8, "05.03.2024", "Abstract Estimate", "CE Highways (C&M) No. 139/2023-2024", "HWD", "vnbr abstract.pdf"),
    (9, "", "Tender Notice", "Tender Notice", "HWD", None),
    (10, "13.06.2024", "COT Proceedings", "No. 10/341-1/COMM/Tech Cell/2024", "HWD", "COT.pdf"),
    (11, "14.06.2024", "Work Order", "T.R.P.169/2023-24/DO-1", "HWD", "Work Order.pdf"),
    (12, "25.06.2024", "Letter to PD, NHAI, Madurai requested NOC for connecting proposed road with NH-44 at Km 2/4", "78/2024/JDO2", "NHAI", "nhai noc 250624.pdf"),
    (13, "28.06.2024", "Agreement Proceedings", "T.R.P.169/2023-24/DO-14", "HWD", "Agt Proc.pdf"),
    (14, "04.07.2024", "Letter from PD, NHAI requesting compliance report to the observations made during inspection", "NHAI/PIU/MDU/NOC/NH-44/2024/1377", "NHAI", "Lr. No. 1377.pdf"),
    (15, "06.07.2024", "Letter from PD, NHAI to Concessionaire requesting comments and consent for NOC", "NHAI/PIU/MDU/NOC/NH-44 & (TN-05)/2024/1380", "NHAI", "nhai pd to con.pdf"),
    (16, "11.07.2024", "Letter from SE, WRD requested marking of Retaining wall alignment with pegs along whole stretch", "D1/C. NOC (Kamarajar Salai)/2024", "WRD", "Lr fr WRD SE 11.07.24.pdf"),
    (17, "12.07.2024", "Compliance report to observations and additional drawings to PD, NHAI", "78/2024/JDO2", "NHAI", "de to pd compliiance 120724.pdf"),
    (18, "23.07.2024", "Inevitability Certificate obtained from the District Collector, Madurai", "Inevitability Certificate", "WRD", "Inevitability Cert.pdf"),
    (19, "03.08.2024", "Letter from PD, NHAI to RO, NHAI recommending for NOC", "NHAI/PIU/MDU/NH-44/North River Bank road/2024/1543", "NHAI", "PD MDU Lr No 1543 dt 03 08 24.pdf"),
    (20, "21.08.2024", "Letter from CE, WRD to Engineer-in-Chief, WRD, Chennai recommending for NOC", "DB/SGDO/C.12119/2024", "WRD", "vnbr noc ce-wrd to eic wrd.pdf"),
    (21, "23.08.2024", "Letter from Engineer-in-Chief, Chennai to The Secretary WRD recommending for NOC", "Letter No.S7(2)/29147/MR-NOC-Highways Department/...", "WRD", "vnbr noc fr eic wrd chn.pdf"),
    (22, "06.09.2024", "Letter from Engineer-in-Chief, Chennai to The Secretary WRD for justification regarding NOC queries", "Letter No.S7(2)/29147/MR-NOC-Highways Department/...", "WRD", "vnbr noc just.pdf"),
    (23, "11.09.2024", "NOC GO from WRD", "G.O.(4D) No. 45/Water Resources (R2) Department, dated : 11.09.2024", "WRD", "G.O. (4D) No. 45.pdf"),
    (24, "18.09.2024", "Letter From PD, NHAI, requesting deposit required Processing fee, Provisional fee & Licence fee for NOC", "NHAI/PIU/MDU/NH-44/North River Bank road/2024/1831", "NHAI", "Lr fr pd to de fee.pdf"),
    (25, "27.09.2024", "Letter from CE/Distribution, TNPDCL, Madurai region requesting to get permission from TNEB", "CE/D/MDU/EE/C/AEE/F.LAND/D.No.191/24", "TNEB", "CE TNEB to CE H.pdf"),
    (26, "09.04.2025", "Letter from PD, NHAI, Communicating In-Principle approval of NOC", "NHAI/PIU/NH-44/North River Bank road/2025/586", "NHAI", "Lr.No.586.pdf"),
]

# Build dataframe (without file path for display)
df = pd.DataFrame(
    [(sl, date, doc, details, dept) for sl, date, doc, details, dept, _ in DOCS],
    columns=["Sl.No", "Date", "Document", "Details", "Department"],
)

# Sidebar: document download
with st.sidebar:
    st.caption("Use the sidebar above to open **Vaigai NBR (Static)** for project details.")
    st.divider()
    st.caption("**Download a document** (PDFs from Vaigai NBR folder)")
    docs_with_file = [(r[0], r[2], r[5]) for r in DOCS if r[5]]
    doc_options = {f"{sl}. {title[:50]}..." if len(title) > 50 else f"{sl}. {title}": (sl, filename) for sl, title, filename in docs_with_file}
    selected = st.selectbox("Choose document", options=["— Select —"] + list(doc_options.keys()), label_visibility="collapsed")
    if selected and selected != "— Select —":
        _sl, filename = doc_options[selected]
        docs_folder = Path(__file__).resolve().parent.parent / "Vaigai NBR"
        pdf_path = docs_folder / filename
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", data=f.read(), file_name=filename, mime="application/pdf", use_container_width=True)
        else:
            st.warning(f"File not found: {filename}")

# Filters
col_filt, _ = st.columns([1, 3])
with col_filt:
    dept_filter = st.multiselect(
        "Filter by Department",
        options=df["Department"].dropna().unique().tolist(),
        default=df["Department"].dropna().unique().tolist(),
        label_visibility="collapsed",
    )
if dept_filter:
    df = df[df["Department"].isin(dept_filter)]

st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("Source: Vaigai North Bank Road Documents — PDFs are in the **Vaigai NBR** project folder.")
