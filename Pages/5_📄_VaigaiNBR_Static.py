"""
Vaigai NBR — Project details (converted from VaigaiNBR.html)
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Vaigai NBR — Project Details", layout="wide")

# Theme color from original HTML
HEADER_BG = "#0A598A"
SIDEBAR_BG = "#0d9984"

st.markdown(
    f"""
    <style>
    div[data-testid="stExpander"] {{ border-left: 4px solid {HEADER_BG}; }}
    .vnbr-header {{ background: {HEADER_BG}; color: white; padding: 0.5rem 1rem; margin: 1rem 0 0.5rem 0; border-radius: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: link to VNBR Documents sub-page
with st.sidebar:
    st.page_link("Pages/6_📄_VNBR_Docs.py", label="VNBR Documents", icon="📋")

st.title("Vaigai North Bank Road Details")

# ---------- Estimate Details ----------
st.markdown('<div class="vnbr-header"><h4>Estimate Details</h4></div>', unsafe_allow_html=True)
e1, e2, e3, e4 = st.columns(4)

with e1:
    with st.container(border=True):
        st.subheader("Name of work")
        st.markdown("""
        Construction of Vaigai North Bank Road Km 0/0-8/0 from Kamarajar Bridge to Varanasi-Kanniyakumari road (NH-44) near Samayanallur  
        *CRIDP 2022-23*  
        *CE No. 139/2023-24   dt: 05.03.2024*
        """)
        st.caption("*1. G.O.(Ms) No. 125, Highways & Minor Ports (HS1) Department, dated 12.07.2022*  \n*2. G.O.(Ms) No. 32, Highways & Minor Ports (HS2) Department, dated 04.03.2024*")

with e2:
    with st.container(border=True):
        st.subheader("Cost Details")
        cost = pd.DataFrame([
            ["Estimate Amount", "Rs 176 Crore"],
            ["Road Work", "Rs 42,61,26,130"],
            ["Sub Estimates", "Rs 90,25,52,352"],
            ["Total Amount", "Rs 132,86,78,488"],
            ["GST 18%", "Rs 23,91,62,128"],
            ["LS Provisions", "Rs 19,21,59,384"],
        ], columns=["Item", "Amount"])
        st.dataframe(cost, use_container_width=True, hide_index=True)

with e3:
    with st.container(border=True):
        st.subheader("Sub Estimates")
        sub = pd.DataFrame([
            ["Minor Bridge Km 2/850", "Rs 2,04,66,248"],
            ["Minor Bridge Km 6/200", "Rs 1,77,13,525"],
            ["Box Culverts (11 Nos)", "Rs 1,47,88,836"],
            ["PCC Retaining wall (450 M)", "Rs 2,02,33,026"],
            ["RCC Retaining wall (7805 M)", "Rs 76,87,54,660"],
            ["Raised Foot Path (8000 M)", "Rs 6,05,96,063"],
        ], columns=["Item", "Amount"])
        st.dataframe(sub, use_container_width=True, hide_index=True)

with e4:
    with st.container(border=True):
        st.subheader("LS Provisions")
        ls = pd.DataFrame([
            ["Labour Welfare Fund", "Rs 1,56,78,407"],
            ["Quality Control Charges", "Rs 1,56,79,000"],
            ["Escalation Provision", "Rs 7,83,93,000"],
            ["Advertisement Charges", "Rs 8,00,000"],
            ["Road Furnitures", "Rs 1,45,00,000"],
            ["Shifting of Utilities", "Rs 50,00,000"],
            ["Land Acquisition", "Rs 3,00,00,000"],
            ["Road Safety Audit", "Rs 15,00,000"],
            ["Turfing & Tree Plantation", "Rs 28,00,000"],
            ["Staircase Provision", "Rs 1,00,00,000"],
            ["Protective wall for Existing Water Pipe Line", "Rs 80,00,000"],
            ["PS and Contingencies", "Rs 98,08,977"],
        ], columns=["Item", "Amount"])
        st.dataframe(ls, use_container_width=True, hide_index=True)

# ---------- Agreement Details ----------
st.markdown('<div class="vnbr-header"><h4>Agreement Details</h4></div>', unsafe_allow_html=True)
a1, a2 = st.columns(2)

with a1:
    with st.container(border=True):
        st.subheader("Agreement Details")
        agt = pd.DataFrame([
            ["Contractor", "M/s Notch India Projects, Madurai"],
            ["Tender Date", "03.05.2024"],
            ["Tender opening", ""],
            ["COT Proceedings", "13.06.2024"],
            ["Work Order", "14.06.2024"],
            ["C.R. Agreement No.2/2024-25", "28.06.2024"],
            ["Contract Value", "Rs 165,89,31,278"],
            ["Tender %", "5.81% above ER 2023-24"],
            ["Period", "20 months"],
            ["Agt completion Date", "27.02.2026"],
            ["EOT Upto", ""],
        ], columns=["Field", "Value"])
        st.dataframe(agt, use_container_width=True, hide_index=True)

with a2:
    with st.container(border=True):
        st.subheader("Work Progress")
        prog = pd.DataFrame([
            ["Road Work", "32.07%", "0% Done"],
            ["Minor Bridge @ Km 2/850", "1.54%", "0% Done"],
            ["Minor Bridge @ Km 6/200", "1.33%", "0% Done"],
            ["Box Culverts (11 Nos)", "1.11%", "0% Done"],
            ["PCC Retaining Wall (450 M)", "1.52%", "0% Done"],
            ["RCC Retaining Wall (7805 M)", "57.86%", "0% Done"],
            ["Raised Foot Path (8000 M)", "4.56%", "0% Done"],
        ], columns=["Item", "%", "Status"])
        st.dataframe(prog, use_container_width=True, hide_index=True)

# ---------- Project NOC Details ----------
st.markdown('<div class="vnbr-header"><h4>Project NOC Details</h4></div>', unsafe_allow_html=True)
n1, n2 = st.columns(2)

with n1:
    with st.container(border=True):
        st.subheader("Diary of Events (WRD NOC)")
        wrd = pd.DataFrame([
            [1, "21.02.2022", "Hon'ble CM Announcement in Govt. function at Madurai"],
            [2, "12.07.2022", "GO (Ms) No. 125, Highways and Minor Ports (HS1) Dept for Administrative Sanction of Rs 95.94 Crore"],
            [3, "20.02.2023", "Letter to SE, WRD Vaigai basin requested joint inspection"],
            [4, "21.02.2023", "Joint inspection by SE, WRD and SE (H) C&M for NOC"],
            [5, "28.02.2023", "Letter from SE, WRD Vaigai basin recommending construction of Retaining wall along River bund"],
            [6, "30.03.2023", "RAS Proposal to CE (H) C&M for obtaining RAS for Rs 176 Crore"],
            [7, "06.04.2023", "Requisition letter to SE, WRD for NOC"],
            [8, "04.03.2024", "GO (Ms) No.32, Highways and Minor Ports (HS1) Dept for Revised Administrative Sanction of Rs 176.00 Crore"],
            [9, "05.03.2024", "Technical Sanction by CE (H) C&M for Rs 176.00 Crore"],
            [10, "05.04.2024", "Site Inspection by SE, WRD with DE (H) C&M for NOC and requested to mark the alignment of proposed road and River boundary"],
            [11, "11.07.2024", "Site inspection by SE, WRD with Highways officials for NOC"],
            [12, "11.07.2024", "Letter from SE, WRD requested marking of Retaining wall alignment with pegs along whole stretch of proposed road"],
            [13, "16.07.2024", "Site inspection by SE, WRD and DE (H) C&M along with officials for NOC"],
            [14, "23.07.2024", "Inevitability Certificate obtained from the District Collector, Madurai"],
            [15, "23.07.2024", "Site inspection by SE (H) C&M"],
            [16, "29.07.2024", "Site inspection by SE, WRD on opposite bank for Vaigai North Bank road NOC"],
            [17, "21.08.2024", "Chief Engineer, WRD Superintending Engineer, WRD, Divisional Engineer (H) C&M Joint Inspection for NOC"],
            [18, "21.08.2024", "Letter from CE, WRD, Madurai to Engineer-in-Chief, Chennai recommending for NOC"],
            [19, "23.08.2024", "Letter from Engineer-in-Chief, Chennai to The Secretary WRD recommending for NOC"],
            [20, "06.09.2024", "Letter from Engineer-in-Chief, Chennai to The Secretary WRD for justification regarding NOC queries."],
            [21, "11.09.2024", "NOC GO from WRD, G.O.(4D) No. 45/Water Resources (R2) Department, dated : 11.09.2024"],
            [23, "28.10.2024", "Site inspection by EE, AEE & AE, WRD with ADE, AE, Highways and Contractors for Alignment"],
        ], columns=["Sl.No", "Date", "Events"])
        st.dataframe(wrd, use_container_width=True, hide_index=True)

with n2:
    with st.container(border=True):
        st.subheader("Diary of Events (NHAI NOC)")
        nhai = pd.DataFrame([
            [1, "25.06.2024", "Letter to PD, NHAI, Madurai requested NOC for connecting proposed road with NH-44 at Km 2/4 (Madurai-Kayathar section)"],
            [2, "04.07.2024", "Joint inspection of PD, NHAI and ADE (H) C&M"],
            [3, "04.07.2024", "Letter from PD, NHAI requesting compliance report to the observations made during inspection"],
            [4, "12.07.2024", "Compliance report to observations and additional drawings to PD, NHAI"],
            [5, "23.07.2024", "Site inspection by SE (H) C&M"],
            [6, "31.07.2024", "Site inspection by CE (H) C&M, Chennai for proposed road alignment"],
            [7, "03.08.2024", "Letter from PD, NHAI to RO, NHAI recommending for NOC"],
            [8, "18.09.2024", "Letter From PD, NHAI, requesting deposit required Processing fee, Provisional fee & Licence fee for NOC"],
            [9, "09.04.2025", "Letter From PD, NHAI, Communicating In-Principle NOC"],
        ], columns=["Sl.No", "Date", "Events"])
        st.dataframe(nhai, use_container_width=True, hide_index=True)

# ---------- Project LA Details ----------
st.markdown('<div class="vnbr-header"><h4>Project LA Details</h4></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Diary of Events (LA)")
    la = pd.DataFrame([
        [1, "", "LPS Submitted to DRO"],
        [2, "", "15(2) Notification sent to CLA"],
        [3, "30.01.2025", "15(2) Notification Published in Newspapers"],
        [4, "", "15(1) notification sent to CLA"],
        [5, "03.06.2025", "15(1) Notification Proposal approved by CLA"],
        [6, "23.06.2025", "15(1) Notification Published in Gazette No."],
        [7, "29.07.2025", "19(2) Enquiry conducted"],
        [8, "06.08.2025", "19(3) sent to CLA for Approval"],
        [9, "", "19(3) Approved by CLA"],
        [10, "10.05.2025", "19(5) Enquiry Conducted"],
    ], columns=["Sl.No", "Date", "Events"])
    st.dataframe(la, use_container_width=True, hide_index=True)

# ---------- Project Progress Details ----------
st.markdown('<div class="vnbr-header"><h4>Project Progress Details</h4></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Project Construction Events")
    const = pd.DataFrame([
        [1, "21.02.2022", "Hon'ble CM Announcement in Govt. function at Madurai"],
        [2, "12.07.2022", "GO (Ms) No. 125, Highways and Minor Ports (HS1) Dept for Administrative Sanction of Rs 95.94 Crore"],
        [3, "30.03.2023", "RAS Proposal to CE (H) C&M for obtaining RAS for Rs 176 Crore"],
        [4, "04.03.2024", "GO (Ms) No.32, Highways and Minor Ports (HS1) Dept for Revised Administrative Sanction of Rs 176.00 Crore"],
        [5, "05.03.2024", "Technical Sanction by CE (H) C&M for Rs 176.00 Crore"],
        [6, "28.06.2024", "Agreement executed by Notch India Projects"],
        [7, "23.07.2024", "Inevitability Certificate obtained from the District Collector, Madurai"],
        [8, "23.07.2024", "Site inspection by SE (H) C&M"],
        [9, "29.07.2024", "DE requested The Corporation Commissioner for Joint inspection for Pipeline and air valve chamber encountered along proposed road alignment"],
        [10, "31.07.2024", "Site inspection by CE (H) C&M, Chennai for proposed road alignment"],
        [11, "23.08.2024", "DE requested The District Collector, Madurai for Encroachment Eviction and River boundary survey in Vaigai North Bank."],
        [12, "06.09.2024", "Site Inspection by Deputy Secretary, Ministry of Youth Welfare and Sports Development"],
        [13, "27.09.2024", "Letter from CE / TNPDCL, Madurai region requesting to get permission from TNEB"],
    ], columns=["Sl.No", "Date", "Events"])
    st.dataframe(const, use_container_width=True, hide_index=True)

st.divider()
st.caption("Source: Vaigai North Bank Road project — Usilampatti Highways")
