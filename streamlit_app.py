"""
Vaigai North Bank Road Project — Streamlit App
Run: streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Vaigai North Bank Road — RCC RW",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Side navigation with all pages
pg = st.navigation([
    st.Page("Home.py", title="Home", icon="🛣️", default=True),
    st.Page("Pages/1_📒_Mbook.py", title="Mbook", icon="📒"),
    st.Page("Pages/2_📊_Progress.py", title="Progress", icon="📊"),
    st.Page("Pages/3_📐_Overlap_Gap.py", title="Overlap & Gap", icon="📐"),
    st.Page("Pages/4_📈_Timeline.py", title="Timeline", icon="📈"),
    st.Page("Pages/5_📄_VaigaiNBR_Static.py", title="Vaigai NBR (Static)", icon="📄"),
    st.Page("Pages/6_📄_VNBR_Docs.py", title="VNBR Documents", icon="📋"),
])
pg.run()
