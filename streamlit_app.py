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

# Side navigation with all pages (ASCII paths for Cloud compatibility)
pg = st.navigation([
    st.Page("Home.py", title="Home", icon="🛣️", default=True),
    st.Page("Pages/1_Mbook.py", title="Mbook", icon="📒"),
    st.Page("Pages/2_Progress.py", title="Progress", icon="📊"),
    st.Page("Pages/3_Overlap_Gap.py", title="Overlap & Gap", icon="📐"),
    st.Page("Pages/4_Timeline.py", title="Timeline", icon="📈"),
    st.Page("Pages/5_VaigaiNBR_Static.py", title="Vaigai NBR (Static)", icon="📄"),
    st.Page("Pages/6_VNBR_Docs.py", title="VNBR Documents", icon="📋"),
])
pg.run()
