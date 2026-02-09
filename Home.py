"""
Vaigai North Bank Road Project — Home
"""
import streamlit as st

st.title("Vaigai North Bank Road Project")
st.markdown("**RCC Retaining Wall (RW) — Analytics Dashboard**")

st.divider()

st.markdown("""
Welcome to the Vaigai North Bank Road Project dashboard.
""")

# Link to static web page in side navigation
with st.sidebar:
    st.page_link("Pages/5_📄_VaigaiNBR_Static.py", label="Vaigai NBR (Static page)", icon="📄")

st.info("Data source: **RCC RW data.xlsx** (sheet: RCC RW)")
