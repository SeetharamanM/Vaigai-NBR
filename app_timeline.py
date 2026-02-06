"""
Vaigai North Bank Road Project — Timeline by Estimate (Streamlit)
Shows timeline of each estimate for items, filtered by Bill No.
Run: streamlit run app_timeline.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="Timeline by Estimate — RCC RW",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCEL_PATH = Path(__file__).parent / "RCC RW data.xlsx"


@st.cache_data
def load_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name="RCC RW", header=0)
    df.columns = [
        "Estimate", "Est_Length", "Bill_No", "Item", "Height",
        "Stretch", "Length", "Mbook", "Pages", "Date"
    ]
    df = df[df["Estimate"] != "Estimate"].copy()
    df = df.dropna(how="all")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["Est_Length"] = pd.to_numeric(df["Est_Length"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def main():
    df = load_data()
    valid = df.dropna(subset=["Length"]).copy()
    valid = valid[valid["Item"].notna() & (valid["Item"] != "Item")]

    # Only 2024+ for timeline
    valid_dates = valid[valid["Date"].notna() & (valid["Date"].dt.year >= 2024)].copy()
    valid_dates["MonthDate"] = valid_dates["Date"].dt.to_period("M").dt.to_timestamp()

    bill_list = sorted(
        valid_dates["Bill_No"].dropna()
        .loc[valid_dates["Bill_No"].astype(str).str.contains("Bill", na=False)]
        .unique()
        .tolist(),
        key=lambda x: str(x),
    )
    estimates_list = sorted(valid_dates["Estimate"].dropna().unique().tolist())

    st.title("Vaigai North Bank Road Project")
    st.caption("Timeline of each stretch by items (filtered by Bill No)")

    # Sidebar: Filter by Bill No
    st.sidebar.header("Filter by Bill No")
    filter_bills = st.sidebar.multiselect(
        "Select one or more bills",
        options=bill_list,
        default=[],
        help="Leave empty to show all bills.",
    )
    st.sidebar.divider()
    st.sidebar.caption("Data: RCC RW data.xlsx")

    # Apply bill filter
    timeline_df = valid_dates.copy()
    if filter_bills:
        timeline_df = timeline_df[timeline_df["Bill_No"].isin(filter_bills)]
        st.info(f"Showing data for bill(s): **{', '.join(filter_bills)}**")
    else:
        st.caption("Showing all bills. Select bills in the sidebar to filter.")

    if len(timeline_df) == 0:
        st.warning("No data for the selected bills. Try other bills or clear the filter.")
        return

    # Aggregate by Estimate, Date, Item, Stretch (date on x-axis)
    by_est_date_item = (
        timeline_df.groupby(["Estimate", "Date", "Item", "Stretch"], dropna=False)
        .agg(total_length=("Length", "sum"))
        .reset_index()
    )
    by_est_date_item = by_est_date_item[
        by_est_date_item["Item"].notna() & (by_est_date_item["Item"] != "Item")
    ]
    by_est_date_item["Stretch"] = by_est_date_item["Stretch"].astype(str)

    # Optional: filter to one estimate for a compact view
    view_mode = st.radio("View", ["All estimates (expandable)", "Single estimate"], horizontal=True)
    if view_mode == "Single estimate":
        selected_est = st.selectbox("Select estimate", options=estimates_list, index=0)
        estimates_to_show = [selected_est]
    else:
        estimates_to_show = estimates_list

    for est in estimates_to_show:
        sub = by_est_date_item[by_est_date_item["Estimate"] == est].sort_values(["Stretch", "Date"])
        if len(sub) == 0:
            continue
        n_stretches = sub["Stretch"].nunique()
        chart_height = max(280, n_stretches * 28)

        if view_mode == "All estimates (expandable)":
            with st.expander(f"**{est}**", expanded=False):
                fig = px.scatter(
                    sub,
                    x="Date",
                    y="Stretch",
                    color="Item",
                    hover_data={"total_length": ":.1f"},
                    title=f"Timeline — {est} (stretches on y-axis)",
                    labels={"Date": "Date", "Stretch": "Stretch", "Item": "Item", "total_length": "Length (m)"},
                )
                fig.update_layout(height=chart_height, margin=dict(t=30, l=80), xaxis_tickformat="%d-%b-%Y", xaxis_title="Date", yaxis_title="Stretch")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader(est)
            fig = px.scatter(
                sub,
                x="Date",
                y="Stretch",
                color="Item",
                hover_data={"total_length": ":.1f"},
                title=f"Timeline — {est} (stretches on y-axis)",
                labels={"Date": "Date", "Stretch": "Stretch", "Item": "Item", "total_length": "Length (m)"},
            )
            fig.update_layout(height=chart_height, margin=dict(t=30, l=80), xaxis_tickformat="%d-%b-%Y", xaxis_title="Date", yaxis_title="Stretch")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
