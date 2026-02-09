"""Timeline by Estimate — RCC RW."""
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "RCC RW data.xlsx"


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
    df["Mbook"] = df["Mbook"].astype(str).str.strip()
    return df


df = load_data()
valid = df.dropna(subset=["Length"]).copy()
valid = valid[valid["Item"].notna() & (valid["Item"] != "Item")]

valid_dates = valid[valid["Date"].notna() & (valid["Date"].dt.year >= 2024)].copy()
valid_dates["MonthDate"] = valid_dates["Date"].dt.to_period("M").dt.to_timestamp()
valid_dates["Month"] = valid_dates["Date"].dt.to_period("M").astype(str)

bill_list = sorted(
    valid_dates["Bill_No"].dropna()
    .loc[valid_dates["Bill_No"].astype(str).str.contains("Bill", na=False)]
    .unique()
    .tolist(),
    key=lambda x: str(x),
)
estimates_list = sorted(valid_dates["Estimate"].dropna().unique().tolist())
months_list = sorted(valid_dates["Month"].dropna().unique().tolist())
mbooks_list = sorted(valid_dates["Mbook"].dropna().replace("", pd.NA).dropna().unique().tolist())
stretches_list = sorted(valid_dates["Stretch"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist(), key=lambda x: str(x))

st.title("Vaigai North Bank Road Project")
st.caption("Timeline of each stretch by items (filtered by Bill No)")

# Sidebar filters
st.sidebar.header("Filters")
filter_estimates = st.sidebar.multiselect(
    "Estimate",
    options=estimates_list,
    default=[],
    help="Leave empty for all.",
    key="tl_filter_est",
)
filter_bills = st.sidebar.multiselect(
    "Bill No",
    options=bill_list,
    default=[],
    help="Leave empty for all.",
    key="timeline_bills",
)
filter_months = st.sidebar.multiselect(
    "Month",
    options=months_list,
    default=[],
    help="Leave empty for all.",
    key="tl_filter_month",
)
filter_mbooks = st.sidebar.multiselect(
    "Mbook",
    options=mbooks_list,
    default=[],
    help="Leave empty for all.",
    key="tl_filter_mbook",
)
filter_stretches = st.sidebar.multiselect(
    "Stretch",
    options=stretches_list,
    default=[],
    help="Leave empty for all.",
    key="tl_filter_stretch",
)
st.sidebar.divider()
st.sidebar.caption("Data: RCC RW data.xlsx")

# Apply filters
timeline_df = valid_dates.copy()
if filter_estimates:
    timeline_df = timeline_df[timeline_df["Estimate"].isin(filter_estimates)]
if filter_bills:
    timeline_df = timeline_df[timeline_df["Bill_No"].isin(filter_bills)]
if filter_months:
    timeline_df = timeline_df[timeline_df["Month"].isin(filter_months)]
if filter_mbooks:
    timeline_df = timeline_df[timeline_df["Mbook"].isin(filter_mbooks)]
if filter_stretches:
    timeline_df = timeline_df[timeline_df["Stretch"].astype(str).str.strip().isin(filter_stretches)]

if filter_bills:
    st.info(f"Showing data for bill(s): **{', '.join(filter_bills)}**")
else:
    st.caption("Showing all bills. Select filters in the sidebar to narrow down.")

if len(timeline_df) == 0:
    st.warning("No data for the selected filters. Try clearing some filters.")
    st.stop()

# --- Date (x) vs Item (y): colored square per item ---
ITEM_ORDER = ["Earthwork", "PCC", "Steel", "Footing", "Below sill", "Above sill"]

def _item_sort_idx(item):
    s = str(item).strip().lower()
    for i, name in enumerate(ITEM_ORDER):
        if s == name.lower():
            return i
    return len(ITEM_ORDER)

st.divider()
st.subheader("Date × Item (colored square per item)")
agg_date_item = timeline_df.groupby(["Date", "Item"], dropna=False).agg(total_length=("Length", "sum")).reset_index()
agg_date_item = agg_date_item[agg_date_item["Item"].notna() & (agg_date_item["Item"].astype(str).str.strip() != "Item")]
if len(agg_date_item) > 0:
    present_items = agg_date_item["Item"].dropna().unique().tolist()
    item_order = sorted(present_items, key=_item_sort_idx)
    fig_squares = px.scatter(
        agg_date_item,
        x="Date",
        y="Item",
        color="Item",
        category_orders={"Item": item_order},
        hover_data={"total_length": ":.1f", "Date": "|%d-%b-%Y"},
        title="Date (x) vs Item (y) — one square per item",
    )
    fig_squares.update_traces(marker=dict(symbol="square", size=14, line=dict(width=0.5)))
    fig_squares.update_layout(
        height=max(320, agg_date_item["Item"].nunique() * 28),
        margin=dict(t=30, l=100),
        xaxis_title="Date",
        yaxis_title="Item",
        xaxis_tickformat="%d-%b-%Y",
        yaxis=dict(categoryorder="array", categoryarray=item_order),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_squares, use_container_width=True)
else:
    st.caption("No data for chart.")

# --- Timeline by estimate (stretch scatter) ---
st.divider()
st.subheader("Timeline by estimate (stretches on y-axis)")

by_est_date_item = (
    timeline_df.groupby(["Estimate", "Date", "Item", "Stretch"], dropna=False)
    .agg(total_length=("Length", "sum"))
    .reset_index()
)
by_est_date_item = by_est_date_item[
    by_est_date_item["Item"].notna() & (by_est_date_item["Item"] != "Item")
]
by_est_date_item["Stretch"] = by_est_date_item["Stretch"].astype(str)

view_mode = st.radio("View", ["All estimates (expandable)", "Single estimate"], horizontal=True, key="tl_view")
if view_mode == "Single estimate":
    selected_est = st.selectbox("Select estimate", options=estimates_list, index=0, key="tl_est")
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
