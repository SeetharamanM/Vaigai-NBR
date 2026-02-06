"""
Vaigai North Bank Road Project — Mbook & Pages analysis (Streamlit)
Analyse Mbook pages and date for each item; timeline by estimate (stretches on y-axis, date on x-axis).
Filter by estimate and bill no.
Run: streamlit run app_mbook.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="RCC RW — Mbook & Pages",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
EXCEL_PATH = BASE / "RCC RW data.xlsx"
CSV_PATH = BASE / "RCC RW data.csv"


@st.cache_data
def load_data():
    if EXCEL_PATH.exists():
        df = pd.read_excel(EXCEL_PATH, sheet_name="RCC RW", header=0)
        df.columns = [
            "Estimate", "Est_Length", "Bill_No", "Item", "Height",
            "Stretch", "Length", "Mbook", "Pages", "Date",
        ]
    else:
        df = pd.read_csv(CSV_PATH)
        df.columns = [
            "Estimate", "Est_Length", "Bill_No", "Item", "Height",
            "Stretch", "Length", "Mbook", "Pages", "Date",
        ]
    df = df[df["Estimate"].astype(str).str.strip() != "Estimate"].copy()
    df = df.dropna(how="all")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["Est_Length"] = pd.to_numeric(df["Est_Length"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Mbook"] = df["Mbook"].astype(str).str.strip()
    return df


def main():
    df = load_data()
    df = df[df["Item"].notna() & (df["Item"].astype(str).str.strip() != "Item")].copy()

    st.title("Mbook & Pages — RCC RW")
    st.caption("Analyse Mbook pages and date for each item. Filter by estimate and bill no.")

    # Sidebar: filter by estimate
    st.sidebar.header("Filters")
    estimates = ["All"] + sorted(df["Estimate"].dropna().astype(str).unique().tolist())
    selected_estimate = st.sidebar.selectbox(
        "Estimate",
        options=estimates,
        index=0,
        help="Choose an estimate to filter items.",
    )

    # Sidebar: filter by bill no
    bill_options = sorted(
        df["Bill_No"].dropna()
        .loc[df["Bill_No"].astype(str).str.contains("Bill", na=False)]
        .unique()
        .tolist(),
        key=lambda x: str(x),
    )
    selected_bills = st.sidebar.multiselect(
        "Bill No",
        options=bill_options,
        default=[],
        help="Leave empty to show all bills.",
    )

    filtered = df.copy()
    if selected_estimate != "All":
        filtered = filtered[filtered["Estimate"].astype(str) == selected_estimate]
    if selected_bills:
        filtered = filtered[filtered["Bill_No"].isin(selected_bills)]
    filtered = filtered.copy()

    if selected_estimate != "All" or selected_bills:
        st.sidebar.caption(f"Showing **{len(filtered)}** rows.")
    else:
        st.sidebar.caption(f"Showing all **{len(filtered)}** rows.")

    st.sidebar.divider()
    st.sidebar.caption("Data: RCC RW data.xlsx / RCC RW data.csv")

    if filtered.empty:
        st.warning("No data for the selected filters.")
        return

    if selected_bills:
        st.info(f"Filtered by bill(s): **{', '.join(selected_bills)}**")

    # Summary by Mbook
    st.subheader("Summary by Mbook")
    by_mbook = (
        filtered.groupby("Mbook", dropna=False)
        .agg(
            Items=("Item", "count"),
            Date_min=("Date", "min"),
            Date_max=("Date", "max"),
            Pages_sample=("Pages", "first"),
        )
        .reset_index()
    )
    by_mbook["Date range"] = by_mbook.apply(
        lambda r: f"{r['Date_min'].strftime('%d-%b-%Y') if pd.notna(r['Date_min']) else '—'} → {r['Date_max'].strftime('%d-%b-%Y') if pd.notna(r['Date_max']) else '—'}",
        axis=1,
    )
    display_mbook = by_mbook[["Mbook", "Items", "Date range", "Pages_sample"]].copy()
    display_mbook = display_mbook.rename(columns={"Pages_sample": "Pages (sample)"})
    st.dataframe(display_mbook, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Items: Mbook, Pages, Date")

    # Table columns: Estimate, Bill No, Item, Stretch, Mbook, Pages, Date
    show_cols = ["Estimate", "Bill_No", "Item", "Stretch", "Length", "Mbook", "Pages", "Date"]
    show_cols = [c for c in show_cols if c in filtered.columns]
    table_df = filtered[show_cols].copy()
    table_df["Date"] = table_df["Date"].dt.strftime("%d-%b-%Y")
    table_df = table_df.fillna("")

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ---------- Timeline by estimate (from app_timeline.py) ----------
    st.divider()
    st.subheader("Timeline by estimate (stretches on y-axis, date on x-axis)")
    valid_timeline = filtered.dropna(subset=["Length"]).copy()
    valid_dates = valid_timeline[valid_timeline["Date"].notna() & (valid_timeline["Date"].dt.year >= 2024)].copy()
    if len(valid_dates) > 0:
        by_est_date_item = (
            valid_dates.groupby(["Estimate", "Date", "Item", "Stretch"], dropna=False)
            .agg(total_length=("Length", "sum"))
            .reset_index()
        )
        by_est_date_item = by_est_date_item[
            by_est_date_item["Item"].notna() & (by_est_date_item["Item"].astype(str).str.strip() != "Item")
        ]
        by_est_date_item["Stretch"] = by_est_date_item["Stretch"].astype(str)
        estimates_list = sorted(by_est_date_item["Estimate"].dropna().unique().tolist())
        view_mode = st.radio("View", ["All estimates (expandable)", "Single estimate"], horizontal=True, key="timeline_view")
        if view_mode == "Single estimate":
            selected_est = st.selectbox("Select estimate", options=estimates_list, index=0, key="timeline_est")
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
                    fig.update_traces(marker=dict(symbol="square", size=16))
                    fig.update_layout(height=chart_height, margin=dict(t=30, l=80), xaxis_tickformat="%d-%b-%Y", xaxis_title="Date", yaxis_title="Stretch")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"**{est}**")
                fig = px.scatter(
                    sub,
                    x="Date",
                    y="Stretch",
                    color="Item",
                    hover_data={"total_length": ":.1f"},
                    title=f"Timeline — {est} (stretches on y-axis)",
                    labels={"Date": "Date", "Stretch": "Stretch", "Item": "Item", "total_length": "Length (m)"},
                )
                fig.update_traces(marker=dict(symbol="square", size=16))
                fig.update_layout(height=chart_height, margin=dict(t=30, l=80), xaxis_tickformat="%d-%b-%Y", xaxis_title="Date", yaxis_title="Stretch")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No timeline data (need dates from 2024 onward and valid Length). Adjust filters or data.")

    # Optional: download filtered data
    st.download_button(
        label="Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="rcc_rw_mbook_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
