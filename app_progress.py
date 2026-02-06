"""
Vaigai North Bank Road Project — RCC Retaining Wall Progress (Streamlit)
Month-wise overall length progress and balance to be completed by March 2026.
Run: streamlit run app_progress.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
st.set_page_config(
    page_title="RCC RW Progress — Vaigai North Bank",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCEL_PATH = Path(__file__).parent / "RCC RW data.xlsx"
TARGET_MONTH = "2026-03"  # March 2026


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

    # Only 2024+ for progress
    valid_dates = valid[valid["Date"].notna() & (valid["Date"].dt.year >= 2024)].copy()
    valid_dates["Month"] = valid_dates["Date"].dt.to_period("M").astype(str)
    valid_dates["MonthDate"] = valid_dates["Date"].dt.to_period("M").dt.to_timestamp()

    # Total estimated length (linear m)
    total_est = valid_dates.groupby("Estimate")["Est_Length"].first().sum()

    # Month-wise completed length: use Earthwork as reference for linear metres
    monthly = (
        valid_dates[valid_dates["Item"] == "Earthwork"]
        .groupby("Month")
        .agg(completed=("Length", "sum"))
        .reset_index()
        .sort_values("Month")
    )
    if len(monthly) == 0:
        st.warning("No Earthwork data from 2024 onward. Cannot compute progress.")
        return

    monthly["MonthDate"] = pd.to_datetime(monthly["Month"] + "-01")
    monthly["Cumulative"] = monthly["completed"].cumsum()
    total_completed = monthly["Cumulative"].iloc[-1]
    balance = max(0, total_est - total_completed)

    # Months remaining to March 2026 (from last data month)
    last_month = monthly["Month"].iloc[-1]
    try:
        last_dt = pd.to_datetime(last_month + "-01")
        target_dt = pd.to_datetime(TARGET_MONTH + "-01")
        months_left = max(0, (target_dt.year - last_dt.year) * 12 + (target_dt.month - last_dt.month))
        required_per_month = (balance / months_left) if months_left > 0 else 0
    except Exception:
        months_left = 0
        required_per_month = 0

    st.title("Vaigai North Bank Road Project")
    st.caption("RCC Retaining Wall — Overall length progress (month-wise)")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total estimated (m)", f"{total_est:,.0f}")
    col2.metric("Completed (m)", f"{total_completed:,.0f}")
    col3.metric("Balance to complete by Mar 2026 (m)", f"{balance:,.0f}")
    col4.metric("Progress", f"{(total_completed / total_est * 100):.1f}%" if total_est else "—")
    if months_left > 0 and balance > 0:
        col5.metric("Required per month to target (m)", f"{required_per_month:,.0f}")
    else:
        col5.metric("Months to Mar 2026", str(months_left) if months_left else "—")

    st.divider()

    # Balance callout
    st.subheader("Balance to be completed by March 2026")
    st.info(f"**{balance:,.0f} m** of linear length (Earthwork basis) remains to be completed by March 2026.")
    if months_left > 0 and balance > 0:
        st.caption(f"At **{required_per_month:,.0f} m/month** from latest data ({last_month}), target is achievable.")

    st.divider()

    # Chart: stacked bars for cumulative (each month = different color)
    st.subheader("Month-wise overall length progress (stacked = cumulative)")
    n_months = len(monthly)
    # Distinct color per month (cycle if more months than colors)
    color_list = px.colors.qualitative.Set3 + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
    month_dates = monthly["MonthDate"].tolist()
    completed = monthly["completed"].tolist()
    cumulative = monthly["Cumulative"].tolist()
    fig = go.Figure()
    for i in range(n_months):
        # At each x, this trace contributes only from month i onward (for stacking)
        y_vals = [0] * i + [completed[i]] * (n_months - i)
        ts = month_dates[i]
        label = ts.strftime("%b %Y") if hasattr(ts, "strftime") else str(ts)
        fig.add_trace(
            go.Bar(
                x=month_dates,
                y=y_vals,
                name=label,
                marker_color=color_list[i % len(color_list)],
                legendgroup=label,
            )
        )
    fig.add_hline(
        y=total_est,
        line_dash="dash",
        line_color="gray",
        annotation_text="Total estimated",
    )
    # Progress percentage on top of each bar
    if total_est and total_est > 0:
        for i in range(n_months):
            pct = cumulative[i] / total_est * 100
            fig.add_annotation(
                x=month_dates[i],
                y=cumulative[i],
                text=f"{pct:.1f}%",
                showarrow=False,
                yanchor="bottom",
                yshift=4,
                font=dict(size=11),
            )
    fig.update_layout(
        height=420,
        margin=dict(t=40),
        xaxis_tickformat="%b %Y",
        xaxis_title="Month",
        yaxis_title="Cumulative length (m)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="stack",
    )
    fig.update_xaxes(tickangle=0)
    st.plotly_chart(fig, use_container_width=True)

    # Optional: show data table
    with st.expander("Month-wise data"):
        show_df = monthly[["Month", "completed", "Cumulative"]].copy()
        show_df.columns = ["Month", "Monthly completed (m)", "Cumulative (m)"]
        st.dataframe(show_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
