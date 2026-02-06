"""
Vaigai North Bank Road Project — RCC RW Dashboard (Plotly Dash)
Run: python app_dash.py
"""
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

EXCEL_PATH = Path(__file__).parent / "RCC RW data.xlsx"

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Vaigai North Bank Road — RCC RW"


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
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


df_full = load_data()
valid_full = df_full.dropna(subset=["Length"]).copy()
valid_full = valid_full[valid_full["Item"].notna() & (valid_full["Item"] != "Item")]
valid_full["YearMonth"] = valid_full["Date"].dt.to_period("M").astype(str)
valid_dates_full = valid_full[valid_full["Date"].notna() & (valid_full["Date"].dt.year >= 2024)].copy()
valid_dates_full["MonthDate"] = valid_dates_full["Date"].dt.to_period("M").dt.to_timestamp()

total_est = valid_full.groupby("Estimate")["Est_Length"].first().sum()
total_completed_linear = valid_full[valid_full["Item"] == "Earthwork"]["Length"].sum()
progress_pct = (total_completed_linear / total_est * 100) if total_est and total_est > 0 else 0
total_length = valid_full["Length"].sum()
n_records = len(valid_full)
bill_series = valid_full["Bill_No"].dropna()
bill_series = bill_series[bill_series.astype(str).str.contains("Bill", na=False)]
unique_bills = bill_series.nunique()
n_sections = valid_full["Estimate"].nunique()

items_list = ["All"] + sorted(valid_full["Item"].dropna().unique().tolist())
estimates_list = ["All"] + sorted(valid_full["Estimate"].dropna().unique().tolist())
bill_list = sorted(
    valid_full["Bill_No"].dropna()
    .loc[valid_full["Bill_No"].astype(str).str.contains("Bill", na=False)]
    .unique()
    .tolist(),
    key=lambda x: str(x),
)


def make_kpi_card(label, value, id_suffix=""):
    return dbc.Card(
        dbc.CardBody([html.H6(label, className="text-muted"), html.H4(value, className="mb-0")]),
        className="shadow-sm",
        id=f"kpi-{id_suffix}" if id_suffix else None,
    )


def build_item_chart(valid):
    by_item = valid.groupby("Item", dropna=False).agg(completed=("Length", "sum")).reset_index()
    by_item = by_item[by_item["Item"].notna() & (by_item["Item"] != "Item")]
    total_est_local = valid.groupby("Estimate")["Est_Length"].first().sum()
    by_item["progress_pct"] = (by_item["completed"] / total_est_local * 100).round(1) if total_est_local and total_est_local > 0 else 0
    fig = px.bar(by_item, x="completed", y="Item", orientation="h", text="progress_pct", title="Work by Item Type — Progress vs Total")
    fig.add_vline(x=float(total_est_local), line_dash="dash", line_color="gray", annotation_text="Total estimated")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=20), xaxis_title="Length (m)", height=340)
    return fig


def build_timeline_chart(valid_dates, filter_bills):
    timeline_df = valid_dates.copy()
    if filter_bills:
        timeline_df = timeline_df[timeline_df["Bill_No"].isin(filter_bills)]
    if len(timeline_df) == 0:
        return go.Figure().add_annotation(text="No data for selected bills", x=0.5, y=0.5, showarrow=False)
    by_month_item = (
        timeline_df.groupby(["MonthDate", "Item"], dropna=False)
        .agg(total_length=("Length", "sum"))
        .reset_index()
    )
    by_month_item = by_month_item[by_month_item["Item"].notna() & (by_month_item["Item"] != "Item")]
    by_month_item = by_month_item.sort_values("MonthDate")
    if len(by_month_item) == 0:
        return go.Figure().add_annotation(text="No item breakdown", x=0.5, y=0.5, showarrow=False)
    fig = px.line(
        by_month_item, x="MonthDate", y="total_length", color="Item",
        facet_col="Item", facet_col_wrap=3,
        title="Timeline by Item (filter by Bill No)",
        labels={"MonthDate": "Month", "total_length": "Length (m)", "Item": "Item"},
    )
    fig.update_layout(height=500, margin=dict(t=40))
    fig.update_xaxes(tickformat="%b %Y")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] if "=" in a.text else a.text))
    return fig


# Initial figures
fig_item_init = build_item_chart(valid_full)
fig_timeline_init = build_timeline_chart(valid_dates_full, [])

app.layout = dbc.Container(
    [
        dbc.Row([
            html.H1("Vaigai North Bank Road Project", className="mt-3"),
            html.P("RCC Retaining Wall (RW) — Progress & Bill-wise Data", className="text-muted"),
        ]),
        dbc.Row([
            dbc.Col(make_kpi_card("Total Length (m)", f"{total_length:,.0f}", "length"), md=2),
            dbc.Col(make_kpi_card("Records", f"{n_records:,}", "records"), md=2),
            dbc.Col(make_kpi_card("Bills", unique_bills, "bills"), md=2),
            dbc.Col(make_kpi_card("Sections", n_sections, "sections"), md=2),
            dbc.Col(make_kpi_card("Progress", f"{progress_pct:.1f}%", "progress"), md=2),
        ], className="g-3 mb-3"),
        html.Hr(),
        dbc.Row([
            html.H5("Progress vs Total"),
            html.P(f"Completed: {total_completed_linear:,.0f} m (Earthwork) / Total estimated: {total_est:,.0f} m", className="text-muted small"),
            dbc.Progress(value=min(progress_pct, 100), label=f"{progress_pct:.1f}%", style={"height": "24px"}),
        ], className="mb-4"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H5("Filters"),
                html.Label("Item", className="form-label"),
                dcc.Dropdown(id="filter-item", options=[{"label": x, "value": x} for x in items_list], value="All", clearable=False),
                html.Label("Section (Estimate)", className="form-label mt-2"),
                dcc.Dropdown(id="filter-estimate", options=[{"label": x, "value": x} for x in estimates_list], value="All", clearable=False),
                html.Label("Filter by Bills (timeline & table)", className="form-label mt-2"),
                dcc.Dropdown(id="filter-bills", options=[{"label": x, "value": x} for x in bill_list], value=[], multi=True),
                html.Label("Search", className="form-label mt-2"),
                dcc.Input(id="filter-search", type="text", placeholder="Stretch, Bill, Mbook...", className="form-control"),
            ], md=2, className="border-end"),
            dbc.Col([
                dcc.Graph(id="graph-item", figure=fig_item_init, config={"responsive": True}),
                html.H5("Timeline by Item (filter by Bill No)", className="mt-4"),
                dcc.Graph(id="graph-timeline", figure=fig_timeline_init, config={"responsive": True}),
                html.H5("Detail Records", className="mt-4"),
                html.Div(id="table-container"),
            ], md=10),
        ], className="g-3"),
    ],
    fluid=True,
    className="py-3",
)


@callback(
    [Output("graph-item", "figure"), Output("graph-timeline", "figure"), Output("table-container", "children")],
    [Input("filter-item", "value"), Input("filter-estimate", "value"), Input("filter-bills", "value"), Input("filter-search", "value")],
)
def update_dashboard(filter_item, filter_estimate, filter_bills, search):
    filter_bills = filter_bills or []
    df = df_full.copy()
    if filter_bills:
        df = df[df["Bill_No"].isin(filter_bills)]
    valid_dates = valid_dates_full.copy()
    if filter_bills:
        valid_dates = valid_dates[valid_dates["Bill_No"].isin(filter_bills)]

    # Work by Item Type always uses full data (progress vs total)
    fig_item = build_item_chart(valid_full)
    fig_timeline = build_timeline_chart(valid_dates, filter_bills)

    table_df = df.copy()
    if filter_item and filter_item != "All":
        table_df = table_df[table_df["Item"] == filter_item]
    if filter_estimate and filter_estimate != "All":
        table_df = table_df[table_df["Estimate"] == filter_estimate]
    if search and search.strip():
        q = search.strip().lower()
        mask = (
            table_df["Stretch"].astype(str).str.lower().str.contains(q, na=False)
            | table_df["Bill_No"].astype(str).str.lower().str.contains(q, na=False)
            | table_df["Mbook"].astype(str).str.lower().str.contains(q, na=False)
            | table_df["Pages"].astype(str).str.lower().str.contains(q, na=False)
        )
        table_df = table_df[mask]
    table_df = table_df.copy()
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    table_df = table_df.fillna("")

    table_df_display = table_df.head(500)
    table = dbc.Table.from_dataframe(table_df_display, striped=True, bordered=True, hover=True, size="sm", responsive=True)
    wrapper = html.Div(table, style={"maxHeight": "400px", "overflowY": "auto"})
    if len(table_df) > 500:
        wrapper = html.Div([wrapper, html.P(f"Showing first 500 of {len(table_df)} records.", className="text-muted small mt-2")])

    return fig_item, fig_timeline, wrapper


if __name__ == "__main__":
    app.run(debug=True, port=8050)
