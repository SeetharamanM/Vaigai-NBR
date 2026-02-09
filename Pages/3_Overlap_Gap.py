"""Overlap & Gap Analysis — RCC RW, stretch-wise."""
import re
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "RCC RW data.xlsx"
CSV_PATH = PROJECT_ROOT / "RCC RW data.csv"

TOTAL_LENGTH_M = 8150
EXCLUDED_ESTIMATE_OVERLAP = "RCC RW (0-280) (mid)"


def is_excluded_estimate(estimate_str):
    if pd.isna(estimate_str):
        return False
    s = str(estimate_str).strip()
    return s == EXCLUDED_ESTIMATE_OVERLAP or ("(0-280)" in s and "(mid)" in s.lower())


def parse_stretch(stretch_str):
    if pd.isna(stretch_str) or not stretch_str or str(stretch_str).strip() == "nan":
        return None, None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$", str(stretch_str).strip())
    if not match:
        return None, None
    try:
        start = int(float(match.group(1)))
        end = int(float(match.group(2)))
        if start <= end:
            return start, end
    except (ValueError, TypeError):
        pass
    return None, None


def get_no_wall_ranges(df, max_chainage=TOTAL_LENGTH_M):
    rows = df[
        (df["Item"].astype(str).str.strip().str.lower() == "no wall")
        | (df["Mbook"].astype(str).str.strip().str.lower() == "wall")
    ]
    intervals = []
    for _, row in rows.iterrows():
        start, end = parse_stretch(row.get("Stretch"))
        if start is not None and end is not None:
            start = max(0, min(start, max_chainage))
            end = max(0, min(end, max_chainage))
            if end > start:
                intervals.append((start, end))
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def in_no_wall(m, no_wall_ranges):
    return any(a <= m < b for a, b in no_wall_ranges)


def compute_coverage(intervals, max_chainage, resolution=1):
    n_bins = int(max_chainage // resolution) + 1
    coverage = [0] * n_bins
    for start, end in intervals:
        if start is None or end is None:
            continue
        i0 = max(0, int(start // resolution))
        i1 = min(n_bins, int((end + resolution - 0.5) // resolution))
        for i in range(i0, i1):
            coverage[i] += 1
    chainage_index = list(range(0, n_bins * resolution, resolution))[:n_bins]
    return chainage_index, coverage


def coverage_to_ranges(chainage_index, coverage, resolution, min_count=0, max_count=None):
    if max_count is None and coverage:
        max_count = max(coverage)
    elif max_count is None:
        max_count = 0
    ranges = []
    in_range = False
    start_m = None
    for i, m in enumerate(chainage_index):
        c = coverage[i] if i < len(coverage) else 0
        ok = min_count <= c <= max_count
        if ok and not in_range:
            start_m = m
            in_range = True
        elif not ok and in_range:
            ranges.append((start_m, m))
            in_range = False
    if in_range and chainage_index:
        end_m = chainage_index[-1] + resolution
        ranges.append((start_m, end_m))
    return ranges


def stretch_label(row):
    return f"{row.get('Estimate', '')} | {row.get('Stretch', '')}"


@st.cache_data
def load_data():
    if EXCEL_PATH.exists():
        df = pd.read_excel(EXCEL_PATH, sheet_name="RCC RW", header=0)
    else:
        df = pd.read_csv(CSV_PATH)
    df.columns = [
        "Estimate", "Est_Length", "Bill_No", "Item", "Height",
        "Stretch", "Length", "Mbook", "Pages", "Date",
    ]
    df = df[df["Estimate"].astype(str).str.strip() != "Estimate"].copy()
    df = df.dropna(how="all")
    df["Stretch"] = df["Stretch"].astype(str).str.strip()
    return df


@st.cache_data
def run_overlap_gap_analysis(df, chunk_size=1000, resolution=1):
    seen = set()
    stretches = []
    for _, row in df.iterrows():
        if is_excluded_estimate(row.get("Estimate")):
            continue
        start, end = parse_stretch(row.get("Stretch"))
        if start is None or end is None:
            continue
        key = (str(row.get("Estimate", "")).strip(), str(row.get("Stretch", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        stretches.append((start, end, row.to_dict()))

    if not stretches:
        no_wall_ranges = get_no_wall_ranges(df, TOTAL_LENGTH_M)
        return {
            "chunk_df": pd.DataFrame(columns=["chunk_label", "chunk_start", "chunk_end", "gap_m", "single_m", "overlap_m", "no_wall_m"]),
            "gap_ranges": [],
            "overlap_ranges": [],
            "no_wall_ranges": no_wall_ranges,
            "stretch_details": pd.DataFrame(),
            "stretches": stretches,
            "max_chainage": TOTAL_LENGTH_M,
        }

    max_chainage = TOTAL_LENGTH_M
    no_wall_ranges = get_no_wall_ranges(df, max_chainage)
    intervals = [(s[0], s[1]) for s in stretches]
    chainage_index, coverage = compute_coverage(intervals, max_chainage, resolution)
    n_chunks = int((max_chainage + chunk_size - 1) // chunk_size)

    rows = []
    for c in range(n_chunks):
        chunk_start = c * chunk_size
        chunk_end = min((c + 1) * chunk_size, max_chainage)
        gap_m = single_m = overlap_m = no_wall_m = 0
        for i, m in enumerate(chainage_index):
            if chunk_start <= m < chunk_end:
                if in_no_wall(m, no_wall_ranges):
                    no_wall_m += resolution
                else:
                    cnt = coverage[i]
                    if cnt == 0:
                        gap_m += resolution
                    elif cnt == 1:
                        single_m += resolution
                    else:
                        overlap_m += resolution
        rows.append({
            "chunk_label": f"{chunk_start}-{chunk_end}",
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "gap_m": gap_m,
            "single_m": single_m,
            "overlap_m": overlap_m,
            "no_wall_m": no_wall_m,
            "total_m": gap_m + single_m + overlap_m + no_wall_m,
        })
    chunk_df = pd.DataFrame(rows)
    gap_ranges = coverage_to_ranges(chainage_index, coverage, resolution, min_count=0, max_count=0)
    overlap_ranges_raw = coverage_to_ranges(chainage_index, coverage, resolution, min_count=2, max_count=None)
    overlap_ranges = []
    for (a, b) in overlap_ranges_raw:
        covering = []
        for start, end, row in stretches:
            if start < b and end > a:
                covering.append(stretch_label(row))
        overlap_ranges.append({"start": a, "end": b, "length_m": b - a, "stretches": covering})

    stretch_details = []
    for start, end, row in stretches:
        overlap_m = single_m = 0
        for i, m in enumerate(chainage_index):
            if start <= m < end:
                cnt = coverage[i]
                if cnt >= 2:
                    overlap_m += resolution
                elif cnt == 1:
                    single_m += resolution
        stretch_details.append({
            "Estimate": row.get("Estimate"),
            "Stretch": row.get("Stretch"),
            "Bill_No": row.get("Bill_No"),
            "Start": start,
            "End": end,
            "Length_m": end - start,
            "Single_m": single_m,
            "Overlap_m": overlap_m,
        })

    return {
        "chunk_df": chunk_df,
        "gap_ranges": gap_ranges,
        "overlap_ranges": overlap_ranges,
        "no_wall_ranges": no_wall_ranges,
        "stretch_details": pd.DataFrame(stretch_details),
        "stretches": stretches,
        "max_chainage": max_chainage,
    }


def build_bar_chart(chunk_df, chunk_size=1000):
    if chunk_df.empty:
        return go.Figure()
    df = chunk_df.sort_values("chunk_start", ascending=False).copy()
    no_wall = df["no_wall_m"].values if "no_wall_m" in df.columns else [0] * len(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Gap", y=df["chunk_label"], x=df["gap_m"], orientation="h", marker_color="#dc3545"))
    fig.add_trace(go.Bar(name="Single", y=df["chunk_label"], x=df["single_m"], orientation="h", marker_color="#28a745"))
    fig.add_trace(go.Bar(name="Overlap", y=df["chunk_label"], x=df["overlap_m"], orientation="h", marker_color="#fd7e14"))
    fig.add_trace(go.Bar(name="No wall", y=df["chunk_label"], x=df["no_wall_m"], orientation="h", marker_color="#6c757d"))
    fig.update_layout(
        barmode="stack",
        xaxis_title=f"Chainage (m) — 0 to {chunk_size} m per chunk (max length {TOTAL_LENGTH_M} m)",
        yaxis_title="Km chunk",
        xaxis=dict(range=[0, chunk_size], dtick=100),
        margin=dict(l=100),
        height=max(400, len(df) * 32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Coverage by chunk (Gap | Single | Overlap | No wall) — stretch-wise",
    )
    return fig


def build_gap_over_chainage_chart(gap_ranges, no_wall_ranges, max_chainage, x_min=None, x_max=None):
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = max_chainage if max_chainage > 0 else 1000
    x_min = max(0, min(x_min, max_chainage - 1))
    x_max = max(x_min + 50, min(x_max, max_chainage)) if max_chainage > 0 else x_min + 1000

    fig = go.Figure()
    y_center, y_h = 0.5, 0.35
    for a, b in no_wall_ranges:
        if b <= x_min or a >= x_max:
            continue
        s, e = max(a, x_min), min(b, x_max)
        fig.add_shape(
            type="rect",
            x0=s, x1=e,
            y0=y_center - y_h / 2, y1=y_center + y_h / 2,
            xref="x", yref="y",
            line=dict(width=0),
            fillcolor="rgba(108, 117, 125, 0.85)",
        )
    for a, b in gap_ranges:
        if b <= x_min or a >= x_max:
            continue
        s, e = max(a, x_min), min(b, x_max)
        fig.add_shape(
            type="rect",
            x0=s, x1=e,
            y0=y_center - y_h / 2, y1=y_center + y_h / 2,
            xref="x", yref="y",
            line=dict(width=0),
            fillcolor="rgba(220, 53, 69, 0.85)",
        )
    for a, b in no_wall_ranges:
        if b > x_min and a < x_max:
            fig.add_annotation(
                x=(a + b) / 2, y=y_center - y_h / 2 - 0.08,
                text=f"No wall {a}–{b}", showarrow=False, font=dict(size=9),
                yref="y", xref="x",
            )
    for a, b in gap_ranges:
        if b > x_min and a < x_max:
            fig.add_annotation(
                x=(a + b) / 2, y=y_center + y_h / 2 + 0.12,
                text=f"Gap {a}–{b}", showarrow=False, font=dict(size=10),
                yref="y", xref="x",
            )
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max], dtick=100 if (x_max - x_min) <= 1500 else (200 if (x_max - x_min) <= 3000 else 500), title="Chainage (m)"),
        yaxis=dict(range=[-0.25, 1], showticklabels=False, title=""),
        height=240,
        margin=dict(t=60, b=50, l=60, r=40),
        title=f"Gaps and No wall over chainage (view {x_min}–{x_max} m of 0–{max_chainage} m)",
        plot_bgcolor="rgba(0,0,0,0.03)",
    )
    return fig


# --- UI ---
st.title("Overlap & Gap Analysis — RCC RW")
st.caption(f"Max length **{TOTAL_LENGTH_M} m**. Stretch-wise; no wall excluded from gap/single/overlap calculation. **Excluded:** {EXCLUDED_ESTIMATE_OVERLAP} (different location).")

df_raw = load_data()
st.sidebar.caption(f"**Excluded from overlap:** {EXCLUDED_ESTIMATE_OVERLAP} (different location).")
chunk_size = st.sidebar.number_input("Chunk size (m)", min_value=500, max_value=2000, value=1000, step=100, key="og_chunk")
resolution = st.sidebar.selectbox("Resolution (m)", [1, 5, 10], index=0, key="og_res")

result = run_overlap_gap_analysis(df_raw, chunk_size=chunk_size, resolution=resolution)
chunk_df = result["chunk_df"]
gap_ranges = result["gap_ranges"]
overlap_ranges = result["overlap_ranges"]
no_wall_ranges = result["no_wall_ranges"]
stretch_details = result["stretch_details"]
max_chainage = result["max_chainage"]

if chunk_df.empty:
    st.warning("No valid Stretch data. Ensure Stretch column has values like '500-603'.")
    st.stop()

st.sidebar.caption(f"**Max length:** {TOTAL_LENGTH_M} m. No wall excluded from gap/single/overlap calculation.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total gap (m)", int(chunk_df["gap_m"].sum()))
col2.metric("Total single (m)", int(chunk_df["single_m"].sum()))
col3.metric("Total overlap (m)", int(chunk_df["overlap_m"].sum()))
col4.metric("No wall (m)", int(chunk_df["no_wall_m"].sum()))
col5.metric("Total chainage (m)", TOTAL_LENGTH_M)

st.subheader("Gaps and No wall over chainage")
view_start, view_end = st.slider(
    "Chainage range to view (m)",
    min_value=0,
    max_value=max(1, max_chainage),
    value=(0, min(2000, max(1, max_chainage))),
    step=100,
    key="og_gap_slider",
)
gap_fig = build_gap_over_chainage_chart(gap_ranges, no_wall_ranges, max_chainage, x_min=view_start, x_max=view_end)
st.plotly_chart(gap_fig, use_container_width=True)

st.subheader("Coverage by km chunk")
fig = build_bar_chart(chunk_df, chunk_size=chunk_size)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Gap ranges (stretch-wise)")
if not gap_ranges:
    st.info("No gaps: every chainage meter is covered by at least one stretch.")
else:
    st.dataframe(
        pd.DataFrame([{"Start (m)": a, "End (m)": b, "Length (m)": b - a} for a, b in gap_ranges]),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("No wall portions (excluded from calculation)")
if not no_wall_ranges:
    st.caption("No chainage marked as No wall in the data.")
else:
    st.dataframe(
        pd.DataFrame([{"Start (m)": a, "End (m)": b, "Length (m)": b - a} for a, b in no_wall_ranges]),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Overlap ranges (stretch-wise) — which stretches cover each")
if not overlap_ranges:
    st.info("No overlaps: no chainage is covered by more than one stretch.")
else:
    for r in overlap_ranges:
        with st.expander(f"Chainage **{r['start']}–{r['end']} m** ({r['length_m']} m) — {len(r['stretches'])} stretches"):
            for s in r["stretches"]:
                st.text(s)
    st.dataframe(
        pd.DataFrame([{"Start (m)": r["start"], "End (m)": r["end"], "Length (m)": r["length_m"], "Stretches": len(r["stretches"])} for r in overlap_ranges]),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Per-stretch breakdown (stretch-wise, not item-wise)")
sd = stretch_details.rename(columns={"Length_m": "Length (m)", "Single_m": "Single (m)", "Overlap_m": "Overlap (m)"})
for col in ["Length (m)", "Single (m)", "Overlap (m)"]:
    if col in sd.columns:
        sd[col] = sd[col].astype(int)
st.dataframe(sd, use_container_width=True, hide_index=True)
