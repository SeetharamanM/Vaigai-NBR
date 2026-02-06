"""
Vaigai North Bank Road Project — Overlap & Gap Analysis (Streamlit)
Stretches only: one per unique (Estimate, Stretch); gaps and overlaps by chainage.
Run: streamlit run app_overlap_gap.py
"""
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

EXCEL_PATH = Path(__file__).parent / "RCC RW data.xlsx"
CSV_PATH = Path(__file__).parent / "RCC RW data.csv"

# Total chainage length (m) for analysis; chainage beyond this is not considered.
TOTAL_LENGTH_M = 8150

# Estimate treated as separate location: its stretches are excluded from overlap/gap coverage
# (overlaps with this estimate are not counted).
MID_ESTIMATE_KEY = "RCC RW (0-280) (mid)"


def is_mid_estimate(estimate_str):
    """True if this estimate is the separate-location (0-280) mid estimate."""
    if pd.isna(estimate_str):
        return False
    s = str(estimate_str).strip()
    return s == MID_ESTIMATE_KEY or "(0-280)" in s and "(mid)" in s.lower()


def is_no_wall(row_dict):
    """True if this stretch is marked as no wall (separate location / no RCC wall)."""
    item = str(row_dict.get("Item", "") or "").strip().lower()
    mbook = str(row_dict.get("Mbook", "") or "").strip().lower()
    return item == "no wall" or mbook == "wall"


def merge_intervals(intervals, max_end):
    """Merge overlapping (start, end) intervals and clip to [0, max_end]. Returns sorted list."""
    if not intervals:
        return []
    sorted_i = sorted([(max(0, s), min(e, max_end)) for s, e in intervals if s < max_end and e > 0])
    if not sorted_i:
        return []
    out = [list(sorted_i[0])]
    for s, e in sorted_i[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


st.set_page_config(
    page_title="Overlap & Gap Analysis — RCC RW",
    page_icon="📊",
    layout="wide",
)

st.title("Overlap & Gap Analysis — RCC RW")
st.caption(
    "Total length **8150 m**. Stretches only: one per unique (Estimate, Stretch). "
    "**RCC RW (0-280) (mid)** is a separate location — overlaps with it are not counted. "
    "Gaps = no stretch; Overlaps = 2+ stretches; **No wall** = chainage marked as no wall (data)."
)


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
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df["Stretch"] = df["Stretch"].astype(str).str.strip()
    return df


def parse_stretch(stretch_str):
    """Parse 'start-end' to (start, end) in meters. Returns (None, None) if invalid."""
    if pd.isna(stretch_str) or not stretch_str or stretch_str == "nan":
        return None, None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$", stretch_str.strip())
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


def compute_coverage_by_meter(intervals, max_chainage, resolution=1):
    """
    For each meter (or resolution step) from 0 to max_chainage, count how many
    intervals cover it. Returns arrays: chainage_index, coverage_count.
    """
    n_bins = int(max_chainage // resolution) + 1
    coverage = [0] * n_bins
    for start, end in intervals:
        if start is None or end is None:
            continue
        i0 = max(0, int(start // resolution))
        i1 = min(n_bins, int((end + resolution - 0.5) // resolution))
        for i in range(i0, i1):
            coverage[i] += 1
    return list(range(0, n_bins * resolution, resolution))[:n_bins], coverage


def chunk_label(chunk_start, chunk_end):
    return f"{chunk_start}-{chunk_end}"


def coverage_to_ranges(chainage_index, coverage, resolution, min_count=0, max_count=None):
    """Merge coverage array into contiguous ranges where count in [min_count, max_count] (inclusive)."""
    if max_count is None:
        max_count = max(coverage) if coverage else 0
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
            ranges.append((start_m, m))  # end exclusive at m
            in_range = False
    if in_range:
        end_m = chainage_index[-1] + resolution if chainage_index else 0
        ranges.append((start_m, end_m))
    return ranges


def stretch_label(row):
    return f"{row.get('Estimate', '')} | {row.get('Stretch', '')} | {row.get('Item', '')}"


@st.cache_data
def run_overlap_gap_analysis(df, chunk_size=1000, resolution=1):
    """
    Stretch-only: one interval per unique (Estimate, Stretch). Overlaps = chainage covered by
    2+ distinct stretches; gaps = chainage with no stretch. Total length capped at TOTAL_LENGTH_M.
    Stretches in RCC RW (0-280) (mid) are excluded from coverage (separate location — overlap with them not considered).
    """
    seen = set()  # (Estimate, Stretch) for deduplication
    stretches = []  # (start, end, row_dict) — unique stretches only
    for _, row in df.iterrows():
        start, end = parse_stretch(row.get("Stretch"))
        if start is None or end is None:
            continue
        key = (str(row.get("Estimate", "")).strip(), str(row.get("Stretch", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        stretches.append((start, end, row.to_dict()))

    if not stretches:
        return {
            "chunk_df": pd.DataFrame(columns=["chunk_label", "chunk_start", "chunk_end", "gap_m", "single_m", "overlap_m"]),
            "gap_ranges": [],
            "overlap_ranges": [],
            "no_wall_ranges": [],
            "chainage_segments": [],
            "stretch_details": pd.DataFrame(),
            "stretches": [],
        }

    # Intervals for coverage: exclude (0-280) mid estimate (separate location; overlap with it not considered)
    intervals_for_coverage = [
        (s[0], s[1]) for s in stretches
        if not is_mid_estimate(s[2].get("Estimate"))
    ]
    # Total chainage fixed at TOTAL_LENGTH_M (8150 m)
    max_chainage = TOTAL_LENGTH_M
    n_chunks = (max_chainage + chunk_size - 1) // chunk_size

    chainage_index, coverage = compute_coverage_by_meter(intervals_for_coverage, max_chainage, resolution)

    # Chunk summary (last chunk ends at max_chainage, not beyond)
    rows = []
    for c in range(n_chunks):
        chunk_start = c * chunk_size
        chunk_end = min((c + 1) * chunk_size, max_chainage)
        gap_m = single_m = overlap_m = 0
        for i, m in enumerate(chainage_index):
            if chunk_start <= m < chunk_end:
                cnt = coverage[i]
                if cnt == 0:
                    gap_m += resolution
                elif cnt == 1:
                    single_m += resolution
                else:
                    overlap_m += resolution
        rows.append({
            "chunk_label": chunk_label(chunk_start, chunk_end),
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "gap_m": gap_m,
            "single_m": single_m,
            "overlap_m": overlap_m,
            "total_m": gap_m + single_m + overlap_m,
        })
    chunk_df = pd.DataFrame(rows)

    # Gap ranges (stretch-wise: chainage with no stretch)
    gap_ranges = coverage_to_ranges(chainage_index, coverage, resolution, min_count=0, max_count=0)

    # Overlap ranges (chainage with 2+ stretches) and which stretches cover each
    overlap_ranges_raw = coverage_to_ranges(chainage_index, coverage, resolution, min_count=2, max_count=None)
    overlap_ranges = []
    for (a, b) in overlap_ranges_raw:
        covering = []
        for start, end, row in stretches:
            if start < b and end > a:  # overlaps [a, b)
                covering.append(stretch_label(row))
        overlap_ranges.append({"start": a, "end": b, "length_m": b - a, "stretches": covering})

    # Per-stretch: how many m of each stretch are in overlap vs single
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
    stretch_details_df = pd.DataFrame(stretch_details)

    # No-wall ranges (from data: Item "No wall" or Mbook "wall"); clip to max_chainage
    no_wall_ranges = [
        (s[0], s[1]) for s in stretches
        if is_no_wall(s[2])
    ]
    no_wall_ranges = merge_intervals(no_wall_ranges, max_chainage)

    # Per-meter segment type for strip chart: no_wall | gap | single | overlap
    n_bins = len(chainage_index)
    segment_type = []
    for i in range(n_bins):
        m = chainage_index[i]
        in_no_wall = any(a <= m < b for a, b in no_wall_ranges)
        if in_no_wall:
            segment_type.append("no_wall")
        else:
            c = coverage[i] if i < len(coverage) else 0
            if c == 0:
                segment_type.append("gap")
            elif c == 1:
                segment_type.append("single")
            else:
                segment_type.append("overlap")

    # Merge consecutive same type into (start, end, type) for drawing
    chainage_segments = []
    i = 0
    while i < n_bins:
        t = segment_type[i]
        start_m = chainage_index[i]
        j = i
        while j < n_bins and segment_type[j] == t:
            j += 1
        end_m = chainage_index[j - 1] + resolution if j > 0 else start_m
        if j < n_bins:
            end_m = chainage_index[j]  # exclusive end
        else:
            end_m = min(chainage_index[-1] + resolution, max_chainage)
        chainage_segments.append((start_m, end_m, t))
        i = j

    return {
        "chunk_df": chunk_df,
        "gap_ranges": gap_ranges,
        "overlap_ranges": overlap_ranges,
        "no_wall_ranges": no_wall_ranges,
        "chainage_segments": chainage_segments,
        "stretch_details": stretch_details_df,
        "stretches": stretches,
    }


def build_bar_chart(chunk_df, chunk_size=1000):
    """
    Horizontal bar: y = km chunks (0-1000, 1000-2000, ...), x = length in meters (0 to chunk_size).
    Stacked: Gap (red), Single (green), Overlap (orange).
    """
    if chunk_df.empty:
        return go.Figure()

    # Reverse so 0-1000 is at top
    df = chunk_df.sort_values("chunk_start", ascending=False).copy()

    fig = go.Figure()

    # Stacked horizontal bar: x = [gap, single, overlap] per chunk
    fig.add_trace(
        go.Bar(
            name="Gap",
            y=df["chunk_label"],
            x=df["gap_m"],
            orientation="h",
            marker_color="rgba(220, 53, 69, 0.85)",
            text=df["gap_m"].astype(int),
            textposition="inside",
            texttemplate="%{text}m",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Single coverage",
            y=df["chunk_label"],
            x=df["single_m"],
            orientation="h",
            marker_color="rgba(40, 167, 69, 0.85)",
            text=df["single_m"].astype(int),
            textposition="inside",
            texttemplate="%{text}m",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Overlap",
            y=df["chunk_label"],
            x=df["overlap_m"],
            orientation="h",
            marker_color="rgba(253, 126, 20, 0.85)",
            text=df["overlap_m"].astype(int),
            textposition="inside",
            texttemplate="%{text}m",
        )
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Chainage (m) — 0 to {} m per chunk".format(chunk_size),
        yaxis_title="Km chunk",
        xaxis=dict(range=[0, chunk_size], dtick=100),
        margin=dict(l=100),
        height=max(400, len(df) * 32),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Coverage by chainage chunk (Gap | Single | Overlap)",
    )
    return fig


# Default visible chainage window (m) for the strip chart.
CHAINAGE_STRIP_WINDOW = 1200


def build_chainage_strip_chart(chainage_segments, max_chainage, gap_ranges, no_wall_ranges, x_min=None, x_max=None):
    """
    One horizontal strip: x = chainage 0 to max_chainage. Colored segments: Gap, No wall, Single, Overlap.
    Annotations above the strip for gap and no-wall chainages. x_min, x_max = visible window (use slider to pan).
    """
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = min(CHAINAGE_STRIP_WINDOW, max_chainage)
    x_min = max(0, min(x_min, max_chainage - 1))
    x_max = max(x_min + 100, min(x_max, max_chainage))

    fig = go.Figure()
    colors = {"gap": "rgba(220, 53, 69, 0.85)", "no_wall": "rgba(108, 117, 125, 0.85)", "single": "rgba(40, 167, 69, 0.85)", "overlap": "rgba(253, 126, 20, 0.85)"}
    y_center = 0.5
    y_h = 0.4  # strip height in y units

    for start, end, seg_type in chainage_segments:
        fig.add_shape(
            type="rect",
            x0=start, x1=end, y0=y_center - y_h / 2, y1=y_center + y_h / 2,
            xref="x", yref="y",
            line=dict(width=0),
            fillcolor=colors.get(seg_type, "gray"),
        )

    # Annotations above strip (only for ranges that overlap visible window)
    annots = []
    for a, b in gap_ranges:
        if b > x_min and a < x_max:
            annots.append(dict(x=(a + b) / 2, y=y_center + y_h / 2 + 0.15, text=f"Gap {a}–{b}", showarrow=False, font=dict(size=10), yref="y", xref="x"))
    for a, b in no_wall_ranges:
        if b > x_min and a < x_max:
            annots.append(dict(x=(a + b) / 2, y=y_center + y_h / 2 + 0.15, text=f"No wall {a}–{b}", showarrow=False, font=dict(size=10), yref="y", xref="x"))

    # Visible range from slider; dtick scales with window size
    window = x_max - x_min
    dtick = 100 if window <= 1500 else (200 if window <= 3000 else 500)
    fig.update_layout(
        xaxis=dict(
            range=[x_min, x_max],
            dtick=dtick,
            title="Chainage (m)",
        ),
        yaxis=dict(range=[-0.2, 1], showticklabels=False, title=""),
        showlegend=True,
        height=280,
        margin=dict(t=70, b=50, l=60, r=40),
        title="Gaps and No-wall along chainage (view: {}–{} m of 0–{} m)".format(x_min, x_max, max_chainage),
        annotations=annots,
        plot_bgcolor="rgba(0,0,0,0.02)",
    )
    # Legend via dummy traces
    for label, seg_type in [("Gap", "gap"), ("No wall", "no_wall"), ("Single coverage", "single"), ("Overlap", "overlap")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, marker=dict(size=12, color=colors.get(seg_type, "gray"), symbol="square")))
    return fig


# --- UI ---
df_raw = load_data()
st.sidebar.caption(f"**Total length:** {TOTAL_LENGTH_M} m. **Excluded from overlap:** {MID_ESTIMATE_KEY}.")
chunk_size = st.sidebar.number_input("Chunk size (m)", min_value=500, max_value=2000, value=1000, step=100)
resolution = st.sidebar.selectbox("Resolution (m)", [1, 5, 10], index=0, help="Meter step for gap/overlap computation")

result = run_overlap_gap_analysis(df_raw, chunk_size=chunk_size, resolution=resolution)
chunk_df = result["chunk_df"]
gap_ranges = result["gap_ranges"]
overlap_ranges = result["overlap_ranges"]
no_wall_ranges = result["no_wall_ranges"]
chainage_segments = result["chainage_segments"]
stretch_details = result["stretch_details"]

if chunk_df.empty:
    st.warning("No valid Stretch data found. Ensure Stretch column has values like '500-603'.")
    st.stop()

# Summary metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_gap = int(chunk_df["gap_m"].sum())
    st.metric("Total gap (m)", total_gap)
with col2:
    total_single = int(chunk_df["single_m"].sum())
    st.metric("Total single coverage (m)", total_single)
with col3:
    total_overlap = int(chunk_df["overlap_m"].sum())
    st.metric("Total overlap (m)", total_overlap)
with col4:
    total_chainage = int(chunk_df["total_m"].sum())
    st.metric("Total chainage in chunks (m)", total_chainage)
with col5:
    total_no_wall = sum(b - a for a, b in no_wall_ranges)
    st.metric("No wall (m)", total_no_wall)

# Chainage strip: gaps and no-wall directly above chainage (slider to pan)
st.subheader("Gaps and No-wall along chainage")
view_start, view_end = st.slider(
    "Chainage range to view (m) — drag to pan along 0–{} m".format(TOTAL_LENGTH_M),
    min_value=0,
    max_value=TOTAL_LENGTH_M,
    value=(0, min(CHAINAGE_STRIP_WINDOW, TOTAL_LENGTH_M)),
    step=100,
    help="Select start and end chainage to zoom; chart shows this segment clearly.",
)
strip_fig = build_chainage_strip_chart(
    chainage_segments, TOTAL_LENGTH_M, gap_ranges, no_wall_ranges,
    x_min=view_start, x_max=view_end,
)
st.plotly_chart(strip_fig, use_container_width=True)

# Bar chart: x = 0–chunk_size m, y = km chunks
fig = build_bar_chart(chunk_df, chunk_size=chunk_size)
st.plotly_chart(fig, use_container_width=True)

# --- No wall ranges ---
st.subheader("No wall portions (chainage)")
if not no_wall_ranges:
    st.caption("No chainage marked as 'No wall' in the data.")
else:
    no_wall_table = pd.DataFrame(
        [{"Chainage start (m)": a, "Chainage end (m)": b, "Length (m)": b - a} for a, b in no_wall_ranges]
    )
    st.dataframe(no_wall_table, use_container_width=True, hide_index=True)

# --- Stretch-wise: Gap ranges (chainage with no stretch) ---
st.subheader("Gap ranges (stretch-wise)")
if not gap_ranges:
    st.info("No gaps: every chainage meter is covered by at least one stretch.")
else:
    gap_table = pd.DataFrame(
        [{"Chainage start (m)": a, "Chainage end (m)": b, "Length (m)": b - a} for a, b in gap_ranges]
    )
    st.dataframe(gap_table, use_container_width=True, hide_index=True)

# --- Stretch-wise: Overlap ranges and which stretches cover each ---
st.subheader("Overlap ranges (stretch-wise) — which stretches cover each")
if not overlap_ranges:
    st.info("No overlaps: no chainage is covered by more than one stretch.")
else:
    for r in overlap_ranges:
        with st.expander(f"Chainage **{r['start']}–{r['end']} m** (length {r['length_m']} m) — {len(r['stretches'])} stretches"):
            for s in r["stretches"]:
                st.text(s)
    overlap_table = pd.DataFrame(
        [
            {
                "Chainage start (m)": r["start"],
                "Chainage end (m)": r["end"],
                "Length (m)": r["length_m"],
                "Stretches (count)": len(r["stretches"]),
            }
            for r in overlap_ranges
        ]
    )
    st.dataframe(overlap_table, use_container_width=True, hide_index=True)

# --- Per-stretch: overlap vs single portion ---
st.subheader("Per-stretch breakdown (overlap vs single coverage)")
sd_display = stretch_details.rename(
    columns={"Length_m": "Length (m)", "Single_m": "Single (m)", "Overlap_m": "Overlap (m)"}
).copy()
for col in ["Length (m)", "Single (m)", "Overlap (m)"]:
    if col in sd_display.columns:
        sd_display[col] = sd_display[col].astype(int)
st.dataframe(sd_display, use_container_width=True, hide_index=True)

# Per-chunk summary table
st.subheader("Per-chunk breakdown")
chunk_display = chunk_df[["chunk_label", "gap_m", "single_m", "overlap_m", "total_m"]].rename(
    columns={"gap_m": "Gap (m)", "single_m": "Single (m)", "overlap_m": "Overlap (m)", "total_m": "Total (m)"}
)
for col in ["Gap (m)", "Single (m)", "Overlap (m)", "Total (m)"]:
    chunk_display[col] = chunk_display[col].astype(int)
st.dataframe(chunk_display, use_container_width=True, hide_index=True)
