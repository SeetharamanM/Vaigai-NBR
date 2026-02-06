# Mbook & Pages — RCC RW App

Streamlit app for the **Vaigai North Bank Road Project**: Mbook & Pages analysis and timeline by estimate for the RCC Retaining Wall (RW) data.

---

## Overview

This app lets you:

- View **Mbook** summary and **Items** detail (Mbook, Pages, Date) for the RCC RW project.
- Filter by **Estimate** and **Bill No**.
- View **timeline charts** by estimate (date on x-axis, stretches on y-axis, square markers).
- Download the filtered data as CSV.

---

## Data source

- **Primary:** `RCC RW data.xlsx` (sheet **RCC RW**).
- **Fallback:** `RCC RW data.csv` if the Excel file is not present.

Expected columns: `Estimate`, `Est_Length`, `Bill_No`, `Item`, `Height`, `Stretch`, `Length`, `Mbook`, `Pages`, `Date`.

---

## How to run

```bash
cd "e:\OneDrive\Cursor\RCC RW dashboard"
streamlit run app_mbook.py
```

Then open the URL shown in the terminal (e.g. **http://localhost:8501**).

**Requirements:** `streamlit`, `pandas`, `plotly`. If using Excel: `openpyxl`.

---

## Features

### 1. Sidebar filters

| Filter    | Type        | Description                                      |
|----------|-------------|--------------------------------------------------|
| Estimate | Select box  | One estimate or **All**.                         |
| Bill No  | Multi-select| One or more bills; empty = all bills.            |

All sections (Mbook summary, table, timeline) use the same filtered dataset.

### 2. Summary by Mbook

Table with:

- **Mbook** — Mbook id.
- **Items** — Count of item rows.
- **Date range** — Min and max date for that Mbook.
- **Pages (sample)** — Sample pages value.

### 3. Items: Mbook, Pages, Date

Detail table with: **Estimate**, **Bill_No**, **Item**, **Stretch**, **Length**, **Mbook**, **Pages**, **Date** (formatted).

### 4. Timeline by estimate

- Uses only rows with valid **Length** and **Date** from **2024 onward**.
- **View:**
  - **All estimates (expandable)** — One expander per estimate with a chart inside.
  - **Single estimate** — Dropdown to pick one estimate, then one chart.
- **Chart:**
  - **X-axis:** Date.
  - **Y-axis:** Stretch.
  - **Color:** Item (e.g. Earthwork, PCC, Steel).
  - **Markers:** Square, size 16; hover shows Length (m).
- If no 2024+ data remains after filtering, a short message is shown instead of charts.

### 5. Download

- **Download filtered data (CSV)** — Exports the current filtered table as `rcc_rw_mbook_filtered.csv`.

---

## File

- **App script:** `app_mbook.py`
- **Docs:** `README-Mbook.md` (this file)

---

## Project

Vaigai North Bank Road Project — RCC Retaining Wall (RW).
