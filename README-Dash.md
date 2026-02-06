# Vaigai North Bank Road — Dash Dashboard

Plotly Dash version of the RCC RW dashboard.

## Setup

```bash
pip install -r requirements-dash.txt
```

## Run

```bash
python app_dash.py
```

Open **http://127.0.0.1:8050** in your browser.

## Features

- **KPIs**: Total length, Records, Bills, Sections, Progress %
- **Progress vs Total**: Overall progress bar (Earthwork completed vs total estimated)
- **Work by Item Type**: Bar chart with progress % vs total estimated (dashed line)
- **Timeline by Item**: Line charts per item type (facet grid), filtered by Bill No
- **Filters** (sidebar): Item, Section (Estimate), Bills (multi), Search
- **Detail table**: Filtered records (scrollable, first 500 shown)

Data is loaded from `RCC RW data.xlsx` (sheet "RCC RW") on startup.
