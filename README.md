# Vaigai North Bank Road — RCC RW

Analytics dashboard for the RCC Retaining Wall (RW) project.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. **Sign in**: Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app**: Click **Create app**.
3. **Configure**:
   - **Repository**: `SeetharamanM/Vaigai-NBR`
   - **Branch**: `main`
   - **Main file path**: **`streamlit_app.py`** (required — using `Home.py` will show only the home page with no sidebar navigation)
4. **Deploy**: Click **Deploy**. The app will be available at `https://[your-app].streamlit.app` within a few minutes.

## Data

Uses `RCC RW data.xlsx` (sheet: RCC RW) and `RCC RW data.csv` from the repository.
