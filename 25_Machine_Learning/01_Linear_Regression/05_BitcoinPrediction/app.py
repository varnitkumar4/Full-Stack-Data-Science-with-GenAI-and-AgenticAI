from pathlib import Path
from datetime import date, timedelta
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# BTC FORECAST AI — Streamlit dashboard
# Keep app.py, crypto_data.csv, and both .pkl files together.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "crypto_08_OCT_2026.csv"
NEXT_DAY_MODEL_PATH = BASE_DIR / "btc_next_day_lr.pkl"
NEXT_7DAY_MODEL_PATH = BASE_DIR / "btc_next_7day_mlp.pkl"

st.set_page_config(
    page_title="BTC Forecast AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Styling --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --bg:#0b1020; --panel:#111a2e; --line:rgba(148,163,184,.16); --muted:#94a3b8; --text:#edf2ff; --purple:#8b5cf6; --cyan:#22d3ee; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background: radial-gradient(ellipse at 10% 0%, rgba(91,33,182,.16), transparent 35%), #0b1020; color:var(--text); }
    [data-testid="stHeader"] { background:rgba(11,16,32,.82); }
    [data-testid="stSidebar"] { background:#0d1426; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:#e5e7eb; }
    .block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1500px; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
    .hero { border:1px solid rgba(139,92,246,.28); background:linear-gradient(120deg,rgba(91,33,182,.23),rgba(17,26,46,.92) 55%,rgba(34,211,238,.08)); padding:26px 28px; border-radius:22px; margin-bottom:22px; }
    .eyebrow { color:#a78bfa; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:700; line-height:1.15; margin:.35rem 0 .55rem; color:#f8fafc; }
    .hero-sub { color:#aab7cf; font-size:.98rem; max-width:780px; }
    .pill { display:inline-block; border:1px solid rgba(34,211,238,.3); color:#67e8f9; background:rgba(34,211,238,.08); border-radius:999px; padding:5px 10px; font-size:.75rem; margin:0 5px 0 0; }
    .metric-card { background:linear-gradient(145deg,rgba(20,31,54,.96),rgba(14,22,39,.96)); border:1px solid var(--line); border-radius:16px; padding:17px 18px; min-height:115px; }
    .metric-label { color:#9baac4; font-size:.82rem; margin-bottom:8px; }
    .metric-value { color:#f8fafc; font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; line-height:1.25; }
    .metric-note { color:#7dd3fc; font-size:.75rem; margin-top:7px; }
    .section-head { margin:1.1rem 0 .65rem; }
    .subtle { color:#94a3b8; font-size:.88rem; }
    .stButton > button[kind="primary"] { background:linear-gradient(90deg,#7c3aed,#6366f1); border:0; border-radius:10px; font-weight:700; min-height:2.8rem; }
    .stButton > button { border-radius:10px; }
    div[data-testid="stForm"] { background:rgba(17,26,46,.7); border:1px solid var(--line); border-radius:16px; padding:18px; }
    div[data-testid="stAlert"] { border-radius:12px; }
    .footer { border-top:1px solid var(--line); margin-top:28px; padding-top:14px; color:#75839d; font-size:.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Helpers --------------------------
@st.cache_data(show_spinner=False)
def load_data(path_string: str, modified_time: float) -> pd.DataFrame:
    df = pd.read_csv(path_string)
    return df


@st.cache_resource(show_spinner=False)
def load_pickle(path_string: str, modified_time: float):
    with open(path_string, "rb") as f:
        return pickle.load(f)


def find_column(df: pd.DataFrame, candidates):
    normalized = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    # fallback: match if a column contains all meaningful words
    for c in df.columns:
        low = str(c).lower().replace("_", " ")
        if any(candidate.lower() in low for candidate in candidates):
            return c
    return None


def money(value):
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"${float(value):,.2f}"


def make_metric(label, value, note=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def plot_price_chart(df, date_col, close_col, volume_col=None, title="Bitcoin price history"):
    plot_df = df.copy()
    if date_col:
        plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce", utc=True)
        plot_df = plot_df.dropna(subset=[date_col]).sort_values(date_col)
        x = plot_df[date_col]
    else:
        x = plot_df.index

    fig = make_subplots(
        rows=2 if volume_col else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28] if volume_col else [1],
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=pd.to_numeric(plot_df[close_col], errors="coerce"),
            mode="lines", name="BTC Close",
            line=dict(color="#8b5cf6", width=2.2),
            fill="tozeroy", fillcolor="rgba(139,92,246,.09)",
            hovertemplate="%{x}<br>Close: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1
    )
    if volume_col:
        fig.add_trace(
            go.Bar(
                x=x, y=pd.to_numeric(plot_df[volume_col], errors="coerce"),
                name="BTC Volume", marker_color="rgba(34,211,238,.55)",
                hovertemplate="%{x}<br>Volume: %{y:,.2f}<extra></extra>",
            ), row=2, col=1
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="rgba(148,163,184,.08)")
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#edf2ff")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aab7cf", family="DM Sans"),
        margin=dict(l=8, r=8, t=52, b=8), height=430,
        hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.08)", showline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.08)", tickprefix="$" if not volume_col else "", row=1, col=1)
    return fig


# -------------------------- Sidebar -------------------------
with st.sidebar:
    st.markdown("## ₿ BTC Forecast AI")
    st.caption("Machine-learning crypto dashboard")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Next-Day Forecast", "7-Day Forecast", "Data Explorer", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Model status**")
    for label, path in [("Next-day model", NEXT_DAY_MODEL_PATH), ("7-day model", NEXT_7DAY_MODEL_PATH)]:
        if path.exists():
            st.success(f"{label}: Ready", icon="✅")
        else:
            st.error(f"{label}: File missing", icon="⚠️")
    st.caption("Data and model files are loaded from this app's folder.")

# --------------------------- Header ------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Crypto intelligence · ML forecasting</div>
      <div class="hero-title">BTC Forecast <span style="color:#a78bfa">AI</span></div>
      <div class="hero-sub">Explore Bitcoin market history and generate experimental next-day and next-7-day closing-price estimates from your saved machine-learning models.</div>
      <div style="margin-top:16px"><span class="pill">LIVE-READY UI</span><span class="pill">REGRESSION</span><span class="pill">STREAMLIT</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Load data safely -------------------
df = None
data_error = None
if DATA_PATH.exists():
    try:
        df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        data_error = f"Could not read crypto_data.csv: {exc}"
else:
    data_error = "crypto_data.csv was not found beside app.py."

if df is not None:
    date_col = find_column(df, ["Date", "Datetime", "Timestamp"])
    close_col = find_column(df, ["Close (BTC)", "BTC_Close", "BTC Close", "Close"])
    volume_col = find_column(df, ["Volume (BTC)", "BTC_Volume", "BTC Volume", "Volume"])
    if close_col:
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    if volume_col:
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if close_col:
        df = df.dropna(subset=[close_col])
        if date_col:
            df = df.sort_values(date_col).reset_index(drop=True)

    if close_col is None:
        st.warning("CSV loaded, but BTC closing-price column was not detected. Check the column names in crypto_data.csv.")
else:
    date_col, close_col, volume_col = None, None, None
    st.warning(data_error)
    st.info("Dashboard layout is ready. Add crypto_data.csv and the model files beside app.py to enable data and predictions.")

# ------------------------- Overview ------------------------
if page == "Overview":
    st.markdown("### Market snapshot")
    if df is not None and close_col and not df.empty:
        latest = df.iloc[-1]
        latest_close = float(latest[close_col])
        previous_close = float(df.iloc[-2][close_col]) if len(df) > 1 else latest_close
        change = latest_close - previous_close
        change_pct = (change / previous_close * 100) if previous_close else 0
        latest_volume = float(latest[volume_col]) if volume_col and pd.notna(latest[volume_col]) else np.nan
        latest_date = latest[date_col].strftime("%d %b %Y") if date_col and pd.notna(latest[date_col]) else "Latest row"

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1: make_metric("LATEST BTC CLOSE", money(latest_close), latest_date)
        with c2: make_metric("DAILY CHANGE", f"{change:+,.2f}", f"{change_pct:+.2f}% vs previous row")
        with c3: make_metric("BTC VOLUME", f"{latest_volume:,.0f}" if np.isfinite(latest_volume) else "—", "Latest available row")
        with c4: make_metric("DATA POINTS", f"{len(df):,}", "Rows available in CSV")

        st.markdown('<div class="section-head"></div>', unsafe_allow_html=True)
        st.plotly_chart(plot_price_chart(df, date_col, close_col, volume_col), use_container_width=True, config={"displayModeBar": False})

        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.markdown("#### Recent market data")
            display_cols = [c for c in [date_col, close_col, volume_col] if c]
            recent = df[display_cols].tail(8).copy()
            if date_col:
                recent[date_col] = recent[date_col].dt.strftime("%Y-%m-%d")
            st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Forecast workspace")
            st.markdown(
                "Use the forecast pages to enter the latest BTC close and volume, then generate a model estimate. "
                "Predictions are experimental and should be compared with a simple baseline."
            )
            if NEXT_DAY_MODEL_PATH.exists() and NEXT_7DAY_MODEL_PATH.exists():
                st.success("Both model files are present.")
            else:
                st.warning("One or more model files are missing.")
    else:
        st.markdown("#### Get started")
        st.write("Place these files in the same folder as `app.py`:")
        st.code("app.py\ncrypto_data.csv\nbtc_next_day_lr.pkl\nbtc_next_7day_mlp.pkl", language="text")
        st.caption("The dashboard will automatically populate when the CSV is available.")

# ---------------------- Forecast pages ---------------------
elif page in ["Next-Day Forecast", "7-Day Forecast"]:
    is_7day = page == "7-Day Forecast"
    st.markdown("### " + ("Next 7-day closing-price estimate" if is_7day else "Next-day closing-price estimate"))
    st.markdown(
        '<p class="subtle">Enter values that would actually be known at prediction time. Keep the same feature order and preprocessing used during model training.</p>',
        unsafe_allow_html=True,
    )
    model_path = NEXT_7DAY_MODEL_PATH if is_7day else NEXT_DAY_MODEL_PATH
    horizon_label = "7-day" if is_7day else "next-day"

    latest_close = float(df[close_col].iloc[-1]) if df is not None and close_col and not df.empty else 0.0
    latest_volume = float(df[volume_col].iloc[-1]) if df is not None and volume_col and not df.empty and pd.notna(df[volume_col].iloc[-1]) else 0.0
    latest_date = df[date_col].iloc[-1].date() if df is not None and date_col and not df.empty and pd.notna(df[date_col].iloc[-1]) else date.today()

    form_col, result_col = st.columns([1, 1.1], gap="large")
    with form_col:
        st.markdown("#### Input features")
        with st.form("forecast_form"):
            input_close = st.number_input("Latest BTC Close (USD)", min_value=0.0, value=max(latest_close, 0.01), step=100.0, format="%.2f")
            input_volume = st.number_input("Latest BTC Volume", min_value=0.0, value=max(latest_volume, 0.0), step=100000.0, format="%.2f")
            input_date = st.date_input("Last observed market date", value=latest_date)
            submitted = st.form_submit_button("Generate forecast", type="primary", use_container_width=True)
        st.caption("Defaults are taken from the last row of crypto_data.csv when available.")

    with result_col:
        st.markdown("#### Forecast result")
        if submitted:
            if not model_path.exists():
                st.error(f"Model file not found: {model_path.name}")
            else:
                try:
                    model = load_pickle(str(model_path), model_path.stat().st_mtime)
                    features = pd.DataFrame([[input_close, input_volume]], columns=["Close (BTC)", "Volume (BTC)"])
                    # The saved estimator must contain any scaler/preprocessing used at training time,
                    # or the same scaler must be applied here before prediction.
                    prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
                    forecast_date = input_date + timedelta(days=7 if is_7day else 1)
                    delta = prediction - input_close
                    delta_pct = (delta / input_close * 100) if input_close else 0
                    st.markdown(
                        f"""<div class="metric-card" style="border-color:rgba(34,211,238,.32);">
                        <div class="metric-label">PREDICTED BTC CLOSE · {horizon_label.upper()}</div>
                        <div class="metric-value" style="font-size:2.1rem;color:#67e8f9;">{money(prediction)}</div>
                        <div class="metric-note">Estimated date: {forecast_date.strftime('%d %b %Y')}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    a, b = st.columns(2)
                    with a: make_metric("CHANGE VS INPUT CLOSE", f"{delta:+,.2f}", f"{delta_pct:+.2f}% (not a guaranteed return)")
                    with b: make_metric("MODEL HORIZON", "7 days" if is_7day else "1 day", "Based on the selected saved model")
                    compare = pd.DataFrame({
                        "Type": ["Latest/input close", "Model estimate"],
                        "BTC price (USD)": [input_close, prediction],
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=compare["Type"], y=compare["BTC price (USD)"], marker_color=["#64748b", "#8b5cf6"], text=[money(v) for v in compare["BTC price (USD)"]], textposition="outside"))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#aab7cf"), height=300, margin=dict(l=5,r=5,t=20,b=5),
                        yaxis=dict(title="USD", gridcolor="rgba(148,163,184,.08)", tickprefix="$"),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.download_button(
                        "Download forecast CSV",
                        data=pd.DataFrame([{
                            "last_observed_date": input_date.isoformat(),
                            "forecast_date": forecast_date.isoformat(),
                            "horizon": horizon_label,
                            "input_btc_close": input_close,
                            "input_btc_volume": input_volume,
                            "predicted_btc_close": prediction,
                        }]).to_csv(index=False).encode("utf-8"),
                        file_name=f"btc_{horizon_label}_forecast.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.warning(
                        "Check that this model was trained with exactly these two features, in this order: "
                        "`Close (BTC)`, `Volume (BTC)`. If training used a separate StandardScaler, load and apply that same scaler here."
                    )
        else:
            st.markdown(
                f"""<div class="metric-card">
                <div class="metric-label">YOUR {horizon_label.upper()} FORECAST</div>
                <div class="metric-value">Ready when you are</div>
                <div class="metric-note">Enter the latest known market inputs and click Generate forecast.</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            st.info("The result is a model estimate, not a live exchange price or a guarantee of future performance.")

# ------------------------ Data Explorer --------------------
elif page == "Data Explorer":
    st.markdown("### Data explorer")
    if df is not None:
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        search = st.text_input("Filter columns by name", placeholder="e.g. BTC, Volume, Date")
        columns = [c for c in df.columns if search.lower() in str(c).lower()] if search else list(df.columns)
        st.dataframe(df[columns].tail(250), use_container_width=True, hide_index=True)
        st.download_button(
            "Download displayed data",
            data=df[columns].to_csv(index=False).encode("utf-8"),
            file_name="crypto_data_export.csv",
            mime="text/csv",
        )
        if close_col:
            st.markdown("#### Summary statistics")
            stats = df[close_col].describe().to_frame(name="BTC Close")
            st.dataframe(stats, use_container_width=True)
    else:
        st.warning(data_error or "No data available.")

# ---------------------------- About ------------------------
elif page == "About":
    st.markdown("### About this project")
    st.write(
        "BTC Forecast AI is a local demonstration dashboard for a Bitcoin price-regression project. "
        "It visualizes historical data and runs locally saved models for next-day and next-7-day estimates."
    )
    st.markdown("#### Components")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Data**")
        st.caption("Historical crypto data from your CSV file.")
    with c2:
        st.markdown("**Models**")
        st.caption("Linear Regression for next-day and MLP-based model for 7-day forecast.")
    with c3:
        st.markdown("**Interface**")
        st.caption("Streamlit · Plotly · Local pickle models")
    st.warning(
        "Important: This is an educational project, not financial advice. Crypto prices are volatile. "
        "Model estimates can be wrong and should not be used alone for trading decisions."
    )
    st.markdown("#### Expected folder structure")
    st.code(
        "your_project/\n"
        "├── app.py\n"
        "├── ccrypto_08_OCT_2026.csv\n"
        "├── btc_next_day_lr.pkl\n"
        "└── btc_next_7day_mlp.pkl",
        language="text",
    )

st.markdown(
    '<div class="footer">BTC Forecast AI · Local machine-learning demo · Predictions are estimates, not financial advice.</div>',
    unsafe_allow_html=True,
)
