
import pickle
from pathlib import Path

import gradio as gr
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="BTC Forecast AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------
# Load trained models
# -----------------------------
@st.cache_resource
def load_models():
    with open(BASE_DIR / "btc_next_day_lr.pkl", "rb") as f:
        next_day_model = pickle.load(f)

    with open(BASE_DIR / "btc_next_7day_mlp.pkl", "rb") as f:
        next_7day_model = pickle.load(f)

    return next_day_model, next_7day_model


try:
    next_day_model, next_7day_model = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# -----------------------------
# Prediction function
# -----------------------------
def predict_btc(close, volume, horizon):
    if close <= 0 or volume < 0:
        raise ValueError(
            "Close must be positive and Volume cannot be negative."
        )

    input_data = pd.DataFrame(
        [[close, volume]],
        columns=["Close (BTC)", "Volume (BTC)"]
    )

    if horizon == "Next Day":
        prediction = next_day_model.predict(input_data)[0]
    else:
        prediction = next_7day_model.predict(input_data)[0]

    return float(prediction)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, #16294a 0%, transparent 30%),
        linear-gradient(135deg, #080d18, #0b1220 65%, #101b2d);
    color: #f1f5f9;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px;
    border: 1px solid #263852;
    border-radius: 22px;
    background: linear-gradient(120deg, #142641, #101827 75%);
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #ffffff;
}

.hero p {
    color: #9fb0c9;
    font-size: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #17253a, #101827);
    border: 1px solid #293b55;
    padding: 23px;
    border-radius: 18px;
    min-height: 140px;
}

.metric-label {
    color: #9aabc2;
    font-size: 14px;
    margin-bottom: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    overflow-wrap: anywhere;
}

.metric-note {
    color: #36d6a5;
    font-size: 12px;
    margin-top: 9px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}

div[data-testid="stForm"] {
    background: #101a2a;
    border: 1px solid #293b55;
    padding: 22px;
    border-radius: 18px;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #8b5cf6, #4f8cff);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    border: 1px solid #a5b4fc;
    color: white;
}

div[data-testid="stSidebar"] {
    background: #0c1422;
    border-right: 1px solid #25344b;
}

.footer {
    color: #8495ae;
    font-size: 12px;
    text-align: center;
    padding: 22px 0;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.markdown("## ₿ BTC Forecast AI")
st.sidebar.caption("Machine Learning Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Next-Day Prediction",
        "7-Day Prediction",
        "Gradio Playground",
        "About Models"
    ]
)

st.sidebar.divider()
st.sidebar.success("Models loaded successfully")
st.sidebar.caption("Local machine learning application")


# -----------------------------
# Hero section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>₿ BTC Forecast AI</h1>
    <p>
        Bitcoin price forecasting powered by Machine Learning.
        Explore next-day and 7-day model predictions in one dashboard.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Reusable prediction form
# -----------------------------
def prediction_form(horizon):
    st.markdown(
        f'<div class="section-title">{horizon} Forecast</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the latest Bitcoin closing price and trading volume."
    )

    with st.form(f"form_{horizon}"):
        col1, col2 = st.columns(2)

        with col1:
            close = st.number_input(
                "BTC Close Price (USD)",
                min_value=0.01,
                value=81676.33,
                step=100.0,
                format="%.2f"
            )

        with col2:
            volume = st.number_input(
                "BTC Volume",
                min_value=0.0,
                value=1000000000.0,
                step=1000000.0,
                format="%.2f"
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:
        try:
            result = predict_btc(close, volume, horizon)

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Predicted BTC Close
                        </div>
                        <div class="metric-value">
                            ${result:,.2f}
                        </div>
                        <div class="metric-note">
                            {horizon} model output
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                change = result - close
                change_pct = (change / close) * 100

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Change vs Input Price
                        </div>
                        <div class="metric-value">
                            {change_pct:+.2f}%
                        </div>
                        <div class="metric-note">
                            ${change:+,.2f} USD
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption(
                "Change is relative to your input price, not a guaranteed "
                "future market return."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------
# Dashboard page
# -----------------------------
if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Model Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Forecast Horizons</div>
            <div class="metric-value">1D / 7D</div>
            <div class="metric-note">Two prediction models</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Next-Day Model</div>
            <div class="metric-value">Linear Regression</div>
            <div class="metric-note">btc_next_day_lr.pkl</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">7-Day Model</div>
            <div class="metric-value">MLP Regressor</div>
            <div class="metric-note">StandardScaler pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Explore Predictions</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.info(
            "**Next-Day Prediction**\n\n"
            "Estimate the next-day closing price using Linear Regression."
        )
        if st.button("Open Next-Day Model", use_container_width=True):
            st.session_state["navigation"] = "Next-Day Prediction"
            st.rerun()

    with right:
        st.info(
            "**7-Day Prediction**\n\n"
            "Estimate the 7-day target using the trained MLP pipeline."
        )
        if st.button("Open 7-Day Model", use_container_width=True):
            st.session_state["navigation"] = "7-Day Prediction"
            st.rerun()

    st.caption(
        "Enter realistic inputs from the same data distribution used "
        "during model training."
    )


# -----------------------------
# Prediction pages
# -----------------------------
elif page == "Next-Day Prediction":
    prediction_form("Next Day")

elif page == "7-Day Prediction":
    prediction_form("Next 7 Days")


# -----------------------------
# Gradio Playground
# -----------------------------
elif page == "Gradio Playground":

    st.markdown(
        '<div class="section-title">Gradio Interactive Playground</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use the embedded Gradio interface to compare both trained models."
    )

    def gradio_predict(close, volume, horizon):
        result = predict_btc(close, volume, horizon)
        return f"${result:,.2f} USD"

    demo = gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue"
        ),
        title="BTC Forecast Playground"
    )

    with demo:
        gr.Markdown("# ₿ BTC Forecast Playground")
        gr.Markdown(
            "Enter the latest BTC Close and Volume to generate a forecast."
        )

        with gr.Row():
            g_close = gr.Number(
                label="BTC Close Price (USD)",
                value=81676.33,
                minimum=0.01
            )
            g_volume = gr.Number(
                label="BTC Volume",
                value=1000000000.0,
                minimum=0
            )

        g_horizon = gr.Radio(
            ["Next Day", "Next 7 Days"],
            value="Next Day",
            label="Prediction Horizon"
        )

        g_button = gr.Button("Predict BTC Price", variant="primary")
        g_output = gr.Textbox(label="Predicted BTC Close")

        g_button.click(
            fn=gradio_predict,
            inputs=[g_close, g_volume, g_horizon],
            outputs=g_output
        )

    # Start Gradio only once, even when Streamlit reruns.
    if not globals().get("_GRADIO_STARTED", False):
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=7862,
                share=False,
                prevent_thread_lock=True,
                quiet=True
            )
            globals()["_GRADIO_STARTED"] = True
        except Exception as e:
            st.error(f"Could not start Gradio: {e}")

    components.iframe(
    "http://127.0.0.1:7862",
    height=650,
    scrolling=True
    )

    st.caption(
        "Gradio runs locally on port 7861; Streamlit runs on port 8501."
    )


# -----------------------------
# Model information
# -----------------------------
elif page == "About Models":

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    model_info = pd.DataFrame({
        "Property": [
            "Next-Day Model",
            "7-Day Model",
            "Input Feature 1",
            "Input Feature 2",
            "Prediction Target"
        ],
        "Details": [
            "Linear Regression",
            "MLPRegressor + StandardScaler",
            "Close (BTC)",
            "Volume (BTC)",
            "BTC closing price"
        ]
    })

    st.dataframe(model_info, hide_index=True, use_container_width=True)

    st.warning(
        "The app uses the exact two input features expected by your "
        "saved models. Confirm that the 7-day model was trained against "
        "the intended 7-day target before interpreting its output."
    )

    st.markdown("### Important")
    st.write(
        "The models use only Close and Volume as inputs. They do not "
        "automatically fetch live market data or guarantee future prices."
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    BTC Forecast AI · Streamlit + Gradio · Local ML Dashboard
</div>
""", unsafe_allow_html=True)
