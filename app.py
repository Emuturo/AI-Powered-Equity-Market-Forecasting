import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import data_loader
import preprocess
import xgb_models
import styles
import eda
from market_trends import show_market_trends  

# 📌 Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("financial_economic_data_cleaned.csv", parse_dates=["Date"], index_col="Date")
    return df

# 🎨 Sidebar with Icons & Expander
st.sidebar.title("🔍 Navigation")

# Sidebar Navigation Options
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "📊 Exploratory Data Analysis (EDA)",
        "🤖 AI Predictions",
        "📈 Market Trends",
        "⚙️ Settings"
    ]
)

# 🎭 Expander for Additional Info
with st.sidebar.expander("ℹ️ About This App"):
    st.write("""
    - **Version:** 1.0  
    - **Developed by:** Eric_Muchoki  
    - **Purpose:** Predict S&P 500 and NASDAQ stock trends using AI.  
    """)

# Display content based on sidebar selection
if page == "🏠 Home":
    st.header("🏠 Welcome to AI-Powered Equity Market Indices Predictions")
    st.write("Use the sidebar to navigate through the app.")

    # Append the detailed S&P 500 and NASDAQ messages here
    st.markdown("""
        **S&P 500:** This index represents the performance of 500 large companies listed on stock exchanges in the United States, offering a broad view of the overall market's health and economic trends. It's a crucial benchmark for investors seeking to understand the general direction of the U.S. equity market.  
        **NASDAQ:** Focused on technology and growth companies, the NASDAQ is a dynamic index that reflects the cutting edge of innovation and technological advancement. It's a key indicator for those interested in the future of tech, biotech, and high-growth sectors.
    """)

elif page == "📊 Exploratory Data Analysis (EDA)":
    eda.show_eda()  # Call the EDA function from eda.py

elif page == "🤖 AI Predictions":
    st.header("🤖 AI-Powered Market Indices Predictions")
    st.write("Make stock market forecasts using Machine Learning models.")

elif page == "📈 Market Trends":
    st.header("📈 Latest Market Trends")
    st.write("Live financial data, news sentiment, and analytics.")
    
    # Call Market Trends Page
    try:
        show_market_trends()
    except Exception as e:
        st.error(f"⚠️ Failed to load Market Trends: {e}")

elif page == "⚙️ Settings":
    st.header("⚙️ App Settings")
    st.write("Customize your experience (Dark Mode, Data Refresh, etc.).")

# Apply custom CSS
st.markdown(styles.load_css(), unsafe_allow_html=True)

# 🔦 Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
            .stApp {background-color: #1e1e1e;}
            .stMarkdown h1, h2, h3 {color: white;}
            .stButton>button {background-color: #007bff; color: white;}
            .stButton>button:hover {background-color: #0056b3;}
        </style>
    """, unsafe_allow_html=True)

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 🔮 Load trained models
model_sp500_lstm = load_model("best_lstm_model_S&P_500.h5")
model_nasdaq_lstm = load_model("best_lstm_model_NASDAQ.h5")

# Load preprocessing pipeline (for LSTM)
pipeline = joblib.load('preprocessing_pipeline.pkl')

st.title("📊 AI-Powered Stock Market Indices Prediction")

# 📌 Sidebar Model Selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "XGBoost"]
)

# Model message
model_messages = {
    "LSTM": "LSTM-Powered Forecasting – Unlocking the Future of Markets, One Sequence at a Time!",
    "XGBoost": "Powering Smarter Market Predictions with XGBoost – Precision, Speed, and Intelligence for Every Trade.",
}
st.markdown(f"_{model_messages.get(model_choice, 'Select a model for prediction.')}_")

st.sidebar.header("User Input Features")
use_latest_data = st.sidebar.checkbox("Use latest data for prediction", value=True)

# 📡 Load latest data
latest_data = data_loader.get_combined_data()
if latest_data is None:
    st.error("Failed to fetch data. Please check your data sources and try again.")
    st.stop()

# 🔄 Preprocessing
if model_choice == "LSTM":
    sp500_input = preprocess.preprocess_data(latest_data.copy(), 'SP500').iloc[-5:, :]
    nasdaq_input = preprocess.preprocess_data(latest_data.copy(), 'NASDAQ').iloc[-5:, :]
else:
    try:
        sp500_input = xgb_models.preprocess_xgb(latest_data.copy(), "S&P 500")
        nasdaq_input = xgb_models.preprocess_xgb(latest_data.copy(), "NASDAQ")

        # Ensure at least 6 rows for XGBoost lag features
        sp500_input = sp500_input.iloc[-6:, :]
        nasdaq_input = nasdaq_input.iloc[-6:, :]
    except ValueError as e:
        st.error(f"XGBoost preprocessing error: {e}")
        st.stop()

# 📌 Validate Input Data for LSTM
if model_choice == "LSTM":
    required_columns = ['NASDAQ Close', 'cy10', 'cm3', 'Trade_Weighted_Dollar_Index']
    missing_columns_sp500 = [col for col in required_columns if col not in sp500_input.columns]
    missing_columns_nasdaq = [col for col in required_columns if col not in nasdaq_input.columns]

    if missing_columns_sp500:
        st.error(f"Error: S&P 500 input data is missing columns: {missing_columns_sp500}")
        st.stop()
    if missing_columns_nasdaq:
        st.error(f"Error: NASDAQ input data is missing columns: {missing_columns_nasdaq}")
        st.stop()

# 🛠 Expand dataset if rows are missing (XGBoost)
if model_choice == "XGBoost":
    if len(sp500_input) < 6:
        st.warning(f"⚠️ Not enough rows ({len(sp500_input)}) for S&P 500 XGBoost prediction. Expanding dataset...")
        sp500_input = sp500_input.ffill()
    if len(nasdaq_input) < 6:
        st.warning(f"⚠️ Not enough rows ({len(nasdaq_input)}) for NASDAQ XGBoost prediction. Expanding dataset...")
        nasdaq_input = nasdaq_input.ffill()

# 🔄 Data Transformation
sp500_input_transformed = pipeline.transform(sp500_input) if model_choice == "LSTM" else sp500_input
nasdaq_input_transformed = pipeline.transform(nasdaq_input) if model_choice == "LSTM" else nasdaq_input

# 🔮 LSTM Prediction Function
def predict_lstm(model, input_data):
    input_reshaped = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    prediction = model.predict(input_reshaped)
    return ["📈 Positive" if p > 0.64803946 else "📉 Negative" for p in prediction]

# 📌 Make Predictions
if st.button("Predict"):
    if model_choice == "LSTM":
        sp500_predictions = predict_lstm(model_sp500_lstm, sp500_input_transformed)
        nasdaq_predictions = predict_lstm(model_nasdaq_lstm, nasdaq_input_transformed)
    else:
        sp500_predictions = xgb_models.predict_xgb(xgb_models.xgb_sp500, sp500_input_transformed)
        nasdaq_predictions = xgb_models.predict_xgb(xgb_models.xgb_nasdaq, nasdaq_input_transformed)

    # 📅 Get the last date for prediction
    sp500_last_date = sp500_input.index[-1].strftime('%Y-%m-%d')
    nasdaq_last_date = nasdaq_input.index[-1].strftime('%Y-%m-%d')

    st.subheader("Predictions")
    st.write(f"📅 Prediction Date: {sp500_last_date} (S&P 500)")
    st.write(f"📊 **S&P 500 Market Direction:** {sp500_predictions[-1]}")
    st.write(f"📅 Prediction Date: {nasdaq_last_date} (NASDAQ)")
    st.write(f"📊 **NASDAQ Market Direction:** {nasdaq_predictions[-1]}")