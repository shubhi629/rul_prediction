
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.process import clean_and_extract_features
from src.predict import run_production_inference

st.set_page_config(page_title="Jet Engine RUL Predictor", layout="wide")

st.title("Predictive Maintenance: Jet Engine RUL Dashboard")
st.write("Evaluate engine sensor configurations to predict the Remaining Useful Life (RUL) using the trained LSTM model.")

# ==========================================
# SIDEBAR - NAVIGATION & SELECTION
# ==========================================
st.sidebar.header("Operational Mode")
app_mode = st.sidebar.radio("Choose testing method:", ["Batch CSV Upload", "Manual Unit Simulator"])

# ==========================================
# MODE 1: BATCH CSV UPLOAD
# ==========================================
if app_mode == "Batch CSV Upload":
    st.sidebar.subheader("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (e.g. train_FD001.csv)", type=["csv"])

    if uploaded_file is not None:

        @st.cache_data
        def load_and_process(file):
            raw_data = pd.read_csv(file)
            return clean_and_extract_features(raw_data)

        raw_data = pd.read_csv(uploaded_file)
        st.success("Raw data loaded successfully!")

        with st.spinner("Processing data windows and running LSTM inference..."):
            try:
                processed_df, X_features = clean_and_extract_features(raw_data)
                predictions, y_true = run_production_inference(processed_df, X_features)

                if len(predictions) == 0:
                    st.error("Not enough data rows matching the required sequence window size (30 cycles).")
                else:
                    processed_df = processed_df.iloc[len(processed_df) - len(predictions):].copy()
                    processed_df['Predicted_RUL'] = predictions

                    st.subheader("Current Uploaded Engine Summary")
                    col1, col2, col3 = st.columns(3)

                    latest_cycle = int(processed_df['cycle'].max())
                    current_prediction = int(predictions[-1])

                    col1.metric("Max Active Run Cycle", f"{latest_cycle} cycles")
                    col2.metric("Latest Predicted RUL", f"{current_prediction} cycles")

                    status = "Safe Operations" if current_prediction > 50 else "Maintenance Warning" if current_prediction > 25 else "CRITICAL FAILURE RISK"
                    col3.metric("Engine Health Status", status)

                    if len(y_true) > 0 and not np.all(y_true == 0):
                        st.markdown("---")
                        st.subheader("Model Prediction Accuracy")

                        mae = mean_absolute_error(y_true, predictions)
                        rmse = np.sqrt(mean_squared_error(y_true, predictions))
                        r2 = r2_score(y_true, predictions)

                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} Cycles")
                        m_col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} Cycles")
                        m_col3.metric("R2 Score", f"{r2 * 100:.1f}%")

                    st.markdown("---")
                    st.subheader("True vs Predicted RUL")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.scatter(y_true, predictions, alpha=0.3, color='dodgerblue', s=10)
                    ax.plot([0, 125], [0, 125], 'r--', label='Perfect Prediction', linewidth=2)
                    ax.set_xlabel("True RUL (Cycles)")
                    ax.set_ylabel("Predicted RUL (Cycles)")
                    ax.set_title("True vs Predicted RUL - LSTM Model")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("Please upload a raw turbine dataset via the sidebar to start the analysis.")

# ==========================================
# MODE 2: MANUAL UNIT SIMULATOR
# ==========================================
else:
    st.subheader("Manual Interactive Testing Simulator")
    st.write("Simulate manual telemetry profiles. Adjust critical target parameters below to observe model sensitivity.")

    profile = st.sidebar.selectbox("Base Degradation Profile", ["Brand New / Stable", "Moderate Degradation", "Near-Failure State"])

    if profile == "Brand New / Stable":
        base_s2, base_s3, base_s4, base_s11 = 642.0, 1585.0, 1400.0, 47.2
    elif profile == "Moderate Degradation":
        base_s2, base_s3, base_s4, base_s11 = 642.8, 1591.0, 1410.0, 47.6
    else:
        base_s2, base_s3, base_s4, base_s11 = 643.5, 1602.0, 1428.0, 48.1

    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Sensor Adjustments")

    s2 = st.sidebar.slider("Sensor 2 (HP Compressor Outlet Temp)", 640.0, 645.0, base_s2, 0.05)
    s3 = st.sidebar.slider("Sensor 3 (LPT Outlet Temp)", 1575.0, 1615.0, base_s3, 0.5)
    s4 = st.sidebar.slider("Sensor 4 (Total HPC Outlet Pressure)", 1390.0, 1440.0, base_s4, 0.5)
    s11 = st.sidebar.slider("Sensor 11 (Static Pressure)", 46.5, 48.5, base_s11, 0.05)

    if st.button("Compute LSTM Simulated Prediction"):
        with st.spinner("Processing synthesized sequence vectors..."):
            try:
                columns = [str(i) for i in range(26)]
                sim_data = []

                for cycle in range(1, 62):
                    factor = cycle / 61.0
                    row = [0.0] * 26

                    row[0] = 1.0
                    row[1] = float(cycle)
                    row[2] = 0.0020
                    row[3] = 0.0003
                    row[4] = 100.0

                    for s_idx in range(1, 22):
                        row[s_idx + 4] = 1300.0 if s_idx in [3, 4, 8, 13] else 500.0 if s_idx in [2, 7, 11, 12, 17] else 40.0

                    row[5] = float(base_s2 + (s2 - base_s2) * factor)
                    row[6] = float(base_s3 + (s3 - base_s3) * factor)
                    row[7] = float(base_s4 + (s4 - base_s4) * factor)
                    row[14] = float(base_s11 + (s11 - base_s11) * factor)

                    sim_data.append(row)

                simulated_df = pd.DataFrame(sim_data, columns=columns)
                processed_df, X_features = clean_and_extract_features(simulated_df)
                predictions, _ = run_production_inference(processed_df, X_features)

                if len(predictions) > 0:
                    sim_result = int(predictions[-1])

                    st.markdown("---")
                    st.subheader("Resulting Model Output")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Simulated Final Cycle", "Cycle 61")
                    m2.metric("LSTM Predicted Remaining RUL", f"{sim_result} Cycles")

                    sim_status = "Safe / Healthy Profile" if sim_result > 50 else "Maintenance Required" if sim_result > 25 else "CRITICAL RISK DETECTED"
                    m3.metric("Evaluated Status", sim_status)

                    st.info(f"Based on your manual adjustments, the model projects a remaining lifespan of {sim_result} cycles before the engine reaches critical thresholds.")
                else:
                    st.error("Data tracking sequence shape error. Could not extract structural sequence array.")

            except Exception as e:
                st.error(f"Simulator Error: {str(e)}")