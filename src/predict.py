import os
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st

SEQ_LEN = 30

@st.cache_resource
def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

def make_sequences(df, feature_cols, seq_len=SEQ_LEN):
    X_seqs = []
    y_seqs = []

    for unit, group in df.groupby('unit'):
        group = group.sort_values('cycle') if 'cycle' in group.columns else group
        X = group[feature_cols].values
        y = group['RUL'].values

        if len(X) >= seq_len:
            for i in range(len(X) - seq_len):
                X_seqs.append(X[i: i + seq_len])
                y_seqs.append(y[i + seq_len])

    return np.array(X_seqs), np.array(y_seqs)

def run_production_inference(processed_df, feature_cols):
    model_path = os.path.join('models', 'rul_lstm_model_final.h5')
    scaler_path = os.path.join('models', 'rul_lstm_model_final_scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at: {model_path}")

    # Scale using training scaler
    X_data = processed_df[feature_cols].values

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        try:
            X_scaled = scaler.transform(X_data)
        except Exception:
            X_scaled = X_data
    else:
        X_scaled = X_data

    # Rebuild temp_df with scaled values for sequence building
    temp_df = processed_df[['unit', 'cycle', 'RUL']].copy()
    scaled_cols = [f'feat_{i}' for i in range(X_scaled.shape[1])]
    for idx, col in enumerate(scaled_cols):
        temp_df[col] = X_scaled[:, idx]

    X_3D, y_true = make_sequences(temp_df, scaled_cols)

    if len(X_3D) == 0:
        return np.array([]), np.array([])

    model = load_keras_model(model_path)
    predictions = model.predict(X_3D, verbose=0)

    return predictions.flatten(), y_true