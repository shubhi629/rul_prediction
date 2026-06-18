import numpy as np
import pandas as pd
import pickle
import os

WINDOW_SIZE = 30

def clean_and_extract_features(df):
    df_cleaned = df.copy()
    df_cleaned.columns = [str(col) for col in df_cleaned.columns]

    if 'engine_id' in df_cleaned.columns:
        df_cleaned.rename(columns={'engine_id': 'unit'}, inplace=True)
    elif '0' in df_cleaned.columns:
        rename_dict = {'0': 'unit', '1': 'cycle'}
        for i in range(1, 22):
            rename_dict[str(i + 4)] = f'sensor_{i}'
        df_cleaned.rename(columns=rename_dict, inplace=True)

    drop_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    df_cleaned.drop(columns=[c for c in drop_cols if c in df_cleaned.columns], inplace=True, errors='ignore')

    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    sensor_cols = [c for c in sensor_cols if c in df_cleaned.columns]

    if 'RUL' not in df_cleaned.columns:
        max_cycle = df_cleaned.groupby('unit')['cycle'].max()
        df_cleaned['RUL'] = df_cleaned['unit'].map(max_cycle) - df_cleaned['cycle']
        df_cleaned['RUL'] = df_cleaned['RUL'].clip(upper=125)

    # Vectorized rolling features — no Python loops
    processed_parts = []

    for unit, engine_data in df_cleaned.groupby('unit'):
        engine_data = engine_data.sort_values('cycle').reset_index(drop=True)

        feat_df = pd.DataFrame()
        feat_df['unit'] = engine_data['unit']
        feat_df['cycle'] = engine_data['cycle']
        feat_df['RUL'] = engine_data['RUL']

        for sensor in sensor_cols:
            col = engine_data[sensor].astype(float)
            feat_df[f'{sensor}_mean'] = col.rolling(WINDOW_SIZE).mean()
            feat_df[f'{sensor}_std'] = col.rolling(WINDOW_SIZE).std().fillna(0)
            feat_df[f'{sensor}_min'] = col.rolling(WINDOW_SIZE).min()
            feat_df[f'{sensor}_max'] = col.rolling(WINDOW_SIZE).max()
            # Vectorized trend using linear regression slope
            feat_df[f'{sensor}_trend'] = col.rolling(WINDOW_SIZE).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
            )

        # Drop rows with NaN (first WINDOW_SIZE-1 rows)
        feat_df = feat_df.dropna().reset_index(drop=True)
        processed_parts.append(feat_df)

    processed_df = pd.concat(processed_parts, ignore_index=True)

    features_path = os.path.join('models', 'rul_lstm_model_final_features.pkl')
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            TOP_SENSORS = pickle.load(f)
    else:
        TOP_SENSORS = [c for c in processed_df.columns if c not in ['unit', 'RUL', 'cycle']]

    available = [f for f in TOP_SENSORS if f in processed_df.columns]

    return processed_df, available