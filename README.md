# rul_prediction#
Predictive Maintenance: Jet Engine RUL Prediction

Ever wondered how airlines know when to service an engine *before* it fails? This project tackles exactly that — predicting how many cycles a jet engine has left before it needs maintenance, using real NASA sensor data and a stacked LSTM neural network.


The model takes in time-series sensor readings from jet engines and predicts the **Remaining Useful Life (RUL)** — essentially, how many more operational cycles the engine can handle before failure. The deployed dashboard lets you either upload your own CSV data or manually tweak sensor values to see how the model responds in real time.

##  Live Demo

 [Click here to open the live app](https://engine-rul-predictor.streamlit.app)

## Dataset

Uses the **NASA CMAPSS FD001 dataset** — a standard benchmark in predictive maintenance research. It contains multivariate time-series sensor readings from 100 jet engines run to failure under a single operating condition.

## Model

- Architecture: Stacked LSTM (2 layers) with dropout regularization
- Input: Sliding window of 30 cycles × 20 features
- Output: Predicted RUL (continuous)
- Top 20 sensors selected via Random Forest feature importance (train only)

## Performance

| Metric | Score |
|--------|-------|
| R² Score | 0.8841 |
| MAE | 9.57 cycles |
| RMSE | 13.97 cycles |

## Tech Stack

- Python, TensorFlow, scikit-learn, NumPy, pandas
- Streamlit for the interactive dashboard
- Deployed on Streamlit Cloud

## Run Locally

git clone https://github.com/shubhi629/rul_prediction
cd rul_prediction
pip install -r requirements.txt
streamlit run app.py

## Project Structure

rul_prediction/
├── app.py              # Streamlit dashboard
├── src/
│   ├── predict.py      # Inference pipeline
│   └── process.py      # Feature engineering
├── models/             # Saved LSTM model + scaler
├── data/               # NASA CMAPSS dataset
└── RUL_final.ipynb     # Training notebook

## Dashboard Features

**Batch Mode** — Upload a raw turbofan CSV and get full predictions with MAE, RMSE, R² metrics and a true vs predicted scatter plot.

**Simulator Mode** — Manually adjust 4 critical sensor readings (HP Compressor Temp, LPT Outlet Temp, HPC Outlet Pressure, Static Pressure) across three degradation profiles to see how RUL changes in real time.
