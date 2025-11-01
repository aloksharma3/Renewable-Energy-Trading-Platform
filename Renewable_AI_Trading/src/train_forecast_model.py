import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from glob import glob
import os

def load_latest_data():
    files = sorted(glob("data/processed/forecast_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError("No processed forecast files found.")
    return pd.read_csv(files[0])

def train_model(df, feature_cols, target_col, model_name):
    X = df[feature_cols]
    y = df[target_col]

    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, f"models/{model_name}.joblib")
    print(f"[✓] Trained and saved model: {model_name}")
    return model

def bootstrap_prediction(model, X, n_rounds=30):
    preds = []
    for _ in range(n_rounds):
        sample_idx = np.random.choice(len(X), len(X), replace=True)
        preds.append(model.predict(X.iloc[sample_idx]))
    return np.mean(preds, axis=0), np.std(preds, axis=0)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = load_latest_data()

    features = ["temp", "wind_speed", "irradiance"]

    model_out = train_model(df, features, "energy_output", "gen_model")
    model_price = train_model(df, features, "market_price", "price_model")
    model_demand = train_model(df, features, "grid_demand", "demand_model")

    print("[✓] All models trained and saved.")
