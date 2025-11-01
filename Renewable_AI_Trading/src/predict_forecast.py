import joblib
import pandas as pd
import numpy as np
from glob import glob

def load_latest_weather():
    files = sorted(glob("data/raw/raw_weather_*.csv"), reverse=True)
    return pd.read_csv(files[0])

def predict(df, model, name):
    X = df[["temp", "wind_speed", "irradiance"]]
    pred = model.predict(X)
    df[f"predicted_{name}"] = pred
    return df

if __name__ == "__main__":
    df = load_latest_weather()

    model_gen = joblib.load("models/gen_model.joblib")
    model_price = joblib.load("models/price_model.joblib")
    model_demand = joblib.load("models/demand_model.joblib")

    df = predict(df, model_gen, "output")
    df = predict(df, model_price, "price")
    df = predict(df, model_demand, "demand")

    output_path = f"data/processed/forecast_ml_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    print(f"[✓] Forecasted with ML models and saved to {output_path}")
