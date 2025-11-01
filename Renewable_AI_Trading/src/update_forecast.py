from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from datetime import datetime
from data_loader import fetch_weather_data
from src.predict_forecast import predict
import joblib

load_dotenv()
api_key = os.getenv("WEATHER_API_KEY")

# Fetch weather
df_raw = fetch_weather_data(api_key)


# Load models
model_gen = joblib.load("models/gen_model.joblib")
model_price = joblib.load("models/price_model.joblib")
model_demand = joblib.load("models/demand_model.joblib")

# Predict values
df_raw = predict(df_raw, model_gen, "output")
df_raw = predict(df_raw, model_price, "price")
df_raw = predict(df_raw, model_demand, "demand")

# Save to raw and processed folders with timestamp
# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_raw.to_csv(f"data/raw/raw_weather_{timestamp}.csv", index=False)
df_raw.to_csv(f"data/processed/forecast_ml_{timestamp}.csv", index=False)
print(f"[✓] Saved ML-based forecast to data/processed/forecast_ml_{timestamp}.csv")
