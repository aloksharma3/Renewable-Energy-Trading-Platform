import os
import pandas as pd
import joblib
from fastapi import FastAPI
from datetime import datetime, timedelta

# Load models
gen_model = joblib.load("models/gen_model.joblib")
price_model = joblib.load("models/price_model.joblib")
demand_model = joblib.load("models/demand_model.joblib")

app = FastAPI()

@app.post("/forecast/batch")
def batch_forecast():
    input_file = "data/raw_weather.csv"
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        return {"error": f"Failed to read input CSV: {e}"}

    # Run predictions
    df["predicted_output"] = gen_model.predict(df[["temp", "wind_speed", "irradiance"]])
    df["predicted_price"] = price_model.predict(df[["temp", "wind_speed", "irradiance"]])
    df["grid_demand"] = demand_model.predict(df[["temp", "wind_speed", "irradiance"]])

    # Add timestamps spaced 5 minutes apart
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))

    df["timestamp"] = [now + timedelta(minutes=5 * i) for i in range(len(df))]

    # Reorder columns
    df = df[["timestamp", "predicted_output", "predicted_price", "grid_demand"]]

    # Save file
    filename = f"forecast_{now:%Y%m%d_%H%M}.csv"
    df.to_csv(os.path.join(output_dir, filename), index=False)

    return {"message": "Forecast saved", "file": filename, "rows": len(df)}
