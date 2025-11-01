# weather_service/main.py

import requests
import pandas as pd
import os
from datetime import datetime
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Environment variables from .env
API_KEY = os.getenv("WEATHER_API_KEY")
LAT = os.getenv("LAT", "42.3601")  # Default: Boston
LON = os.getenv("LON", "-71.0589")
OUTPUT_FILE = "data/raw_weather.csv"

def fetch_weather():
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    )
    response = requests.get(url)
    data = response.json()

    temp = data["main"]["temp"]
    wind_speed = data["wind"]["speed"]
    cloud_coverage = data["clouds"]["all"]
    # Approximate solar irradiance
    irradiance = 500 + 100 * (1 - cloud_coverage / 100.0)

    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temp": temp,
        "wind_speed": wind_speed,
        "irradiance": irradiance
    }])

    return df

@app.post("/update")
def update_weather():
    try:
        df = fetch_weather()
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        return {"message": "Weather data updated", "data": df.to_dict(orient="records")[0]}
    except Exception as e:
        return {"error": str(e)}
