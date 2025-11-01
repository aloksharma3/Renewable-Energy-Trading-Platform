import pandas as pd
import requests

def fetch_weather_data(api_key, lat=32.7767, lon=-96.7970):  # Dallas example
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    res = requests.get(url)
    data = res.json()
    weather_list = []
    for item in data["list"]:
        weather_list.append({
            "timestamp": item["dt_txt"],
            "temp": item["main"]["temp"],
            "wind_speed": item["wind"]["speed"],
            "irradiance": item["clouds"]["all"]
        })

    return pd.DataFrame(weather_list)
