import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import json
import time

MAX_RETRIES = 3
WAIT_SECONDS = 5  # wait between retries

def fetch_json(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()  # raise exception for 4xx/5xx
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {WAIT_SECONDS} seconds...")
                time.sleep(WAIT_SECONDS)
            else:
                raise RuntimeError(f"Failed to fetch data after {MAX_RETRIES} attempts.")
            

def main():
    with open("city_config/gothenburg_femman.json") as f:
        city_config = json.load(f)

    CITY_NAME = city_config["city_name"]
    LAT = city_config["city_lat"]
    LON = city_config["city_lon"]
    SENSOR = city_config["sensors"][0]  # only one station
    FG_VERSIONS = city_config["fg_versions"]

    project = hopsworks.login()  
    fs = project.get_feature_store()

    START_DATE = "2019-11-01"
    END_DATE = "2025-11-01"

    # air_quality_url = (
    #     f"https://open-meteo.com/en/docs/air-quality-api?"
    #     f"latitude={LAT}&longitude={LON}"
    #     f"&past_days={92}"
    #     f"&hourly=pm2_5"       # target variable
    # )

    # features
    historical_weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        f"&hourly=wind_speed_100m,wind_direction_100m,wind_gusts_10m,wind_direction_10m,wind_speed_10m,temperature_2m"
    )

    weather_data = fetch_json(historical_weather_url)

    weather_df = pd.DataFrame(weather_data["hourly"])
    print(f"Fetched {len(weather_df)} weather records")
    weather_df["datetime"] = pd.to_datetime(weather_df["time"])
    weather_df.drop(columns=["time"], inplace=True)

    aqi_csv_path = "data/goteborg-femman-air-quality.csv"  # CSV for Femman
    aqi_df = pd.read_csv(aqi_csv_path)
    aqi_df["datetime"] = pd.to_datetime(aqi_df["date"])    # rename column to datetime
    aqi_df.drop(columns=["date"], inplace=True)

    # Add sensor id as a column
    aqi_df["sensor_id"] = SENSOR["id"]
    aqi_df["sensor_name"] = SENSOR["display_name"]

    # Register as feature groups:
    weather_fg = fs.get_or_create_feature_group(
        name="weather",
        version=FG_VERSIONS["weather"],
        description=f"Historical weather data for {CITY_NAME}",
        primary_key=["datetime"],
        event_time="datetime"
    )

    aqi_fg = fs.get_or_create_feature_group(
        name="air_quality",
        description="Air Quality characteristics of each day for {CITY_NAME} ({SENSOR['display_name']})",
        version=FG_VERSIONS["air_quality"],
        primary_key=["datetime"],        
        event_time="datetime"
    )


    # Backfill historical data:
    weather_fg.insert(weather_df, write_options={"wait_for_job": False})
    aqi_fg.insert(aqi_df, write_options={"wait_for_job": False})

    print(f"Backfill complete! Feature Groups for {CITY_NAME} registered.")


if __name__ == "__main__":
    main()