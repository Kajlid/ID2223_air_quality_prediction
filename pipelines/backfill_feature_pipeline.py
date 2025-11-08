import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import json
import time
import numpy as np
from dotenv import load_dotenv
import great_expectations as ge

MAX_RETRIES = 3
WAIT_SECONDS = 5  # wait between retries

def fetch_json(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()  
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {WAIT_SECONDS} seconds...")
                time.sleep(WAIT_SECONDS)
            else:
                raise RuntimeError(f"Failed to fetch data after {MAX_RETRIES} attempts.")
            
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespaces in beginning of column names to make sure they are compatible with Hopsworks"""
    df.columns = (
        df.columns
        .str.strip()            
        .str.lower()            # lowercase everything
        .str.replace(" ", "_")  # replace spaces with underscores
    )
    # Hopsworks requires names to start with a letter:
    df.columns = [
        col if col[0].isalpha() else f"f_{col.lstrip('_')}"
        for col in df.columns
    ]
    return df

def validate_with_expectations(df, expectation_suite, name="dataset"):
    """
    Run Great Expectations validation before ingestion.
    Prints results and raises an error if validation fails.
    """
    ge_df = ge.from_pandas(df)
    validation_result = ge_df.validate(expectation_suite=expectation_suite)
    
    if not validation_result.success:
        print(f"Validation failed for {name}")
        for res in validation_result.results:
            if not res.success:
                print(f" - {res.expectation_config.expectation_type} failed: {res.result}")
        raise ValueError(f"Validation failed for {name}. Check the data.")
    else:
        print(f"Validation passed for {name}")


def main():
    with open("city_config/gothenburg_femman.json") as f:
        city_config = json.load(f)

    CITY_NAME = city_config["city_name"]
    LAT = city_config["city_lat"]
    LON = city_config["city_lon"]
    SENSOR = city_config["sensors"][0]  # only one station
    FG_VERSIONS = city_config["fg_versions"]

    load_dotenv()
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    hopsworks_project = os.getenv("HOPSWORKS_PROJECT")
    project = hopsworks.login(project=hopsworks_project, api_key_value=hopsworks_key)  
    fs = project.get_feature_store()

    # Last 6 years
    START_DATE = "2019-11-06"
    END_DATE = "2025-11-06"

    # Features
    historical_weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        f"&daily=wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,temperature_2m_max"
        "&timezone=Europe%2FBerlin"    # same timezone as Sweden     
    )

    weather_data = fetch_json(historical_weather_url)

    weather_df = pd.DataFrame(weather_data["daily"])
    print(f"Fetched {len(weather_df)} weather records")
    weather_df["date"] = pd.to_datetime(weather_df["time"], yearfirst=True)
    weather_df["country"] = os.getenv("AQICN_COUNTRY")
    weather_df["city"] = os.getenv("AQICN_CITY")
    weather_df["street"] = os.getenv("AQICN_STREET")
    weather_df.drop(columns=["time"], inplace=True)
    
    # Data validation
    weather_expectation_suite = ge.core.ExpectationSuite(
        expectation_suite_name="weather_expectation_suite"
    )

    # Check for null values
    weather_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "temperature_2m_max"}
        )
    )

    # Check for reasonable range
    weather_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "temperature_2m_max",
                "min_value": -60.0,
                "max_value": 60.0
            }
        )
    )

    aqi_csv_path = "data/goteborg-femman-air-quality.csv"  # CSV for Femman
    df_aq = (
    pd.read_csv(aqi_csv_path)
    .pipe(clean_column_names)
    )
    df_aq = df_aq[["date", "pm25"]] 
    df_aq.rename(columns={"pm25": "pm2_5", "date": "date"}, inplace=True)  
    
    # Replace missing rows with nan
    df_aq["pm2_5"].replace(" ", np.nan, inplace=True)  
    
    # Make sure the value is a double
    df_aq["pm2_5"] = df_aq["pm2_5"].astype(float) 
    
    # Missing PM2.5 values are filled from the nearest available measurement
    # df_aq["pm2_5"].interpolate(method='nearest', inplace=True)
    df_aq.dropna(inplace=True)
    df_aq["date"] = pd.to_datetime(df_aq["date"], yearfirst=True)       # convert to datetime   
    # df_aq.drop(columns=["date"], inplace=True)

    # Add location information as columns
    df_aq["country"] = os.getenv("AQICN_COUNTRY")
    df_aq["city"] = os.getenv("AQICN_CITY")
    df_aq["street"] = os.getenv("AQICN_STREET")   # does not have to be an actual street, just a location
    
    # Data validation
    aq_expectation_suite = ge.core.ExpectationSuite(
        expectation_suite_name="aq_expectation_suite"
    )
    
    # Check for null values
    aq_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "pm2_5"}
        )
    )

    # Check for reasonable PM2.5 range
    aq_expectation_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "pm2_5",
                "min_value": 0.0,
                "max_value": 500.0,
                "strict_min": True
            }
        )
    )     
    
    # Align both datasets by overlapping dates (keep only dates that are in both dataframes)
    common_dates = set(weather_df["date"]).intersection(df_aq["date"])
    weather_df = weather_df[weather_df["date"].isin(common_dates)]
    df_aq = df_aq[df_aq["date"].isin(common_dates)]
    
    print(f"Aligned datasets: {len(common_dates)} common days found")

    # Register as feature groups:
    weather_fg = fs.get_or_create_feature_group(
        name='weather',
        description='Historical daily weather observations and weather forecasts',
        version=FG_VERSIONS["weather"],
        primary_key=['city'],
        event_time="date",
        expectation_suite = weather_expectation_suite
    )

    air_quality_fg = fs.get_or_create_feature_group(
        name='air_quality',
        description=f"Air Quality observations daily for {CITY_NAME} ({SENSOR['display_name']})",
        version=FG_VERSIONS["air_quality"],
        primary_key=['city'],
        expectation_suite = aq_expectation_suite,
        event_time="date",
    )
    
    # Validate data locally before ingestion
    validate_with_expectations(weather_df, weather_expectation_suite, name="weather data")
    validate_with_expectations(df_aq, aq_expectation_suite, name="air quality data")

    # Backfill historical data:
    weather_fg.insert(weather_df)
    air_quality_fg.insert(df_aq)

    print(f"Backfill complete! Feature Groups for {CITY_NAME} registered.")


if __name__ == "__main__":
    main()