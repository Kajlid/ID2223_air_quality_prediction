import streamlit as st
import pandas as pd
import hopsworks
import json
import os
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import pytz
import sys
sys.path.append(".")
import util

berlin_tz = pytz.timezone("Europe/Berlin")
today = pd.Timestamp(datetime.datetime.now().date(), tz=berlin_tz)
today_str = today.strftime("%Y-%m-%d")

st.title("Air Quality Forecast Dashboard")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

@st.cache_resource
def get_hopsworks_project(api_key, project_name):
    return hopsworks.login(project=project_name, api_key_value=api_key)

@st.cache_resource
def load_model(model_dir):
    xgb_model = XGBRegressor()
    xgb_model.load_model(model_dir)
    return xgb_model

project = get_hopsworks_project(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load city configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this script
CONFIG_PATH = os.path.join(BASE_DIR, "../city_config/gothenburg_femman.json")
print(os.path.abspath(CONFIG_PATH))
with open(CONFIG_PATH) as f:
    city_config = json.load(f)
    
country = city_config["city_name"]
city = city_config["country_name"]
street = city_config["street_name"]
FG_VERSIONS = city_config["fg_versions"]
model_name = city_config["model_registry"]["name"]
model_version = city_config["model_registry"]["version"]

# Load model
retrieved_model = mr.get_model(name=model_name, version=model_version)
saved_model_dir = retrieved_model.download()

xgb_model = load_model(Path(saved_model_dir) / "model.json")

# Selection of forecast settings 
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Number of days to forecast", 1, 7, 1)

# Read features
monitor_fg = fs.get_feature_group(
    name="aq_predictions", 
    version=FG_VERSIONS["aq_predictions"]
    )

forecast_horizon = forecast_days
forecast_df = monitor_fg.filter(monitor_fg.days_before_forecast_day <= forecast_days).read().iloc[:forecast_days]
forecast_df["city"] = city
forecast_df["street"] = street
forecast_df["country"] = country

# Start prediction from tomorrow
forecast_df['date'] = pd.date_range(start=today + pd.Timedelta(days=1), periods=forecast_days, freq='D')
forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date
forecast_df['days_before_forecast_day'] = range(1, forecast_horizon + 1)

# Plot forecast
fig = util.plot_air_quality_forecast(city, street, forecast_df, f"air_quality_model/daily_plots/forecast_{today_str}.png")  # saves the plot
st.pyplot(fig)

forecast_df = forecast_df.rename(columns={"date": "Date", "predicted_pm25": "Predicted PM2.5"})

st.write("### Forecast Data")
st.dataframe(forecast_df.reset_index(drop=True)[['Date', 'Predicted PM2.5']].sort_values(["Date"], ascending=True))

if st.checkbox("Show Hindcast (1-day prior predictions vs actual PM2.5 readings)"):
    
    monitoring_df = monitor_fg.filter(
        (monitor_fg.days_before_forecast_day == 1)
    ).read()
    
    air_quality_fg = fs.get_feature_group(name="air_quality", version=FG_VERSIONS["air_quality"])
    air_quality_df = air_quality_fg.read()
    outcome_df = air_quality_df[['date', 'pm2_5']]
    outcome_df = outcome_df[outcome_df['date'] < today]  # because there were placeholder values in the original data for future dates, we only want previous values
    preds_df = monitoring_df[['date', 'predicted_pm25']]
    
    hindcast_df = pd.merge(preds_df, outcome_df, on="date", how="inner")
    hindcast_df = hindcast_df.sort_values(["date"], ascending=True)
    hindcast_df['date'] = pd.to_datetime(hindcast_df['date']).dt.date
    hindcast_df = hindcast_df.rename(columns={"date": "Date", "predicted_pm25": "Predicted PM2.5", "pm2_5": "PM2.5"})
    
    st.write("### Hindcast Data")
    st.dataframe(hindcast_df.reset_index(drop=True))
    
    hindcast_df_plot = hindcast_df.rename(columns={"Date": "date", "Predicted PM2.5":"predicted_pm25", "PM2.5":"pm2_5"})
    fig2 = util.plot_air_quality_forecast(city, street, hindcast_df_plot, f"air_quality_model/daily_plots/hindcast_{today_str}.png", hindcast=True)
    st.pyplot(fig2)
