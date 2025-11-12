import streamlit as st
import pandas as pd
import hopsworks
import json
import os
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import sys
sys.path.append(".")
import util

today = pd.Timestamp(datetime.datetime.now().date())
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

# Instead of generating new predictions, we will use the one's stored in Hopsworks
forecast_df = monitor_fg.filter(monitor_fg.days_before_forecast_day <= forecast_days).read()
forecast_df = forecast_df.sort_values("date")
forecast_df["city"] = city
forecast_df["street"] = street
forecast_df["country"] = country

# Set the first prediction day to today
forecast_horizon = len(forecast_df)
forecast_df['date'] = pd.date_range(
    start=today + pd.Timedelta(days=1),        # start from tomorrow
    periods=forecast_horizon,
    freq='D'
)

# Reset days_before_forecast_day to start from 0 (tomorrow)
# forecast_df['days_before_forecast_day'] = range(0, forecast_horizon)


# Plot forecast
fig = util.plot_air_quality_forecast(city, street, forecast_df, f"air_quality_model/daily_plots/forecast_{today_str}.png")  # saves the plot
st.pyplot(fig)

st.write("### Forecast Data")
st.dataframe(forecast_df[['date', 'predicted_pm25']])

if st.checkbox("Show Hindcast (1-day prior predictions vs actual)"):
    monitor_fg = fs.get_feature_group(name="aq_predictions", version=1)
    monitoring_df = monitor_fg.filter(monitor_fg.days_before_forecast_day == 1).read()
    
    air_quality_fg = fs.get_feature_group(name="air_quality", version=1)
    air_quality_df = air_quality_fg.read()
    outcome_df = air_quality_df[['date', 'pm2_5']]
    preds_df = monitoring_df[['date', 'predicted_pm25']]
    
    hindcast_df = pd.merge(preds_df, outcome_df, on="date", how="inner")
    hindcast_df = hindcast_df.sort_values("date")
    
    st.write("### Hindcast Data")
    st.dataframe(hindcast_df)
    
    fig2 = util.plot_air_quality_forecast(city, street, hindcast_df, f"air_quality_model/daily_plots/hindcast_{today_str}.png", hindcast=True)
    st.pyplot(fig2)
