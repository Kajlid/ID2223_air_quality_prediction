---
title: Air Quality Prediction
emoji: 💨
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.2"
app_file: dashboard/streamlit_app.py
pinned: true
---

# Air Quality Prediction Pipeline

This repository contains scripts and notebooks for downloading, processing, and predicting air quality (PM2.5) from specific air quality sensors in Sweden. The pipelines use Hopsworks Feature Store and GitHub Actions to automate daily updates.

The dashboard app showing predictions of PM2.5 is deployed on HuggingFace and can be viewed here: http://0.0.0.0:8501. 

## Prerequisites
- Python 3.10+

- Conda or virtualenv

- Required Python packages (install via requirements.txt)


## Setup
1. Clone the repository:
```
git clone https://github.com/AxelHolst/ID2223_lab1_legendariskt_basta_gruppen.git
cd ID2223_lab1_legendariskt_basta_gruppen
```
2. Create a .env file in the root repository (either by by copying .env.example or by copying the lines below to existing .env file) and paste your Hopsworks API key and project name there (instructions below):
```
HOPSWORKS_API_KEY=<your_api_key_here>
HOPSWORKS_PROJECT=<your_project_name_here>
AQICN_COUNTRY=Sweden
AQICN_CITY=Göteborg
AQICN_STREET=Femman
```
#### Getting a Hopsworks API Key:

    1. Log in to your Hopsworks project.

    2. Go to Account Settings → API.

    3. Generate a new API key.

    4. Add it to your .env file


3. Add these secrets in your Account Settings in Hopsworks (Account Settings -> Secrets -> New Secret). If you forked the repository, also write the secrets into the GitHub settings (Settings -> Secrets and variables -> Actions -> New repository secret). If you just cloned the repository, you can skip this step.


## Backfill Feature Pipeline

The backfill feature pipeline downloads historical data (6 years if available) and populates Hopsworks Feature Groups.

#### Steps performed:

1. Fetches historical weather data from Open-Meteo.

2. Loads historical air quality CSV from [AQICN](https://aqicn.org/historical/#city:sweden/goteborg-femman).

3. Cleans data (e.g. convert PM2.5 to float, time to datetime and rename columns).

4. Registers two Feature Groups in Hopsworks:
    - weather
    - air_quality

#### Run the pipeline:
```
python -u pipelines/backfill_feature_pipeline.py
```

## Daily Feature Pipeline

The daily feature pipeline updates Feature Groups with the latest data and a weather prediction for the next 7 days.

#### Steps performed:

1. Fetches yesterday’s weather and air quality data.

2. Fetches weather forecasts for the next 7 days.

3. Updates Feature Groups in Hopsworks:
    - weather
    - air_quality
    - weather_forecast_features


#### Run the pipeline:
In order to open it in Jupyter Notebook:
```
jupyter notebook pipelines/2_air_quality_feature_pipeline.ipynb
```
Running it without opening Jupyter Notebook in the browser: 
```
jupyter notebook pipelines/2_air_quality_feature_pipeline.ipynb --no-browser 
```


## Training Pipeline
This pipeline trains an XGBRegressor (XGBoost) to predict air quality (pm25) and register the model with Hopsworks.

#### Steps performed:
1. Selecting features for training the data and create a Feature View.

2. Splitting the training data into train/test data sets based on a time-series split.

3. Training an XGBRegressor model to predict pm2.5.

4. Saving the trained model and performance metrics as well as plots in a model registry (project.get_model_registry()).

#### Run the pipeline:
```
jupyter notebook pipelines/3_air_quality_training_pipeline.ipynb
```

## Batch Inference Pipeline
A batch inference pipeline that creates a dashboard. Downloads the trained model from Hopsworks and plots a dashboard that predicts the air quality for the next 7 days for the chosen location.

#### Steps performed:
1. Downloading the model from Model Registry.

2. Getting Weather Forecast Features with Feature View.

3. Making new predictions and saving the predictions to a new feature group for monitoring.

4. Creating a forecast graph and a hindcast graph.


#### Run the pipeline:
```
jupyter notebook pipelines/4_batch_inference_pipeline.ipynb
```


## Running the UI

The dashboard UI in dashboard/streamlit_app.py can be launched with: 
```streamlit run dashboard/streamlit_app.py```

Before running the UI, you may need to load your environment variables from your local .env file. You can do this with:
```export $(cat .env | xargs)```.
This ensures that required variables are available to the application. If you don’t do this, you might be prompted to manually set your environment variables in the shell.


## Automating with GitHub Actions
Ensure your repository contains .github/workflows/ files for automated updates. This project uses air-quality-daily ( daily_feature_pipeline.yml) that is run on a daily basis, air-quality-train (training_pipeline.yml) that is run on a weekly basis and air-quality-batch-inference (batch_inference_pipeline.yml) that is run on a daily basis (after the daily feature pipeline).

Use the secrets from GitHub Secrets.

Merge workflows from feature branches to main to activate scheduled runs.

## Notes
- All PM2.5 columns must be floats; empty strings or NaNs should be handled with interpolation or removal.

- Increment feature group versions in city_config if schemas change.

- Check GitHub Actions and Hopsworks logs for troubleshooting.

- Plots from the UI (forecast and hindcast graphs) are stored in air_quality_model/daily_plots.
