# Air Quality Prediction Pipeline

This repository contains scripts and notebooks for downloading, processing, and predicting air quality (PM2.5) from specific air quality sensors in Sweden. The pipelines use Hopsworks Feature Store and GitHub Actions to automate daily updates.

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
2. Create a .env file in the root repository and put your Hopsworks API key there (instructions below):
```
HOPSWORKS_API_KEY=<your_api_key_here>
```
### Hopsworks API Key

1. Log in to your Hopsworks project.

2. Go to Account Settings → API.

3. Generate a new API key.

4. Add it to your .env file


## Backfill Feature Pipeline

The backfill feature pipeline downloads historical data (6 years if available) and populates Hopsworks Feature Groups.

#### Steps performed:

1. Fetch historical weather data from Open-Meteo.

2. Load historical air quality CSV from [AQICN](https://aqicn.org/historical/#city:sweden/goteborg-femman).

3. Clean data (e.g. convert PM2.5 to float, time to datetime and rename columns).

4. Register two Feature Groups in Hopsworks:
    - weather
    - air_quality

#### Run the pipeline:
```
python -u pipelines/backfill_feature_pipeline.py
```

## Daily Feature Pipeline

The daily feature pipeline updates Feature Groups with the latest data and a weather prediction for the next 7 days.

#### Steps performed:

1. Fetch yesterday’s weather and air quality data.

2. Fetch weather forecasts for the next 7 days.

3. Update Feature Groups in Hopsworks:
    - weather
    - air_quality
    - weather_forecast_features


#### Run the pipeline once:
```
python -u pipelines/daily_feature_pipeline.py
```

In this repository, we have set up a GitHub workflow that runs the script every day at 06:00.


## Training Pipeline
To be added.

## Batch Inference Pipeline
To be added.

## Automating with GitHub Actions
1. Ensure your repository contains .github/workflows/<workflow>.yml for daily updates. This project uses daily_feature_pipeline.yml.

Use the Hopsworks API key from GitHub Secrets (HOPSWORKS_API_KEY).

Schedule runs via cron, e.g., daily at 06:00 UTC:
```
on:
  schedule:
    - cron: "0 6 * * *"
```

4. Merge workflows from feature branches to main to activate scheduled runs.

## Notes
- All PM2.5 columns must be floats; empty strings or NaNs should be handled with interpolation or removal.

- Increment feature group versions in city_config if schemas change.

- Check GitHub Actions and Hopsworks logs for troubleshooting.
