import os
import json
import argparse
from datetime import datetime, timedelta
import requests
import pandas as pd
import hopsworks

OPEN_METEO_AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"

def load_city_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_pm25_column(df):
    lower = {c.lower(): c for c in df.columns}
    for k in ["pm2_5","pm25","pm2.5","value","median"]:
        if k in lower:
            df = df.rename(columns={lower[k]:"pm2_5"})
            break
    if "pm2_5" not in df.columns:
        raise ValueError(f"No pm2_5-like column in {list(df.columns)}")
    return df

def normalize_date_column(df):
    lower = {c.lower(): c for c in df.columns}
    for k in ["date","day","time","timestamp"]:
        if k in lower:
            df = df.rename(columns={lower[k]:"date"})
            break
    if "date" not in df.columns:
        raise ValueError("No date column found")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().dt.tz_localize(None).dt.date
    return df

def fetch_weather_daily(lat, lon, start=None, end=None, forecast_days=7):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max","wind_speed_10m_max","wind_gusts_10m_max","wind_direction_10m_dominant"],
        "timezone": "UTC"
    }
    if start and end:
        params["start_date"] = start
        params["end_date"] = end
    else:
        params["forecast_days"] = forecast_days
    r = requests.get(OPEN_METEO_WEATHER, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()
    daily = d.get("daily", {})
    if not daily:
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time"])
    df = df.rename(columns={
        "temperature_2m_max":"temp_max",
        "wind_speed_10m_max":"wind_speed_max",
        "wind_gusts_10m_max":"wind_gusts_max",
        "wind_direction_10m_dominant":"wind_direction_dominant"
    })
    return df

def fetch_pm25_forecast(lat, lon, forecast_days=7):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["pm2_5_mean"],
        "timezone": "UTC",
        "forecast_days": forecast_days
    }
    r = requests.get(OPEN_METEO_AIR_QUALITY, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()
    daily = d.get("daily", {})
    if not daily:
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time"])
    df = df.rename(columns={"pm2_5_mean": "pm2_5"})
    return df

def ge_filter_bad_rows(df):
    if "pm2_5" in df.columns:
        df = df[(df["pm2_5"] >= 0) & (df["pm2_5"] <= 500)]
    if "temp_max" in df.columns:
        df = df[(df["temp_max"] > -70) & (df["temp_max"] < 70)]
    for c in ["wind_speed_max","wind_gusts_max"]:
        if c in df.columns:
            df = df[(df[c] >= 0) & (df[c] < 80)]
    return df

def to_feature_group(df, fs, name, version, description, primary_key, event_time):
    if df.empty:
        return
    df = ge_filter_bad_rows(df.copy())
    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time=event_time
    )
    if event_time in df.columns:
        df[event_time] = pd.to_datetime(df[event_time])
    fg.insert(df)

def load_observed_pm25_from_csvs(data_dir, city_name, sensors):
    import os, pandas as pd
    rows = []
    for s in sensors:
        sid = s["id"]
        fp = os.path.join(data_dir, f"{sid}.csv")
        if not os.path.exists(fp):
            continue
        raw = pd.read_csv(fp)
        raw = normalize_date_column(raw)
        raw = normalize_pm25_column(raw)
        tmp = raw[["date","pm2_5"]].copy()
        tmp["city_name"] = city_name
        tmp["sensor_id"] = sid
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["city_name","sensor_id","date","pm2_5"])
    df = pd.concat(rows, ignore_index=True)
    df = df.groupby(["city_name","sensor_id","date"], as_index=False)["pm2_5"].mean()
    return df[["city_name","sensor_id","date","pm2_5"]]

def main(args):
    cfg = load_city_config(args.city_config)
    city = cfg["city_name"]
    lat, lon = cfg["city_lat"], cfg["city_lon"]
    sensors = cfg["sensors"]
    fg_versions = cfg.get("fg_versions", {"weather":1,"air_quality":1})
    forecast_days = int(cfg.get("forecast_days", 7))

    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    if args.backfill:
        end = datetime.utcnow().date()
        start = end - timedelta(days=400)

        weather_hist = fetch_weather_daily(lat, lon, start=start.isoformat(), end=end.isoformat())
        all_w = []
        for s in sensors:
            wdf = weather_hist.copy()
            wdf["city_name"] = city
            wdf["sensor_id"] = s["id"]
            all_w.append(wdf[["city_name","sensor_id","date","wind_speed_max","wind_gusts_max","wind_direction_dominant","temp_max"]])
        weather_df = pd.concat(all_w, ignore_index=True)
        to_feature_group(
            weather_df, fs,
            name="weather", version=fg_versions["weather"],
            description="Daily weather (replicated per sensor for alignment)",
            primary_key=["city_name","sensor_id"],
            event_time="date"
        )

        aq_obs = load_observed_pm25_from_csvs("data", city, sensors)
        if not aq_obs.empty:
            to_feature_group(
                aq_obs, fs,
                name="air_quality", version=fg_versions["air_quality"],
                description="Observed daily PM2.5 from AQICN CSVs",
                primary_key=["city_name","sensor_id"],
                event_time="date"
            )
        else:
            print("WARNING: No observed PM2.5 CSVs found in ./data; hindcast will be empty.")
    else:
        yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
        wday = fetch_weather_daily(lat, lon, start=yesterday, end=yesterday)
        all_w = []
        for s in sensors:
            wdf = wday.copy()
            wdf["city_name"] = city
            wdf["sensor_id"] = s["id"]
            all_w.append(wdf[["city_name","sensor_id","date","wind_speed_max","wind_gusts_max","wind_direction_dominant","temp_max"]])
        weather_df = pd.concat(all_w, ignore_index=True)
        to_feature_group(
            weather_df, fs,
            name="weather", version=fg_versions["weather"],
            description="Daily weather features",
            primary_key=["city_name","sensor_id"],
            event_time="date"
        )

        aq_fore = fetch_pm25_forecast(lat, lon, forecast_days=forecast_days)
        all_aq = []
        for s in sensors:
            adf = aq_fore.copy()
            adf["city_name"] = city
            adf["sensor_id"] = s["id"]
            all_aq.append(adf[["city_name","sensor_id","date","pm2_5"]])
        aq_df = pd.concat(all_aq, ignore_index=True)
        to_feature_group(
            aq_df, fs,
            name="air_quality", version=fg_versions["air_quality"],
            description="Forecast PM2.5 (Open-Meteo) replicated per sensor",
            primary_key=["city_name","sensor_id"],
            event_time="date"
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-config", required=True)
    ap.add_argument("--backfill", type=int, default=0, help="1 = ~400d historical backfill")
    args = ap.parse_args()
    main(args)
