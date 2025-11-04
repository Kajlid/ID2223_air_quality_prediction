import os
import json
import argparse
from datetime import datetime, timedelta
import joblib
import pandas as pd
import hopsworks

def load_city_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(args):
    cfg = load_city_config(args.city_config)
    city = cfg["city_name"]
    fv_name = cfg["feature_view"]["name"]
    fv_version = cfg["feature_view"]["version"]

    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fv = fs.get_feature_view(name=fv_name, version=fv_version)

    start_time = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    feat_df = fv.get_batch_data(start_time=start_time)
    if feat_df.empty:
        raise SystemExit("No feature rows to infer on — run feature pipeline.")

    mr = project.get_model_registry()
    model = mr.get_model("air_quality_model")
    local_dir = model.download()
    model_path = os.path.join(local_dir, "xgb_pipeline.pkl")
    model = joblib.load(model_path)

    pred_df = feat_df.sort_values("date").copy()
    pred_df = pred_df.groupby(["city_name","sensor_id"], as_index=False, group_keys=False).apply(lambda d: d.tail(14))

    feature_cols = [c for c in pred_df.columns if c not in ["pm2_5"]]
    y_hat = model.predict(pred_df[feature_cols])
    pred_df["pm2_5_pred"] = y_hat

    os.makedirs("artifacts", exist_ok=True)
    pred_df.to_csv("artifacts/predictions.csv", index=False)
    print("Wrote artifacts/predictions.csv with columns:", list(pred_df.columns))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-config", required=True)
    args = ap.parse_args()
    main(args)
