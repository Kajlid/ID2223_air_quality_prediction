import os
import argparse
import joblib
import pandas as pd
import numpy as np
import hopsworks
from hsfs.feature_store import FeatureStore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

def main(fv_version):
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
    fs: FeatureStore = project.get_feature_store()
    fg_aq = fs.get_feature_group(name="air_quality", version=1)
    fg_w  = fs.get_feature_group(name="weather", version=1)

    q = fg_aq.select(["city_name","sensor_id","date","pm2_5"]).join(fg_w.select_all())
    fv = fs.get_or_create_feature_view(
        name="air_quality_fv",
        version=fv_version,
        description="PM2.5 forecasting features per (city,sensor)",
        labels=["pm2_5"],
        query=q
    )

    df = fv.get_batch_data()
    if df.empty:
        raise SystemExit("No data from Feature View. Run feature pipeline.")
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=7)
    df = df[df["date"] <= cutoff]

    df = df.set_index(["city_name","sensor_id","date"]).sort_index()
    for l in [1,2,3]:
        df[f"pm2_5_lag{l}"] = df.groupby(level=[0,1])["pm2_5"].shift(l)
    df = df.reset_index().dropna()

    y = df["pm2_5"].values
    X = df.drop(columns=["pm2_5"])

    cat_cols = ["city_name","sensor_id","wind_direction_dominant"]
    num_cols = [c for c in X.columns if c not in cat_cols and c != "date"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )
    model = Pipeline(steps=[
        ("pre", preprocessor),
        ("rgr", XGBRegressor(
            n_estimators=450,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ))
    ])

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print({"mae": mae, "r2": r2})

    mr = project.get_model_registry()
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/xgb_pipeline.pkl"
    joblib.dump(model, model_path)

    input_schema = Schema(X_test)
    output_schema = Schema(pd.Series(y_test, name="pm2_5"))
    aq_model = mr.python.create_model(
        name="air_quality_model",
        version=None,
        metrics={"mae": mae, "r2": r2},
        description="XGBoost pipeline for daily PM2.5 with per-sensor features and lags",
        input_example=X_test.iloc[:1],
        model_schema=ModelSchema(input_schema=input_schema, output_schema=output_schema)
    )
    aq_model.save(model_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-view-version", type=int, default=1)
    args = ap.parse_args()
    main(args.feature_view_version)
