import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_city_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(args):
    cfg = load_city_config(args.city_config)
    city = cfg["city_name"]
    os.makedirs(args.out_dir, exist_ok=True)

    pred_path = "artifacts/predictions.csv"
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{pred_path} not found. Run batch_inference_pipeline first.")

    df = pd.read_csv(pred_path, parse_dates=["date"])
    sensors = sorted(df["sensor_id"].dropna().unique().tolist())

    for sid in sensors:
        sub = df[df["sensor_id"] == sid].sort_values("date")
        plt.figure()
        plt.plot(sub["date"], sub["pm2_5_pred"], label="Predicted PM2.5")
        if "pm2_5" in sub.columns and not sub["pm2_5"].isna().all():
            plt.scatter(sub["date"], sub["pm2_5"], s=12, label="Observed PM2.5")
        plt.title(f"{city} — PM2.5 Forecast & Hindcast (Sensor {sid})")
        plt.xlabel("Date")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        out_png = os.path.join(args.out_dir, f"pm25_{sid}.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=160)
        plt.close()

    # Minimal HTML without inline CSS braces to avoid tooling issues
    with open(os.path.join(args.out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><title>{}</title></head><body>".format(city + " Air Quality Forecast"))
        f.write("<h1>{}</h1>".format(city + " — Air Quality Forecast"))
        f.write("<p>Forecast & hindcast for all configured sensors.</p><ul>")
        for sid in sensors:
            f.write("<li><a href='pm25_{}.png'>Sensor {}</a></li>".format(sid, sid))
        f.write("</ul></body></html>")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    main(args)
