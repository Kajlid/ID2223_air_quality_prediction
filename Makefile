.PHONY: setup backfill daily train predict dashboard all

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

backfill:
	python pipelines/feature_daily_pipeline.py --city-config city_config/stockholm.json --backfill 1

daily:
	python pipelines/feature_daily_pipeline.py --city-config city_config/stockholm.json

train:
	python pipelines/training_pipeline.py --feature-view-version 1

predict:
	python pipelines/batch_inference_pipeline.py --city-config city_config/stockholm.json

dashboard:
	python dashboard/generate_dashboard.py --city-config city_config/stockholm.json --out-dir docs

all: daily train predict dashboard
