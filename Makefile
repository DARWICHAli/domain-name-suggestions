.PHONY: all data train eval edges report api test

all: data train eval

data:
	python scripts/synthesize.py

train:
	python scripts/train.py --config configs/train.yaml

eval:
	python scripts/evaluate.py --config configs/eval.yaml

edges:
	python scripts/discover_edges.py

holdout:
	python scripts/create_holdout.py

report:
	jupyter nbconvert --to html notebooks/06_reporting.ipynb --output artifacts/report.html

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q
