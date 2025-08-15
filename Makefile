.PHONY: all data prep train eval edges report api

all: data prep train eval

data:
	python scripts/synthesize.py --config configs/data.yaml

prep:
	python scripts/prepare_sft.py \
		--input data/processed/train.jsonl \
		--output data/processed/train_sft.jsonl
train:
	python scripts/train.py --config configs/train.yaml

eval:
	python scripts/evaluate.py --config configs/eval.yaml

edges:
	python scripts/discover_edges.py --config configs/edges.yaml

report:
	jupyter nbconvert --to html notebooks/06_reporting.ipynb --output artifacts/report.html

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
