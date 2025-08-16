import json, random
from pathlib import Path

train_p = Path("data/processed/train.jsonl")
edges_p = Path("data/eval/edge_cases.jsonl")
out_p   = Path("data/processed/train_v1.jsonl")

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

train = load(train_p)
edges = load(edges_p)

k = max(1, int(0.5 * len(edges)))  # ~50%
picked = random.sample(edges, k)

for ex in picked:
    ex.setdefault("meta", {})["source"] = "edge"

merged = train + picked

with open(out_p, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Merged {len(train)} + {len(picked)} -> {out_p}")
