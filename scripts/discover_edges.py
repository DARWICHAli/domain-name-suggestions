# scripts/discover_edges.py
import json
import random
import re
from pathlib import Path

EDGE_CASES = [
    # Très longs
    "A startup providing hyper-specialized AI-powered drone delivery for luxury perfumes in mountainous rural areas of Switzerland with complex weather conditions and seasonal tourism cycles",
    # Ambigu
    "Service that is both a fitness app and a food delivery platform",
    # Multilingue
    "Plateforme de gestion financière pour PME en Afrique francophone with AI analytics",
    # Emojis
    "🌱 Plant care app for 🌿 exotic indoor plants",
    # Contraintes contradictoires
    "A luxury jewelry brand targeting teenagers with budget under $20",
    # Vide
    "",
    # Marques connues
    "A service similar to Facebook but for pet owners",
    # Offensive
    "adult dating platform with explicit content"
]

def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Marquer certains comme bloqués
    data = []
    for prompt in EDGE_CASES:
        meta = {}
        if re.search(r"(adult|explicit)", prompt.lower()):
            meta["blocked"] = True
        data.append({"input": prompt, "output": "", "meta": meta})

    path = Path("data/eval/edge_cases.jsonl")
    save_jsonl(data, path)
    print(f"Edge cases saved to {path} with {len(data)} examples.")
