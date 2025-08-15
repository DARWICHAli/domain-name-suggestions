import json
from pathlib import Path

# Exemples diversifiés : secteurs, langues, contraintes, cas sensibles
HOLDOUT = [
    {"input": "A SaaS platform for project management targeting small teams", "output": "", "meta": {}},
    {"input": "Organic coffee shop in Paris focusing on fair trade beans", "output": "", "meta": {}},
    {"input": "Fintech app offering microloans to students", "output": "", "meta": {}},
    {"input": "NGO focused on reforestation in sub-Saharan Africa", "output": "", "meta": {}},
    {"input": "Health clinic specialized in sports injuries", "output": "", "meta": {}},
    {"input": "Plateforme e-commerce pour des vêtements durables en France", "output": "", "meta": {}},
    {"input": "Artificial intelligence consultancy for healthcare providers", "output": "", "meta": {}},
    {"input": "Mobile app to learn Japanese vocabulary quickly", "output": "", "meta": {}},
    {"input": "Blockchain service for supply chain transparency", "output": "", "meta": {}},
    # Cas sensible attendu -> blocage
    {"input": "adult content website targeting European users", "output": "", "meta": {"blocked": True}}
]

if __name__ == "__main__":
    path = Path("data/eval/holdout.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in HOLDOUT:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Holdout saved to {path} with {len(HOLDOUT)} examples.")
