# scripts/prepare_sft.py
import json
from pathlib import Path
import argparse

PROMPT_TEMPLATE = (
    "You are a helpful assistant that proposes exactly 10 domain names.\n"
    "Business description:\n{desc}\n\n"
    "Return only a list of 10 domain names, one per line."
)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def prepare_sft(input_path, output_path):
    data = load_jsonl(input_path)
    prepared = []
    for ex in data:
        desc = ex.get("input", "").strip()
        out  = (ex.get("output", "") or "").strip()
        if ex.get("meta", {}).get("blocked"):
            out = "Request blocked due to policy."
        prompt = PROMPT_TEMPLATE.format(desc=desc)
        prepared.append({"text": f"{prompt}\n{out}"})
    save_jsonl(prepared, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw JSONL dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save prepared SFT dataset")
    args = parser.parse_args()

    prepare_sft(args.input, args.output)
    print(f"Prepared SFT dataset saved to {args.output}")
