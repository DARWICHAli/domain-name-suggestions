import json, random
from pathlib import Path

BUSINESS_TYPES = ["SaaS", "coffee shop", "fintech app", "NGO", "health clinic"]
TLD_OPTIONS = [".com", ".io", ".fr"]
STYLES = ["brandable", "minimal", "descriptive"]
SENSITIVE = ["adult content", "hate speech", "gambling"]

random.seed(42)

def synthesize_dataset(n=100):
    data = []
    for _ in range(n):
        if random.random() < 0.1:
            desc = random.choice(SENSITIVE)
            output = ""
            meta = {"blocked": True}
        else:
            desc = f"{random.choice(BUSINESS_TYPES)} in {random.choice(['Paris','NY','Tokyo'])}"
            tlds = random.sample(TLD_OPTIONS, k=random.randint(1,len(TLD_OPTIONS)))
            meta = {"tld_prefs": tlds, "style": random.choice(STYLES)}
            output = "\n".join([f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5,10)))}{t}" for t in tlds])
        data.append({"input": desc, "output": output, "meta": meta})
    return data

# if __name__ == "__main__":
#     Path("data/processed").mkdir(parents=True, exist_ok=True)
#     dataset = synthesize_dataset(200)
#     with open("data/processed/train.jsonl", "w") as f:
#         for row in dataset:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    dataset = synthesize_dataset(200)
    split_idx = int(0.9 * len(dataset))
    train_set = dataset[:split_idx]
    valid_set = dataset[split_idx:]

    with open("data/processed/train.jsonl", "w") as f:
        for row in train_set:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open("data/processed/valid.jsonl", "w") as f:
        for row in valid_set:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")