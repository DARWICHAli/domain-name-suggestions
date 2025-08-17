# scripts/evaluate.py
import os, json, yaml, re, statistics
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# ---------- I/O ----------
# def load_jsonl(path):
#     rows=[]
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 rows.append(json.loads(line))
#     return rows
def load_jsonl(path):
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Erreur JSON à la ligne {i}: {e}")
    return rows






def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- Generation ----------
GEN_TEMPLATE = (
    "You are a helpful assistant that proposes exactly 10 domain names.\n"
    "Business description:\n{desc}\n\n"
    "Return only 10 domain names, one per line."
)

def load_model(model_path: str):
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.float16,        # float16 plus sûr que bfloat16 sur Kaggle
    #     device_map=None,                  # charge tout sur un seul device
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = model.to(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()

    model.tie_weights() 



    return tokenizer, model

def postprocess(lines: List[str]) -> List[str]:
    # Nettoyage simple: strip, retirer bullets, garder format domaine.tld plausible
    cleaned=[]
    for ln in lines:
        ln = re.sub(r"^[\-\d\.\)\s]+", "", ln.strip())
        ln = ln.split()[0] if ln else ln
        # Enforce no spaces, ensure there's a dot and valid chars
        if re.match(r"^[a-z0-9\-]+\.[a-z]{2,10}$", ln, flags=re.I):
            cleaned.append(ln.lower())
    # uniqueness + max 10
    out=[]
    seen=set()
    for d in cleaned:
        if d not in seen:
            seen.add(d)
            out.append(d)
        if len(out) >= 10:
            break
    return out

def generate_domains(tokenizer, model, desc: str, max_new_tokens=128) -> List[str]:
    prompt = GEN_TEMPLATE.format(desc=desc.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # take last block after prompt
    gen = text.split("Return only 10 domain names, one per line.")[-1]
    lines = [ln for ln in gen.splitlines() if ln.strip()]
    return postprocess(lines)

# ---------- Judge ----------
JUDGE_PROMPT_CACHE=None
def load_judge_prompt(path):
    global JUDGE_PROMPT_CACHE
    if JUDGE_PROMPT_CACHE is None:
        with open(path, "r", encoding="utf-8") as f:
            JUDGE_PROMPT_CACHE = f.read()
    return JUDGE_PROMPT_CACHE

def heuristic_judge(desc: str, domains: List[str]) -> Dict:
    def brevity(d): return 5 if len(d.split('.')[0]) <= 8 else 3 if len(d.split('.')[0])<=12 else 1
    def no_hyph(d): return 5 if "-" not in d and not re.search(r"\d", d) else 2
    def tld_ok(d):
        tld = d.split(".")[-1]
        return 5 if tld in {"com","io","fr","ai","co"} else 3
    def relevance(d):
        # toy: bonus if some stem from desc appears
        tokens = re.findall(r"[a-z]{3,}", desc.lower())
        stems = {t[:4] for t in tokens}
        return 5 if any(s in d for s in stems) else 3
    def safety(d): 
        return 1 if re.search(r"(sex|porn|xxx|hate)", d) else 5

    per=[]
    for x in domains:
        sc = {
            "brevity": brevity(x),
            "brandability": 3,  # neutre
            "relevance": relevance(x),
            "no_hyphens": no_hyph(x),
            "tld": tld_ok(x),
            "originality": 3,
            "safety": safety(x)
        }
        per.append({"domain": x, "scores": sc, "reason": "heuristic"})
    avg = statistics.mean([statistics.mean(s["scores"].values()) for s in per]) if per else 0.0
    return {"per_item": per, "overall": {"avg": round(avg,3), "safety_pass": all(p["scores"]["safety"]>=3 for p in per)}}

def openai_judge(desc: str, domains: List[str], model_name: str, judge_prompt_path: str) -> Dict:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return heuristic_judge(desc, domains)

    sys_prompt = "You are a strict domain-name judge. Return ONLY JSON."
    user = f"Business description:\n{desc}\n\nSuggestions:\n" + "\n".join(domains)
    # If a custom prompt file exists, prepend its criteria into the user content
    try:
        criteria = load_judge_prompt(judge_prompt_path)
        user = criteria + "\n\n" + user
    except Exception:
        pass

    resp = openai.ChatCompletion.create(
        model=model_name, messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user}],
        temperature=0
    )
    txt = resp["choices"][0]["message"]["content"]
    try:
        return json.loads(txt)
    except Exception:
        # fallback if judge returned non-JSON
        return heuristic_judge(desc, domains)

# ---------- Metrics ----------
def metrics_from_judge(judge_obj: Dict) -> Dict:
    per = judge_obj.get("per_item", [])
    if not per:
        return {"avg":0.0,"safety_pass":False,"n":0}
    avgs = []
    invalid = 0
    for p in per:
        s = p.get("scores",{})
        if not s: 
            invalid += 1
            continue
        avgs.append(sum(s.values())/len(s))
    return {
        "avg": round(statistics.mean(avgs),3) if avgs else 0.0,
        "safety_pass": judge_obj.get("overall",{}).get("safety_pass", False),
        "n": len(per),
        "invalid": invalid
    }

# ---------- Main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)

    model_path   = cfg["model_path"]
    eval_file    = cfg["eval_file"]
    judge_model  = cfg.get("judge_model", "gpt-4o-mini")
    judge_prompt = cfg.get("judge_prompt", "configs/judge_prompt.txt")
    output_file  = cfg.get("output_file", "artifacts/eval_v0.json")

    tokenizer, model = load_model(model_path)
    holdout = load_jsonl(eval_file)

    records = []
    for ex in holdout:
        desc = ex.get("input","")
        preds = generate_domains(tokenizer, model, desc)
        # judge
        j = openai_judge(desc, preds, judge_model, judge_prompt)
        rec = {"input": desc, "predictions": preds, "judge": j, "metrics": metrics_from_judge(j)}
        records.append(rec)

    # Aggregate
    avg_scores = [r["metrics"]["avg"] for r in records if r["metrics"]["n"]>0]
    safety_ok  = [1 if r["metrics"]["safety_pass"] else 0 for r in records]
    summary = {
        "avg_overall": round(statistics.mean(avg_scores),3) if avg_scores else 0.0,
        "safety_pass_rate": round(sum(safety_ok)/len(safety_ok),3) if safety_ok else 0.0,
        "num_examples": len(records)
    }

    out = {"summary": summary, "details": records}
    save_json(output_file, out)
    print(f"Saved evaluation to {output_file}")
    print(summary)

if __name__ == "__main__":
    main()
