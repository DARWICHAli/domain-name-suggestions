# scripts/train.py
import os, json, yaml, math
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer
from peft import LoraConfig

PROMPT_TEMPLATE = (
    "You are a helpful assistant that proposes exactly 10 domain names.\n"
    "Business description:\n{desc}\n\n"
    "Return only a list of 10 domain names, one per line."
)

def load_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_dataset(jsonl_path: str):
    data = load_jsonl(jsonl_path)
    texts=[]
    for ex in data:
        desc = ex.get("input","").strip()
        out  = (ex.get("output","") or "").strip()
        # ex.meta may mark blocked examples; keep them as refusals
        if ex.get("meta",{}).get("blocked"):
            out = "Request blocked due to policy."
        prompt = PROMPT_TEMPLATE.format(desc=desc)
        texts.append({"text": f"{prompt}\n{out}"})
    return Dataset.from_list(texts)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    out_dir    = cfg["output_dir"]
    logging_dir= cfg.get("logging_dir", "artifacts/logs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = build_dataset(cfg["data"]["train_file"])
    val_file = cfg["data"].get("val_file")
    eval_ds  = build_dataset(val_file) if val_file else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit loading (QLoRA-friendly)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA
    lora = cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get("target_modules", ["q_proj","v_proj"])
    )

    tr = cfg.get("training", {})
    args_tr = TrainingArguments(
        output_dir=out_dir,
        logging_dir=logging_dir,
        learning_rate=tr.get("learning_rate", 2e-5),
        per_device_train_batch_size=tr.get("batch_size", 4),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 4),
        num_train_epochs=tr.get("num_train_epochs", 3),
        bf16=tr.get("bf16", True),
        gradient_checkpointing=tr.get("gradient_checkpointing", True),
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        logging_steps=25,
        report_to=["none"],
        seed=tr.get("seed", 42)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args_tr,
        peft_config=peft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        packing=False,
        max_seq_length=tokenizer.model_max_length
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
