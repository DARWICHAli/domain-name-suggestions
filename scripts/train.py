# scripts/train.py
import os, json, yaml, math
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import pandas as pd

import mlflow
from mlflow.models.signature import infer_signature

from transformers.integrations import MLflowCallback
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
    
    print("Tokenizing datasets...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    # 4-bit QLoRA configuration
    if cfg["training"].get("qlora", False):
        print("Setting up QLoRA 4-bit configuration...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_cfg = None


    # # 4-bit loading (QLoRA-friendly)
    # bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
    #                              bnb_4bit_use_double_quant=True,
    #                              bnb_4bit_quant_type="nf4",
    #                              bnb_4bit_compute_dtype=torch.bfloat16)
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto", # cant use GPU on my machine
        #torch_dtype=torch.float16,     # safest on CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Configuring model for LoRA...")

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
        learning_rate=float(tr.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(tr.get("batch_size", 1)),  # reduce for memory
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 4)),
        num_train_epochs=int(tr.get("num_train_epochs", 3)),
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
        bf16=cfg["training"]["bf16"],
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        report_to=["mlflow"],             # Enable MLflow logging
        disable_tqdm=False,               # Ensure tqdm progress bar is shown
        seed=int(tr.get("seed", 42))
    )


    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=args_tr,
    #     peft_config=peft_cfg,
    #     train_dataset=train_ds,
    #     eval_dataset=eval_ds,
    #     dataset_text_field="text",
    #     packing=False,
    #     max_seq_length=tokenizer.model_max_length
    # )

    mlflow.set_experiment("domain_name_suggestions")
    mlflow.start_run()


    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    if eval_ds:
        eval_ds = eval_ds.map(tokenize_fn, batched=True)

    
    trainer = SFTTrainer(
    model=model,
    args=args_tr,
    peft_config=peft_cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    #dataset_text_field="text",
    #packing=False,
    #max_seq_length=tokenizer.model_max_length
)
    
    print("Starting training...")
    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    

    # Create a small input example (tokenized)
    #input_example = pd.DataFrame([train_ds[0]["input_ids"]])
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    short_input_ids = train_ds[0]["input_ids"][:10]
    input_tensor = torch.tensor([short_input_ids], dtype=torch.long).to(device)
    input_example = pd.DataFrame(input_tensor.cpu().numpy())

    # Model output
    #output = model(input_tensor).logits.detach().numpy()
    # Inférence sans gradient
    with torch.no_grad():
        # Use autocast with the correct device and dtype
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with torch.amp.autocast(device.type, dtype=dtype):
            output_tensor = model(input_tensor).logits
    output = output_tensor.detach().cpu().numpy()


    # Infer signature from input example and model output
    signature = infer_signature(input_tensor.cpu().numpy(), output)

    # Log model with signature and input example
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="domain-name-suggester",
        signature=signature,
        input_example=input_example
    )

    # Optionally log the config file
    mlflow.log_artifact(args.config)

    # End the MLflow run
    mlflow.end_run()
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
