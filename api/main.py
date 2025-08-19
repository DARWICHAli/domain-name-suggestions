from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from scripts import safety_filter as sf



app = FastAPI(title="Domain Name Suggester API")

# Charger ton modèle fine-tuné
MODEL_PATH = "artifacts/model_v3b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",torch_dtype="auto" )

# Créer un pipeline HuggingFace
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer#,
    #device=0 if torch.cuda.is_available() else -1,
)

class BusinessDesc(BaseModel):
    description: str

PROMPT_TEMPLATE = (
    "You are a helpful assistant that proposes exactly 10 domain names.\n"
    "Business description:\n{desc}\n\n"
    "Return only a list of 10 domain names, one per line."
)

@app.post("/generate")
def generate_names(req: BusinessDesc):
    prompt = PROMPT_TEMPLATE.format(desc=req.description)
    outputs = generator(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    print(outputs)
    # Extraire uniquement le texte généré
    result = outputs[0]["generated_text"].split("\n")[1:]
    # Nettoyer les lignes vides
    result = [r.strip() for r in result if r.strip()]

    result = sf.process_request(req.description, result)


    if result["status"] == "blocked":
        raise HTTPException(status_code=400, detail=result["message"])

    return {"domain_names": result}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}