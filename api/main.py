from fastapi import FastAPI
from pydantic import BaseModel
from scripts.safety_filter import process_request
import random

app = FastAPI()

class Req(BaseModel):
    business_description: str

@app.post("/suggest")
def suggest(req: Req):
    """
    Génère des suggestions de noms de domaine et applique le filtrage sécurité.
    """
    # Génération factice (à remplacer par appel au modèle fine-tuné)
    generated = [
        f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}.com"
        for _ in range(5)
    ]
    # Passage par le garde-fous
    return process_request(req.business_description, generated)
