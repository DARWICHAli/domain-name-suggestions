from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

class Req(BaseModel):
    business_description: str

@app.post("/suggest")
def suggest(req: Req):
    if any(word in req.business_description.lower() for word in ["adult", "porn", "xxx"]):
        return {"suggestions": [], "status": "blocked", "message": "Inappropriate request"}
    domains = [f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}.com" for _ in range(5)]
    return {"suggestions": [{"domain": d, "confidence": round(random.uniform(0.7, 0.95),2)} for d in domains],
            "status": "success"}
