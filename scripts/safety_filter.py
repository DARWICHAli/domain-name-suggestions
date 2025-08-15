import re
from typing import List, Dict

BANNED_KEYWORDS = [
    r"\badult\b", r"\bporn\b", r"\bxxx\b", r"\bsex\b",
    r"\bhate\b", r"\bracist\b", r"\bnazi\b"
]
BANNED_BRANDS = [
    "facebook", "google", "microsoft", "apple", "amazon"
]
ALLOWED_TLDS = {"com", "io", "fr", "ai", "co"}

def is_safe_input(text: str) -> bool:
    if not text.strip():
        return True  # vide = pas bloqué par défaut
    lower_text = text.lower()
    # Mots interdits
    for pattern in BANNED_KEYWORDS:
        if re.search(pattern, lower_text):
            return False
    return True

def is_safe_output(domain: str) -> bool:
    d = domain.lower().strip()
    # Marque interdite
    if any(brand in d for brand in BANNED_BRANDS):
        return False
    # Mots interdits
    for pattern in BANNED_KEYWORDS:
        if re.search(pattern, d):
            return False
    # TLD valide
    parts = d.split(".")
    if len(parts) < 2 or parts[-1] not in ALLOWED_TLDS:
        return False
    return True

def filter_suggestions(domains: List[str]) -> List[str]:
    """Retourne seulement les domaines conformes."""
    clean = []
    seen = set()
    for d in domains:
        if d not in seen and is_safe_output(d):
            seen.add(d)
            clean.append(d)
    return clean

def process_request(description: str, suggestions: List[str]) -> Dict:
    """Retourne dict conforme à la spec API."""
    if not is_safe_input(description):
        return {
            "suggestions": [],
            "status": "blocked",
            "message": "Inappropriate request"
        }
    safe_domains = filter_suggestions(suggestions)
    return {
        "suggestions": [{"domain": d, "confidence": 0.9} for d in safe_domains],
        "status": "success"
    }
