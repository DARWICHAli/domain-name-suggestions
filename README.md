# domain-name-suggestions

## 📌 Description
Ce projet implémente un système complet pour générer des noms de domaine via un LLM fine-tuné, avec :
- Génération de données synthétiques
- Fine-tuning LoRA/QLoRA
- Évaluation automatique via LLM-judge
- Tests sur edge cases
- Garde-fous de sécurité
- API REST (FastAPI)

---

## 📂 Structure du dépôt
```
.
├── api/                   # API FastAPI
├── artifacts/             # Checkpoints & résultats évaluation
├── configs/               # Configurations YAML et prompts juge
├── data/
│   ├── processed/          # Données entraînement SFT
│   └── eval/               # Jeux d'évaluation (holdout, edge cases)
├── notebooks/             # Analyses & reporting
├── scripts/               # Scripts de génération, entraînement, évaluation
├── tests/                 # Tests pytest (sécurité)
├── Makefile                # Automatisation commandes
└── README.md
```

---

## ⚙️ Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 🚀 Pipeline d’exécution

### Vue d’ensemble
                ┌─────────────────────────────────────┐
                │           Données synthétiques      │
                │    scripts/synthesize.py (make data)│
                └─────────────────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Préparation SFT                    │
                │  scripts/prepare_sft.py  (make prep)│
                │  ➜ train_sft.jsonl                  │
                └─────────────────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Entraînement LoRA/QLoRA            │
                │  scripts/train.py  (make train)     │
                │  ➜ artifacts/model_vX/              │
                └─────────────────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Jeux d'évaluation                  │
                │   - scripts/create_holdout.py       │
                │   - scripts/discover_edges.py       │
                │  (make holdout / make edges)        │
                └─────────────────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Évaluation automatique             │
                │  scripts/evaluate.py  (make eval)   │
                │  ➜ LLM-judge ou heuristique         │
                │  ➜ artifacts/eval_vX.json           │
                └─────────────────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │  Garde-fous sécurité                │
                │  scripts/safety_filter.py           │
                │  tests/test_safety.py  (make test)  │
                └─────────────────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────────┐
                │           API finale                │
                │     api/main.py  (make api)         │
                │  ➜ intègre safety_filter            │
                │  ➜ sert modèle fine-tuné            │
                └─────────────────────────────────────┘


### 1. Générer le dataset d’entraînement
```bash
make data
```
- **scripts/synthesize.py** : crée un dataset synthétique `data/processed/train.jsonl`.

---

### 2. Préparer le dataset SFT
```bash
make prep
```
- **scripts/prepare_sft.py** : formate les données pour l’entraînement LoRA.

---

### 3. (Optionnel) Créer le holdout
Deux options :
- **Par script** :
```bash
make holdout
```
- **En copiant un fichier existant** dans `data/eval/holdout.jsonl`.

---

### 4. (Optionnel) Générer les edge cases
```bash
make edges
```
- **scripts/discover_edges.py** : crée `data/eval/edge_cases.jsonl`.

---

### 5. Entraîner le modèle LoRA/QLoRA
```bash
make train
```
- **scripts/train.py** : fine-tune un modèle open-source (Llama 3, Mistral…).
- Sauvegarde le checkpoint dans `artifacts/model_v0/`.

---

### 6. Évaluer le modèle
```bash
make eval
```
- **scripts/evaluate.py** : génère des suggestions et les note avec un LLM-judge (OpenAI si clé dispo, sinon heuristique locale).
- Résultats sauvegardés dans `artifacts/eval_v0.json`.

---

### 7. Tester la sécurité
```bash
make test
```
- **tests/test_safety.py** : vérifie que le filtrage bloque bien les contenus sensibles.

---

### 8. Lancer l’API
```bash
make api
```
- **api/main.py** : API REST pour recevoir une description et renvoyer des noms filtrés.
- Exemple :
```bash
curl -X POST "http://127.0.0.1:8000/suggest" -H "Content-Type: application/json" -d '{"business_description":"Coffee shop in Paris"}'
```

---

## 🔒 Sécurité
- **Entrée** : blocage sur mots-clés interdits (`adult`, `porn`, `hate`, etc.).
- **Sortie** : suppression des doublons, blocage des marques connues, TLD restreints (`.com`, `.io`, `.fr`, `.ai`, `.co`).

---

## 📊 Résultats attendus
Après entraînement :
- Score moyen > baseline
- Taux de sécurité ≥ 95%
- Couverture correcte des TLD préférés
- Moins de 5% de suggestions invalides

---

## 📑 Rapport final
À rédiger dans `REPORT.md` :
- Méthodologie
- Résultats initiaux et finaux
- Analyse edge cases
- Recommandations pour déploiement

---

## 👤 Auteur
Projet réalisé par Ali DARWICH dans le cadre du **AI Engineer Homework**.