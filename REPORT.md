# Rapport – Fine-tuning LLM pour la génération de noms de domaine

## 1. Introduction
Ce projet a pour objectif de fine-tuner un modèle de langage afin de générer
des listes de **10 noms de domaine** valides, pertinents et sûrs à partir d’une description d’entreprise.

L’approche suivie repose sur :
- la préparation d’un dataset synthétique,
- l’utilisation de LoRA/QLoRA pour un fine-tuning efficace,
- l’évaluation systématique via un holdout set et des edge cases.

---

## 2. Pipeline global
Le pipeline de traitement est résumé ci-dessous :

[Synthèse données] 
    → [Split train/valid/holdout] 
    → [Fine-tuning (QLoRA)] 
    → [Évaluation holdout + edge cases] 
    → [Comparaison des expériences]



---

## 3. Expériences menées

```text
### Expérience 1 – Baseline
- **Modèle** : Llama-3.2-8B-Instruct (QLoRA)
- **Dataset** : train.jsonl
- **Config** : LoRA (r=16, α=32), 1 epoch, batch_size=2
- **But** : établir un point de référence.

### Expérience 2 – Dataset enrichi + QLoRA
- **Modèle** : Llama-3.2-3B-Instruct
- **Dataset** : train_v1.jsonl (intégrant 20% d’edge cases)
- **Config** : LoRA (r=16, α=32), 4 epochs, batch_size=4
- **But** : améliorer robustesse et sécurité.

### Expérience 3A – Hyperparamètres (epochs ↓, lr ↓)
- **Modèle** : Llama-3.2-3B-Instruct
- **Dataset** : train_v1.jsonl
- **Config** : LoRA (r=16, α=32), 2 epochs, lr=1e-5, batch_size=6
- **But** : tester la généralisation avec moins d’epochs.

### Expérience 3B – LoRA plus expressif
- **Modèle** : Llama-3.2-3B-Instruct
- **Dataset** : train_v1.jsonl
- **Config** : LoRA (r=32, α=64), 2 epochs, lr=2e-5, batch_size=4
- **But** : accroître la capacité d’adaptation.

### Expérience 4 – Modèle plus grand + plus d’epochs
- **Modèle** : Llama-3.1-8B-Instruct
- **Dataset** : train_v1.jsonl
- **Config** : LoRA (r=16, α=32), 10 epochs, batch_size=1
- **But** : Augmenter le nombre de params, et l'epochs.

```
---

## 4. Résultats comparatifs

| Expérience | Modèle                   | Dataset        | Epochs | LoRA (r, α)  | Loss|mean_token_accuracy| Score Holdout | Score Edge | Safety Pass % | 
|------------|-------------------------------|-----------------|-------------|-----------------|----|-------------------|---------------|------------|---------------|
| Exp 1  | Llama-3.2-8B-Instruct (QLoRA) | train.jsonl    | 1  | (16,32) | 0.1901 | 0.9693 | 3.803 | 3.889 | 70% | 
| Exp 2  | Llama-3.2-3B-Instruct (QLoRA) | train_v1.jsonl | 4  | (16,32) | 0.2172 | 0.9675 | 3.792 | N/A | 50% |
| Exp 3A | Llama-3.2-3B-Instruct (QLoRA) | train_v1.jsonl | 2  | (16,32) | 0.2172 | 0.9589 | 3.775 | 3.841 | 80% |
| Exp 3B | Llama-3.2-3B-Instruct (QLoRA) | train_v1.jsonl | 2  | (32,64) | 0.2537 | 0.9663 | 3.839 | 3.9 | 80% |
| Exp 4  | Llama-3.1-8B-Instruct (QLoRA) | train_v1.jsonl | 10 | (16,32) | 0.1227 | 0.9814 | 3.873 | 3.989 | 80% |

---

## 5. Analyse

- **Exp1 (baseline)** : fournit un premier modèle fonctionnel, mais avec un **taux de sécurité faible (70%)** et des scores modestes.  
- **Exp2 (dataset enrichi, 4 epochs)** : légère baisse du taux de sécurité (50%), probablement liée à un **sur-apprentissage** ; le dataset enrichi n’a pas apporté l’amélioration attendue avec ces hyperparamètres.  
- **Exp3A (2 epochs, lr plus faible)** : meilleure généralisation, **taux de sécurité à 80%** avec un compromis intéressant entre qualité et robustesse.  
- **Exp3B (LoRA plus large, r=32, α=64)** : amélioration nette des scores sur **holdout et edge cases**, tout en maintenant la sécurité à 80%. Cela montre que l’augmentation de la capacité des adaptateurs aide le modèle à mieux apprendre sans dégrader la stabilité.  
- **Exp4 (8B, 10 epochs)** : meilleurs résultats globaux sur toutes les métriques, mais au prix d’un **coût mémoire élevé** et d’un **risque d’overfit**. Ce setup est moins pratique pour une utilisation courante.  


---

## 6. Conclusion
En conclusion :  
- Le **dataset enrichi seul (Exp2)** n’a pas suffi : il doit être combiné à un choix judicieux d’hyperparamètres.  
- Les **configurations LoRA plus larges (Exp3B)** apportent un gain réel en qualité et sécurité tout en restant abordables en ressources.  
- L’entraînement long sur un modèle plus grand (**Exp4**) donne les meilleurs résultats absolus, mais il est coûteux et peu scalable.  

👉 Ainsi, **le meilleur compromis est Exp3B**, qui combine de bonnes performances, une sécurité élevée (80%), et un coût d’entraînement raisonnable.  
**Exp4** reste la meilleure configuration en absolu, mais ne serait recommandée que si les ressources (GPU/temps) ne sont pas une contrainte.  

---

## 7. Limites et perspectives

### Limites
- **Dataset** : les données utilisées sont en grande partie synthétiques, ce qui peut limiter la diversité et la représentativité des cas réels.  
- **Évaluation** : les métriques reposent principalement sur des heuristiques et un LLM-judge, sans véritable évaluation humaine de la pertinence ou de la créativité des noms générés.  
- **Overfitting** : certains réglages (Exp2 et Exp4) montrent une tendance à l’overfit, ce qui réduit la capacité du modèle à généraliser.  

### Perspectives
- **Données réelles** : collecter un corpus plus large et plus varié de descriptions d’entreprises réelles afin d’améliorer la robustesse.  
- **Évaluation avancée** : mettre en place une évaluation humaine et/ou basée sur des critères business (ex. pertinence marketing, disponibilité réelle des noms de domaine).  
- **Optimisation mémoire** : explorer d’autres techniques de fine-tuning efficaces en ressources (ex. LoRA + quantization mixte, adapters spécifiques).  
- **RLHF / RLAIF** : intégrer une étape de Reinforcement Learning from Human/AI Feedback pour aligner le modèle sur des préférences de qualité et de sécurité plus fines.  


En définitive, ce projet a montré qu’un fine-tuning léger (QLoRA + LoRA) permet d’adapter efficacement des modèles de grande taille à une tâche spécifique, tout en équilibrant performance, sécurité et contraintes de ressources. Les perspectives ouvertes laissent entrevoir des applications concrètes et extensibles dans des contextes réels.
