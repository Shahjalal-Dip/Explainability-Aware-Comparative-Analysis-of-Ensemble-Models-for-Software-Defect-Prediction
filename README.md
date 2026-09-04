# Explainability-Aware Comparative Analysis of Ensemble Models for Software Defect Prediction

## Research Question
> "Do Random Forest, XGBoost, and LightGBM agree on what makes code defective?"

---

## Project Structure

```
SDP_Research/
│
├── data/                          ← All datasets (CSV)
│   ├── CM1.csv                    (498 instances, 9.83% defects)
│   ├── KC1.csv                    (2,109 instances, 15.45% defects)
│   ├── JM1.csv                    (10,885 instances, 19.35% defects)
│   └── PC1.csv                    (1,109 instances, 6.94% defects)
│
├── notebooks/                     ← Run these IN ORDER
│   ├── step1_generate_datasets.py    Generate NASA-mirrored datasets
│   ├── step2_train_and_evaluate.py   5-fold CV + SMOTE + 6 metrics
│   ├── step3_shap_consistency.py     SHAP analysis across all 4 datasets
│   └── step4_feature_correlation.py  Multicollinearity explanation
│
├── results/
│   ├── figures/                   ← All publication figures (PNG)
│   │   ├── fig1_performance_bars.png
│   │   ├── fig2_f1_heatmap.png
│   │   ├── fig3_shap_consistency_heatmaps.png
│   │   ├── fig4_consistency_trend.png
│   │   ├── fig5_shap_top5_all.png
│   │   ├── fig6_feature_correlation_matrix.png
│   │   └── fig7_multicollinearity_pairs.png
│   │
│   └── metrics/                   ← All numerical results (JSON)
│       ├── performance_results.json
│       ├── shap_consistency_all.json
│       └── feature_correlation.json
│
└── README.md                      ← This file
```

---

## How to Run (In Order)

```bash
# 1. Install dependencies
pip install scikit-learn xgboost lightgbm shap imbalanced-learn \
            pandas numpy matplotlib seaborn scipy

# 2. Generate datasets
python notebooks/step1_generate_datasets.py

# 3. Train models and evaluate performance
python notebooks/step2_train_and_evaluate.py

# 4. SHAP explainability and consistency analysis
python notebooks/step3_shap_consistency.py

# 5. Feature correlation and multicollinearity analysis
python notebooks/step4_feature_correlation.py
```

---

## Key Results

### Performance (5-Fold CV with SMOTE)

| Dataset | Best Model | F1     | AUC-ROC | MCC    |
|---------|-----------|--------|---------|--------|
| CM1     | LightGBM  | 0.7710 | 0.9799  | 0.7508 |
| KC1     | RF        | 0.8354 | 0.9819  | 0.8059 |
| JM1     | RF        | 0.8396 | 0.9822  | 0.8009 |
| PC1     | LightGBM  | 0.8458 | 0.9924  | 0.8365 |

### SHAP Consistency (Spearman ρ)

| Dataset | RF ↔ XGB | RF ↔ LGB | XGB ↔ LGB |
|---------|----------|----------|-----------|
| CM1     | 0.488    | 0.554    | 0.963     |
| KC1     | 0.299    | 0.324    | 0.969     |
| JM1     | 0.196    | 0.282    | 0.866     |
| PC1     | 0.378    | 0.380    | 0.928     |
| **Mean**| **0.340**| **0.385**| **0.932** |

**Key finding:** XGBoost and LightGBM agree almost perfectly on feature
importance (ρ = 0.932). Random Forest diverges from both (ρ = 0.340–0.385).

### Mechanistic Explanation

Four software metric pairs have **perfect Pearson correlation (r = 1.000)**:

| Pair              | Relationship         |
|-------------------|----------------------|
| `e` ↔ `t`         | t = e ÷ 18           |
| `v` ↔ `b`         | b = v ÷ 3000         |
| `v_g` ↔ `branchCount` | Definitional     |
| `loc` ↔ `lOCode`  | lOCode ⊂ loc         |

Random Forest (bagging, no regularization) cannot suppress redundant features
and distributes importance across correlated pairs. XGBoost and LightGBM
(L1/L2 regularization + EFB) select one member per pair and zero the other,
producing near-identical, more reliable SHAP explanations.

---

## Dependencies

```
scikit-learn >= 1.3
xgboost      >= 2.0
lightgbm     >= 4.0
shap         >= 0.43
imbalanced-learn >= 0.11
pandas       >= 2.0
numpy        >= 1.24
matplotlib   >= 3.7
seaborn      >= 0.12
scipy        >= 1.11
```

---

## Target Journals

- **IEEE Access** (fast review, open access)
- **Empirical Software Engineering** — Springer (Q1)
- **Information and Software Technology** — Elsevier (Q1)
- **MDPI Applied Sciences** (fast turnaround)
