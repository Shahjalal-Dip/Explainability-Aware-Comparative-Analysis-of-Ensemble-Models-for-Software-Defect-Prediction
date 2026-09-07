Evaluating SHAP Explanation Consistency in Ensemble Models for Software Defect Prediction
This project investigates whether different machine learning models trained on the same software defect prediction task agree with each other's explanations, not just their predictions. Four classifiers — Random Forest (RF), XGBoost (XGB), LightGBM (LGBM), and a Multi-Layer Perceptron (MLP) — are trained on the NASA JM1 software metrics dataset, and their SHAP (SHapley Additive exPlanations) values are compared across models, across repeated runs of the same model, and with bootstrap-based robustness checks.
The central contribution is a Cross-Model Explanation Consistency Index (CECI): a composite metric that quantifies how much models agree on why a module is predicted as defective, combining global feature-ranking agreement with local (per-instance) explanation agreement.
> Evaluates whether SHAP explanations agree across ensemble models (Random Forest, XGBoost, LightGBM) and a neural network (MLP) trained on the NASA JM1 software defect dataset. Introduces a Cross-Model Explanation Consistency Index (CECI) to quantify agreement, with bootstrap robustness and statistical validation.
Key Findings
Overall CECI: 0.776 (77.6%) — a moderately-high-to-high level of cross-model explanation agreement, but not perfect consistency.
Most consistent pair: RF–XGB (CECI = 0.876). Least consistent pair: RF–MLP (CECI = 0.684).
Most consistent model overall: XGB. Least consistent (most divergent) model: MLP.
Global (population-level) feature-importance rankings are fairly stable across model families; local (instance-level) explanations are systematically less consistent — the weakest component is local Spearman rank agreement.
CECI is highly stable under bootstrap resampling (5,000 samples; 95% CI = [0.7738, 0.7779]), so the result is not an artifact of the specific test split.
Predictive performance and explanation consistency diverge: the best model by CV F1 (RF) is not the model with the most consistent SHAP explanations (XGB). Model choice is not "explanation-neutral."
Statistical validation (Wilcoxon signed-rank test, Holm-Bonferroni corrected, α = 0.05) confirms all 6 pairwise local-SHAP correlations are significantly non-zero, including the weakest pair (RF–MLP).
Full narrative writeup: `data/results/final_experimental_results/FINAL_EXPERIMENTAL_RESULTS_REPORT.txt`
Dataset
NASA JM1 — a public software defect prediction dataset consisting of static code metrics (McCabe complexity, Halstead metrics, lines of code, etc.) extracted from C/C++ modules, labeled defective/non-defective. The raw file lives at `data/raw/jm1.csv`; a cleaned version is produced by the pipeline at `data/processed/JM1_cleaned.csv`.
Project Structure
```
.
├── data/
│   ├── raw/                        # Original JM1 dataset
│   ├── processed/                  # Cleaned data + train/dev/test splits
│   └── results/                    # CSV/JSON/TXT outputs from every pipeline stage
├── figures/                        # All generated plots (EDA, performance, SHAP, etc.)
├── models/
│   ├── final/                      # Final trained RF, XGB, LGBM, MLP models
│   ├── intra_model_shap_stability/ # 5 repeated-training runs per model (stability check)
│   └── rf_rid/                     # Reduced-feature RF variant (multicollinearity control)
├── scripts/                        # Numbered, sequential pipeline (01 → 20), see below
└── requirements.txt
```
Pipeline
Scripts are numbered in the order they're meant to be run. Each stage reads from `data/` and/or `models/`, and writes its outputs back into the corresponding `data/results/<stage>/` and `figures/<stage>/` folders.
Script	Purpose
`01_dataset_inspection.py`	Initial inspection of the raw JM1 dataset
`02_data_cleaning.py`	Cleaning pass (missing values, type fixes)
`02a_duplicate_conflict_analysis.py`	Detect duplicate/conflicting records
`02b_finalize_cleaning.py`	Finalize the cleaned dataset
`03_eda.py`	Exploratory data analysis (distributions, class balance, correlations)
`04_baseline_models.py`	Baseline model training (RF, XGB, LGBM, MLP)
`04a_train_test_spilt.py`	Fixed, stratified 80/20 development/test split (seed = 42)
`04b_baseline_models_development.py`	Cross-validated baseline training on the development set
`05_smote_experiment.py`	SMOTE class-balancing experiment
`06_smote_vs_no_smote_comparison.py`	Compare SMOTE vs. no-SMOTE performance
`07_hyperparameter_tuning.py`	Hyperparameter search for each model
`08_final_model_selection.py`	Select final model configurations by CV performance
`09_retrain_final_models.py`	Retrain final models on the full development set
`10_independent_test_evaluation.py`	Evaluate final models on the held-out test set
`11_multicolinearity_analysis.py`	VIF / Pearson correlation analysis of input features
`12_rf_vs_rf_rid.py`	Compare full RF vs. a reduced-feature RF (multicollinearity control)
`13_shap_explanation_analysis.py`	Compute SHAP values for all final models
`13b_rf_vs_rf_rid_shap.py`	SHAP comparison between full RF and reduced-feature RF
`14_intra_model_shap_stability.py`	Same-model SHAP stability across 5 repeated training runs
`15_shap_stability_index.py`	Aggregate intra-model stability into a single index
`16_unified_cross_model_shap.py`	Cross-model SHAP comparison (global + local)
`17_statistical_validation_shap_consistency.py`	Significance testing for cross-model SHAP agreement
`18_cross_model_explanation_consistency_index.py`	Compute the overall CECI metric
`19_final_ceci_robustness_analysis.py`	Bootstrap robustness analysis of CECI
`20_final_experimental_results.py`	Compile final results tables and the summary report
Run them in order from the project root, e.g.:
```bash
python scripts/01_dataset_inspection.py
python scripts/02_data_cleaning.py
...
python scripts/20_final_experimental_results.py
```
> **Note:** these scripts were developed and run on Windows; a couple of the earlier scripts (e.g. `13_shap_explanation_analysis.py`) contain a hardcoded absolute `PROJECT_ROOT` path. Update that path (or switch it to the `Path(__file__).resolve().parent.parent` pattern used in the other scripts) before running on a different machine.
Setup
```bash
git clone <your-repo-url>
cd "Evaluating SHAP Explanation Consistency in Ensemble Models for Software Defect Prediction"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```
Requires Python 3.10+.
Repository Notes
`models/*.pkl` and everything under `.venv/` are excluded from version control (see `.gitignore`) — trained model artifacts are fairly large (~100 MB) and are reproducible by re-running the pipeline; the virtual environment should never be committed.
All figures in `figures/` and result tables in `data/results/` are pipeline outputs and can be regenerated by re-running the corresponding numbered script.
Citation
If you use this work, please cite it as:
> *Evaluating SHAP Explanation Consistency in Ensemble Models for Software Defect Prediction*, applied to the NASA JM1 dataset.