import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       roc_auc_score, matthews_corrcoef)
from imblearn.over_sampling    import SMOTE
from xgboost                   import XGBClassifier
from lightgbm                  import LGBMClassifier

# Paths
BASE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE, '..', 'data')
FIG_DIR     = os.path.join(BASE, '..', 'results', 'figures')
METRICS_DIR = os.path.join(BASE, '..', 'results', 'metrics')
os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Settings 
DATASETS    = ['CM1', 'KC1', 'JM1', 'PC1']
MODEL_NAMES = ['Random Forest', 'XGBoost', 'LightGBM']
N_FOLDS     = 5
SEED        = 42
COLORS      = {
    'Random Forest': '#2196F3',
    'XGBoost':       '#FF5722',
    'LightGBM':      '#4CAF50',
}


def get_models(neg_pos_ratio: float) -> dict:
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            scale_pos_weight=neg_pos_ratio,   # handles imbalance
            eval_metric='logloss',
            random_state=SEED,
            verbosity=0,
            n_jobs=-1,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        ),
    }


def evaluate_fold(model, X_test, y_test) -> dict:
    """Compute all 6 metrics for one fold."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1':        f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC':   roc_auc_score(y_test, y_prob),
        'MCC':       matthews_corrcoef(y_test, y_pred),
    }


# Main experiment loop 
print("=" * 60)
print("Model Training & Performance Evaluation")
print("=" * 60)

RESULTS = {}
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for ds_name in DATASETS:
    path = os.path.join(DATA_DIR, f'{ds_name}.csv')
    df   = pd.read_csv(path)
    X    = df.drop('defects', axis=1).values.astype(float)
    y    = df['defects'].values.astype(int)
    ratio = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))

    print(f"\nDataset: {ds_name}  "
          f"(n={len(df):,}, defect_rate={y.mean():.3f}, "
          f"neg/pos ratio={ratio:.1f})")

    RESULTS[ds_name] = {}
    models = get_models(ratio)

    for m_name, clf in models.items():
        fold_metrics = {k: [] for k in
                        ['Accuracy','Precision','Recall','F1','AUC-ROC','MCC']}

        for fold_idx, (train_idx, test_idx) in \
                enumerate(skf.split(X, y), start=1):

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # SMOTE applied only on training fold 
            X_tr_sm, y_tr_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

            clf.fit(X_tr_sm, y_tr_sm)
            fold_result = evaluate_fold(clf, X_te, y_te)

            for metric, val in fold_result.items():
                fold_metrics[metric].append(val)

        # Average across folds
        avg = {m: round(float(np.mean(vals)), 4)
               for m, vals in fold_metrics.items()}
        RESULTS[ds_name][m_name] = avg

        print(f"  {m_name:20s} | "
              f"Acc={avg['Accuracy']:.4f}  "
              f"F1={avg['F1']:.4f}  "
              f"AUC={avg['AUC-ROC']:.4f}  "
              f"MCC={avg['MCC']:.4f}")

out_path = os.path.join(METRICS_DIR, 'performance_results.json')
with open(out_path, 'w') as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults saved → {out_path}")

METRICS_PLOT = ['Accuracy', 'F1', 'AUC-ROC', 'MCC']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Across NASA Datasets\n(5-Fold CV with SMOTE)',
             fontsize=14, fontweight='bold')

for ax, ds in zip(axes.flatten(), DATASETS):
    x = np.arange(len(METRICS_PLOT))
    w = 0.25
    for i, m_name in enumerate(MODEL_NAMES):
        vals  = [RESULTS[ds][m_name][m] for m in METRICS_PLOT]
        bars  = ax.bar(x + i * w, vals, w,
                       label=m_name, color=COLORS[m_name],
                       alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, rotation=45)

    ax.set_title(f'Dataset: {ds}', fontweight='bold')
    ax.set_xticks(x + w)
    ax.set_xticklabels(METRICS_PLOT, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, 'fig1_performance_bars.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 1 saved → {fig1_path}")

f1_df = pd.DataFrame({
    ds: {m: RESULTS[ds][m]['F1'] for m in MODEL_NAMES}
    for ds in DATASETS
}).T

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(f1_df, annot=True, fmt='.4f', cmap='Blues',
            ax=ax, linewidths=0.5,
            vmin=0.60, vmax=1.00,
            annot_kws={'size': 11})
ax.set_title('F1 Score Heatmap: Models × Datasets', fontweight='bold')
ax.set_xlabel('Model')
ax.set_ylabel('Dataset')
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, 'fig2_f1_heatmap.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 2 saved → {fig2_path}")

print("\nTraining & Evaluate complete.")
 