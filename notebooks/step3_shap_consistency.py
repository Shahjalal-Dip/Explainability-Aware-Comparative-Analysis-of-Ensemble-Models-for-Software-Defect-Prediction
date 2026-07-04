import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score, roc_auc_score
from imblearn.over_sampling  import SMOTE
from scipy.stats             import spearmanr
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
import shap

BASE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE, '..', 'data')
FIG_DIR     = os.path.join(BASE, '..', 'results', 'figures')
METRICS_DIR = os.path.join(BASE, '..', 'results', 'metrics')
os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

DATASETS    = ['CM1', 'KC1', 'JM1', 'PC1']
MODEL_NAMES = ['Random Forest', 'XGBoost', 'LightGBM']
PAIRS       = [('Random Forest','XGBoost'),
               ('Random Forest','LightGBM'),
               ('XGBoost','LightGBM')]
PAIR_IDX    = [(0,1),(0,2),(1,2)]
SEED        = 42
COLORS      = {
    'Random Forest': '#2196F3',
    'XGBoost':       '#FF5722',
    'LightGBM':      '#4CAF50',
}
DEFECT_RATES = {'CM1':'9.83%','KC1':'15.45%','JM1':'19.35%','PC1':'6.94%'}


def get_shap_values(model, X_test: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)

    if isinstance(sv, list):
        return sv[1]
    elif sv.ndim == 3:
        return sv[:, :, 1]
    else:
        return sv

print("=" * 60)
print("SHAP Explainability & Consistency Analysis")
print("=" * 60)

all_corr_matrices = {}   
all_shap_ranks    = {}   
top5_per_dataset  = {}   

for ds_name in DATASETS:
    print(f"\n{'─'*50}")
    print(f"  Dataset: {ds_name}")
    print(f"{'─'*50}")

    df            = pd.read_csv(os.path.join(DATA_DIR, f'{ds_name}.csv'))
    X             = df.drop('defects', axis=1).values.astype(float)
    y             = df['defects'].values.astype(int)
    feature_names = list(df.drop('defects', axis=1).columns)
    ratio         = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))

    # 80/20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # SMOTE on training set only
    X_tr_sm, y_tr_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    print(f"  Train: {X_tr_sm.shape[0]:,} (after SMOTE)  "
          f"Test: {X_te.shape[0]:,}")

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=SEED, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=100, scale_pos_weight=ratio,
            eval_metric='logloss', verbosity=0,
            random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMClassifier(
            n_estimators=100, class_weight='balanced',
            verbose=-1, random_state=SEED, n_jobs=-1),
    }

  
    mean_abs_shap = {}  
    rank_series   = {}   

    for m_name, clf in models.items():
        clf.fit(X_tr_sm, y_tr_sm)

        # performance checking
        y_pred  = clf.predict(X_te)
        y_prob  = clf.predict_proba(X_te)[:, 1]
        f1  = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob)
        print(f"  {m_name:20s}  F1={f1:.4f}  AUC={auc:.4f}")

        # SHAP computation
        sv       = get_shap_values(clf, X_te)
        mean_abs = np.abs(sv).mean(axis=0)
        mean_abs_shap[m_name] = mean_abs

        ranks = pd.Series(mean_abs, index=feature_names).rank(ascending=False)
        rank_series[m_name] = ranks

    # Spearman rank correlation matrix 
    corr_mat = np.zeros((3, 3))
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            rho, _ = spearmanr(rank_series[m1], rank_series[m2])
            corr_mat[i, j] = round(float(rho), 4)

    print(f"\n  Spearman ρ  |  RF↔XGB={corr_mat[0,1]:.3f}  "
          f"RF↔LGB={corr_mat[0,2]:.3f}  "
          f"XGB↔LGB={corr_mat[1,2]:.3f}")

   # Top-5 features 
    top5 = {}
    for m_name in MODEL_NAMES:
        idx  = np.argsort(mean_abs_shap[m_name])[::-1][:5]
        top5[m_name] = [
            (feature_names[i], round(float(mean_abs_shap[m_name][i]), 4))
            for i in idx
        ]
        feats_str = ", ".join(
            [f"{f}({v:.3f})" for f, v in top5[m_name]])
        print(f"  Top-5 {m_name:18s}: {feats_str}")

    all_corr_matrices[ds_name] = corr_mat.tolist()
    all_shap_ranks[ds_name] = {
        m: {f: float(rank_series[m][f]) for f in feature_names}
        for m in MODEL_NAMES
    }
    top5_per_dataset[ds_name] = top5

print(f"\n{'='*60}")
print("SUMMARY: Spearman ρ across all datasets")
print(f"{'='*60}")
print(f"  {'Dataset':6s}  {'RF↔XGB':>8}  {'RF↔LGB':>8}  {'XGB↔LGB':>9}")
for ds in DATASETS:
    m = all_corr_matrices[ds]
    print(f"  {ds:6s}  {m[0][1]:>8.3f}  {m[0][2]:>8.3f}  {m[1][2]:>9.3f}")

avgs = {
    'RF↔XGB':   np.mean([all_corr_matrices[d][0][1] for d in DATASETS]),
    'RF↔LGB':   np.mean([all_corr_matrices[d][0][2] for d in DATASETS]),
    'XGB↔LGB':  np.mean([all_corr_matrices[d][1][2] for d in DATASETS]),
}
print(f"  {'MEAN':6s}  {avgs['RF↔XGB']:>8.3f}  {avgs['RF↔LGB']:>8.3f}"
      f"  {avgs['XGB↔LGB']:>9.3f}")

out_path = os.path.join(METRICS_DIR, 'shap_consistency_all.json')
with open(out_path, 'w') as f:
    json.dump({'ranks':        all_shap_ranks,
               'correlations': all_corr_matrices,
               'top5':         top5_per_dataset}, f, indent=2)
print(f"\nSHAP results saved → {out_path}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'SHAP Feature Rank Consistency Across All Datasets\n(Spearman Correlation)',
    fontsize=14, fontweight='bold')

labels = ['RF', 'XGB', 'LGB']
for ax, ds in zip(axes.flatten(), DATASETS):
    mat    = np.array(all_corr_matrices[ds])
    annots = [[f"{mat[i,j]:.3f}" for j in range(3)] for i in range(3)]
    sns.heatmap(mat, annot=annots, fmt='', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels,
                vmin=0.0, vmax=1.0, ax=ax, linewidths=1.5,
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    sub = (f"RF↔XGB: {mat[0,1]:.3f}  |  "
           f"RF↔LGB: {mat[0,2]:.3f}  |  "
           f"XGB↔LGB: {mat[1,2]:.3f}")
    ax.set_title(f'Dataset: {ds}\n{sub}', fontweight='bold', fontsize=10)

plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, 'fig3_shap_consistency_heatmaps.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 3 saved → {fig3_path}")

# Consistency Trend Line
fig, ax = plt.subplots(figsize=(10, 5))
pair_cfg = [
    ('RF ↔ XGBoost',          (0,1), '#E91E63', 'o--'),
    ('RF ↔ LightGBM',         (0,2), '#9C27B0', 's--'),
    ('XGBoost ↔ LightGBM',   (1,2), '#00BCD4', '^-'),
]
x = np.arange(len(DATASETS))
for label, (i, j), color, style in pair_cfg:
    rhos = [all_corr_matrices[d][i][j] for d in DATASETS]
    ax.plot(x, rhos, style, color=color, linewidth=2.5,
            markersize=9, label=label, zorder=3)
    for xi, rho in zip(x, rhos):
        ax.annotate(f'{rho:.3f}', xy=(xi, rho),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=10,
                    fontweight='bold', color=color)

ax.axhspan(0.8, 1.0, alpha=0.08, color='green',  label='High (ρ>0.8)')
ax.axhspan(0.5, 0.8, alpha=0.08, color='yellow', label='Moderate')
ax.axhspan(0.0, 0.5, alpha=0.08, color='red',    label='Low (ρ<0.5)')

x_labels = [f'{ds}\n({DEFECT_RATES[ds]} defects)' for ds in DATASETS]
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel('Spearman Rank Correlation (ρ)', fontsize=12)
ax.set_title('SHAP Feature Importance Consistency by Model Pair',
             fontsize=13, fontweight='bold')
ax.set_ylim(-0.05, 1.15)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, 'fig4_consistency_trend.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 4 saved → {fig4_path}")

# Top-5 SHAP features across all datasets 
fig, axes = plt.subplots(4, 3, figsize=(16, 20))
fig.suptitle('Top-5 SHAP Feature Importance — All Datasets & Models',
             fontsize=14, fontweight='bold', y=1.01)

for row, ds in enumerate(DATASETS):
    for col, m_name in enumerate(MODEL_NAMES):
        ax       = axes[row, col]
        features = [x[0] for x in top5_per_dataset[ds][m_name]]
        values   = [x[1] for x in top5_per_dataset[ds][m_name]]
        color    = list(COLORS.values())[col]
        ax.barh(range(len(features))[::-1], values,
                color=color, alpha=0.82, edgecolor='white')
        ax.set_yticks(range(len(features))[::-1])
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Mean |SHAP value|', fontsize=8)
        if row == 0:
            ax.set_title(m_name, fontsize=11,
                         fontweight='bold', color=color)
        if col == 0:
            ax.set_ylabel(ds, fontsize=11, fontweight='bold',
                          rotation=0, labelpad=35, va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.25)

plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, 'fig5_shap_top5_all.png')
plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 5 saved → {fig5_path}")
print("\nSHAP Consistency complete.")
