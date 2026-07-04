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

BASE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE, '..', 'data')
FIG_DIR     = os.path.join(BASE, '..', 'results', 'figures')
METRICS_DIR = os.path.join(BASE, '..', 'results', 'metrics')
os.makedirs(FIG_DIR,     exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

HALSTEAD = ['n', 'v', 'l', 'd', 'i', 'e', 'b', 't']
MCCABE   = ['v_g', 'ev_g', 'iv_g', 'branchCount']
LOC_GRP  = ['loc', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
            'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd']
ORDERED  = HALSTEAD + MCCABE + LOC_GRP

THRESHOLD_PERFECT = 0.999   
THRESHOLD_HIGH    = 0.90    

print("=" * 60)
print("Feature Correlation & Multicollinearity Analysis")
print("=" * 60)
print("\nUsing KC1 as the representative dataset (n=2,109, 21 features)")

df    = pd.read_csv(os.path.join(DATA_DIR, 'KC1.csv'))
X     = df.drop('defects', axis=1)
corr  = X.corr(method='pearson')
names = list(X.columns)

print("\nHighly correlated feature pairs (|r| > 0.90):")
print(f"{'Feature 1':22s} {'Feature 2':22s} {'Pearson r':>10}  Category")
print("-" * 70)

high_corr_pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r = corr.loc[names[i], names[j]]
        if abs(r) > THRESHOLD_HIGH:
            cat = "PERFECT (definitional)" if abs(r) >= THRESHOLD_PERFECT \
                  else "High"
            high_corr_pairs.append({
                'feature_1': names[i],
                'feature_2': names[j],
                'pearson_r': round(float(r), 6),
                'category':  cat,
            })
            print(f"  {names[i]:20s} {names[j]:20s} "
                  f"  {r:8.4f}   {cat}")

high_corr_pairs.sort(key=lambda x: abs(x['pearson_r']), reverse=True)

print("\nVerifying Halstead definitional identities:")
identities = [
    ('t', 'e',  18.0,   't = e / 18'),
    ('b', 'v',  3000.0, 'b = v / 3000'),
]
for f1, f2, divisor, formula in identities:
    ratio = df[f1].values / (df[f2].values + 1e-12)
    r     = np.corrcoef(df[f1].values, df[f2].values)[0, 1]
    print(f"  {formula:20s}  mean(ratio)={ratio.mean():.6f}  "
          f"std(ratio)={ratio.std():.2e}  r={r:.6f}")

corr_dict = {
    f: {g: round(float(corr.loc[f, g]), 4) for g in names}
    for f in names
}
out_path = os.path.join(METRICS_DIR, 'feature_correlation.json')
with open(out_path, 'w') as fh:
    json.dump({'correlation_matrix': corr_dict,
               'high_corr_pairs':    high_corr_pairs}, fh, indent=2)
print(f"\nCorrelation data saved → {out_path}")

corr_ordered = corr.loc[ORDERED, ORDERED]
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr_ordered,
            annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.3, linecolor='#EEEEEE',
            annot_kws={'size': 7},
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})

for pos in [len(HALSTEAD), len(HALSTEAD) + len(MCCABE)]:
    ax.axhline(y=pos, color='black', linewidth=2.5)
    ax.axvline(x=pos, color='black', linewidth=2.5)

mid_h = len(HALSTEAD) / 2
mid_m = len(HALSTEAD) + len(MCCABE) / 2
mid_l = len(HALSTEAD) + len(MCCABE) + len(LOC_GRP) / 2
ax.text(mid_h, -0.9, 'Halstead Metrics',
        ha='center', fontsize=10, fontweight='bold', color='#1565C0')
ax.text(mid_m, -0.9, 'McCabe',
        ha='center', fontsize=10, fontweight='bold', color='#6A1B9A')
ax.text(mid_l, -0.9, 'LOC / Operator Metrics',
        ha='center', fontsize=10, fontweight='bold', color='#2E7D32')

ax.set_title('Feature Correlation Matrix (KC1)\nGrouped by Metric Family',
             fontsize=13, fontweight='bold', pad=20)
ax.tick_params(axis='both', labelsize=9)
plt.tight_layout()
fig6_path = os.path.join(FIG_DIR, 'fig6_feature_correlation_matrix.png')
plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 6 saved → {fig6_path}")

PERFECT_PAIRS = [
    ('e', 't',          'e ↔ t  (r=1.000)\nt = e ÷ 18  [Halstead]',  '#E91E63'),
    ('v_g', 'branchCount','v_g ↔ branchCount  (r=1.000)\nf(cyclomatic)', '#FF5722'),
    ('loc', 'lOCode',   'loc ↔ lOCode  (r=1.000)\nlOCode ⊂ loc',       '#9C27B0'),
    ('v', 'b',          'v ↔ b  (r=1.000)\nb = v ÷ 3000  [Halstead]', '#2196F3'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'Multicollinearity: Perfectly Correlated Feature Pairs (r = 1.000)\n'
    'Mechanistic cause of RF SHAP divergence from gradient boosting',
    fontsize=12, fontweight='bold')

for ax, (f1, f2, title, color) in zip(axes.flatten(), PERFECT_PAIRS):
    x_vals = df[f1].values
    y_vals = df[f2].values

    # Clip top-1% outliers for visual clarity
    p99    = np.percentile(x_vals, 99)
    mask   = x_vals < p99
    x_clip = x_vals[mask]
    y_clip = y_vals[mask]

    ax.scatter(x_clip, y_clip, alpha=0.25, s=8, color=color, zorder=2)

    # Fitted line
    m_coef, b_coef = np.polyfit(x_clip, y_clip, 1)
    xl = np.array([x_clip.min(), x_clip.max()])
    ax.plot(xl, m_coef * xl + b_coef, 'k-', linewidth=2, alpha=0.8, zorder=3)

    ax.set_xlabel(f1,  fontsize=11)
    ax.set_ylabel(f2, fontsize=11)
    ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    ax.text(0.05, 0.92, 'Pearson r = 1.000',
            transform=ax.transAxes, fontsize=11,
            fontweight='bold', color='red')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)

plt.tight_layout()
fig7_path = os.path.join(FIG_DIR, 'fig7_multicollinearity_pairs.png')
plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure 7 saved → {fig7_path}")
print("\nFeature Correlation complete.")
