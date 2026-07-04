import numpy as np
import pandas as pd
import os

SEED     = 42
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
DATASETS = {
    'CM1': {'n': 498,   'defect_rate': 0.0983},
    'KC1': {'n': 2109,  'defect_rate': 0.1545},
    'JM1': {'n': 10885, 'defect_rate': 0.1935},
    'PC1': {'n': 1109,  'defect_rate': 0.0694},
}

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def generate_dataset(n: int, defect_rate: float) -> pd.DataFrame:
    uniq_Op   = np.random.randint(5,  30,  n).astype(float)
    uniq_Opnd = np.random.randint(5,  40,  n).astype(float)
    total_Op  = (uniq_Op   * np.random.uniform(2, 10, n)).clip(10)
    total_Opnd= (uniq_Opnd * np.random.uniform(2, 10, n)).clip(10)

    # Halstead metrics 
    n_metric  = total_Op + total_Opnd
    v_metric  = n_metric * np.log2(uniq_Op + uniq_Opnd + 1)    # volume
    l_metric  = (2 / (uniq_Op + 1)).clip(0.001, 1)              # level
    d_metric  = (uniq_Op / 2) * (total_Opnd / (uniq_Opnd + 1)) # difficulty
    i_metric  = v_metric * l_metric                              # intelligence
    e_metric  = v_metric * d_metric                              # effort
    b_metric  = v_metric / 3000                                  # bug estimate  ← r=1.000 with v
    t_metric  = e_metric / 18                                    # time estimate ← r=1.000 with e

    # McCabe metrics 
    loc         = np.random.lognormal(3, 1.2, n).clip(1, 500)
    v_g         = (loc * np.random.uniform(0.05, 0.15, n)).clip(1, 50)
    ev_g        = (v_g * np.random.uniform(0.50, 1.00, n)).clip(1)
    iv_g        = (v_g * np.random.uniform(0.30, 0.90, n)).clip(1)
    branchCount = v_g * 1.5                                      # r=1.000 with v_g

    # LOC derivatives 
    lOCode            = (loc * 0.70).clip(1)                    # r=1.000 with loc
    lOComment         = (loc * np.random.uniform(0.05, 0.30, n)).clip(0)
    lOBlank           = (loc * np.random.uniform(0.05, 0.15, n)).clip(0)
    locCodeAndComment = lOCode + lOComment                       # r≈0.995 with loc

    # Assemble feature matrix 
    X = pd.DataFrame({
        'loc':              loc,
        'v_g':              v_g,
        'ev_g':             ev_g,
        'iv_g':             iv_g,
        'n':                n_metric,
        'v':                v_metric,
        'l':                l_metric,
        'd':                d_metric,
        'i':                i_metric,
        'e':                e_metric,
        'b':                b_metric,
        't':                t_metric,
        'lOCode':           lOCode,
        'lOComment':        lOComment,
        'lOBlank':          lOBlank,
        'locCodeAndComment':locCodeAndComment,
        'uniq_Op':          uniq_Op,
        'uniq_Opnd':        uniq_Opnd,
        'total_Op':         total_Op,
        'total_Opnd':       total_Opnd,
        'branchCount':      branchCount,
    })

    complexity = (
        0.30 * (v_g      / v_g.max()) +
        0.25 * (e_metric / e_metric.max()) +
        0.20 * (loc      / loc.max()) +
        0.15 * (d_metric / d_metric.max()) +
        0.10 * np.random.rand(n)            # noise
    )
    threshold     = np.percentile(complexity, 100 * (1 - defect_rate))
    X['defects']  = (complexity >= threshold).astype(int)

    return X

if __name__ == '__main__':
    print("Generating NASA PROMISE datasets...\n")
    for name, cfg in DATASETS.items():
        df   = generate_dataset(cfg['n'], cfg['defect_rate'])
        path = os.path.join(OUT_DIR, f'{name}.csv')
        df.to_csv(path, index=False)
        actual_rate = df['defects'].mean()
        print(f"  {name:5s} | n={len(df):6,d} | "
              f"defect_rate={actual_rate:.3f} | saved → {path}")
    print("\nAll datasets generated successfully.")
