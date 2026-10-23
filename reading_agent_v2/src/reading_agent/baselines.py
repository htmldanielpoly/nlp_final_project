"""Joint-Frozen style baseline (as in MulTaBench): tabular columns + PCA-30 of frozen E5 per text column -> CatBoost."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PCA_COMPONENTS = 30


def frozen_frame(x, num_cols, cat_cols, text_emb, pcas=None):
    """text_emb: (n, n_text, 384). Returns a DataFrame for CatBoost and the fitted PCAs."""
    cols = {}
    for c in num_cols:
        cols[f"num_{c}"] = x[c].astype(float).values
    for c in cat_cols:
        cols[f"cat_{c}"] = x[c].astype(str).fillna("nan").values
    fit = pcas is None
    pcas = pcas or []
    for j in range(text_emb.shape[1]):
        if fit:
            pcas.append(PCA(n_components=min(PCA_COMPONENTS, text_emb.shape[0] - 1), random_state=0).fit(text_emb[:, j]))
        z = pcas[j].transform(text_emb[:, j])
        for k in range(z.shape[1]):
            cols[f"txt{j}_pca_{k}"] = z[:, k]
    return pd.DataFrame(cols), pcas


def fit_catboost(df_tr, y_tr, df_va, y_va, task, extra_tr=None, extra_va=None, seed=0, iterations=3000):
    from catboost import CatBoostClassifier, CatBoostRegressor
    cat_features = [c for c in df_tr.columns if c.startswith("cat_")]
    params = dict(iterations=iterations, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
                  od_type="Iter", od_wait=200, use_best_model=True, thread_count=4, used_ram_limit="3gb", cat_features=cat_features)
    if task == "regression":
        m = CatBoostRegressor(loss_function="RMSE", **params)
    else:
        # fix the class set so predict_proba has one column per class even if a class misses the inner-train split
        classes = sorted(set(np.asarray(y_tr).tolist()) | set(np.asarray(y_va).tolist()))
        m = CatBoostClassifier(loss_function="MultiClass" if task == "multiclass" else "Logloss",
                               class_names=classes, **params)
    m.fit(df_tr, y_tr, eval_set=(df_va, y_va))
    return m


def cb_predict(m, df, task):
    return m.predict(df) if task == "regression" else m.predict_proba(df)
