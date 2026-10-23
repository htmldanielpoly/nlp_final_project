"""Hugging Face MulTaBench datasets with current or paper-style split rules.

Paper-style splitting does not establish identity with the paper's Kaggle CSV rows.
Record the resolved data revision and row fingerprints before claiming paired external results.
"""
import json
from dataclasses import dataclass
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, train_test_split

TEXT_DATASETS = {
    "BIN_TEXT_FAKE_JOB_POSTING": "core-text-cls-fake-job-posting",
    "BIN_TEXT_JIGSAW_TOXICITY": "core-text-cls-jigsaw-toxicity",
    "BIN_TEXT_KICKSTARTER_FUNDING": "core-text-cls-kickstarter-funding",
    "MUL_TEXT_DATA_SCIENTIST_SALARY": "core-text-cls-data-scientist-salary",
    "MUL_TEXT_MICHELIN_RESTAURANTS": "core-text-cls-michelin-restaurants",
    "MUL_TEXT_PRODUCT_SENTIMENT": "core-text-cls-product-sentiment",
    "MUL_TEXT_SPOTIFY_GENRES": "core-text-cls-spotify-genres",
    "MUL_TEXT_US_ACCIDENTS": "core-text-cls-us-accidents",
    "MUL_TEXT_WINE_REVIEW": "core-text-cls-wine-review",
    "MUL_TEXT_WOMEN_CLOTHING_REVIEW": "core-text-cls-women-clothing-review",
    "REG_TEXT_BABIES_PRICES": "core-text-reg-babies-prices",
    "REG_TEXT_BOOK_PRICE": "core-text-reg-book-price",
    "REG_TEXT_BOOK_READABILITY": "core-text-reg-book-readability",
    "REG_TEXT_MERCARI_MARKETPLACE": "core-text-reg-mercari-marketplace",
    "REG_TEXT_MONTGOMERY_SALARIES": "core-text-reg-montgomery-salaries",
    "REG_TEXT_SCIMAGOJR_IMPACT": "core-text-reg-scimagojr-impact",
    "REG_TEXT_ROTTEN_TOMATOES": "core-text-reg-rotten-tomatoes",
    "REG_TEXT_VANCOUVER_SALARIES": "core-text-reg-vancouver-salaries",
    "REG_TEXT_VIDEO_GAMES_SALES": "core-text-reg-video-games-sales",
    "REG_TEXT_ZOMATO_RESTAURANTS": "core-text-reg-zomato-restaurants",
}

FOLDS, REPEATS, SEED = 3, 2, 42
MAX_TRAIN_ROWS, MAX_TEST_ROWS = 10_000, 5_000


@dataclass
class Dataset:
    name: str
    x: pd.DataFrame
    y: pd.Series
    task: str  # "binary" | "multiclass" | "regression"
    num_cols: list
    cat_cols: list
    text_cols: list
    source_revision: str | None = None
    source: str = "huggingface"

    @property
    def is_cls(self):
        return self.task != "regression"


def load_dataset(name: str, revision: str | None = None) -> Dataset:
    from huggingface_hub import snapshot_download
    if revision is None:  # v2: default to the revision every v1 run used (identical rows)
        from reading_agent2.revisions import PINNED_REVISIONS
        revision = PINNED_REVISIONS.get(name)
    path = snapshot_download(repo_id=f"multabench/{TEXT_DATASETS[name]}", repo_type="dataset", revision=revision)
    df = pd.read_parquet(join(path, "data.parquet"))
    with open(join(path, "metadata.json")) as f:
        meta = json.load(f)
    y = df[meta["target"]]
    x = df.drop(columns=[meta["target"]])
    for c in x.columns:  # parquet gives None for missing strings
        if x[c].dtype == object:
            x[c] = x[c].where(x[c].notna(), np.nan)
    if name.startswith("REG"):
        task = "regression"
    else:
        task = "binary" if y.nunique() == 2 else "multiclass"
    # column types follow MulTaBench's detect_feature_types: stored dtypes decide
    num, cat, text = [], [], []
    for c in x.columns:
        s = x[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            # dates -> numeric timestamp (MulTaBench extracts date parts; a single ordinal is enough here)
            s = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
            x[c] = (s - pd.Timestamp("1970-01-01")).dt.days.astype(float)
            num.append(c)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            x[c] = s.astype(float)
            num.append(c)
        else:
            text.append(c)
    return Dataset(name, x, y, task, num, cat, text, source_revision=Path(path).name)


def _smallest_class_count(y):
    return np.unique(y, return_counts=True)[1].min()


def _cap(idx, y, is_cls, size, seed):
    if len(idx) <= size:
        return idx
    stratify = y[idx] if is_cls and _smallest_class_count(y[idx]) >= 2 else None
    kept, _ = train_test_split(idx, train_size=size, random_state=seed, stratify=stratify)
    return kept


def get_split(y, is_cls: bool, fold: int, protocol: str = "current"):
    if protocol not in {"current", "paper"}:
        raise ValueError(f"Unknown protocol: {protocol}")
    if fold < 0 or fold >= (5 if protocol == "paper" else FOLDS * REPEATS):
        raise ValueError(f"Invalid fold {fold} for {protocol} protocol")
    if protocol == "paper":
        return get_split_paper(y, is_cls, fold)
    y = np.asarray(y)
    cv_cls = RepeatedStratifiedKFold if is_cls and _smallest_class_count(y) >= FOLDS else RepeatedKFold
    cv = cv_cls(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    train, test = list(cv.split(np.zeros(len(y)), y))[fold]
    train = _cap(train, y, is_cls, MAX_TRAIN_ROWS, SEED + fold)
    test = _cap(test, y, is_cls, MAX_TEST_ROWS, SEED + fold)
    return np.sort(train), np.sort(test)


# ---------------------------------------------------------------- paper protocol (MulTaBench tag paper_version)
# Port of tabstar.preprocessing.splits (SEED=42) + multabench/baselines/preprocessing/sampling.py at paper_version:
# subsample to 10K train + min(10%, 2K) test, then one seeded holdout per fold (folds 0-4). These are the splits
# used by the paper leaderboard. Exact identity additionally requires the original data and row ordering.
PAPER_SEED, PAPER_TEST_RATIO, PAPER_MAX_TEST = 42, 0.1, 2000


def _paper_do_split(x, y, test_size, is_cls, fold):
    rs = PAPER_SEED + fold
    if not is_cls:
        return train_test_split(x, y, test_size=test_size, random_state=rs)
    if y.nunique() == 2:
        rng = np.random.RandomState(rs)
        picked = [rng.choice(y[y == v].index.tolist()) for v in sorted(set(y))]
        x_rest, y_rest = x.drop(index=picked), y.drop(index=picked)
        x_tr, x_te, y_tr, y_te = train_test_split(x_rest, y_rest, test_size=test_size, random_state=rs, stratify=y_rest)
        x_te = pd.concat([x_te, x.loc[picked]])
        x_te = x_te.loc[x_te.sample(frac=1, random_state=rs).index]
        return x_tr, x_te, y_tr, y.loc[x_te.index]
    vc = y.value_counts()
    if vc.min() <= 1:
        single = y.isin(vc[vc == 1].index)
        x_rest, y_rest = x[~single], y[~single]
        test_size = max(test_size, y_rest.nunique())
        x_tr, x_te, y_tr, y_te = train_test_split(x_rest, y_rest, test_size=test_size, random_state=rs, stratify=y_rest)
        x_tr = pd.concat([x_tr, x[single]])
        x_tr = x_tr.loc[x_tr.sample(frac=1, random_state=rs).index]
        return x_tr, x_te, y.loc[x_tr.index], y_te
    return train_test_split(x, y, test_size=test_size, random_state=rs, stratify=y)


def get_split_paper(y, is_cls: bool, fold: int, train_examples: int = MAX_TRAIN_ROWS):
    """Positional (train, test) indices of the paper-version split. Only the row index is split, so x is a dummy frame."""
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    x = pd.DataFrame(index=y.index)
    n_test = min(int(len(y) * PAPER_TEST_RATIO), PAPER_MAX_TEST)
    if len(y) > train_examples + n_test:  # subsample_dataset
        x, _, y, _ = _paper_do_split(x, y, len(y) - train_examples - n_test, is_cls, fold)
    test_size = min(int(len(y) * PAPER_TEST_RATIO), PAPER_MAX_TEST)  # split_to_test
    if len(x) > train_examples:
        test_size = len(x) - train_examples
    x_tr, x_te, _, _ = _paper_do_split(x, y, test_size, is_cls, fold)
    return np.sort(x_tr.index.values), np.sort(x_te.index.values)
