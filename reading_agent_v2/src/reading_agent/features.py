"""Turn a text-tabular DataFrame into tensors for the reading agent.

Per row:
  x_num  (n_num,)   quantile-normalised numerics (NaN -> 0) and m_num, the missing mask
  x_cat  (n_cat,)   category ids (0 = unknown/missing)
  items  (K, 384)   frozen E5 embedding of each readable text item ("passage: <col>: <chunk>")
  item_field / item_pos / item_mask (K,)
Readable items are (text column, chunk) pairs: short fields give one item, long fields are split into
sentence chunks.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .chunking import chunk_text
from .embed import EmbeddingStore, embed


def _is_missing(t):
    return t is None or (isinstance(t, float) and t != t) or not str(t).strip()


@dataclass
class Prepared:
    x_num: np.ndarray
    m_num: np.ndarray
    x_cat: np.ndarray
    items: np.ndarray
    item_field: np.ndarray
    item_pos: np.ndarray
    item_mask: np.ndarray
    y: np.ndarray | None = None

    def subset(self, idx):
        return Prepared(*(None if a is None else a[idx] for a in
                          (self.x_num, self.m_num, self.x_cat, self.items, self.item_field, self.item_pos,
                           self.item_mask, self.y)))

    def __len__(self):
        return len(self.item_mask)


@dataclass
class FeaturePipeline:
    num_cols: list
    cat_cols: list
    text_cols: list
    namespace: str                 # embedding-store name (one per dataset)
    item_mode: str = "chunk"       # "chunk" or "field" (one item per text column)
    max_items: int = 8             # items kept per row (round robin over fields)
    max_chunks: int = 8            # chunks per field (the tail is merged); fixed so cached chunk strings stay valid
    max_words: int = 48
    max_cat: int = 100
    field_meta: bool = False       # add [present, log word count] per text field to the table (known before reading)
    _qt: QuantileTransformer = None
    _cat_maps: list = field(default_factory=list)
    col_name_emb: np.ndarray = None
    field_name_emb: np.ndarray = None
    _len_stats: tuple = None

    def fit(self, df: pd.DataFrame):
        if self.num_cols:
            self._qt = QuantileTransformer(n_quantiles=min(1000, len(df)), output_distribution="normal",
                                           subsample=10**9, random_state=0)
            self._qt.fit(df[self.num_cols].astype(float).values)
        self._cat_maps = []
        for c in self.cat_cols:
            vc = df[c].dropna().astype(str).value_counts()
            self._cat_maps.append({v: i + 1 for i, v in enumerate(vc.index[: self.max_cat])})
        if self.field_meta:
            ll = self._log_words(df)
            self._len_stats = (ll.mean(0), ll.std(0) + 1e-6)
        self.col_name_emb = embed([str(c) for c in self.num_cols] + self._meta_names() + [str(c) for c in self.cat_cols],
                                  prefix="query: ")
        self.field_name_emb = embed([str(c) for c in self.text_cols], prefix="query: ")
        return self

    def _meta_names(self):
        if not self.field_meta:
            return []
        return [f"{c} present" for c in self.text_cols] + [f"{c} length" for c in self.text_cols]

    def _log_words(self, df):
        if not self.text_cols:
            return np.zeros((len(df), 0), np.float32)
        return np.stack([[0.0 if _is_missing(v) else np.log1p(len(str(v).split())) for v in df[c].values]
                         for c in self.text_cols], 1).astype(np.float32)

    @property
    def n_num_in(self):
        """Numeric tokens seen by the table encoder (real numerics + text-field meta)."""
        return len(self.num_cols) + len(self._meta_names())

    @property
    def cat_cardinalities(self):
        return [len(m) + 1 for m in self._cat_maps]

    def _row_items(self, row_texts):
        out = []
        for f, t in enumerate(row_texts):
            if _is_missing(t):
                continue
            col = self.text_cols[f]
            if self.item_mode == "field":
                out.append((f"passage: {col}: {str(t).strip()}", f, 0))
            else:
                for p, ch in enumerate(chunk_text(t, self.max_words, self.max_chunks)):
                    out.append((f"passage: {col}: {ch}", f, p))
        if len(out) > self.max_items:  # keep a balanced number of items per field (round robin)
            by_f = {}
            for it in out:
                by_f.setdefault(it[1], []).append(it)
            kept, i = [], 0
            while len(kept) < self.max_items:
                for f in sorted(by_f):
                    if i < len(by_f[f]) and len(kept) < self.max_items:
                        kept.append(by_f[f][i])
                i += 1
            out = sorted(kept, key=lambda x: (x[1], x[2]))
        return out

    def transform(self, df: pd.DataFrame, y=None) -> Prepared:
        n = len(df)
        if self.num_cols:
            raw = df[self.num_cols].astype(float).values
            m_num = np.isnan(raw).astype(np.float32)
            x_num = np.nan_to_num(self._qt.transform(raw), nan=0.0).astype(np.float32)
        else:
            x_num = np.zeros((n, 0), np.float32)
            m_num = np.zeros((n, 0), np.float32)
        if self.field_meta and self.text_cols:
            ll = self._log_words(df)
            meta = np.concatenate([(ll > 0) * 2.0 - 1, (ll - self._len_stats[0]) / self._len_stats[1]], 1)
            x_num = np.concatenate([x_num, meta.astype(np.float32)], 1)
            m_num = np.concatenate([m_num, np.zeros(meta.shape, np.float32)], 1)
        x_cat = np.zeros((n, len(self.cat_cols)), np.int64)
        for j, (c, mp) in enumerate(zip(self.cat_cols, self._cat_maps)):
            x_cat[:, j] = [0 if _is_missing(v) else mp.get(str(v), 0) for v in df[c].values]

        K = self.max_items
        rows = ([self._row_items(list(r)) for r in df[self.text_cols].itertuples(index=False, name=None)]
                if self.text_cols else [[] for _ in range(n)])
        flat = [it[0] for r in rows for it in r]
        vecs = EmbeddingStore(self.namespace).get(flat) if flat else np.zeros((0, 384), np.float32)
        items = np.zeros((n, K, 384), np.float16)
        item_field = np.zeros((n, K), np.int64)
        item_pos = np.zeros((n, K), np.int64)
        item_mask = np.zeros((n, K), np.float32)
        k = 0
        for i, r in enumerate(rows):
            for j, (_, f, p) in enumerate(r):
                items[i, j] = vecs[k]
                item_field[i, j] = f
                item_pos[i, j] = min(p, K - 1)
                item_mask[i, j] = 1
                k += 1
        return Prepared(x_num, m_num, x_cat, items, item_field, item_pos, item_mask,
                        None if y is None else np.asarray(y))


def whole_text_embeddings(df: pd.DataFrame, text_cols, namespace) -> np.ndarray:
    """Joint-Frozen style: one E5 vector per (row, text column) of the full field text. (n, n_text, 384)"""
    store = EmbeddingStore(namespace)
    out = np.zeros((len(df), len(text_cols), 384), np.float32)
    for j, c in enumerate(text_cols):
        vals = df[c].values
        texts = [f"passage: {c}: {'' if _is_missing(v) else str(v).strip()}" for v in vals]
        out[:, j] = store.get(texts)
    return out
