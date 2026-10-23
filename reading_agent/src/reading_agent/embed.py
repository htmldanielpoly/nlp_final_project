"""Frozen E5-small-v2 embeddings with a persistent per-namespace store (string -> 384-d vector).

Every distinct string is embedded once, so folds, chunkings and baselines all share one cache.
Texts are embedded as MulTaBench does: "passage: <col>: <text>", mean pooling, max 512 tokens.
"""
import os
import pickle
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/e5-small-v2"
MODEL_REVISION = os.environ.get("READING_AGENT_E5_REVISION", "ffb93f3bd4047442299a41ebb6fa998a38507c52")  # snapshot used for all results
DIM = 384
CACHE_DIR = Path(os.environ.get("READING_AGENT_CACHE", Path.home() / ".cache" / "reading_agent"))

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(os.cpu_count() or 4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device, revision=MODEL_REVISION)
        if device == "cuda":
            _model.half()  # fp16 inference; vectors are stored as fp16 anyway
        _model.max_seq_length = 512
    return _model


class EmbeddingStore:
    def __init__(self, namespace: str):
        self.path = CACHE_DIR / f"{namespace}.pkl"
        self.index: dict[str, int] = {}
        self.vecs = np.zeros((0, DIM), np.float16)
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.index, self.vecs = pickle.load(f)

    def save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump((self.index, self.vecs), f, protocol=4)
        os.replace(tmp, self.path)

    def add(self, texts, batch_size: int = 32, save_every: int = 4000):
        missing = sorted({t for t in texts if t not in self.index}, key=len)  # length-sorted -> less padding
        if not missing:
            return
        import torch
        model = _get_model()
        for s in range(0, len(missing), save_every):
            part = missing[s:s + save_every]
            with torch.inference_mode():
                v = model.encode(part, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(part) > 2000, convert_to_numpy=True).astype(np.float16)
            base = len(self.vecs)
            self.vecs = np.concatenate([self.vecs, v])
            for i, t in enumerate(part):
                self.index[t] = base + i
            self.save()

    def get(self, texts) -> np.ndarray:
        self.add(texts)
        return self.vecs[[self.index[t] for t in texts]].astype(np.float32)


def embed(texts, prefix: str = "passage: ", namespace: str = "names") -> np.ndarray:
    """Convenience wrapper (used for short strings such as column names)."""
    if not len(texts):
        return np.zeros((0, DIM), np.float32)
    return EmbeddingStore(namespace).get([prefix + t for t in texts])


def release_model():
    """Free the E5 weights once all embeddings are cached (keeps peak RAM low during training)."""
    global _model
    _model = None
    import gc
    gc.collect()
