"""The v2 model as one estimator: TabLoRA readers + Joint-Frozen TabM / TabPFN-2.5 heads, blended.

    model = TabLoRAStack(num_cols, cat_cols, text_cols, task="binary").fit(train_df, y)   # any text-tabular data (GPU)
    model.save("artifact/");  TabLoRAStack.load("artifact/").predict_proba(test_df)

An artifact holds the fitted preprocessing (`preprocessing.pkl`: feature pipeline and the PCAs of
the frozen frame), the K cross-fitted TabLoRA readers (`reader_k.pt`: reader, LoRA and gate
parameters only; the frozen E5 layers are rebuilt from the pinned checkpoint), the Joint-Frozen
TabM (`tabm.pt`), the TabPFN-2.5 context (`context.parquet`: the frozen-frame training rows and
their targets, which is what a fitted TabPFN consists of) and the blend weights (`manifest.json`).

Inference from the published benchmark runs (downloads one run from the Hub):

    python -m reading_agent2.estimator2 --repo htmldaniel/reading-agent-v2-multabench \
        --dataset REG_TEXT_BOOK_PRICE --fold 0 --input rows.csv --output preds.csv

`export` assembles artifacts from the training runs (stack features, tabcond, heads) and
`check` replays a run's stored test predictions from raw rows.
"""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from reading_agent.baselines import frozen_frame
from reading_agent.embed import MODEL_NAME, MODEL_REVISION, release_model
from reading_agent.features import FeaturePipeline, whole_text_embeddings
from reading_agent.run import score
from reading_agent.train import Config

from .stack import PRIMARY, ROOT, _full_probs, _ordinal, design, fit_tabm, fit_tabpfn, prepare, reader_feats, tabm_predict
from . import tabcond as tc

FORMAT = 2
HEADS = ("tabm", "tabpfn")


def _device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class ItemStatesAny(tc.ItemStates):
    """tc.ItemStates (fp16 on CUDA, as in training) that also runs on CPU in fp32."""

    def __init__(self, texts, device, batch_size=256):
        if str(device).startswith("cuda"):
            return super().__init__(texts, device, batch_size)
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        self.e5 = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).eval()
        lengths, chunks = np.zeros(len(texts), np.int64), [None] * len(texts)
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                enc = tok(texts[s:s + batch_size], max_length=tc.MAX_TOKENS, truncation=True, padding=True, return_tensors="pt")
                hs = self.e5(**enc, output_hidden_states=True).hidden_states[-tc.N_TOP - 1]
                for j, n in enumerate(enc["attention_mask"].sum(1).tolist()):
                    chunks[s + j], lengths[s + j] = hs[j, :n], n
        self.lengths = torch.as_tensor(lengths)
        self.offsets = torch.as_tensor(np.r_[0, np.cumsum(lengths)[:-1]])
        self.flat, self.device = torch.cat(chunks), device


def _devices_to_cpu(obj, seen=None):
    """pytabkit keeps the training device in nested interfaces (`device`, `device_`); point them at the CPU."""
    seen = set() if seen is None else seen
    if id(obj) in seen or not hasattr(obj, "__dict__") or isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return
    seen.add(id(obj))
    for k, v in vars(obj).items():
        if k in ("device", "device_") and str(v).startswith("cuda"):
            setattr(obj, k, torch.device("cpu") if isinstance(v, torch.device) else "cpu")
        elif isinstance(v, (list, tuple)):
            for x in v:
                _devices_to_cpu(x, seen)
        elif isinstance(v, dict):
            for x in v.values():
                _devices_to_cpu(x, seen)
        else:
            _devices_to_cpu(v, seen)


def _id_matrix(layout, tid, K):
    m = np.zeros((len(layout), K), np.int64)
    for i, r in enumerate(layout):
        m[i, : len(r)] = [tid[t] for t in r]
    return m


def _tabm_frame(F, flags, medians):
    """TabM input: frozen frame, missing flags for the training-time flagged columns, training medians
    (every numeric column, so rows with gaps unseen in training still get complete inputs)."""
    X, cats = design(F, None)
    for c in flags:
        X[f"{c}_missing"] = X[c].isna().astype(float)
    for c, med in medians.items():
        X[c] = X[c].fillna(med)
    return X, cats


class TabLoRAStack:
    def __init__(self, num_cols, cat_cols, text_cols, *, task="binary", namespace="generic", max_items=8,
                 n_splits=3, gate=True, heads=HEADS, seed=0, adapt_epochs=10, lora_lr=1e-3):
        if task not in {"binary", "multiclass", "regression"}:
            raise ValueError("task must be binary, multiclass or regression")
        self.task, self.n_splits, self.gate, self.heads, self.seed = task, n_splits, gate, tuple(heads), seed
        self.adapt_epochs, self.lora_lr = adapt_epochs, lora_lr
        self.pipe = FeaturePipeline(list(num_cols), list(cat_cols), list(text_cols), namespace,
                                    max_items=max(max_items, len(text_cols)))
        self.pcas, self.states_dicts, self.tabm, self.context = None, [], None, None
        self.weights, self.flags, self.medians = {}, [], {}
        self.classes_, self.target_mean, self.target_scale, self.n_out = None, 0.0, 1.0, None
        self.config, self.metadata = Config(epochs=100, patience=15), {}

    # ------------------------------------------------------------ readers
    def _reader(self, e5, device):
        cfg = self.config
        model = tc.TabCondReader(n_num=self.pipe.n_num_in, cat_cards=self.pipe.cat_cardinalities,
                                 col_name_emb=self.pipe.col_name_emb, field_name_emb=self.pipe.field_name_emb,
                                 n_out=self.n_out, d=cfg.d, layers=cfg.layers, cell=cfg.cell, dropout=cfg.dropout,
                                 max_items=self.pipe.max_items)
        return model, tc.TopE5(copy.deepcopy(e5), d_tab=cfg.d, gate=self.gate)

    def _reader_outputs(self, frame, P, device):
        """Mean over the K readers of the read-all reader features (probabilities / regression value)."""
        layout = tc.item_layout(self.pipe, frame)
        texts = sorted({t for r in layout for t in r}) or [""]
        tid = {t: i for i, t in enumerate(texts)}
        ids = _id_matrix(layout, tid, self.pipe.max_items)
        states = ItemStatesAny(texts, device)
        feats = 0.0
        for sd in self.states_dicts:
            model, top = self._reader(states.e5, device)
            model.attach(top, states)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            bad = [k for k in missing if not (k.startswith("top.layers.") or ".base." in k)]
            if bad or unexpected:
                raise ValueError(f"reader weights do not match the model: missing {bad[:3]}, unexpected {unexpected[:3]}")
            top.adapt = True
            model.to(device)
            out, _ = tc.outputs(model, P, ids, np.arange(len(frame)), self.task, device)
            feats = feats + reader_feats(out, self.task) / len(self.states_dicts)
            del model, top
        return feats

    # ------------------------------------------------------------ fit (any dataset)
    def fit(self, frame, y, device=None):
        """End-to-end training on a DataFrame: cross-fitted TabLoRA readers, heads, blend weights."""
        device = _device(device)
        if device == "cpu":
            raise RuntimeError("TabLoRA training needs a CUDA GPU (inference also runs on CPU)")
        frame = frame.reset_index(drop=True)
        y = np.asarray(y)
        if self.task == "regression":
            self.target_mean, self.target_scale = float(y.astype(float).mean()), float(y.astype(float).std() + 1e-9)
            y_tr = (y.astype(float) - self.target_mean) / self.target_scale
            self.n_out = 1
        else:
            self.classes_ = np.unique(y)
            y_tr = np.searchsorted(self.classes_, y)
            self.n_out = len(self.classes_)
        strat = y_tr if self.task != "regression" and np.bincount(y_tr).min() >= 2 else None
        i_in, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=self.seed, stratify=strat)
        # frozen frame (structured columns + PCA of each field's whole-text E5 embedding)
        self.pipe.fit(frame)
        E = whole_text_embeddings(frame, self.pipe.text_cols, self.pipe.namespace)
        F, self.pcas = frozen_frame(frame, self.pipe.num_cols, self.pipe.cat_cols, E)
        P = self.pipe.transform(frame, y_tr)
        release_model()
        # cross-fitted TabLoRA readers (as tabcond.run)
        layout = tc.item_layout(self.pipe, frame)
        texts = sorted({t for r in layout for t in r})
        tid = {t: i for i, t in enumerate(texts)}
        ids = _id_matrix(layout, tid, self.pipe.max_items)
        states = tc.ItemStates(texts, device)
        static = np.zeros((len(texts), 384), np.float32)
        probe = tc.TopE5(copy.deepcopy(states.e5), d_tab=self.config.d, gate=self.gate).to(device)
        with torch.no_grad():
            for s in range(0, len(texts), 512):
                st, mk = states.batch(torch.arange(s, min(s + 512, len(texts))))
                static[s:s + len(st)] = probe(st, mk).cpu().numpy()
        del probe
        P.items = static[ids].astype(np.float16) * P.item_mask[..., None]
        use_strat = self.task != "regression" and np.bincount(y_tr).min() >= self.n_splits
        kf = (StratifiedKFold if use_strat else KFold)(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof, self.states_dicts = None, []
        log = lambda s: print(s, flush=True)
        for k, (a, b_idx) in enumerate(kf.split(np.zeros(len(y_tr)), y_tr if use_strat else None)):
            a_in, a_va = train_test_split(a, test_size=0.15, random_state=self.seed + k)
            torch.manual_seed(self.seed + k)
            model, top = self._reader(states.e5, device)
            model.attach(top, states)
            model.to(device)
            reader_params = [p for n, p in model.named_parameters() if not n.startswith("top.")]
            tc.fit(model, P, ids, a_in, a_va, self.task, device, self.config.epochs, self.config.patience,
                   [{"params": reader_params, "lr": self.config.lr}], log, seed=self.seed + k, amp=False)
            top.adapt = True
            adapt = [p for l in top.loras for p in (l.A, l.B)] + (list(top.gate_proj.parameters()) if self.gate else [])
            for p in reader_params:
                p.requires_grad_(False)
            tc.fit(model, P, ids, a_in, a_va, self.task, device, self.adapt_epochs, 3, [{"params": adapt, "lr": self.lora_lr}],
                   log, seed=100 + self.seed + k, amp=True, batch_rows=64)
            out, _ = tc.outputs(model, P, ids, b_idx, self.task, device)
            f = reader_feats(out, self.task)
            oof = np.zeros((len(y_tr), f.shape[1]), np.float32) if oof is None else oof
            oof[b_idx] = f
            self.states_dicts.append(tc.adapted_state(model))
            del model, top
            torch.cuda.empty_cache()
        # heads on the frozen frame and validation-tuned blend weights (as stack.heads)
        full = lambda f: f if self.task == "regression" else _full_probs(f, self.task)
        r_val = full(oof[i_va])
        self.context = F.assign(__y__=y_tr)
        X, cats = design(F, None)
        self.flags = [c for c in X.columns if c not in cats and X[c].isna().any()]
        self.medians = {c: float(0.0 if pd.isna(X[c].median()) else X[c].median()) for c in X.columns if c not in cats}
        self.weights = {}
        for head in self.heads:
            if head == "tabpfn" and self.task != "regression" and self.n_out > 10:
                continue
            if head == "tabm":
                Xi, _ = _tabm_frame(F, self.flags, self.medians)
                self.tabm = fit_tabm(Xi.iloc[i_in], y_tr[i_in], Xi.iloc[i_va], y_tr[i_va], cats, self.task, seed=self.seed)
                val = tabm_predict(self.tabm, Xi.iloc[i_va], self.task)
            else:
                Xo, _ = _ordinal(X, X, cats)
                m = fit_tabpfn(Xo.iloc[i_in], y_tr[i_in], cats, self.task, seed=self.seed)
                val = m.predict(Xo.iloc[i_va]) if self.task == "regression" else m.predict_proba(Xo.iloc[i_va])
            if np.asarray(val).size != r_val.size:
                continue
            ws = np.linspace(0, 1, 21)
            vs = [score(self.task, y_tr[i_va], w * np.asarray(val).reshape(r_val.shape) + (1 - w) * r_val) for w in ws]
            self.weights[head] = float(ws[int(np.argmax(vs))])
        return self

    # ------------------------------------------------------------ predict
    def predict_outputs(self, frame, device=None):
        """Probabilities (classes_ order) or raw-scale regression values, plus the parts that were blended."""
        device = _device(device)
        frame = frame.reset_index(drop=True)
        missing = [c for c in self.pipe.num_cols + self.pipe.cat_cols + self.pipe.text_cols if c not in frame]
        if missing:
            raise ValueError(f"input lacks columns {missing}")
        full = lambda f: np.asarray(f) if self.task == "regression" else _full_probs(np.asarray(f), self.task)
        P = self.pipe.transform(frame, np.zeros(len(frame), np.int64))  # placeholder targets (unused)
        reader = full(self._reader_outputs(frame, P, device))
        E = whole_text_embeddings(frame, self.pipe.text_cols, self.pipe.namespace)
        F, _ = frozen_frame(frame, self.pipe.num_cols, self.pipe.cat_cols, E, self.pcas)
        parts = {"reader": reader}
        for head, w in self.weights.items():
            if head == "tabm":
                X, _ = _tabm_frame(F, self.flags, self.medians)
                out = tabm_predict(self.tabm, X, self.task)
            else:
                ctx = self.context.drop(columns="__y__")
                Xc, cats = design(ctx, None)
                Xn, _ = design(F[ctx.columns], None)
                Xo_tr, Xo = _ordinal(Xc, Xn, cats)
                m = fit_tabpfn(Xo_tr, self.context["__y__"].to_numpy(), cats, self.task, seed=self.seed)
                out = m.predict(Xo) if self.task == "regression" else m.predict_proba(Xo)
            parts[head] = w * np.asarray(out).reshape(reader.shape) + (1 - w) * reader
        blends = [parts[h] for h in self.weights] or [reader]
        out = np.mean(blends, axis=0)
        if self.task == "regression":
            out = out.reshape(-1) * self.target_scale + self.target_mean
        return out, parts

    def predict_proba(self, frame, device=None):
        if self.task == "regression":
            raise ValueError("predict_proba is for classification")
        return self.predict_outputs(frame, device)[0]

    def predict(self, frame, device=None):
        out = self.predict_outputs(frame, device)[0]
        return out if self.task == "regression" else self.classes_[out.argmax(1)]

    # ------------------------------------------------------------ persistence
    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "preprocessing.pkl").open("wb") as f:
            pickle.dump({"pipe": self.pipe, "pcas": self.pcas}, f, protocol=4)
        for k, sd in enumerate(self.states_dicts):
            torch.save(sd, path / f"reader_{k}.pt")
        if self.tabm is not None:
            torch.save(self.tabm, path / "tabm.pt", pickle_protocol=4)
        self.context.to_parquet(path / "context.parquet")
        manifest = {"format_version": FORMAT, "task": self.task, "n_out": self.n_out, "seed": self.seed,
                    "classes": None if self.classes_ is None else [str(c) for c in self.classes_],
                    "target_mean": self.target_mean, "target_scale": self.target_scale,
                    "n_readers": len(self.states_dicts), "gate": self.gate, "config": {k: getattr(self.config, k) for k in ("d", "layers", "cell", "dropout")},
                    "encoder": MODEL_NAME, "encoder_revision": MODEL_REVISION, "max_tokens": tc.MAX_TOKENS,
                    "blend_weights": self.weights, "tabm_flags": self.flags, "tabm_medians": self.medians,
                    "metadata": self.metadata}
        # manifest last: a directory without it is not a complete artifact
        (path / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory, device=None):
        path = Path(directory)
        man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if man["format_version"] != FORMAT:
            raise ValueError("unsupported artifact format")
        with (path / "preprocessing.pkl").open("rb") as f:
            prep = pickle.load(f)
        pipe = prep["pipe"]
        obj = cls(pipe.num_cols, pipe.cat_cols, pipe.text_cols, task=man["task"], namespace=pipe.namespace,
                  max_items=pipe.max_items, n_splits=man["n_readers"], gate=man["gate"], seed=man["seed"])
        obj.pipe, obj.pcas, obj.n_out = pipe, prep["pcas"], man["n_out"]
        for k, v in man["config"].items():
            setattr(obj.config, k, v)
        obj.classes_ = None if man["classes"] is None else np.asarray(man["classes"])
        obj.target_mean, obj.target_scale = man["target_mean"], man["target_scale"]
        obj.weights, obj.flags, obj.medians, obj.metadata = man["blend_weights"], man["tabm_flags"], man["tabm_medians"], man["metadata"]
        loc = "cuda" if _device(device).startswith("cuda") else "cpu"
        obj.states_dicts = [torch.load(path / f"reader_{k}.pt", map_location="cpu", weights_only=True) for k in range(man["n_readers"])]
        if (path / "tabm.pt").exists():
            obj.tabm = torch.load(path / "tabm.pt", map_location=loc, weights_only=False)
            if loc == "cpu":
                _devices_to_cpu(obj.tabm)
        obj.context = pd.read_parquet(path / "context.parquet")
        return obj


# ---------------------------------------------------------------- benchmark runs -> artifacts
def run_id(dataset, fold):
    return f"{dataset}_f{fold}"


def export(dataset, fold, tag="v2", out_root=None):
    """Assemble the artifact of one benchmark run from the stage A/B outputs and the heads (run on the VM)."""
    out_root = Path(out_root or ROOT / "results" / "models" / tag)
    a_dir, b_dir = ROOT / "results" / "stack" / run_id(dataset, fold), ROOT / "results" / "stackB" / run_id(dataset, fold)
    h_dir = ROOT / "results" / "heads" / tag / run_id(dataset, fold)
    info_a = json.loads((a_dir / "info.json").read_text(encoding="utf-8"))
    info_b = json.loads((b_dir / "info.json").read_text(encoding="utf-8"))
    blend = json.loads((h_dir / "blend.json").read_text(encoding="utf-8"))
    ds, x_tr, x_te, y_tr, y_te, te_idx, i_in, i_va, n_out, info = prepare(dataset, fold)
    F_tr = pd.read_parquet(a_dir / "F_tr.parquet")
    with (b_dir / "weights" / "pipe.pkl").open("rb") as f:
        pipe = pickle.load(f)
    E_tr = whole_text_embeddings(x_tr, ds.text_cols, f"e5_{dataset}")
    F_chk, pcas = frozen_frame(x_tr, ds.num_cols, ds.cat_cols, E_tr)
    pd.testing.assert_frame_equal(F_chk, F_tr, check_exact=False, atol=1e-5)  # refit PCAs reproduce stage A
    F_te = pd.read_parquet(a_dir / "F_te.parquet")
    X_tr, cats = design(F_tr, None)
    X_te, _ = design(F_te, None)
    flags = [c for c in X_tr.columns if c not in cats and (X_tr[c].isna().any() or X_te[c].isna().any())]
    obj = TabLoRAStack(ds.num_cols, ds.cat_cols, ds.text_cols, task=ds.task, namespace=f"e5_{dataset}",
                       max_items=pipe.max_items, n_splits=info_b["n_splits"], gate=info_b["gate"], seed=fold)
    obj.pipe, obj.pcas, obj.n_out = pipe, pcas, n_out
    obj.classes_ = None if info["classes"] is None else np.asarray(info["classes"])
    obj.target_mean, obj.target_scale = info["target_mean"], info["target_scale"]
    obj.states_dicts = [torch.load(b_dir / "weights" / f"reader_{k}.pt", map_location="cpu", weights_only=True)
                        for k in range(info_b["n_splits"])]
    tabm_path = h_dir / "tabm.pt"
    obj.tabm = torch.load(tabm_path, map_location="cpu" if not torch.cuda.is_available() else None, weights_only=False) if tabm_path.exists() else None
    obj.weights = blend["weights"].get("stackB", {})
    obj.flags = flags
    obj.medians = {c: float(0.0 if pd.isna(X_tr[c].median()) else X_tr[c].median()) for c in X_tr.columns if c not in cats}
    obj.context = F_tr.assign(__y__=y_tr)
    preds = np.load(ROOT / "results" / "preds" / tag / f"{run_id(dataset, fold)}.npz")
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    obj.metadata = dict(dataset=dataset, fold=fold, protocol="paper", data_revision=info["revision"],
                        train_rows=len(y_tr), test_rows=len(y_te), test_index_sha=sha(te_idx),
                        test_score=float(score(ds.task, y_te, preds[PRIMARY])),
                        method=PRIMARY, versions=dict(stack=info_a["version"], tablora=info_b["version"], heads=blend["version"]))
    path = obj.save(out_root / run_id(dataset, fold))
    print(f"exported {path} (test score {obj.metadata['test_score']:.4f})")
    return path


def check(dataset, fold, tag="v2", root=None, device=None, rows=None, tolerance=None):
    """Replay a run's stored test predictions from raw rows and the artifact (local or downloaded).

    Readers were trained and scored under fp16 autocast on GPU; an fp32 CPU replay differs slightly
    (e.g. 2.6e-3 in standardised target units on Montgomery f0, same score), hence the looser CPU tolerance."""
    if tolerance is None:
        tolerance = 1e-3 if _device(device) == "cuda" else 1e-2
    root = Path(root or ROOT / "results" / "models" / tag / run_id(dataset, fold))
    obj = TabLoRAStack.load(root, device)
    ds, x_tr, x_te, y_tr, y_te, te_idx, *_ = prepare(dataset, fold)
    stored = np.load(ROOT / "results" / "preds" / tag / f"{run_id(dataset, fold)}.npz")
    sel = np.arange(len(y_te)) if rows is None else np.arange(min(rows, len(y_te)))
    out, _ = obj.predict_outputs(x_te.iloc[sel], device)
    ref = np.asarray(stored[PRIMARY])[sel]
    if ds.task == "regression":
        out = (out - obj.target_mean) / obj.target_scale
    diff = float(np.abs(out.reshape(ref.shape) - ref).max())
    full_score = score(ds.task, y_te[sel], out.reshape(ref.shape))
    ok = diff <= tolerance
    print(f"{run_id(dataset, fold)}: max|diff| {diff:.2e} on {len(sel)} rows, replayed score {full_score:.4f} "
          f"(stored {score(ds.task, y_te[sel], ref):.4f}) {'OK' if ok else 'MISMATCH'}")
    return ok


def publish(repo, tag="v2", private=True):
    """Upload results/models/<tag> (+ index.json, model card, code snapshot) to a Hugging Face model repo."""
    import shutil
    from huggingface_hub import HfApi
    root = ROOT / "results" / "models" / tag
    runs = {}
    for man in sorted(root.glob("*/manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        runs[man.parent.name] = dict(task=m["task"], blend_weights=m["blend_weights"], **m["metadata"])
    (root / "index.json").write_text(json.dumps({"tag": tag, "method": PRIMARY, "runs": runs}, indent=1), encoding="utf-8")
    shutil.copyfile(ROOT / "MODEL_CARD.md", root / "README.md")
    code = root / "code"
    shutil.rmtree(code, ignore_errors=True)
    for name in ("src", "tests", "scripts", "reference"):
        shutil.copytree(ROOT / name, code / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "uv.lock", "README.md", ".python-version"):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, code / name)
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo, folder_path=str(root), repo_type="model")
    print(f"published {len(runs)} runs to https://huggingface.co/{repo}")


def fetch(repo, dataset, fold, revision=None):
    from huggingface_hub import snapshot_download
    rid = run_id(dataset, fold)
    local = snapshot_download(repo, revision=revision, allow_patterns=[f"{rid}/*"])
    return Path(local) / rid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("predict", help="predict a CSV/Parquet file with a saved artifact")
    p.add_argument("--artifact")
    p.add_argument("--repo")
    p.add_argument("--dataset")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--revision")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device")
    e = sub.add_parser("export", help="assemble artifacts from the benchmark runs")
    e.add_argument("--dataset", required=True)
    e.add_argument("--folds", default="0,1,2,3,4")
    c = sub.add_parser("check", help="replay stored test predictions")
    c.add_argument("--dataset", required=True)
    c.add_argument("--fold", type=int, default=0)
    c.add_argument("--repo")
    c.add_argument("--rows", type=int)
    c.add_argument("--device")
    u = sub.add_parser("publish", help="upload all exported artifacts to a Hugging Face model repo")
    u.add_argument("--repo", required=True)
    u.add_argument("--public", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "publish":
        publish(args.repo, private=not args.public)
    elif args.cmd == "export":
        for f in map(int, args.folds.split(",")):
            export(args.dataset, f)
    elif args.cmd == "check":
        root = fetch(args.repo, args.dataset, args.fold) if args.repo else None
        raise SystemExit(0 if check(args.dataset, args.fold, root=root, device=args.device, rows=args.rows) else 1)
    elif args.cmd == "predict":
        root = args.artifact or fetch(args.repo, args.dataset, args.fold, args.revision)
        obj = TabLoRAStack.load(root, args.device)
        src = Path(args.input)
        frame = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
        out, _ = obj.predict_outputs(frame, args.device)
        if obj.task == "regression":
            res = pd.DataFrame({"prediction": out})
        else:
            res = pd.DataFrame(out, columns=[f"proba_{c}" for c in obj.classes_])
            res.insert(0, "prediction", obj.classes_[out.argmax(1)])
        res.to_csv(args.output, index=False)
        print(f"wrote {len(res)} predictions to {args.output}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
