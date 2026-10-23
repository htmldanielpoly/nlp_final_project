"""Replay warm-up early-stopping rules on a saved full validation curve.

Usage: python -m reading_agent.es_replay results/history/es_curve/<run_id>.json [...]
The curve must come from a run without effective stopping (``--patience`` >= ``--epochs``),
e.g. ``run --warmup_only 1 --epochs 100 --patience 100``. Mirrors ``train.train_warmup``:
patience counts epochs without a 1e-4 improvement of the (optionally EMA-smoothed) stop
loss, and the restored checkpoint is the lowest raw val loss seen up to the stop.
"""
import argparse
import glob
import json
import math
from numbers import Integral, Real
from pathlib import Path

from .train import Config

RULES = [  # (name, epochs, patience, ema, min_epochs)
    ("p6 (current)", 60, 6, 0.0, 0),
    ("p10", 60, 10, 0.0, 0),
    ("p12", 80, 12, 0.0, 0),
    ("p15", 100, 15, 0.0, 0),
    ("p6 ema.5", 60, 6, 0.5, 0),
    ("p8 ema.7", 80, 8, 0.7, 0),
    ("p10 ema.7", 100, 10, 0.7, 0),
    ("p6 min30", 60, 6, 0.0, 30),
    ("p12 min30", 100, 12, 0.0, 30),
]


def _curve(val):
    values = list(val)
    if not values:
        raise ValueError("validation curve must contain at least one epoch")
    for ep, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"validation loss at epoch {ep} must be a finite number")
    return values


def _config(epochs, patience, ema_ret, min_epochs, sched):
    try:
        return Config(epochs=epochs, patience=patience, early_stop_ema=ema_ret,
                      min_epochs=min_epochs, sched=sched)
    except TypeError as exc:
        raise ValueError(f"invalid replay configuration: {exc}") from exc


def _run(val, cfg):
    """Replay validated inputs; the final flag distinguishes stopping from curve exhaustion."""
    patience = cfg.patience + (4 if cfg.sched == "plateau" else 0)
    best, best_ep, best_stop, ema, bad = math.inf, None, math.inf, None, 0
    for ep, vl in enumerate(val[:cfg.epochs]):
        if vl < best:
            best, best_ep = vl, ep
        ema = vl if ema is None else cfg.early_stop_ema * ema + (1 - cfg.early_stop_ema) * vl
        if ema < best_stop - 1e-4:
            best_stop, bad = ema, 0
        else:
            bad += 1
        if ep + 1 >= cfg.min_epochs and bad >= patience:
            return ep + 1, best, best_ep, True
    return min(cfg.epochs, len(val)), best, best_ep, False


def replay(val, epochs, patience, ema_ret, min_epochs, *, sched="none"):
    """Return ``(epochs_run, best_raw_loss, zero_based_best_epoch)`` on the observed curve.

    Mirrors warm-up's strict 1e-4 patience threshold, EMA retention and completed-epoch
    minimum. Raw checkpoint selection has no 1e-4 threshold. ``plateau`` adds four to
    patience, as training does. Replay cannot predict unobserved epochs or the different
    learning-rate trajectory produced by changing the source run's scheduler.
    """
    cfg = _config(epochs, patience, ema_ret, min_epochs, sched)
    return _run(_curve(val), cfg)[:3]


def longest_stall(val, upto):
    """Longest non-reset streak through zero-based ``upto`` under raw patience semantics.

    The baseline updates only for a strict improvement greater than 1e-4, so tiny new
    raw checkpoint minima can count as stalled epochs. This is not the gap between
    successive exact raw minima, and it does not use EMA.
    """
    val = _curve(val)
    if isinstance(upto, bool) or not isinstance(upto, Integral) or not 0 <= upto < len(val):
        raise ValueError("upto must be a valid zero-based curve epoch")
    best, run, worst = math.inf, 0, 0
    for vl in val[:upto + 1]:
        if vl < best - 1e-4:
            best, run = vl, 0
        else:
            run += 1
            worst = max(worst, run)
    return worst


def _load_history(path):
    """Read current list histories or a history envelope, with optional metadata sidecars."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata, sources = {}, []
    if isinstance(payload, dict):
        if "history" not in payload:
            raise ValueError("history object must contain a 'history' list")
        history = payload["history"]
        metadata = {key: value for key, value in payload.items() if key != "history"}
        sources.append(str(path))
    else:
        history = payload
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a nonempty list")
    values = []
    for ep, row in enumerate(history):
        if not isinstance(row, dict) or "val_loss" not in row:
            raise ValueError(f"history epoch {ep} must contain val_loss")
        if "epoch" in row and row["epoch"] != ep:
            raise ValueError("history epochs must be contiguous and zero-based")
        values.append(row["val_loss"])
    # Do not require sidecars: older diagnostic histories saved only the epoch rows.
    candidates = [path.with_suffix(".meta.json"), path.with_suffix(".metadata.json")]
    if path.parent.parent.name == "history":
        model_dir = path.parent.parent.parent / "models" / path.parent.name / path.stem
        candidates.extend(sorted(model_dir.glob("*/manifest.json")))
    for sidecar in candidates:
        if not sidecar.is_file():
            continue
        extra = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(extra, dict):
            raise ValueError(f"metadata sidecar must be an object: {sidecar}")
        metadata = {**extra.get("metadata", {}), **extra, **metadata}
        sources.append(str(sidecar))
        # All policy manifests share the same warm-up configuration.
        if sidecar.name == "manifest.json":
            break
    return _curve(values), metadata, sources


def main(paths):
    """Print replay tables; accepts paths directly for existing programmatic callers."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("provide at least one history path")
    for p in paths:
        path = Path(p)
        try:
            val, metadata, sources = _load_history(path)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        observed_best = min(val)
        print(f"\n{path.name}: {len(val)} observed epochs, best observed val {observed_best:.4f} "
              f"@ ep {val.index(observed_best)}, longest raw-criterion stall before it "
              f"{longest_stall(val, val.index(observed_best))} epochs")
        source_cfg = metadata.get("config") or metadata.get("args") or {}
        source_epochs = source_cfg.get("epochs")
        source_sched = source_cfg.get("sched", "unknown")
        print(f"  source metadata: {', '.join(sources) if sources else 'unavailable'}; "
              f"early_stopped={metadata.get('early_stopped', 'unknown')}; scheduler={source_sched}")
        if isinstance(source_epochs, Integral) and not isinstance(source_epochs, bool):
            status = ("truncated before configured epoch cap" if len(val) < source_epochs
                      else "reached configured epoch cap")
            print(f"  source coverage: {status} ({len(val)}/{source_epochs})")
        else:
            print("  source coverage: unknown; epoch rows alone do not certify a full curve")
        print("  The best observed value is not a global best; unobserved epochs cannot be replayed.")
        print("  Replay holds the source optimization trajectory fixed (including its learning rates).")
        print(f"  {'rule':<14}{'ran':>5}{'best val':>10}{'@ep':>5}{'gap':>9}  coverage")
        for name, *r in RULES:
            cfg = _config(*r, sched="none")
            ran, best, ep, stopped = _run(val, cfg)
            coverage = ("stopped" if stopped else "epoch cap" if ran == cfg.epochs
                        else f"truncated: needs up to {cfg.epochs} epochs")
            print(f"  {name:<14}{ran:>5}{best:>10.4f}{ep:>5}{best - observed_best:>9.4f}  {coverage}")


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="history JSON paths or glob patterns")
    args = parser.parse_args(argv)
    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"history path/pattern did not match a file: {pattern}")
        paths.extend(matches)
    try:
        main(list(dict.fromkeys(paths)))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli()
