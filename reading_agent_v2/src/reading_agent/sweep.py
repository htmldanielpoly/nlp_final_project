"""Plan and run serial experiments with recipe-aware resume checks.

Use --dry-run to inspect remaining dataset/fold pairs without launching children,
creating logs or loading embedding caches. Cache presence is not completeness.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("READING_AGENT_CACHE", Path.home() / ".cache" / "reading_agent"))
MANAGED_ARGS = {"dataset", "fold", "tag"}
OPERATIONAL_ARGS = {"save_models", "baseline_cache"}
SET_ARGS = {"skip", "key_modes", "extra"}
NUMBER_LIST_ARGS = {"budgets", "lams"}
DIAGNOSTIC_MARKERS = {"warmup_only_done", "_warmup_only_done"}
# Both zeros preserve the historical automatic-thread/non-deterministic behavior.
# No other missing scientific arguments receive an assumed default.
LEGACY_ZERO_DEFAULTS = {"threads", "deterministic"}


def dataset_names(root=ROOT):
    """Read the static registry without importing data.py and its ML dependencies."""
    tree = ast.parse((Path(root) / "src" / "reading_agent" / "data.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TEXT_DATASETS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise ValueError("cannot locate the static TEXT_DATASETS registry")


def run_arguments(rest, root=ROOT):
    """Read literal argparse declarations; never import the training entry point."""
    tree = ast.parse((Path(root) / "src" / "reading_agent" / "run.py").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    converters = {"int": int, "float": float, "str": str}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "ap"
                and node.func.attr == "add_argument"):
            continue
        names = [ast.literal_eval(arg) for arg in node.args]
        if any(name.lstrip("-").replace("-", "_") in MANAGED_ARGS for name in names):
            continue
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg in {"default", "action", "choices", "nargs", "dest"}:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            elif keyword.arg == "type":
                if not isinstance(keyword.value, ast.Name) or keyword.value.id not in converters:
                    raise ValueError("non-literal run argument type; update sweep's safe parser")
                kwargs["type"] = converters[keyword.value.id]
        parser.add_argument(*names, **kwargs)
    return vars(parser.parse_args(rest))


def canonical(value, name):
    if name in SET_ARGS:
        return tuple(sorted(set(filter(None, str(value).split(",")))))
    if name in NUMBER_LIST_ARGS:
        return tuple(sorted(set(float(v) for v in str(value).split(",") if v)))
    return value


def recipe_arguments(arguments):
    return {name: canonical(value, name) for name, value in arguments.items()
            if name not in MANAGED_ARGS | OPERATIONAL_ARGS}


@dataclass
class Manifest:
    complete: set = field(default_factory=set)
    partial: set = field(default_factory=set)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def read_manifest(tag, requested, root=ROOT):
    """A completion marker counts only when recorded arguments match the request."""
    manifest = Manifest()
    path = Path(root) / "results" / f"{tag}.jsonl"
    if not path.exists():
        return manifest
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not any(s.strip() for s in lines[index + 1:]):
                manifest.warnings.append("Ignoring a truncated final result line; it is not a completion marker.")
            else:
                manifest.errors.append(f"Invalid JSON at {path.name}:{index + 1}; repair the archive or use a new tag.")
    requested_recipe = recipe_arguments(requested)
    seen_runs, done_runs, partial_pairs = set(), set(), {}
    missing_provenance = set()
    legacy_defaults = set()
    for row in rows:
        dataset, fold = row.get("dataset"), row.get("fold")
        if dataset is None or fold is None:
            manifest.errors.append("A result record is missing dataset/fold; cannot safely resume this tag.")
            continue
        pair = (dataset, fold)
        run_id = row.get("run_id")
        args = row.get("args")
        if not isinstance(args, dict):
            manifest.errors.append(f"{dataset} fold {fold}: recorded recipe missing; use a new tag.")
            continue
        error_count = len(manifest.errors)
        if row.get("tag", tag) != tag or args.get("tag", tag) != tag:
            manifest.errors.append(f"{dataset} fold {fold}: stored tag does not match archive tag {tag}.")
        if args.get("dataset", dataset) != dataset or args.get("fold", fold) != fold:
            manifest.errors.append(f"{dataset} fold {fold}: inconsistent row and argument identities.")
        if row.get("protocol") != args.get("protocol"):
            manifest.errors.append(f"{dataset} fold {fold}: missing or inconsistent protocol metadata.")
        mismatches = []
        for name, expected in requested_recipe.items():
            if name not in args:
                if name in LEGACY_ZERO_DEFAULTS and expected == 0:
                    legacy_defaults.add(name)
                    continue
                mismatches.append(f"{name}=unrecorded (requested {expected!r})")
                continue
            try:
                observed = canonical(args[name], name)
            except (TypeError, ValueError):
                mismatches.append(f"{name}=invalid")
                continue
            if observed != expected:
                mismatches.append(f"{name}={observed!r} (requested {expected!r})")
        # Stored scientific arguments absent from today's parser are not assumed irrelevant.
        unknown = set(args) - set(requested) - MANAGED_ARGS - OPERATIONAL_ARGS
        mismatches.extend(f"{name}=not supported by current parser" for name in sorted(unknown))
        if mismatches:
            manifest.errors.append(f"{dataset} fold {fold}: recipe mismatch: " + "; ".join(mismatches))
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or any(not provenance.get(k) for k in ("data_source", "data_revision", "train_fingerprint", "test_fingerprint")):
            missing_provenance.add(dataset)
        if not row.get("implementation_version"):
            missing_provenance.add(dataset)
        marker = row.get("method")
        diagnostic = bool(args.get("warmup_only", 0)) or marker in DIAGNOSTIC_MARKERS
        if marker in DIAGNOSTIC_MARKERS:
            manifest.warnings.append(f"{dataset} fold {fold}: diagnostic warm-up marker is not a full completion.")
        if run_id is None:
            manifest.errors.append(f"{dataset} fold {fold}: no run_id; completion cannot be established safely.")
            continue
        identity = (dataset, fold, run_id)
        seen_runs.add(identity)
        partial_pairs[identity] = pair
        if marker == "_done" and not diagnostic and len(manifest.errors) == error_count:
            done_runs.add(identity)
            manifest.complete.add(pair)
    manifest.partial = {partial_pairs[r] for r in seen_runs - done_runs}
    if missing_provenance:
        manifest.warnings.append("Missing data/implementation provenance for: " + ",".join(sorted(missing_provenance)) + ". Recipe matching does not verify original data or code identity.")
    if legacy_defaults:
        manifest.warnings.append("Historical missing settings treated as behavior-preserving zero defaults: " + ",".join(sorted(legacy_defaults)) + ". Explicit nonzero settings require a fresh tag.")
    # Every score row repeats metadata, so report each issue only once.
    manifest.errors = list(dict.fromkeys(manifest.errors))
    manifest.warnings = list(dict.fromkeys(manifest.warnings))
    return manifest


def done_pairs(tag, requested, root=ROOT):
    """Validated replacement for the old marker-only resume helper."""
    manifest = read_manifest(tag, requested, root)
    if manifest.errors:
        raise ValueError("\n".join(manifest.errors))
    return manifest.complete


@dataclass
class Plan:
    requested_pairs: list
    complete: list
    pending: list
    missing_caches: list
    partial: set
    errors: list
    warnings: list


def build_plan(tag, names, folds, requested, last_datasets=(), root=ROOT, cache_dir=CACHE_DIR):
    manifest = read_manifest(tag, requested, root)
    last = [name for name in last_datasets if name in names]
    ordered = [name for name in names if name not in last] + last
    pairs = [(name, fold) for fold in folds for name in ordered]
    complete = [pair for pair in pairs if pair in manifest.complete]
    pending = [pair for pair in pairs if pair not in manifest.complete]
    missing = []
    for name in dict.fromkeys(name for name, _ in pending):
        cache = Path(cache_dir) / f"e5_{name}.pkl"
        # Only filesystem metadata is read: no pickle/cache contents are loaded.
        if not cache.is_file() or cache.stat().st_size == 0:
            missing.append(name)
    warnings = manifest.warnings.copy()
    if requested.get("data_revision") is None:
        warnings.append("Requested data_revision is unpinned; recipe matching cannot ensure unchanged HF data.")
    return Plan(pairs, complete, pending, missing, manifest.partial, manifest.errors, warnings)


def print_plan(plan, tag, requested):
    print(f"Tag: {tag}; protocol: {requested.get('protocol')}; recipe: {recipe_arguments(requested)}", flush=True)
    print(f"Requested: {len(plan.requested_pairs)} dataset/fold pairs; matching complete: {len(plan.complete)}; pending: {len(plan.pending)}", flush=True)
    print("Cache check is presence/size only; it does not prove all required text embeddings are cached.", flush=True)
    for warning in plan.warnings:
        print("WARNING: " + warning, flush=True)
    for error in plan.errors:
        print("BLOCKED: " + error, flush=True)
    if plan.errors:
        print("Use a fresh tag or pass the exact recorded recipe; no children will be launched.", flush=True)
    for name, fold in plan.pending:
        state = "missing-cache" if name in plan.missing_caches else ("retry-partial" if (name, fold) in plan.partial else "pending")
        print(f"  {name} fold {fold}: {state}", flush=True)
    if plan.missing_caches:
        print("Missing/empty cache files: " + ",".join(plan.missing_caches) + "; sweep blocked, not silently skipped.", flush=True)


def execute_plan(plan, tag, rest, requested=None, root=ROOT):
    if plan.errors or plan.missing_caches:
        return 2
    if not plan.pending:
        print("Nothing pending for this recipe.", flush=True)
        return 0
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    requested = run_arguments(rest, root) if requested is None else requested
    failures, successes = [], 0
    for name, fold in plan.pending:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        log = logs / f"{tag}_{name}_f{fold}_{timestamp}_{uuid.uuid4().hex[:8]}.log"
        command = [sys.executable, "-m", "reading_agent.run", "--dataset", name, "--fold", str(fold), "--tag", tag, *rest]
        print(f"=== {name} fold {fold}; log {log}", flush=True)
        try:
            with log.open("x", encoding="utf-8") as handle:
                code = subprocess.call(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        except OSError as error:
            print(f"    launch failed: {error}", flush=True)
            failures.append((name, fold, "launch-error"))
            continue
        print(f"    exit {code}", flush=True)
        if code:
            failures.append((name, fold, code))
        else:
            finished = read_manifest(tag, requested, root)
            if not requested.get("warmup_only") and (finished.errors or (name, fold) not in finished.complete):
                print("    no matching full completion marker; treating child as incomplete", flush=True)
                failures.append((name, fold, "incomplete-results"))
            else:
                successes += 1
    print(f"Sweep child summary: succeeded={successes}; failed={len(failures)}; attempted={len(plan.pending)}", flush=True)
    for name, fold, code in failures:
        print(f"  FAILED {name} fold {fold}: exit {code}", flush=True)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--last-datasets", default="", help="comma-separated datasets moved last within each fold (e.g. MUL_TEXT_SPOTIFY_GENRES)")
    parser.add_argument("--dry-run", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        separator = argv.index("--")
        args = parser.parse_args(argv[:separator])
        rest = argv[separator + 1:]
    else:
        args, rest = parser.parse_known_args(argv)
    try:
        if not args.tag or args.tag in {".", ".."} or any(c in args.tag for c in '/\\<>:"|?*'):
            raise ValueError("tag must be a safe filename component")
        registered = dataset_names(ROOT)
        names = list(dict.fromkeys(filter(None, args.datasets.split(",")))) or registered
        last = list(dict.fromkeys(filter(None, args.last_datasets.split(","))))
        unknown = set(names + last) - set(registered)
        if unknown:
            raise ValueError("unknown datasets: " + ",".join(sorted(unknown)))
        folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
        if not folds:
            raise ValueError("at least one fold is required")
        requested = run_arguments(rest, ROOT)
        protocol = requested.get("protocol")
        if protocol not in {"paper", "current"}:
            raise ValueError("protocol must be paper or current")
        maximum = 4 if protocol == "paper" else 5
        if any(f < 0 or f > maximum for f in folds):
            raise ValueError(f"{protocol} protocol folds must lie in 0..{maximum}")
        plan = build_plan(args.tag, names, folds, requested, last, root=ROOT, cache_dir=CACHE_DIR)
        print_plan(plan, args.tag, requested)
        if args.dry_run:
            print("Dry run: no child processes, log directories or embedding cache loads.", flush=True)
            return 2 if plan.errors or plan.missing_caches else 0
        return execute_plan(plan, args.tag, rest, requested=requested, root=ROOT)
    except (ValueError, OSError, SyntaxError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
