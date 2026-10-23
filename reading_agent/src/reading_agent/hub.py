"""Publish saved run artifacts to, and fetch them from, the Hugging Face Hub.

The Hub repo mirrors `results/models/<tag>/`: one folder per run (`<run_id>/reader`,
`gain_content`, `gain_meta`), plus `index.json` (dataset, fold, task, test scores)
and `README.md` (the model card). Artifacts contain pickled preprocessing; load
only repositories you trust.

    python -m reading_agent.hub index  --tag paper_v2                 # write index.json locally
    python -m reading_agent.hub upload --tag paper_v2 --repo <user>/reading-agent-multabench
    python -m reading_agent.hub fetch  --repo <user>/reading-agent-multabench --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0
"""
import argparse
import json
import shutil
from pathlib import Path

from .run import RESULTS

CARD = Path(__file__).resolve().parents[2] / "MODEL_CARD.md"
REPORTED = ("agent_read_all", "blend_catboost_agent", "catboost_frozen", "gainB_content@0.1", "gainB_meta@0.1")


def build_index(tag):
    """One entry per completed run that has saved artifacts."""
    rows = [json.loads(line) for line in (RESULTS / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {r["run_id"] for r in rows if r.get("method") == "_done"}
    index = {}
    for r in rows:
        run_id = r.get("run_id")
        if run_id not in done or not (RESULTS / "models" / tag / run_id / "reader").is_dir():
            continue
        if run_id not in index:
            manifest = RESULTS / "models" / tag / run_id / "reader" / "manifest.json"
            meta = json.loads(manifest.read_text(encoding="utf-8")).get("metadata", {}) if manifest.is_file() else {}
            prov = meta.get("provenance", {})
            index[run_id] = {"dataset": r["dataset"], "fold": r["fold"], "task": r["task"],
                             "protocol": r.get("protocol"), "artifacts": sorted(
                                 p.name for p in (RESULTS / "models" / tag / run_id).iterdir() if p.is_dir()),
                             "data_revision": prov.get("data_revision"),
                             "train_fingerprint": prov.get("train_fingerprint"),
                             "test_fingerprint": prov.get("test_fingerprint"),
                             "implementation_version": meta.get("implementation_version"),
                             "test_scores": {}}
        entry = index[run_id]
        key = r["method"] + (f"@{r['budget']:g}" if "budget" in r else "")
        if key in REPORTED:
            entry["test_scores"][key] = r["score"]
    return {"tag": tag, "metric": {"binary": "AUROC", "multiclass": "macro one-vs-rest AUROC", "regression": "R2"},
            "runs": dict(sorted(index.items()))}


def write_index(tag):
    path = RESULTS / "models" / tag / "index.json"
    index = build_index(tag)
    path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return path, index


def upload(tag, repo, private=False):
    from huggingface_hub import HfApi
    folder = RESULTS / "models" / tag
    write_index(tag)
    shutil.copyfile(CARD, folder / "README.md")
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=str(folder))
    # Source snapshot that produced these weights, so the repo is usable without another download.
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(CARD.parent), path_in_repo="code",
                      allow_patterns=["src/reading_agent/*.py", "tests/*.py", "scripts/*", "pyproject.toml",
                                      "uv.lock", "README.md", ".python-version", "reference/**"],
                      commit_message="Source snapshot")
    return f"https://huggingface.co/{repo}"


def fetch(repo, dataset, fold, revision=None, local_dir=None):
    """Download one run's artifacts; returns the local run directory."""
    from huggingface_hub import hf_hub_download, snapshot_download
    index = json.loads(Path(hf_hub_download(repo, "index.json", revision=revision)).read_text(encoding="utf-8"))
    matches = [rid for rid, e in index["runs"].items() if e["dataset"] == dataset and e["fold"] == fold]
    if len(matches) != 1:
        raise ValueError(f"expected one run for {dataset} fold {fold}, found {len(matches)}")
    root = snapshot_download(repo, revision=revision, allow_patterns=[f"{matches[0]}/*"], local_dir=local_dir)
    return Path(root) / matches[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "upload"):
        p = sub.add_parser(name)
        p.add_argument("--tag", default="paper_v2")
        if name == "upload":
            p.add_argument("--repo", required=True)
            p.add_argument("--private", action="store_true")
    p = sub.add_parser("fetch")
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--revision")
    p.add_argument("--local_dir")
    args = ap.parse_args(argv)
    if args.cmd == "index":
        path, index = write_index(args.tag)
        print(f"{len(index['runs'])} runs -> {path}")
    elif args.cmd == "upload":
        print(upload(args.tag, args.repo, args.private))
    else:
        print(fetch(args.repo, args.dataset, args.fold, args.revision, args.local_dir))


if __name__ == "__main__":
    main()
