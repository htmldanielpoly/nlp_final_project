# Reading What Matters: Table-Conditioned Selective Text Reading

NLP 097215 final project (Track 1, algorithm), Technion, 2026.

| Path | Contents |
|---|---|
| `paper/` | The short paper (`main.pdf`, source `main.tex`, generated tables/figures) |
| `reading_agent/` | Source code, tests, pinned dependencies (`pyproject.toml` + `uv.lock`) |

Trained weights for all 100 runs (20 MulTaBench text datasets x 5 folds):
<https://huggingface.co/htmldaniel/reading-agent-multabench>.

See [`reading_agent/README.md`](reading_agent/README.md) for installation, the
inference and end-to-end training commands, and how the paper's tables are generated.
