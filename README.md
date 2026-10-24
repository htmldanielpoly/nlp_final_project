# Reading What Matters: Table-Conditioned Text Reading for Text-Tabular Prediction

NLP 097215 final project (Track 1, algorithm), Technion, 2026.

**Submitted version: v2** (`reading_agent_v2/`). v1 is kept unchanged for reference.

| Path | Contents |
|---|---|
| `reading_agent_v2/` | **v2 (submission)**: code, tests, pinned dependencies, evaluation CSVs |
| `reading_agent_v2/paper/` | **v2 paper** (`main.pdf`, source `main.tex`, generated tables) |
| `reading_agent/` | v1: selective-reading agent (code, tests, pinned dependencies) |
| `paper/` | v1 paper |

Trained weights for all 100 runs (20 MulTaBench text datasets x 5 folds):

- v2: <https://huggingface.co/htmldaniel/reading-agent-v2-multabench>
- v1: <https://huggingface.co/htmldaniel/reading-agent-multabench>

v2 (TabLoRA readers, cross-fitted and blended with Joint-Frozen TabM and TabPFN-2.5) reaches
0.913 AUROC / 0.738 R² against v1's 0.900 / 0.702 on identical test rows.
See [`reading_agent_v2/README.md`](reading_agent_v2/README.md) for installation, the inference and
end-to-end training commands, and how the paper's tables are generated
([`reading_agent/README.md`](reading_agent/README.md) for v1).
