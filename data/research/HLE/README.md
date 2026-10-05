# HLE

**Input**: JSONL text-only HLE rows (image rows excluded).

**Evaluation**: The unified evaluator loads `config.json`, applies the prompt in `prompt.py`, runs the model, then uses `hle_official` as the scoring contract. HLE judge adapter.

**Reported metric**: full-credit and task metrics. The raw judge payload is kept in the run directory; aggregate only successful rows.

**Data**: The default path is `${AREX_DATA_ROOT}/HLE/text_items.jsonl` after `${AREX_DATA_ROOT}` expansion. Run `python3 evaluate.py download HLE`.

**Run**:

```bash
python3 evaluate.py HLE --n 10 --save-path runs/hle-10
```

The default `auto` profile uses the HLE agent and judge adapter with the
same one-case concurrency, ten outer rounds, 300-per-round and 1,500-total
budgets, confidence thresholds (95/90), generation settings, and retry values
as the other three headline benchmarks. HLE-specific context truncation and
judge calls remain inside that adapter; supply the external judge with the
three `--judge-*` options.

This directory contains the prompt and judge implementation for the dataset. If the benchmark has `judge_local.py` or `judge_offical.py`, the selected judge mode is loaded from that file; otherwise the scorer metadata in `config.json` is used.
