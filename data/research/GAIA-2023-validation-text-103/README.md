# GAIA-2023-validation-text-103

**Input**: JSONL text-only GAIA rows.

**Evaluation**: The unified evaluator loads `config.json`, applies the prompt in `prompt.py`, runs the model, then uses `gaia_text_103_judge` as the scoring contract. GAIA validation text rubric / WebAgent LLM judge.

**Reported metric**: task-level correctness. The raw judge payload is kept in the run directory; aggregate only successful rows.

**Data**: The default path is `${AREX_DATA_ROOT}/GAIA-2023-validation-text-103/standardized_data.jsonl` after `${AREX_DATA_ROOT}` expansion. Run `python3 evaluate.py download GAIA-2023-validation-text-103`.

**Run**:

```bash
python3 evaluate.py GAIA-2023-validation-text-103 --n 10 --save-path runs/gaia-2023-validation-text-103-10
```

The default `auto` profile matches BrowseComp: one concurrent case, ten outer
rounds, 300 calls per round, 1,500 calls per case, confidence-tiered review
(95/90), and the same thinking, sampling, token, and retry settings. Supply
the external judge with the three `--judge-*` options shown above.

This directory contains the prompt and judge implementation for the dataset. If the benchmark has `judge_local.py` or `judge_offical.py`, the selected judge mode is loaded from that file; otherwise the scorer metadata in `config.json` is used.
