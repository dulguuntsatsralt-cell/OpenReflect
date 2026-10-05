# DeepSearch-QA

**Input**: CSV question/answer rows.

**Evaluation**: The unified evaluator loads `config.json`, applies the prompt in `prompt.py`, runs the model, then uses `deepsearchqa_official` as the scoring contract. DeepSearchQA official Gemini-autorater style judge.

**Reported metric**: official QA score. The raw judge payload is kept in the run directory; aggregate only successful rows.

**Data**: The default path is `${AREX_DATA_ROOT}/DeepSearch-QA/DSQA-full.csv` after `${AREX_DATA_ROOT}` expansion. Run `python3 evaluate.py download DeepSearch-QA`.

**Run**:

```bash
python3 evaluate.py DeepSearch-QA --n 10 --save-path runs/deepsearch-qa-10
```

The default `auto` profile matches BrowseComp: one concurrent case, ten outer
rounds, 300 calls per round, 1,500 calls per case, confidence-tiered review
(95/90), and the same thinking, sampling, token, and retry settings. The
inference model serves summaries; provide the external judge with the three
`--judge-*` options.

This directory contains the prompt and judge implementation for the dataset. If the benchmark has `judge_local.py` or `judge_offical.py`, the selected judge mode is loaded from that file; otherwise the scorer metadata in `config.json` is used.
