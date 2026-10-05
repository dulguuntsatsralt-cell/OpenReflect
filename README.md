# OpenReflect

**A fully specified, reproducible recipe for training long-horizon self-improving agents, built on AREX-2.**

This repository is a fork of [VectorSpaceLab/AREX-2](https://github.com/VectorSpaceLab/AREX-2). The upstream
repository releases only evaluation code. OpenReflect adds the parts the AREX-2 paper
([arXiv:2609.38288](https://arxiv.org/abs/2609.38288)) describes but does not release or fully specify:
environment synthesis and filtering, the agent scaffold, context management for multi-hour runs,
progress-aware loss masking, trajectory selection, and weighted SFT. Every threshold is a config value,
and every component has an ablation config.

Paper: [OpenReflect: A Fully Specified Recipe and Reference Implementation for Training Long-Horizon
Self-Improving Agents](paper/main.pdf) (Dulguun Enkhtsatsral, 2026). LaTeX source is in [`paper/`](paper/).

> Status: research code accompanying the paper. No training runs have been completed yet. The
> hyperparameters and thresholds in `configs/openreflect/default.yaml` are proposed defaults, not tuned
> results.

## What OpenReflect adds

| Gap in the AREX-2 recipe | OpenReflect component | Where |
| --- | --- | --- |
| "Baseline well below the reference" has no number | Noise-aware headroom filter: gap >= max(3 sigma, 5% of abs(reference)) | `src/openreflect/envs/filter.py` |
| Scores on different scales | Baseline-to-reference normalization | `src/openreflect/envs/normalize.py` |
| No contamination check | 13-gram, dataset, and problem-ID decontamination | `src/openreflect/envs/decontam.py` |
| "Process conforms" is unspecified | Isolated, hash-checked scorer plus canary, rule, and LLM audit | `src/openreflect/envs/scorer.py`, `audit.py` |
| Scaffold not released | Fixed tool set with blocking `wait_job` | `src/openreflect/scaffold/tools.py` |
| Context overflow on 12-hour runs | Round Ledger Compaction (RLC) | `src/openreflect/scaffold/rlc.py`, `ledger.py` |
| Train/test context mismatch | One context builder shared by the agent and the training replay | `src/openreflect/scaffold/context.py`, `train/windows.py` |
| "No-progress" and "near-duplicate" turns undefined | PALM: rules R, N, P, plus a provenance set S+ | `src/openreflect/palm/` |
| Acceptance thresholds unstated | tau_ml = 0.9, tau_judge = 0.95, at least 3 rounds, must iterate, at most 2 per env | `src/openreflect/selection/` |
| No training hyperparameters | Full SFT config with weighted loss | `configs/openreflect/default.yaml`, `src/openreflect/train/sft.py` |
| Capabilities measured only by final score | T*, mean gain, best-so-far AUC, recovery rate, wasted-turn share | `src/openreflect/metrics/` |

Design notes, the paper-to-code map, data formats, and a list of what is not implemented yet are in
[docs/openreflect/README.md](docs/openreflect/README.md).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[openreflect,dev]'        # pipeline, scaffold, PALM, tests
# pip install -e '.[openreflect-train]'    # adds torch, transformers, deepspeed for training
python -m pytest tests/openreflect -q
```

## Pipeline

```bash
openreflect filter    --envs data/openreflect/envs.jsonl --benchmarks data/openreflect/benchmarks.jsonl \
                      --out data/openreflect/envs_kept.jsonl
openreflect rollout   --env env.json --model TEACHER --base-url http://teacher/v1 --out data/openreflect/trajs.jsonl
openreflect audit     --envs data/openreflect/envs_kept.jsonl --trajs data/openreflect/trajs.jsonl --out audit.jsonl
openreflect select    --trajs data/openreflect/trajs.jsonl --audits audit.jsonl --out data/openreflect/accepted.jsonl
openreflect build-sft --trajs data/openreflect/accepted.jsonl --out data/openreflect/windows.jsonl
torchrun --nproc_per_node 8 -m openreflect.cli train --config configs/openreflect/default.yaml
openreflect metrics   --trajs runs/eval_trajs.jsonl --budget-s 18000
```

`scripts/openreflect/run_pipeline.sh` runs stages 1 to 5 in order. `scripts/openreflect/run_ablations.sh`
rebuilds training data for each ablation in `configs/openreflect/ablations/`:

| ID | Question | Configs |
| --- | --- | --- |
| A1 | Does PALM beat simpler masks? | `A1_mask_all`, `A1_mask_arex`, `A1_mask_llm_judge`, `A1_palm_lambda_{0,025,1}` |
| A2 | Does RLC matter, and does train/test consistency? | `A2_truncate`, `A2_rlc_test_only` |
| A3 | How strict should acceptance be? | `A3_tau_{07,10}` |
| A4 | Does the headroom filter matter? | `A4_no_filter`, `A4_sigma_only`, `A4_rel_margin_only` |
| A5 | How much does the teacher matter? | `A5_runner_up_teacher` |
| A6 | How much of the gain is contamination? | `A6_no_decontam` |
| A7 | Is transfer caused by iteration? | `A7_single_attempt` + `scripts/openreflect/make_single_attempt.py` |

Evaluation on the six benchmarks uses the upstream AREX-2 harness below, unchanged.

## Layout

```text
src/openreflect/       OpenReflect package (envs, scaffold, palm, selection, train, metrics, cli)
configs/openreflect/   default config, DeepSpeed config, ablation configs
scripts/openreflect/   pipeline and ablation scripts
tests/openreflect/     unit and end-to-end tests
docs/openreflect/      design notes
paper/                 paper LaTeX source and PDF
evaluation/ data/ assets/ docs/evaluation/ scripts/{research,mle,algorithmic}/   upstream AREX-2 (unchanged)
```

## License and attribution

Upstream AREX-2 code is Apache-2.0 (see `LICENSE`) and is kept unchanged except for packaging
(`pyproject.toml`, `setup.py`) and this README. OpenReflect additions are released under the same license.
See `NOTICE` for the list of modifications.

If you use this code, please cite OpenReflect and AREX-2:

```bibtex
@misc{enkhtsatsral2026openreflect,
  title  = {OpenReflect: A Fully Specified Recipe and Reference Implementation for Training Long-Horizon Self-Improving Agents},
  author = {Enkhtsatsral, Dulguun},
  year   = {2026},
  note   = {arXiv identifier to be added after submission}
}


@article{2026arex2,
  title   = {AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks},
  author  = {Qian, Hongjin and Li, Chaofan and Luo, Kun and Wei, Wenqing and Chen, Jianlyu and Lu, Shuqi and Hu, Yuyang and Xiao, Hongwang and Wang, Hui and Li, Chaozhuo and Ye, Qiwei and Dou, Zhicheng and Lian, Defu and Liu, Zheng},
  journal = {arXiv preprint arXiv:2609.38288},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.38288}
}
```

---

# Upstream: AREX-2 evaluation suite

The rest of this README is the upstream AREX-2 README, kept for the evaluation harness.

AREX-2 studies whether an agent can turn more test-time rounds into a better solution. The paper treats this as two connected abilities: **reflection**, which uses feedback to decide what to change, and **long-horizon execution**, which keeps the improvement loop useful over many rounds. We synthesize these trajectories from machine learning engineering and algorithmic programming, where progress can be checked directly, and study how the resulting capability transfers to deep research.

This repository contains the evaluation runners and dataset definitions for the six reported tracks. It keeps benchmark files, model outputs, credentials, and run artifacts outside Git while making the evaluation command simple: choose a dataset, then record the model, endpoint, task range, evaluator mode, commit, and data checksum with every number.

- 🌐 [Project site](https://vectorspacelab.github.io/AREX-2/) — Research overview and benchmark results.
- 📚 [Evaluation guide](data/README.md) — Data preparation, prompts, and scoring for each dataset.
- 🧪 [Experiment notes](scripts/README.md) — Run configurations and experiment notes.
- 🇨🇳 [中文说明](README.zh-CN.md) — 中文安装与评测指南。

## Layout

```text
evaluation/   CLI, benchmark implementations, judge services, and vendor snapshots
data/         dataset configs, preparation catalog, and per-dataset notes
assets/       benchmark PDF/SVG, logo files, and the static project site
docs/         detailed evaluation notes
scripts/      download scripts, run configs, and algorithmic experiments
evaluate.py   short root-level wrapper for selecting research datasets
```

The important rule is that dataset-specific behavior lives in `data/<track>/<dataset>/`: `config.json` describes the input and output fields, `prompt.py` describes the model prompt, and `judge_local.py` or `judge_offical.py` defines the scorer when the benchmark has one. The complete matrix is in [data/README.md](data/README.md).

## Install

Python 3.10+ is required. Install only the dependencies for the track you need:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[research]'   # research benchmarks
# pip install -e '.[frontier]' # Frontier-CS
# pip install -e '.[all]'      # all Python dependencies
python3 evaluate.py doctor
```

## Research evaluation

Copy [the environment template](scripts/configs/model.env.example) to `.env`,
fill in your model, tokenizer, judge, and tool credentials, then load it once:

```bash
cp scripts/configs/model.env.example .env
# Edit .env before loading it.
set -a; source .env; set +a
```

List the registered datasets first:

```bash
python3 evaluate.py list
```

The outer interface only asks for a dataset. The wrapper resolves its config, data path, prompt, loader, and scorer before launching the evaluator:

```bash
# inspect the exact subprocess command, without credentials or network calls
python3 evaluate.py BrowseComp --n 1 --dry-run

# evaluate rows [0, 10)
python3 evaluate.py BrowseComp --n 10 --save-path runs/browsecomp-10

# evaluate rows [100, 120)
python3 evaluate.py HLE --start-index 100 --n 20

# run the same range for two datasets
python3 evaluate.py BrowseComp HLE --n 5 --save-path runs/smoke
```

BrowseComp, GAIA, HLE, and DeepSearch-QA use the `refine-equal` profile when
selected directly. It defaults to one concurrent case, a 300-call per-round
budget with a 1,500-call total cap, confidence-tiered review, and the same
thinking, sampling, token, and retry settings across the four benchmarks. The
judge remains an explicit external service:

```bash
export MODEL_API_KEY=...
export JUDGE_API_KEY=...
python3 evaluate.py BrowseComp --n 10 \
  --model YOUR_MODEL --base-url http://model.example/v1 \
  --tokenizer-path /path/to/tokenizer \
  --judge-model YOUR_JUDGE --judge-base-url http://judge.example/v1 \
  --judge-api-key-env JUDGE_API_KEY \
  --save-path runs/browsecomp-10
```

See [the shared parameter table](docs/evaluation/configuration.md#shared-research-profile)
for the exact values. `--profile default` opts out of this profile;
`--dry-run` shows the expanded command without calling a service.

Prepare the four datasets with a built-in download recipe:

```bash
python3 evaluate.py download BrowseComp
python3 evaluate.py download DeepSearch-QA
python3 evaluate.py download HLE                 # requires HF_TOKEN and access
python3 evaluate.py download GAIA-2023-validation-text-103  # requires HF_TOKEN
python3 evaluate.py download --list
```

By default files are stored under `data/files/`. Use `--data-root PATH` to keep them elsewhere, or `--data-path PATH` for one manually prepared dataset. The selected path is checked before a real run starts. Results are written to `runs/<timestamp>/<dataset>/`; inspect each case's `status`, `judge_raw`, and `score_result` before aggregating scores.

The per-dataset input format, prompt, judge, metric, and preparation command are documented in [data/README.md](data/README.md). Advanced CLI and environment options are in [docs/evaluation/configuration.md](docs/evaluation/configuration.md).

## Frontier-CS algorithmic evaluation

The source for the judge is in `evaluation/frontier/judge/`; the downloaded problem set and run artifacts live in `data/algorithmic/`:

```bash
python3 evaluate.py download algorithmic
python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
```

Solutions are C++17 files. The checker reports per-case results and `scoreRatio`/`scoreRatioUnbounded`; see [data/algorithmic/README.md](data/algorithmic/README.md) for the input layout and judge lifecycle.

## MLE-bench Lite

```bash
MLE_BENCH=$HOME/mle-bench python3 evaluate.py mle leaf-classification --prepare
MLE_BENCH=$HOME/mle-bench python3 evaluate.py mle leaf-classification
MLE_BENCH=$HOME/mle-bench \
  bash scripts/mle/grade.sh runs/<run-dir> leaf-classification
```

The final number comes from the host grader (`grade.log`), including `valid_submission` and the competition score. The harness and its upstream notes are kept in `evaluation/mle/`; the checkout-level lifecycle wrappers are in `scripts/mle/`.

## Results

![AREX benchmark results](assets/performance/arex-v2-benchmark-results.svg)

## Citation

If AREX is useful in your work, please cite:

```bibtex
@article{2026arex2,
  title   = {AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks},
  author  = {Qian, Hongjin and Li, Chaofan and Luo, Kun and Wei, Wenqing and Chen, Jianlyu and Lu, Shuqi and Hu, Yuyang and Xiao, Hongwang and Wang, Hui and Li, Chaozhuo and Ye, Qiwei and Dou, Zhicheng and Lian, Defu and Liu, Zheng},
  journal = {arXiv preprint arXiv:2609.38288},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.38288}
}
```
