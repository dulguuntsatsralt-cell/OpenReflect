# OpenReflect

Training recipe and reference code for agents that improve their own solutions over many hours.

**Paper:** [OpenReflect: A Fully Specified Recipe and Reference Implementation for Training Long-Horizon Self-Improving Agents](paper/main.pdf) · Dulguun Enkhtsatsral, 2026

> **Status.** Research code. The pipeline, scaffold, and loss masking are implemented and tested. No model has been
> trained with it yet, so this repository reports no benchmark results. The values in
> `configs/openreflect/default.yaml` are proposed defaults, not tuned ones.

---

## Overview

[AREX-2](https://arxiv.org/abs/2609.38288) showed that an agent trained on long improvement trajectories keeps getting
better as it is given more rounds. Its paper describes the training recipe in prose and its repository releases only
evaluation code. OpenReflect turns that recipe into something you can run and test: every step is code, every
threshold is a config value, and every design choice has an ablation.

It adds four things:

1. **Environment checks.** Scores are put on a common scale, environments without real headroom are dropped,
   training tasks that overlap the test benchmarks are removed, and the scorer is isolated so the agent cannot game it.
2. **Round Ledger Compaction (RLC).** A structured record of every round, failures included, that replaces old
   turns when a multi-hour run fills the context window.
3. **Progress-Aware Loss Masking (PALM).** Exact rules for which agent turns are trained on: none for repeated, empty,
   or polling turns, full weight for turns that led to a better score.
4. **Matching training and test contexts.** The agent and the training data builder share one context builder, so the
   model trains on exactly what it will see at run time. A test checks this.

## How it works

```text
environments ──► filter + decontaminate ──► teacher rollouts (with RLC) ──► audit ──► select
                                                                                         │
             trained agent ◄── weighted SFT ◄── training windows ◄── PALM weights ◄──────┘
```

| Stage | Command | Code |
| --- | --- | --- |
| Keep environments with real headroom; drop benchmark overlap | `openreflect filter` | `src/openreflect/envs/` |
| Run the teacher agent with the fixed tool set and RLC | `openreflect rollout` | `src/openreflect/scaffold/` |
| Check runs for scorer access, leaked answers, copied solutions | `openreflect audit` | `src/openreflect/envs/audit.py` |
| Keep runs that reached the threshold and actually iterated | `openreflect select` | `src/openreflect/selection/` |
| Weight each turn by whether it made progress | `openreflect weights` | `src/openreflect/palm/` |
| Replay runs into training windows | `openreflect build-sft` | `src/openreflect/train/windows.py` |
| Fine-tune with the weighted loss | `openreflect train` | `src/openreflect/train/sft.py` |
| Measure productive rounds, gain per round, recovery | `openreflect metrics` | `src/openreflect/metrics/` |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[openreflect,dev]'
python -m pytest tests/openreflect -q          # unit tests + an end-to-end run on a toy environment
```

For training, also install `.[openreflect-train]` (torch, transformers, deepspeed). The full data pipeline is in
`scripts/openreflect/run_pipeline.sh`; it expects environment specs, a benchmark list for decontamination, and an
OpenAI-compatible endpoint for the teacher model.

```bash
TEACHER_MODEL=... TEACHER_BASE_URL=http://teacher/v1 bash scripts/openreflect/run_pipeline.sh
torchrun --nproc_per_node 8 -m openreflect.cli train --config configs/openreflect/default.yaml
```

## Key defaults

| Setting | Value | Config key |
| --- | --- | --- |
| Headroom filter | reference minus baseline ≥ max(3σ, 5% of reference) | `filter` |
| Decontamination | shared dataset or problem ID, or ≥ 50% 13-gram overlap | `decontam` |
| Context window, compaction trigger | 128K tokens, compact at 75%, keep last 2 rounds | `rlc` |
| PALM weight for other turns | λ = 0.5 | `palm.lam` |
| Acceptance | normalized score ≥ 0.9 (ML) or 0.95 (judge), ≥ 3 rounds, must improve after round 1 | `selection` |
| Training | Qwen3.8-27B, LR 1e-5 cosine, 2 epochs, batch 64 | `sft` |

## Ablations

Each file in `configs/openreflect/ablations/` changes one thing; `scripts/openreflect/run_ablations.sh` rebuilds the
training data for each.

| ID | Question | Configs |
| --- | --- | --- |
| A1 | Does PALM beat simpler masks? | `A1_mask_all`, `A1_mask_arex`, `A1_mask_llm_judge`, `A1_palm_lambda_{0,025,1}` |
| A2 | Does context compaction matter? | `A2_truncate`, `A2_rlc_test_only` |
| A3 | How strict should acceptance be? | `A3_tau_{07,10}` |
| A4 | Does the headroom filter matter? | `A4_no_filter`, `A4_sigma_only`, `A4_rel_margin_only` |
| A5 | How much does the teacher model matter? | `A5_runner_up_teacher` |
| A6 | How much of the gain is contamination? | `A6_no_decontam` |
| A7 | Do gains come from iterating, or from more training data? | `A7_single_attempt` |

## Evaluation

Benchmarks (MLE-bench Lite, Frontier-CS, BrowseComp, HLE, GAIA, DeepSearchQA) are run with the AREX-2 evaluation
harness, included unchanged. Instructions: [docs/evaluation/AREX2_EVALUATION.md](docs/evaluation/AREX2_EVALUATION.md).

## Repository layout

```text
src/openreflect/        the OpenReflect package
configs/openreflect/    default config, DeepSpeed config, ablations
scripts/openreflect/    pipeline and ablation scripts
tests/openreflect/      unit and end-to-end tests
docs/openreflect/       design notes, paper-to-code map, data formats, what is not implemented yet
paper/                  paper source (LaTeX) and PDF

evaluation/ data/ scripts/{research,mle,algorithmic}/ docs/evaluation/   AREX-2 evaluation harness
```

## Citation

```bibtex
@misc{enkhtsatsral2026openreflect,
  title  = {OpenReflect: A Fully Specified Recipe and Reference Implementation for Training Long-Horizon Self-Improving Agents},
  author = {Enkhtsatsral, Dulguun},
  year   = {2026},
  note   = {arXiv identifier to be added after submission}
}
```

This work builds directly on AREX-2. If you use it, please also cite:

```bibtex
@article{qian2026arex2,
  title   = {AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks},
  author  = {Qian, Hongjin and Li, Chaofan and Luo, Kun and Wei, Wenqing and Chen, Jianlyu and Lu, Shuqi and Hu, Yuyang and Xiao, Hongwang and Wang, Hui and Li, Chaozhuo and Ye, Qiwei and Dou, Zhicheng and Lian, Defu and Liu, Zheng},
  journal = {arXiv preprint arXiv:2609.38288},
  year    = {2026}
}
```

## License

Apache-2.0. OpenReflect started as a fork of [VectorSpaceLab/AREX-2](https://github.com/VectorSpaceLab/AREX-2) (also
Apache-2.0); the evaluation harness comes from there. `NOTICE` lists every change made to upstream files.
