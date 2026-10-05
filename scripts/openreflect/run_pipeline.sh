#!/usr/bin/env bash
# OpenReflect data pipeline (paper Sections 3.1-3.4), one stage per command.
#
# Inputs you provide:
#   data/openreflect/envs.jsonl        environment specs (openreflect.envs.EnvSpec), with
#                                      baseline_score and >= 3 reference_scores filled in
#   data/openreflect/benchmarks.jsonl  evaluation tasks for decontamination (BenchmarkTask)
#   TEACHER_MODEL / TEACHER_BASE_URL   an OpenAI-compatible endpoint for the teacher
#
# Usage: bash scripts/openreflect/run_pipeline.sh [config]
set -euo pipefail
CONFIG=${1:-configs/openreflect/default.yaml}
D=data/openreflect
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-4}
mkdir -p "$D"

echo "== 1. headroom filter + decontamination"
openreflect filter --config "$CONFIG" --envs "$D/envs.jsonl" --benchmarks "$D/benchmarks.jsonl" \
  --out "$D/envs_kept.jsonl"

echo "== 2. teacher rollouts (RLC on, same scaffold as inference)"
mkdir -p "$D/env_specs"
python - "$D/envs_kept.jsonl" "$D/env_specs" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[2])
for line in open(sys.argv[1]):
    if line.strip():
        e = json.loads(line); (out / f"{e['env_id']}.json").write_text(json.dumps(e))
PY
: > "$D/trajs.jsonl"
for spec in "$D"/env_specs/*.json; do
  for k in $(seq 1 "$ROLLOUTS_PER_ENV"); do
    openreflect rollout --config "$CONFIG" --env "$spec" \
      --model "$TEACHER_MODEL" --base-url "$TEACHER_BASE_URL" --temperature 0.7 \
      ${SANDBOX_IMAGE:+--sandbox-image "$SANDBOX_IMAGE"} --out "$D/trajs.jsonl"
  done
done

echo "== 3. process audit"
openreflect audit --config "$CONFIG" --envs "$D/envs_kept.jsonl" --trajs "$D/trajs.jsonl" --out "$D/audit.jsonl"

echo "== 4. selection"
openreflect select --config "$CONFIG" --trajs "$D/trajs.jsonl" --audits "$D/audit.jsonl" --out "$D/accepted.jsonl"

echo "== 5. PALM weights (summary) + training windows"
openreflect weights --config "$CONFIG" --trajs "$D/accepted.jsonl" --out "$D/weights.jsonl"
openreflect build-sft --config "$CONFIG" --trajs "$D/accepted.jsonl" \
  ${TOKENIZER:+--tokenizer "$TOKENIZER"} --out "$D/windows.jsonl"

echo "== 6. train"
echo "Put the AREX deep-research data at $D/arex_research.jsonl (same window format), then:"
echo "  torchrun --nproc_per_node 8 -m openreflect.cli train --config $CONFIG"
