#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

EPOCHS="${EPOCHS:-10}"
RESULTS_JSONL="${SCRIPT_DIR}/results/metrics.jsonl"

MODELS=(
  "google/mt5-small"
  "vinai/bartpho-syllable"
  "VietAI/vit5-base"
  "vinai/bartpho-word"
  "facebook/mbart-large-50-many-to-many-mmt"
  "google/mt5-base"
)

METHODS=(
  "pipeline"
  "multitask"
  "end2end"
)

has_result() {
  local model="$1"
  local method="$2"
  local instr_mode="$3"
  local instr_fixed_id="$4"
  local epochs="$5"

  if [[ ! -f "$RESULTS_JSONL" ]]; then
    return 1
  fi

  python3 - "$RESULTS_JSONL" "$model" "$method" "$instr_mode" "$instr_fixed_id" "$epochs" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
model = sys.argv[2]
method = sys.argv[3]
instr_mode = sys.argv[4]
instr_fixed_id = int(sys.argv[5])
epochs = int(sys.argv[6])

for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (
        row.get("model_name") == model
        and row.get("method") == method
        and row.get("instr_mode", "na") == instr_mode
        and int(row.get("instr_fixed_id", -1)) == instr_fixed_id
        and int(row.get("epochs", -1)) == epochs
    ):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_job() {
  local model="$1"
  local method="$2"
  local instr_mode="$3"
  local instr_fixed_id="$4"

  if has_result "$model" "$method" "$instr_mode" "$instr_fixed_id" "$EPOCHS"; then
    echo "[skip] model=$model method=$method instr_mode=$instr_mode instr_fixed_id=$instr_fixed_id epochs=$EPOCHS"
    return 0
  fi

  echo "[run ] model=$model method=$method instr_mode=$instr_mode instr_fixed_id=$instr_fixed_id epochs=$EPOCHS"
  "$PYTHON_BIN" train_qag_benchmark.py \
    --model_name "$model" \
    --method "$method" \
    --instr_mode "$instr_mode" \
    --instr_fixed_id "$instr_fixed_id" \
    --epochs "$EPOCHS" \
    --do_train --do_eval \
    --fp16
}

# 1) 3 method đầu
for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_job "$model" "$method" "na" -1
  done
done

# 2) instruction: fixed + random
for model in "${MODELS[@]}"; do
  # fixed instruction (id=0)
  run_job "$model" "instruction" "fixed" 0

  # random instruction
  run_job "$model" "instruction" "random" -1
done

echo "Done. Results saved in results/metrics.csv and results/metrics.jsonl"
