#!/usr/bin/env bash
set -euo pipefail

# Run from the extracted package root, or set PACKAGE_ROOT explicitly.
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face model directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
BENCHMARK="${BENCHMARK:-gqa}"
SAMPLES="${SAMPLES:-8}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
TASK_NAME="${TASK_NAME:-${BENCHMARK}_package_smoke}"

case "${BENCHMARK}" in
  gqa|vqav2) ;;
  *)
    echo "BENCHMARK must be gqa or vqav2 (got: ${BENCHMARK})" >&2
    exit 2
    ;;
esac

cd "${PACKAGE_ROOT}"
export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" src/eval.py \
  --config-name=eval.yaml \
  "experiment=eval/${BENCHMARK}/default" \
  "task_name=${TASK_NAME}" \
  "eval.${BENCHMARK}.max_samples=${SAMPLES}" \
  "eval.${BENCHMARK}.sample_seed=${SAMPLE_SEED}" \
  "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}" \
  "model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH}" \
  "model.model_args.device_map=${DEVICE_MAP}"
