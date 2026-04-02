#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
  "Fastweb/FastwebMIIA-7B"
  "Qwen/Qwen3.5-9B"
  "sapienzanlp/Minerva-7B-instruct-v1.0"
  "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
  "utter-project/EuroLLM-9B-Instruct-2512"
  "Qwen/Qwen3-8B"
  "swiss-ai/Apertus-8B-Instruct-2509"
)

for model in "${MODELS[@]}"; do
  echo "Running model: ${model}"
  python "${SCRIPT_DIR}/predict_mmlu.py" --config "configs/json_v1.yaml" --model "${model}"
done

python "${SCRIPT_DIR}/combine_predictions.py" \
  --input-dir "output/mmlu_responses/by_model" \
  --output-file "output/mmlu_responses/test.parquet"
