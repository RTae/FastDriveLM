#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_MODEL="outputs/qwen3vl"
DRAFT_MODEL="outputs/qwen3vl_draft"
DATASET="datasets/DriveLM_nuScenes/split/val"
OUTPUT_DIR="outputs/qwen3vl"
RUN_LABEL="bench50"
MAX_SAMPLES="50"
MAX_NEW_TOKENS="64"
WARMUP_SAMPLES="3"
RUN_BASELINE=1
RUN_VLLM=1
RUN_CUSTOM=1

usage() {
    cat <<'EOF'
Usage: scripts/run_50_sample_benchmark.sh [options]

Run 50-sample benchmark jobs and write warmup-filtered summaries.

Options:
  --target-model PATH     Target model or adapter directory (default: outputs/qwen3vl)
  --draft-model PATH      Draft model or adapter directory (default: outputs/qwen3vl_draft)
  --data PATH             Dataset path (default: datasets/DriveLM_nuScenes/split/val)
  --output-dir PATH       Output directory (default: outputs/qwen3vl)
  --run-label LABEL       Output suffix label (default: bench50)
  --max-samples N         Number of samples to run (default: 50)
  --max-new-tokens N      Max new tokens per sample (default: 64)
  --warmup N              Number of leading samples to drop in summaries (default: 3)
  --skip-baseline         Skip plain Hugging Face baseline run
  --skip-vllm             Skip vLLM run
  --skip-custom           Skip custom speculative VLM run
  -h, --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL="$2"
            shift 2
            ;;
        --data)
            DATASET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run-label)
            RUN_LABEL="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_SAMPLES="$2"
            shift 2
            ;;
        --skip-baseline)
            RUN_BASELINE=0
            shift
            ;;
        --skip-vllm)
            RUN_VLLM=0
            shift
            ;;
        --skip-custom)
            RUN_CUSTOM=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cd "$ROOT_DIR"

export PATH="$ROOT_DIR/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$ROOT_DIR/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR"

summarize() {
    local metrics_path="$1"
    local filtered_path="$2"
    python scripts/summarize_metrics.py "$metrics_path" --warmup "$WARMUP_SAMPLES" --output "$filtered_path"
}

if [[ "$RUN_BASELINE" == "1" ]]; then
    python tools/inference.py \
        --model-path "$TARGET_MODEL" \
        --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
        --data "$DATASET" \
        --output "$OUTPUT_DIR/infer_results_baseline_${RUN_LABEL}.json" \
        --metrics \
        --metrics-output "$OUTPUT_DIR/metrics_baseline_${RUN_LABEL}.json" \
        --max-samples "$MAX_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS"

    summarize \
        "$OUTPUT_DIR/metrics_baseline_${RUN_LABEL}.json" \
        "$OUTPUT_DIR/metrics_baseline_${RUN_LABEL}.warmup${WARMUP_SAMPLES}.json"
fi

if [[ "$RUN_VLLM" == "1" ]]; then
    python tools/inference_sd_vlm_vllm.py \
        --target-model "$TARGET_MODEL" \
        --data "$DATASET" \
        --output "$OUTPUT_DIR/infer_results_sd_vlm_vllm_base_${RUN_LABEL}.json" \
        --metrics \
        --metrics-output "$OUTPUT_DIR/metrics_sd_vlm_vllm_base_${RUN_LABEL}.json" \
        --max-samples "$MAX_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS"

    summarize \
        "$OUTPUT_DIR/metrics_sd_vlm_vllm_base_${RUN_LABEL}.json" \
        "$OUTPUT_DIR/metrics_sd_vlm_vllm_base_${RUN_LABEL}.warmup${WARMUP_SAMPLES}.json"
fi

if [[ "$RUN_CUSTOM" == "1" ]]; then
    python tools/inference_custom_sd_vlm.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --data "$DATASET" \
        --output "$OUTPUT_DIR/infer_results_custom_sd_vlm_${RUN_LABEL}.json" \
        --metrics \
        --metrics-output "$OUTPUT_DIR/infer_results_custom_sd_vlm_${RUN_LABEL}.metrics.json" \
        --max-samples "$MAX_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS"

    summarize \
        "$OUTPUT_DIR/infer_results_custom_sd_vlm_${RUN_LABEL}.metrics.json" \
        "$OUTPUT_DIR/infer_results_custom_sd_vlm_${RUN_LABEL}.warmup${WARMUP_SAMPLES}.metrics.json"
fi

echo
echo "[run_50_sample_benchmark] warmup-filtered summary"
python - "$OUTPUT_DIR" "$RUN_LABEL" "$WARMUP_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
run_label = sys.argv[2]
warmup = sys.argv[3]

paths = [
    output_dir / f"metrics_baseline_{run_label}.warmup{warmup}.json",
    output_dir / f"metrics_sd_vlm_vllm_base_{run_label}.warmup{warmup}.json",
    output_dir / f"infer_results_custom_sd_vlm_{run_label}.warmup{warmup}.metrics.json",
]

rows = []
for path in paths:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows.append((path.name, payload["summary"]))

if not rows:
    raise SystemExit("No warmup-filtered metrics found.")

print(f"{'run':<54} {'samples':>8} {'avg_ttft':>10} {'avg_lat':>10} {'e2e_tok/s':>12} {'decode_tok/s':>14} {'step_ms':>10} {'verify_ms':>10}")
print("-" * 136)
for name, summary in rows:
    print(
        f"{name:<54} "
        f"{summary.get('num_samples', 0):>8} "
        f"{summary.get('avg_ttft_sec', 0.0):>10.2f} "
        f"{summary.get('avg_latency_sec', 0.0):>10.2f} "
        f"{summary.get('end_to_end_throughput_tok_per_sec', 0.0):>12.2f} "
        f"{summary.get('avg_decode_throughput_tok_per_sec', 0.0):>14.2f} "
        f"{summary.get('avg_target_step_time_ms', 0.0):>10.2f} "
        f"{summary.get('avg_target_verify_time_ms', 0.0):>10.2f}"
    )
PY