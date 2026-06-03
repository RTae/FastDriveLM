#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_MODEL="outputs/qwen3vl"
DATASET="datasets/DriveLM_nuScenes/split/val"
OUTPUT_DIR="outputs/qwen3vl"
RUN_LABEL="smoke"
MAX_SAMPLES="10"
MAX_NEW_TOKENS="64"
ATTN_BACKEND="FLASH_ATTN"
MODES=(base prefix spec full)
RUN_BASELINE=1
FULL_TEST=0

usage() {
    cat <<'EOF'
Usage: scripts/run_vllm_report.sh [options]

Run baseline inference plus vLLM ablation modes, then print a summary table.

Options:
  --target-model PATH       Target model or adapter directory (default: outputs/qwen3vl)
  --data PATH               Dataset path (default: datasets/DriveLM_nuScenes/split/val)
  --output-dir PATH         Directory for outputs and metrics (default: outputs/qwen3vl)
  --run-label LABEL         Output suffix such as smoke or full (default: smoke)
  --max-samples N           Number of samples for each run (default: 10)
  --max-new-tokens N        Max new tokens per sample (default: 64)
  --attn-backend NAME       vLLM attention backend (default: FLASH_ATTN)
  --modes "a b c"           Space-separated ablation modes (default: "base prefix spec full")
  --full-test               Run the full validation split and default run-label to full
  --skip-baseline           Skip the Hugging Face baseline run
  -h, --help                Show this help message
EOF
}

run_vllm_mode() {
    local mode="$1"
    local output_path="$OUTPUT_DIR/infer_results_sd_vlm_vllm_${mode}_${RUN_LABEL}.json"
    local metrics_path="$OUTPUT_DIR/metrics_sd_vlm_vllm_${mode}_${RUN_LABEL}.json"

    local cmd=(
        python tools/inference_sd_vlm_vllm.py
        --target-model "$TARGET_MODEL"
        --data "$DATASET"
        --output "$output_path"
        --metrics
        --metrics-output "$metrics_path"
        --max-new-tokens "$MAX_NEW_TOKENS"
    )

    if [[ "$ATTN_BACKEND" != "auto" ]]; then
        cmd+=(--attn-backend "$ATTN_BACKEND")
    fi
    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd+=(--max-samples "$MAX_SAMPLES")
    fi

    case "$mode" in
        base)
            ;;
        prefix)
            cmd+=(--use-prefix-caching)
            ;;
        spec)
            cmd+=(
                --enable-speculative-decoding
                --speculative-method ngram
                --spec-k 2
                --prompt-lookup-min 2
                --prompt-lookup-max 4
            )
            ;;
        full)
            cmd+=(
                --enable-speculative-decoding
                --speculative-method ngram
                --spec-k 2
                --prompt-lookup-min 2
                --prompt-lookup-max 4
                --use-prefix-caching
            )
            ;;
        *)
            echo "Unsupported mode: $mode" >&2
            exit 1
            ;;
    esac

    echo "[run_vllm_report] running $mode: ${cmd[*]}"
    "${cmd[@]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-model)
            TARGET_MODEL="$2"
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
        --attn-backend)
            ATTN_BACKEND="$2"
            shift 2
            ;;
        --modes)
            IFS=' ' read -r -a MODES <<< "$2"
            shift 2
            ;;
        --full-test)
            FULL_TEST=1
            shift
            ;;
        --skip-baseline)
            RUN_BASELINE=0
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

if [[ "$FULL_TEST" == "1" ]]; then
    if [[ "$RUN_LABEL" == "smoke" ]]; then
        RUN_LABEL="full"
    fi
    MAX_SAMPLES=""
    if [[ "$MAX_NEW_TOKENS" == "64" ]]; then
        MAX_NEW_TOKENS="128"
    fi
fi

export PATH="$ROOT_DIR/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$ROOT_DIR/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTPUT_DIR"

if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "[run_vllm_report] running baseline"
    if [[ -n "$MAX_SAMPLES" ]]; then
        python tools/inference.py \
            --model-path "$TARGET_MODEL" \
            --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
            --data "$DATASET" \
            --output "$OUTPUT_DIR/infer_results_baseline_${RUN_LABEL}.json" \
            --metrics \
            --metrics-output "$OUTPUT_DIR/metrics_baseline_${RUN_LABEL}.json" \
            --max-samples "$MAX_SAMPLES" \
            --max-new-tokens "$MAX_NEW_TOKENS"
    else
        python tools/inference.py \
            --model-path "$TARGET_MODEL" \
            --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
            --data "$DATASET" \
            --output "$OUTPUT_DIR/infer_results_baseline_${RUN_LABEL}.json" \
            --metrics \
            --metrics-output "$OUTPUT_DIR/metrics_baseline_${RUN_LABEL}.json" \
            --max-new-tokens "$MAX_NEW_TOKENS"
    fi
fi

echo "[run_vllm_report] running vLLM ablation modes: ${MODES[*]}"
for mode in "${MODES[@]}"; do
    run_vllm_mode "$mode"
done

echo
echo "[run_vllm_report] summary"
python - "$OUTPUT_DIR" "$RUN_LABEL" "$RUN_BASELINE" "${MODES[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
run_label = sys.argv[2]
run_baseline = sys.argv[3] == "1"
modes = sys.argv[4:]

rows = []

def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["summary"]

if run_baseline:
    baseline_path = output_dir / f"metrics_baseline_{run_label}.json"
    if baseline_path.exists():
        rows.append(("baseline", load_summary(baseline_path)))

for mode in modes:
    metrics_path = output_dir / f"metrics_sd_vlm_vllm_{mode}_{run_label}.json"
    if metrics_path.exists():
        rows.append((mode, load_summary(metrics_path)))

if not rows:
    raise SystemExit("No metrics files found to summarize.")

print(f"{'run':<10} {'samples':>8} {'tokens':>10} {'wall_s':>10} {'avg_lat_s':>10} {'e2e_tok/s':>12} {'avg_dec_tok/s':>14}")
print("-" * 82)
for name, summary in rows:
    print(
        f"{name:<10} "
        f"{summary.get('num_samples', 0):>8} "
        f"{summary.get('total_output_tokens', 0):>10} "
        f"{summary.get('total_wall_time_sec', 0.0):>10.2f} "
        f"{summary.get('avg_latency_sec', 0.0):>10.2f} "
        f"{summary.get('end_to_end_throughput_tok_per_sec', 0.0):>12.2f} "
        f"{summary.get('avg_decode_throughput_tok_per_sec', 0.0):>14.2f}"
    )
PY