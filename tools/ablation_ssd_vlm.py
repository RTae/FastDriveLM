"""
Ablation study driver for inference_ssd_vlm.py.

Sweeps a fixed set of attention-backend / prefix-caching configurations,
runs each via a subprocess call to inference_ssd_vlm.py, then prints a
side-by-side comparison table of the aggregate speed metrics.

Usage
-----
python tools/ablation_ssd_vlm.py \
    --target-model outputs/qwen3vl \
    --draft-model  outputs/qwen3vl_draft \
    --data         datasets/DriveLM_nuScenes/split/val \
    --output-dir   outputs/ablation_ssd_vlm \
    [--max-samples 50] \
    [--max-new-tokens 128] \
    [--spec-k 4] \
    [--num-gpus 2] \
    [--max-model-len 2048] \
    [--warmup-steps 2] \
    [--sparge-topk 0.3 0.5 0.7]

The four built-in configurations are:
  1. flash  | prefix-caching=True
  2. flash  | prefix-caching=False
  3. sparge | prefix-caching=False (topk per --sparge-topk values, default 0.5)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_SCRIPT = Path(__file__).resolve().parent / "inference_ssd_vlm.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_configs(sparge_topk_values: list[float]) -> list[dict]:
    """Return the list of ablation configuration dicts."""
    configs = [
        {
            "name": "flash_cache",
            "label": "flash  | cache=ON",
            "use_prefix_caching": True,
            "attn_backend": "flash",
            "sparge_topk": None,
        },
        {
            "name": "flash_no_cache",
            "label": "flash  | cache=OFF",
            "use_prefix_caching": False,
            "attn_backend": "flash",
            "sparge_topk": None,
        },
    ]
    for topk in sparge_topk_values:
        configs.append(
            {
                "name": f"sparge_topk{topk}",
                "label": f"sparge | topk={topk}",
                "use_prefix_caching": False,
                "attn_backend": "sparge",
                "sparge_topk": topk,
            }
        )
    return configs


def _build_subprocess_cmd(cfg: dict, args: argparse.Namespace, metrics_path: Path, output_path: Path) -> list[str]:
    cmd = [
        sys.executable, str(THIS_SCRIPT),
        "--target-model", args.target_model,
        "--draft-model",  args.draft_model,
        "--data",         args.data,
        "--output",       str(output_path),
        "--metrics",
        "--metrics-output", str(metrics_path),
        "--max-new-tokens",  str(args.max_new_tokens),
        "--spec-k",          str(args.spec_k),
        "--num-gpus",        str(args.num_gpus),
        "--max-model-len",   str(args.max_model_len),
        "--attn-backend",    cfg["attn_backend"],
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    if args.start_index:
        cmd += ["--start-index", str(args.start_index)]
    if cfg["use_prefix_caching"]:
        cmd.append("--use-prefix-caching")
    else:
        cmd.append("--no-prefix-caching")
    if cfg["sparge_topk"] is not None:
        cmd += ["--sparge-topk", str(cfg["sparge_topk"])]
    return cmd


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _build_metrics_summary(per_sample_metrics: list[dict]) -> dict:
    if not per_sample_metrics:
        return {
            "num_samples": 0,
            "total_output_tokens": 0,
            "total_wall_time_sec": 0.0,
            "avg_ttft_sec": 0.0,
            "avg_latency_sec": 0.0,
            "avg_prefill_throughput_tok_per_sec": 0.0,
            "avg_decode_throughput_tok_per_sec": 0.0,
            "end_to_end_throughput_tok_per_sec": 0.0,
            "avg_target_step_time_ms": 0.0,
            "avg_target_verify_time_ms": 0.0,
        }
    total_output_tokens = sum(item["output_tokens"] for item in per_sample_metrics)
    total_wall_time = sum(item["latency_sec"] for item in per_sample_metrics)
    return {
        "num_samples": len(per_sample_metrics),
        "total_output_tokens": total_output_tokens,
        "total_wall_time_sec": total_wall_time,
        "avg_ttft_sec": sum(item["ttft_sec"] for item in per_sample_metrics) / len(per_sample_metrics),
        "avg_latency_sec": sum(item["latency_sec"] for item in per_sample_metrics) / len(per_sample_metrics),
        "avg_prefill_throughput_tok_per_sec": sum(item["prefill_throughput_tok_per_sec"] for item in per_sample_metrics) / len(per_sample_metrics),
        "avg_decode_throughput_tok_per_sec": sum(item["decode_throughput_tok_per_sec"] for item in per_sample_metrics) / len(per_sample_metrics),
        "end_to_end_throughput_tok_per_sec": _safe_divide(total_output_tokens, total_wall_time),
        "avg_target_step_time_ms": sum(item["avg_target_step_time_ms"] for item in per_sample_metrics) / len(per_sample_metrics),
        "avg_target_verify_time_ms": sum(item["avg_target_verify_time_ms"] for item in per_sample_metrics) / len(per_sample_metrics),
    }

METRICS_COLS = [
    ("avg_ttft_sec",                        "TTFT (s)",      ".3f"),
    ("avg_latency_sec",                     "Latency (s)",   ".3f"),
    ("avg_prefill_throughput_tok_per_sec",  "Prefill tok/s", ".1f"),
    ("avg_decode_throughput_tok_per_sec",   "Decode tok/s",  ".1f"),
    ("end_to_end_throughput_tok_per_sec",   "E2E tok/s",     ".1f"),
    ("avg_target_step_time_ms",             "Step (ms)",     ".2f"),
    ("avg_target_verify_time_ms",           "Verify (ms)",   ".2f"),
    ("num_samples",                         "Samples",       "d"),
]


def _print_table(results: list[dict]) -> None:
    """Print a fixed-width comparison table to stdout."""
    label_w = max(len(r["label"]) for r in results)
    col_headers = [c[1] for c in METRICS_COLS]
    col_w = [max(len(h), 12) for h in col_headers]

    def row_str(label: str, values: list[str]) -> str:
        cells = [f"{label:<{label_w}}"]
        for v, w in zip(values, col_w):
            cells.append(f"{v:>{w}}")
        return "  ".join(cells)

    header = row_str("Configuration", col_headers)
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("Ablation Study Results")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        summary = r.get("summary", {})
        values = []
        for key, _, fmt in METRICS_COLS:
            val = summary.get(key, float("nan"))
            if fmt == "d":
                values.append(str(int(val)) if val == val else "N/A")
            else:
                try:
                    values.append(format(float(val), fmt))
                except (TypeError, ValueError):
                    values.append("N/A")
        print(row_str(r["label"], values))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study driver for inference_ssd_vlm.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- forwarded to inference_ssd_vlm.py ---
    parser.add_argument("--target-model", default="outputs/qwen3vl",
                        help="Target model or adapter directory")
    parser.add_argument("--draft-model", default="outputs/qwen3vl_draft",
                        help="Draft model or adapter directory")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val",
                        help="Dataset path (load_from_disk)")
    parser.add_argument("--output-dir", default="outputs/ablation_ssd_vlm",
                        help="Root directory for per-config output files")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--spec-k", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap on number of samples per run")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Leading samples excluded from aggregate metrics")
    # --- ablation-specific ---
    parser.add_argument("--sparge-topk", type=float, nargs="+", default=[0.5],
                        metavar="K",
                        help="One or more SpargeAttn sparsity ratios to sweep")
    parser.add_argument("--skip-on-error", action="store_true",
                        help="Continue to next config if a subprocess fails, "
                             "rather than aborting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(args.sparge_topk)
    results = []

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}", flush=True)
        print(f"[ablation] Running config: {cfg['label']}", flush=True)
        print(f"{'='*60}", flush=True)

        cfg_dir = output_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = cfg_dir / "metrics.json"
        output_path  = cfg_dir / "infer_results.json"

        cmd = _build_subprocess_cmd(cfg, args, metrics_path, output_path)
        print(f"[ablation] CMD: {' '.join(cmd)}", flush=True)

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            msg = f"[ablation] Config '{name}' failed with exit code {proc.returncode}"
            if args.skip_on_error:
                print(f"WARNING: {msg} — skipping.", flush=True)
                results.append({"label": cfg["label"], "summary": {}})
                continue
            else:
                print(f"ERROR: {msg}", flush=True)
                sys.exit(proc.returncode)

        summary = {}
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            summary = _build_metrics_summary(samples[args.warmup_steps:])

        results.append({"label": cfg["label"], "summary": summary})

    # Save combined results
    combined_path = output_dir / "ablation_summary.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ablation] Combined summary saved to {combined_path}", flush=True)

    _print_table(results)


if __name__ == "__main__":
    main()
