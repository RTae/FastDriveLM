import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


METHODS_REQUIRING_MODEL = {
    "draft_model": "--draft-model",
    "eagle": "--eagle-model",
    "eagle3": "--eagle3-model",
    "medusa": "--medusa-model",
}


def _run(cmd: list[str]) -> None:
    print(f"[compare_speculative_decoding] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["summary"]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _method_cli_args(args, method: str) -> list[str] | None:
    if method == "baseline":
        return ["--speculative-method", "none"]

    cli = ["--speculative-method", method, "--spec-k", str(args.spec_k)]

    if method in {"ngram", "ngram_gpu"}:
        cli.extend(
            [
                "--prompt-lookup-min",
                str(args.prompt_lookup_min),
                "--prompt-lookup-max",
                str(args.prompt_lookup_max),
            ]
        )
    elif method == "suffix":
        cli.extend(
            [
                "--suffix-max-tree-depth",
                str(args.suffix_max_tree_depth),
                "--suffix-max-cached-requests",
                str(args.suffix_max_cached_requests),
                "--suffix-max-spec-factor",
                str(args.suffix_max_spec_factor),
                "--suffix-min-token-prob",
                str(args.suffix_min_token_prob),
            ]
        )
    elif method in METHODS_REQUIRING_MODEL:
        option = METHODS_REQUIRING_MODEL[method]
        value = getattr(args, option[2:].replace("-", "_"))
        if not value:
            if args.skip_missing_model_methods:
                print(
                    f"[compare_speculative_decoding] skipping {method}: {option} was not provided",
                    flush=True,
                )
                return None
            raise ValueError(f"{option} is required for method {method}")
        cli.extend([option, value])

    if args.draft_tensor_parallel_size is not None:
        cli.extend(["--draft-tensor-parallel-size", str(args.draft_tensor_parallel_size)])
    if args.draft_max_model_len is not None:
        cli.extend(["--draft-max-model-len", str(args.draft_max_model_len)])
    if args.parallel_drafting:
        cli.append("--parallel-drafting")
    if args.disable_padded_drafter_batch:
        cli.append("--disable-padded-drafter-batch")

    return cli


def _build_run_command(args, method: str, output_path: Path, metrics_path: Path) -> list[str] | None:
    method_args = _method_cli_args(args, method)
    if method_args is None:
        return None

    cmd = [
        sys.executable,
        "tools/inference_sd_vlm_vllm.py",
        "--target-model",
        args.target_model,
        "--data",
        args.data,
        "--output",
        str(output_path),
        "--metrics-output",
        str(metrics_path),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-steps",
        str(args.warmup_steps),
        "--spec-k",
        str(args.spec_k),
    ]

    if args.metrics:
        cmd.append("--metrics")
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.start_index:
        cmd.extend(["--start-index", str(args.start_index)])
    if args.num_gpus != 1:
        cmd.extend(["--num-gpus", str(args.num_gpus)])
    if args.max_model_len != 16384:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.dtype != "auto":
        cmd.extend(["--dtype", args.dtype])
    if args.attn_backend != "auto":
        cmd.extend(["--attn-backend", args.attn_backend])
    if args.use_prefix_caching is True:
        cmd.append("--use-prefix-caching")
    elif args.use_prefix_caching is False:
        cmd.append("--no-use-prefix-caching")
    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    if args.enforce_eager is True:
        cmd.append("--enforce-eager")
    elif args.enforce_eager is False:
        cmd.append("--no-enforce-eager")

    cmd.extend(method_args)
    return cmd


def _print_summary(rows: list[tuple[str, dict]]) -> None:
    baseline = next((summary for name, summary in rows if name == "baseline"), None)
    baseline_latency = float(baseline.get("avg_latency_sec", 0.0)) if baseline else 0.0

    print()
    print(
        f"{'method':<12} {'samples':>8} {'avg_lat_s':>10} {'lat_speedup':>12} "
        f"{'e2e_tok/s':>12} {'dec_tok/s':>12} {'accept_len':>11}"
    )
    print("-" * 83)
    for name, summary in rows:
        avg_latency = float(summary.get("avg_latency_sec", 0.0))
        latency_speedup = _safe_divide(baseline_latency, avg_latency) if baseline_latency else 0.0
        print(
            f"{name:<12} "
            f"{int(summary.get('num_samples', 0)):>8} "
            f"{avg_latency:>10.4f} "
            f"{latency_speedup:>12.4f} "
            f"{float(summary.get('end_to_end_throughput_tok_per_sec', 0.0)):>12.4f} "
            f"{float(summary.get('avg_decode_throughput_tok_per_sec', 0.0)):>12.4f} "
            f"{float(summary.get('spec_decode_mean_acceptance_length', 0.0)):>11.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and compare vLLM speculative decoding methods."
    )
    parser.add_argument("--target-model", default="outputs/qwen3vl")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val")
    parser.add_argument("--output-dir", default="outputs/qwen3vl/spec_compare")
    parser.add_argument("--run-label", default="smoke")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "ngram", "ngram_gpu", "suffix", "eagle", "eagle3", "medusa"],
        help="Methods to run: baseline, ngram, ngram_gpu, suffix, draft_model, eagle, eagle3, medusa",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Required subset size for each method. Defaults to 10 samples.",
    )
    parser.add_argument(
        "--allow-full-dataset",
        action="store_true",
        help="Allow --max-samples 0 to run the full dataset. Off by default.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--spec-k", type=int, default=4)
    parser.add_argument("--prompt-lookup-min", type=int, default=2)
    parser.add_argument("--prompt-lookup-max", type=int, default=4)
    parser.add_argument("--suffix-max-tree-depth", type=int, default=24)
    parser.add_argument("--suffix-max-cached-requests", type=int, default=10000)
    parser.add_argument("--suffix-max-spec-factor", type=float, default=1.0)
    parser.add_argument("--suffix-min-token-prob", type=float, default=0.1)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--eagle-model", default=None)
    parser.add_argument("--eagle3-model", default=None)
    parser.add_argument("--medusa-model", default=None)
    parser.add_argument("--skip-missing-model-methods", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attn-backend", default="auto")
    parser.add_argument("--use-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--draft-max-model-len", type=int, default=None)
    parser.add_argument("--parallel-drafting", action="store_true")
    parser.add_argument("--disable-padded-drafter-batch", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue comparing other methods if one method fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples is None or args.max_samples <= 0:
        if not args.allow_full_dataset:
            raise SystemExit(
                "Speculative decoding comparison is subset-only by default. "
                "Pass --max-samples N with N > 0, or pass --allow-full-dataset "
                "with --max-samples 0 if you intentionally want the full split."
            )
        args.max_samples = None

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, dict]] = []
    failed: dict[str, str] = {}
    for method in args.methods:
        output_path = output_dir / f"infer_results_{method}_{args.run_label}.json"
        metrics_path = output_dir / f"metrics_{method}_{args.run_label}.json"
        cmd = _build_run_command(args, method, output_path, metrics_path)
        if cmd is None:
            continue

        try:
            _run(cmd)
        except subprocess.CalledProcessError as exc:
            failed[method] = str(exc)
            if method == "baseline":
                raise SystemExit(
                    "Baseline vLLM run failed, so speculative methods cannot be "
                    "compared reliably. Check the baseline error above first."
                ) from exc
            if not args.continue_on_error:
                raise
            print(f"[compare_speculative_decoding] {method} failed: {exc}", flush=True)
            continue

        rows.append((method, _load_summary(metrics_path)))

    if rows:
        _print_summary(rows)

    if failed:
        print()
        print("[compare_speculative_decoding] failed methods:")
        for method, error in failed.items():
            print(f"- {method}: {error}")

    if not rows:
        raise SystemExit("No methods completed successfully.")


if __name__ == "__main__":
    main()
