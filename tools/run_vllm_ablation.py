import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = REPO_ROOT / "tools" / "inference_sd_vlm_vllm.py"
DEFAULT_MODES = ("base", "prefix", "spec", "full")


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM ablation experiments for FastDriveLM"
    )
    parser.add_argument(
        "--target-model",
        default="outputs/qwen3vl",
        help="Target model or adapter directory",
    )
    parser.add_argument(
        "--data",
        default="datasets/DriveLM_nuScenes/split/val",
        help="Dataset path created by load_from_disk",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qwen3vl",
        help="Directory for inference outputs and metrics",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
        help="Ablation modes to run",
    )
    parser.add_argument(
        "--run-label",
        default="smoke",
        help="Suffix used in output filenames, for example smoke or full",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Optional number of samples to run for each mode; omit manually for full runs by setting a large or custom value",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens per sample",
    )
    parser.add_argument(
        "--attn-backend",
        choices=["auto", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"],
        default="auto",
        help="Attention backend passed through to the main vLLM script",
    )
    parser.add_argument(
        "--spec-k",
        type=int,
        default=2,
        help="Number of speculative draft tokens for ngram-based modes",
    )
    parser.add_argument(
        "--prompt-lookup-min",
        type=int,
        default=2,
        help="Minimum ngram window for ngram-based modes",
    )
    parser.add_argument(
        "--prompt-lookup-max",
        type=int,
        default=4,
        help="Maximum ngram window for ngram-based modes",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to tools/inference_sd_vlm_vllm.py",
    )
    return parser.parse_args()


def _mode_args(mode: str, args: argparse.Namespace) -> list[str]:
    if mode == "base":
        return []
    if mode == "prefix":
        return ["--use-prefix-caching"]
    if mode == "spec":
        return [
            "--enable-speculative-decoding",
            "--speculative-method",
            "ngram",
            "--spec-k",
            str(args.spec_k),
            "--prompt-lookup-min",
            str(args.prompt_lookup_min),
            "--prompt-lookup-max",
            str(args.prompt_lookup_max),
        ]
    if mode == "full":
        return [
            "--enable-speculative-decoding",
            "--speculative-method",
            "ngram",
            "--spec-k",
            str(args.spec_k),
            "--prompt-lookup-min",
            str(args.prompt_lookup_min),
            "--prompt-lookup-max",
            str(args.prompt_lookup_max),
            "--use-prefix-caching",
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def _build_command(mode: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    output_path = output_dir / f"infer_results_sd_vlm_vllm_{mode}_{args.run_label}.json"
    metrics_path = output_dir / f"metrics_sd_vlm_vllm_{mode}_{args.run_label}.json"

    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--target-model",
        args.target_model,
        "--data",
        args.data,
        "--output",
        str(output_path),
        "--metrics",
        "--metrics-output",
        str(metrics_path),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.attn_backend != "auto":
        command.extend(["--attn-backend", args.attn_backend])
    command.extend(_mode_args(mode, args))
    command.extend(args.extra_args)
    return command


def main() -> None:
    args = parse_args()
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        command = _build_command(mode, args, output_dir)
        print(f"[run_vllm_ablation] running {mode}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
