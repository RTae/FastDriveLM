import argparse
import ast
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _command_to_string(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run_command(command: list[str], log_path: Path, dry_run: bool = False) -> None:
    print(f"$ {_command_to_string(command)}", flush=True)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def _maybe_add(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_inference_command(
    args,
    model_path: str,
    output_path: Path,
    metrics_path: Path,
    base_model: str | None = None,
) -> list[str]:
    command = [
        args.python,
        "tools/inference_vllm.py",
        "--model-path",
        model_path,
        "--collate_fn",
        args.collate_fn,
        "--data",
        args.data,
        "--output",
        str(output_path),
        "--metrics",
        "--metrics-output",
        str(metrics_path),
        "--warmup-steps",
        str(args.warmup_steps),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--limit-mm-images",
        str(args.limit_mm_images),
        "--attn-implementation",
        args.attn_implementation,
    ]
    _maybe_add(command, "--base-model", base_model)
    _maybe_add(command, "--max-samples", args.max_samples)
    _maybe_add(command, "--max-new-tokens", args.max_new_tokens)
    return command


def run_inference_pair(args, artifacts: dict[str, Path]) -> dict[str, list[str]]:
    commands = {
        "baseline_inference": build_inference_command(
            args=args,
            model_path=args.baseline_model,
            output_path=artifacts["baseline_output"],
            metrics_path=artifacts["baseline_metrics"],
            base_model=args.baseline_base_model,
        ),
        "quantized_inference": build_inference_command(
            args=args,
            model_path=args.quantized_model,
            output_path=artifacts["quantized_output"],
            metrics_path=artifacts["quantized_metrics"],
            base_model=None,
        ),
    }

    if not args.reuse_existing or not (
        artifacts["baseline_output"].exists() and artifacts["baseline_metrics"].exists()
    ):
        _run_command(
            commands["baseline_inference"],
            artifacts["baseline_log"],
            dry_run=args.dry_run,
        )
    else:
        print("[compare] reusing baseline inference artifacts", flush=True)

    if not args.reuse_existing or not (
        artifacts["quantized_output"].exists() and artifacts["quantized_metrics"].exists()
    ):
        _run_command(
            commands["quantized_inference"],
            artifacts["quantized_log"],
            dry_run=args.dry_run,
        )
    else:
        print("[compare] reusing quantized inference artifacts", flush=True)

    return commands


def parse_evaluation_output(text: str) -> dict[str, Any]:
    def find_float(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    result: dict[str, Any] = {
        "accuracy": find_float(r"accuracy score:\s+([0-9.eE+-]+)"),
        "match": find_float(r"match score:\s+([0-9.eE+-]+)"),
        "final_score": find_float(r"final combined score:\s+([0-9.eE+-]+)"),
    }

    language_match = re.search(r"language score:\s+(\{.*\})", text)
    if language_match:
        try:
            result["language"] = ast.literal_eval(language_match.group(1))
        except (SyntaxError, ValueError):
            result["language"] = None
    else:
        result["language"] = None

    return result


def run_evaluation(args, src: Path, log_path: Path) -> tuple[dict[str, Any], list[str]]:
    command = [
        args.python,
        "tools/evaluation.py",
        "--src",
        str(src),
        "--tgt",
        args.refs,
    ]
    _run_command(command, log_path, dry_run=args.dry_run)

    if args.dry_run:
        return {}, command

    with log_path.open("r", encoding="utf-8") as f:
        return parse_evaluation_output(f.read()), command


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new - old


def build_comparison(
    baseline_metrics: dict[str, Any],
    quantized_metrics: dict[str, Any],
    baseline_eval: dict[str, Any],
    quantized_eval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "speed": {
            "baseline": baseline_metrics,
            "quantized": quantized_metrics,
            "speedup": {
                "decode_throughput": _safe_divide(
                    quantized_metrics.get("avg_decode_throughput_tok_per_sec"),
                    baseline_metrics.get("avg_decode_throughput_tok_per_sec"),
                ),
                "end_to_end_throughput": _safe_divide(
                    quantized_metrics.get("end_to_end_throughput_tok_per_sec"),
                    baseline_metrics.get("end_to_end_throughput_tok_per_sec"),
                ),
                "latency": _safe_divide(
                    baseline_metrics.get("avg_latency_sec"),
                    quantized_metrics.get("avg_latency_sec"),
                ),
            },
        },
        "accuracy": {
            "baseline": baseline_eval,
            "quantized": quantized_eval,
            "delta": {
                "final_score": _delta(
                    quantized_eval.get("final_score"),
                    baseline_eval.get("final_score"),
                ),
                "accuracy": _delta(
                    quantized_eval.get("accuracy"),
                    baseline_eval.get("accuracy"),
                ),
                "match": _delta(
                    quantized_eval.get("match"),
                    baseline_eval.get("match"),
                ),
            },
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_summary(comparison: dict[str, Any]) -> None:
    speed = comparison["speed"]
    accuracy = comparison["accuracy"]
    baseline_speed = speed["baseline"]
    quantized_speed = speed["quantized"]
    baseline_eval = accuracy["baseline"]
    quantized_eval = accuracy["quantized"]

    rows = [
        (
            "avg_latency_sec",
            baseline_speed.get("avg_latency_sec"),
            quantized_speed.get("avg_latency_sec"),
            speed["speedup"].get("latency"),
            "baseline/quantized",
        ),
        (
            "avg_decode_tps",
            baseline_speed.get("avg_decode_throughput_tok_per_sec"),
            quantized_speed.get("avg_decode_throughput_tok_per_sec"),
            speed["speedup"].get("decode_throughput"),
            "quantized/baseline",
        ),
        (
            "e2e_tps",
            baseline_speed.get("end_to_end_throughput_tok_per_sec"),
            quantized_speed.get("end_to_end_throughput_tok_per_sec"),
            speed["speedup"].get("end_to_end_throughput"),
            "quantized/baseline",
        ),
        (
            "final_score",
            baseline_eval.get("final_score"),
            quantized_eval.get("final_score"),
            accuracy["delta"].get("final_score"),
            "delta",
        ),
        (
            "accuracy",
            baseline_eval.get("accuracy"),
            quantized_eval.get("accuracy"),
            accuracy["delta"].get("accuracy"),
            "delta",
        ),
        (
            "match",
            baseline_eval.get("match"),
            quantized_eval.get("match"),
            accuracy["delta"].get("match"),
            "delta",
        ),
    ]

    print("\nComparison summary")
    print(f"{'metric':<22} {'baseline':>12} {'quantized':>12} {'change':>12}  note")
    print("-" * 78)
    for metric, baseline, quantized, change, note in rows:
        print(
            f"{metric:<22} {_fmt(baseline):>12} {_fmt(quantized):>12} {_fmt(change):>12}  {note}"
        )


def build_artifacts(output_dir: Path) -> dict[str, Path]:
    return {
        "baseline_output": output_dir / "baseline_infer_results.json",
        "baseline_metrics": output_dir / "baseline_metrics.json",
        "baseline_log": output_dir / "baseline_inference.log",
        "baseline_eval_log": output_dir / "baseline_eval.log",
        "quantized_output": output_dir / "quantized_infer_results.json",
        "quantized_metrics": output_dir / "quantized_metrics.json",
        "quantized_log": output_dir / "quantized_inference.log",
        "quantized_eval_log": output_dir / "quantized_eval.log",
        "comparison": output_dir / "comparison.json",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference for unquantized and AWQ INT4 Qwen3-VL models, then compare speed and DriveLM scores."
    )
    parser.add_argument("--baseline-model", default="outputs/qwen3vl")
    parser.add_argument(
        "--baseline-base-model",
        default=None,
        help="Optional local base model for baseline adapter inference.",
    )
    parser.add_argument("--quantized-model", default="outputs/qwen3vl_awq_int4")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val")
    parser.add_argument("--refs", default="datasets/DriveLM_nuScenes/refs/val_cot.json")
    parser.add_argument(
        "--collate_fn",
        default="drivelm_nus_qwen3vl_collate_fn_val",
    )
    parser.add_argument("--output-dir", default="outputs/qwen3vl/quant_compare")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--limit-mm-images", type=int, default=6)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip inference if output and metrics JSON files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts(output_dir)

    quantized_path = _resolve_repo_path(args.quantized_model)
    if not quantized_path.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"Quantized model not found: {quantized_path}. "
            "Run tools/quantize_qwen3vl_awq.py first."
        )

    commands = run_inference_pair(args, artifacts)

    baseline_eval, baseline_eval_cmd = run_evaluation(
        args,
        artifacts["baseline_output"],
        artifacts["baseline_eval_log"],
    )
    quantized_eval, quantized_eval_cmd = run_evaluation(
        args,
        artifacts["quantized_output"],
        artifacts["quantized_eval_log"],
    )
    commands["baseline_evaluation"] = baseline_eval_cmd
    commands["quantized_evaluation"] = quantized_eval_cmd

    if args.dry_run:
        return

    baseline_metrics = load_metrics(artifacts["baseline_metrics"])
    quantized_metrics = load_metrics(artifacts["quantized_metrics"])
    comparison = build_comparison(
        baseline_metrics,
        quantized_metrics,
        baseline_eval,
        quantized_eval,
    )
    comparison["artifacts"] = {key: str(path) for key, path in artifacts.items()}
    comparison["commands"] = {
        name: _command_to_string(command)
        for name, command in commands.items()
    }

    with artifacts["comparison"].open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print_summary(comparison)
    print(f"\n[compare] wrote {artifacts['comparison']}", flush=True)


if __name__ == "__main__":
    main()
