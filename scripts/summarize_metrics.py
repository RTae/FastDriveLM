import argparse
import json
from pathlib import Path


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _decode_throughput_for_sample(sample: dict) -> float:
    runner_decode = sample.get("runner_decode_throughput_tok_per_sec")
    if runner_decode is not None:
        return float(runner_decode)
    return float(sample.get("decode_throughput_tok_per_sec", 0.0))


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
        "avg_decode_throughput_tok_per_sec": sum(_decode_throughput_for_sample(item) for item in per_sample_metrics) / len(per_sample_metrics),
        "end_to_end_throughput_tok_per_sec": _safe_divide(total_output_tokens, total_wall_time),
        "avg_target_step_time_ms": sum(item["avg_target_step_time_ms"] for item in per_sample_metrics) / len(per_sample_metrics),
        "avg_target_verify_time_ms": sum(item["avg_target_verify_time_ms"] for item in per_sample_metrics) / len(per_sample_metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute metrics summary after dropping warmup samples.",
    )
    parser.add_argument("metrics_path", help="Metrics JSON with top-level summary and samples fields")
    parser.add_argument("--warmup", type=int, default=3, help="Number of leading samples to drop")
    parser.add_argument("--output", default=None, help="Optional path to write filtered metrics JSON")
    parser.add_argument("--print-json", action="store_true", help="Print filtered JSON instead of a table row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_path)
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    samples = payload.get("samples", [])
    filtered_samples = samples[args.warmup:]
    summary = _build_metrics_summary(filtered_samples)
    filtered_payload = {
        "source": str(metrics_path),
        "warmup_dropped": args.warmup,
        "original_num_samples": len(samples),
        "summary": summary,
        "samples": filtered_samples,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(filtered_payload, handle, indent=2, ensure_ascii=False)

    if args.print_json:
        print(json.dumps(filtered_payload, indent=2, ensure_ascii=False))
        return

    print(
        f"{metrics_path.name:<48} "
        f"drop={args.warmup:<3d} "
        f"samples={summary['num_samples']:<3d} "
        f"avg_ttft={summary['avg_ttft_sec']:.2f}s "
        f"avg_lat={summary['avg_latency_sec']:.2f}s "
        f"e2e={summary['end_to_end_throughput_tok_per_sec']:.2f} tok/s "
        f"decode={summary['avg_decode_throughput_tok_per_sec']:.2f} tok/s "
        f"step={summary['avg_target_step_time_ms']:.2f} ms "
        f"verify={summary['avg_target_verify_time_ms']:.2f} ms"
    )


if __name__ == "__main__":
    main()