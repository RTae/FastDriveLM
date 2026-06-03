import argparse
import json
from pathlib import Path


def _load_summary(path: str) -> dict:
    metrics_path = Path(path)
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError(f"Missing 'summary' block in {metrics_path}")
    return data["summary"]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _get(summary: dict, key: str) -> float:
    if key not in summary:
        raise KeyError(f"Missing metric {key!r} in summary")
    return float(summary[key])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate inference metrics."
    )
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON")
    parser.add_argument("--candidate", required=True, help="Candidate metrics JSON")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Required avg_latency speedup. 1.0 means candidate must be faster.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = _load_summary(args.baseline)
    candidate = _load_summary(args.candidate)

    baseline_latency = _get(baseline, "avg_latency_sec")
    candidate_latency = _get(candidate, "avg_latency_sec")
    baseline_e2e_tps = _get(baseline, "end_to_end_throughput_tok_per_sec")
    candidate_e2e_tps = _get(candidate, "end_to_end_throughput_tok_per_sec")
    baseline_decode_tps = _get(baseline, "avg_decode_throughput_tok_per_sec")
    candidate_decode_tps = _get(candidate, "avg_decode_throughput_tok_per_sec")

    latency_speedup = _safe_divide(baseline_latency, candidate_latency)
    e2e_speedup = _safe_divide(candidate_e2e_tps, baseline_e2e_tps)
    decode_speedup = _safe_divide(candidate_decode_tps, baseline_decode_tps)

    print(f"{'metric':<34} {'baseline':>14} {'candidate':>14} {'speedup':>12}")
    print("-" * 78)
    print(
        f"{'avg_latency_sec':<34} {baseline_latency:>14.4f} "
        f"{candidate_latency:>14.4f} {latency_speedup:>12.4f}x"
    )
    print(
        f"{'end_to_end_throughput_tok_per_sec':<34} {baseline_e2e_tps:>14.4f} "
        f"{candidate_e2e_tps:>14.4f} {e2e_speedup:>12.4f}x"
    )
    print(
        f"{'avg_decode_throughput_tok_per_sec':<34} {baseline_decode_tps:>14.4f} "
        f"{candidate_decode_tps:>14.4f} {decode_speedup:>12.4f}x"
    )

    if "spec_decode_mean_acceptance_length" in candidate:
        print(
            "spec_decode_mean_acceptance_length: "
            f"{float(candidate['spec_decode_mean_acceptance_length']):.4f}"
        )

    if latency_speedup < args.min_speedup:
        raise SystemExit(
            "Speculative decoding did not pass the speedup gate: "
            f"latency speedup {latency_speedup:.4f}x < {args.min_speedup:.4f}x"
        )

    print(
        "PASS: candidate is faster by avg_latency_sec "
        f"({latency_speedup:.4f}x >= {args.min_speedup:.4f}x)"
    )


if __name__ == "__main__":
    main()
