import argparse
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SSD_ROOT = REPO_ROOT / "ssd"


def _ensure_env_defaults() -> None:
    os.environ.setdefault("SSD_HF_CACHE", str(REPO_ROOT / "base_models"))
    os.environ.setdefault("SSD_DATASET_DIR", str(REPO_ROOT / "datasets"))
    venv_bin = REPO_ROOT / ".venv" / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"


_ensure_env_defaults()
if str(SSD_ROOT) not in sys.path:
    sys.path.insert(0, str(SSD_ROOT))

from ssd.ssd import SamplingParams 
from ssd.ssd.engine.llm_engine import LLMEngine


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _resolve_image_path(path: str) -> str:
    image_path = Path(path)
    if image_path.is_absolute():
        return str(image_path)
    candidate = (REPO_ROOT / image_path).resolve()
    if candidate.exists():
        return str(candidate)
    return str(image_path)


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resolve_model_and_lora(model_or_adapter: str) -> tuple[str, str | None]:
    path = Path(model_or_adapter)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    adapter_config_path = path / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"LoRA adapter is missing base_model_name_or_path: {adapter_config_path}")
        base_path = Path(base_model)
        if not base_path.is_absolute():
            base_path = (REPO_ROOT / base_path).resolve()
        return str(base_path), str(path)

    return str(path), None


def _sample_to_prompt(sample: dict) -> tuple[str, list[Image.Image], str, str]:
    question = sample["conversations"][0]["value"]
    image_paths = [_resolve_image_path(path) for path in sample["image_paths"]]
    images = [_load_image(path) for path in image_paths]
    prompt = f"{IMAGE_PLACEHOLDER * len(images)}{question}"
    return prompt, images, question, sample["id"]


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM inference with VLM SSD draft_async path")
    parser.add_argument("--target-model", default="outputs/qwen3vl", help="Target model or adapter directory")
    parser.add_argument("--draft-model", default="outputs/qwen3vl_draft", help="Draft model or adapter directory")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val", help="Dataset path created by load_from_disk")
    parser.add_argument("--output", default="outputs/qwen3vl/infer_results_ssd.json", help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per sample")
    parser.add_argument("--spec-k", type=int, default=4, help="Lookahead for speculative decoding")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs to expose to the VLM SSD path")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--start-index", type=int, default=0, help="Start sample index")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional number of samples to run")
    parser.add_argument("--metrics", action="store_true", help="Print per-sample and aggregate speed metrics")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON path for aggregate and per-sample metrics")
    return parser.parse_args()


def _build_engine(args):
    target_model, target_lora = _resolve_model_and_lora(args.target_model)
    draft_model, draft_lora = _resolve_model_and_lora(args.draft_model)
    return LLMEngine(
        target_model,
        draft=draft_model,
        lora_path=target_lora,
        draft_lora_path=draft_lora,
        is_vlm=True,
        speculate=True,
        draft_async=True,
        num_gpus=args.num_gpus,
        speculate_k=args.spec_k,
        max_model_len=args.max_model_len,
    )


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


def main():
    args = parse_args()

    dataset = load_from_disk(args.data)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    engine = _build_engine(args)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    per_sample_metrics = []
    for sample in tqdm(dataset, desc="SSD VLM Inference", dynamic_ncols=True):
        prompt, images, question, sample_id = _sample_to_prompt(sample)

        first_token_time = [None]

        def on_tokens(seq_id, new_ids):
            if new_ids and first_token_time[0] is None:
                first_token_time[0] = time.perf_counter()

        t0 = time.perf_counter()
        outputs, metrics = engine.generate(
            [prompt],
            [SamplingParams(temperature=0.0, max_new_tokens=args.max_new_tokens)],
            use_tqdm=False,
            stream_callback=on_tokens,
            images=[images],
        )
        t1 = time.perf_counter()

        output_tokens = len(outputs[0]["token_ids"])
        ttft_sec = (first_token_time[0] - t0) if first_token_time[0] is not None else 0.0
        decode_time_sec = t1 - (first_token_time[0] or t0)
        latency_sec = t1 - t0
        sample_metrics = {
            "id": sample_id,
            "output_tokens": output_tokens,
            "ttft_sec": ttft_sec,
            "latency_sec": latency_sec,
            "decode_time_sec": decode_time_sec,
            "prefill_tokens": metrics.get("prefill_total_tokens", 0),
            "prefill_time_sec": metrics.get("prefill_total_time", 0.0),
            "prefill_throughput_tok_per_sec": _safe_divide(metrics.get("prefill_total_tokens", 0), metrics.get("prefill_total_time", 0.0)),
            "decode_throughput_tok_per_sec": _safe_divide(output_tokens, decode_time_sec),
            "end_to_end_throughput_tok_per_sec": _safe_divide(output_tokens, latency_sec),
            "runner_decode_throughput_tok_per_sec": _safe_divide(metrics.get("decode_total_tokens", 0), metrics.get("decode_total_time", 0.0)),
            "avg_target_step_time_ms": _safe_divide(sum(metrics.get("target_step_times", [])) * 1000.0, len(metrics.get("target_step_times", []))),
            "avg_target_verify_time_ms": _safe_divide(sum(metrics.get("target_verify_times", [])) * 1000.0, len(metrics.get("target_verify_times", []))),
        }
        per_sample_metrics.append(sample_metrics)

        results.append(
            {
                "id": sample_id,
                "question": question,
                "answer": outputs[0]["text"],
            }
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if args.metrics:
            print(
                f"[metrics] id={sample_id} tok={output_tokens} latency={latency_sec:.2f}s "
                f"ttft={ttft_sec:.2f}s prefill_tps={sample_metrics['prefill_throughput_tok_per_sec']:.2f} "
                f"decode_tps={sample_metrics['decode_throughput_tok_per_sec']:.2f} "
                f"runner_decode_tps={sample_metrics['runner_decode_throughput_tok_per_sec']:.2f} "
                f"step_ms={sample_metrics['avg_target_step_time_ms']:.2f} verify_ms={sample_metrics['avg_target_verify_time_ms']:.2f} "
                f"e2e_tps={sample_metrics['end_to_end_throughput_tok_per_sec']:.2f}",
                flush=True,
            )

    summary = _build_metrics_summary(per_sample_metrics)
    if args.metrics:
        print(
            f"[metrics] samples={summary['num_samples']} total_tokens={summary['total_output_tokens']} "
            f"total_time={summary['total_wall_time_sec']:.2f}s avg_ttft={summary['avg_ttft_sec']:.2f}s "
            f"avg_latency={summary['avg_latency_sec']:.2f}s avg_prefill_tps={summary['avg_prefill_throughput_tok_per_sec']:.2f} "
            f"avg_decode_tps={summary['avg_decode_throughput_tok_per_sec']:.2f} avg_step_ms={summary['avg_target_step_time_ms']:.2f} "
            f"avg_verify_ms={summary['avg_target_verify_time_ms']:.2f} e2e_tps={summary['end_to_end_throughput_tok_per_sec']:.2f}",
            flush=True,
        )

    if args.metrics_output:
        metrics_output_path = Path(args.metrics_output)
        if not metrics_output_path.is_absolute():
            metrics_output_path = REPO_ROOT / metrics_output_path
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": per_sample_metrics}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()