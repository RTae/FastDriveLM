import argparse
import inspect
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _resolve_image_path(path: str) -> str:
    image_path = Path(path)
    if image_path.is_absolute():
        return str(image_path)

    candidate = (REPO_ROOT / image_path).resolve()
    if candidate.exists():
        return str(candidate)

    return str(image_path)


def _load_images(image_paths: list[str]) -> list[Image.Image]:
    return [Image.open(_resolve_image_path(path)).convert("RGB") for path in image_paths]


def _resolve_model_and_lora(model_or_adapter: str) -> tuple[str, str | None, str | None]:
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
            repo_base = (REPO_ROOT / base_path).resolve()
            if repo_base.exists():
                base_path = repo_base

        return str(base_path), str(path), str(base_path)

    return str(path), None, str(path)


def _torch_dtype_from_arg(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return None


def _merged_model_dir(lora_path: str) -> Path:
    adapter_dir = Path(lora_path)
    return adapter_dir / ".vllm_merged_model"


def _copy_tokenizer_assets(source_dir: str, destination_dir: Path) -> None:
    asset_names = [
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for name in asset_names:
        source_path = Path(source_dir) / name
        if source_path.exists():
            shutil.copy2(source_path, destination_dir / name)


def _merge_lora_for_vllm(base_model: str, lora_path: str, dtype_name: str) -> str:
    merged_dir = _merged_model_dir(lora_path)
    marker_path = merged_dir / "merge_meta.json"
    expected_meta = {
        "base_model": str(Path(base_model).resolve()),
        "adapter_path": str(Path(lora_path).resolve()),
    }

    if merged_dir.exists() and marker_path.exists():
        with marker_path.open("r", encoding="utf-8") as f:
            current_meta = json.load(f)
        if current_meta == expected_meta and (merged_dir / "config.json").exists():
            _copy_tokenizer_assets(base_model, merged_dir)
            print(f"[inference_sd_vlm_vllm] reusing merged model cache: {merged_dir}", flush=True)
            return str(merged_dir)

    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[inference_sd_vlm_vllm] merging LoRA adapter for vLLM: {lora_path}", flush=True)
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    torch_dtype = _torch_dtype_from_arg(dtype_name)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    base = AutoModelForImageTextToText.from_pretrained(base_model, **model_kwargs)
    merged = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    processor.save_pretrained(merged_dir)
    _copy_tokenizer_assets(base_model, merged_dir)

    generation_config_path = Path(base_model) / "generation_config.json"
    if generation_config_path.exists():
        shutil.copy2(generation_config_path, merged_dir / "generation_config.json")

    with marker_path.open("w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2)

    return str(merged_dir)


def _load_model_type(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("model_type")


def _validate_vllm_model_support(model_path: str) -> None:
    import vllm

    model_type = _load_model_type(model_path)
    vllm_version = getattr(vllm, "__version__", "unknown")
    if model_type == "qwen3_vl" and vllm_version == "0.1.2":
        raise RuntimeError(
            "Installed vLLM 0.1.2 does not support Qwen3-VL or multimodal inference. "
            "The script can merge LoRA adapters into a full model cache, but Qwen3-VL still requires a newer vLLM build."
        )


def _sample_to_vllm_request(sample: dict, processor) -> tuple[dict, str, str]:
    question = sample["conversations"][0]["value"]
    images = _load_images(sample["image_paths"])
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": question}],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, question, sample["id"]


def _load_generation_config(source: str) -> GenerationConfig:
    try:
        return GenerationConfig.from_pretrained(source, local_files_only=True)
    except OSError:
        return GenerationConfig()


def _build_sampling_params(args, generation_config: GenerationConfig):
    from vllm import SamplingParams

    kwargs: dict[str, object] = {"max_tokens": args.max_new_tokens}
    if getattr(generation_config, "do_sample", False):
        kwargs["temperature"] = float(getattr(generation_config, "temperature", 1.0) or 1.0)
        kwargs["top_p"] = float(getattr(generation_config, "top_p", 1.0) or 1.0)
        top_k = getattr(generation_config, "top_k", None)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
    else:
        kwargs["temperature"] = 0.0

    return SamplingParams(**kwargs)


def _metric_attr(metrics, *names: str) -> float | None:
    if metrics is None:
        return None

    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _filter_llm_kwargs(llm_cls, kwargs: dict) -> dict:
    accepted = set(inspect.signature(llm_cls).parameters)
    dropped = sorted(key for key in kwargs if key not in accepted and kwargs[key] is not None)
    if dropped:
        print(
            f"[inference_sd_vlm_vllm] ignoring unsupported vLLM args: {', '.join(dropped)}",
            flush=True,
        )
    return {key: value for key, value in kwargs.items() if key in accepted and value is not None}


def _ensure_spawn_start_method() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    current = mp.get_start_method(allow_none=True)
    if current != "spawn":
        mp.set_start_method("spawn", force=True)


def _derive_ttft_sec(request_output, started: float, finished: float) -> float:
    metrics = getattr(request_output, "metrics", None)
    first_token_time = _metric_attr(metrics, "first_token_time", "first_token_ts")
    arrival_time = _metric_attr(metrics, "arrival_time", "queued_ts", "scheduled_ts")
    if first_token_time is None:
        return 0.0
    if started <= first_token_time <= finished:
        return first_token_time - started
    if arrival_time is not None:
        candidate = first_token_time - arrival_time
        if 0.0 <= candidate <= finished - started:
            return candidate
    return 0.0


def _build_sample_metrics(request_output, started: float, finished: float) -> dict:
    completion = request_output.outputs[0]
    output_tokens = len(completion.token_ids)
    prompt_tokens = len(request_output.prompt_token_ids or [])
    latency_sec = finished - started
    ttft_sec = _derive_ttft_sec(request_output, started, finished)
    decode_time_sec = max(latency_sec - ttft_sec, 0.0)
    return {
        "output_tokens": output_tokens,
        "ttft_sec": ttft_sec,
        "latency_sec": latency_sec,
        "decode_time_sec": decode_time_sec,
        "prefill_tokens": prompt_tokens,
        "prefill_time_sec": ttft_sec,
        "prefill_throughput_tok_per_sec": _safe_divide(prompt_tokens, ttft_sec),
        "decode_throughput_tok_per_sec": _safe_divide(output_tokens, decode_time_sec),
        "end_to_end_throughput_tok_per_sec": _safe_divide(output_tokens, latency_sec),
        "runner_decode_throughput_tok_per_sec": _safe_divide(output_tokens, decode_time_sec),
        "avg_target_step_time_ms": _safe_divide(decode_time_sec * 1000.0, output_tokens),
        "avg_target_verify_time_ms": 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM inference with vLLM")
    parser.add_argument("--target-model", default="outputs/qwen3vl", help="Target model or adapter directory")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val", help="Dataset path created by load_from_disk")
    parser.add_argument("--output", default="outputs/qwen3vl/infer_results_sd_vlm_vllm.json", help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per sample")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=16384, help="Maximum model length")
    parser.add_argument("--start-index", type=int, default=0, help="Start sample index")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional number of samples to run")
    parser.add_argument("--metrics", action="store_true", help="Print per-sample and aggregate speed metrics")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON path for aggregate and per-sample metrics")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto", help="vLLM dtype")
    parser.add_argument("--max-lora-rank", type=int, default=64, help="Maximum LoRA rank exposed to vLLM")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Maximum concurrent sequences for vLLM")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None, help="Pass through to vLLM when supported")
    return parser.parse_args()


def main():
    _ensure_spawn_start_method()

    args = parse_args()

    model_reference, lora_path, processor_source = _resolve_model_and_lora(args.target_model)
    if lora_path is not None:
        model_reference = _merge_lora_for_vllm(processor_source, lora_path, args.dtype)
        processor_source = model_reference
        lora_path = None

    _validate_vllm_model_support(model_reference)

    from vllm import LLM

    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    generation_config = _load_generation_config(processor_source)
    sampling_params = _build_sampling_params(args, generation_config)

    llm_kwargs = {
        "model": model_reference,
        "tokenizer": processor_source,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.num_gpus,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
    }
    llm = LLM(**_filter_llm_kwargs(LLM, llm_kwargs))

    dataset = load_from_disk(args.data)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    per_sample_metrics = []
    for sample in tqdm(dataset, desc="vLLM DriveLM Inference", dynamic_ncols=True):
        request, question, sample_id = _sample_to_vllm_request(sample, processor)

        started = time.perf_counter()
        outputs = llm.generate([request], sampling_params=sampling_params, use_tqdm=False)
        finished = time.perf_counter()

        request_output = outputs[0]
        answer = request_output.outputs[0].text
        sample_metrics = {"id": sample_id, **_build_sample_metrics(request_output, started, finished)}
        per_sample_metrics.append(sample_metrics)

        results.append({"id": sample_id, "question": question, "answer": answer})
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if args.metrics:
            print(
                f"[metrics] id={sample_id} tok={sample_metrics['output_tokens']} latency={sample_metrics['latency_sec']:.2f}s "
                f"ttft={sample_metrics['ttft_sec']:.2f}s prefill_tps={sample_metrics['prefill_throughput_tok_per_sec']:.2f} "
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