from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_BASED_SPEC_METHODS = {"draft_model", "eagle", "eagle3", "medusa"}


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


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _resolve_maybe_local_reference(reference: str) -> str:
    path = Path(reference)
    if path.is_absolute():
        return str(path)

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    # Keep Hugging Face model IDs untouched.
    return reference


def _read_adapter_config(adapter_dir: Path) -> dict:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return {}
    with adapter_config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _has_adapter_artifacts(path: Path) -> bool:
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        return False
    return (path / "adapter_model.safetensors").exists() or any(
        path.glob("adapter_model*.bin")
    )


def _resolve_model_and_lora(model_or_adapter: str) -> tuple[str, str | None, str]:
    path = _resolve_repo_path(model_or_adapter)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {path}")

    if _has_adapter_artifacts(path):
        adapter_config = _read_adapter_config(path)
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(
                f"LoRA adapter is missing base_model_name_or_path: {path / 'adapter_config.json'}"
            )
        base_model = _resolve_maybe_local_reference(base_model)
        return base_model, str(path), base_model

    return str(path), None, str(path)


def _load_model_type(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("model_type")


def _looks_multimodal_model(model_path: str) -> bool:
    model_type = _load_model_type(model_path)
    if model_type is None:
        return False
    return any(marker in model_type.lower() for marker in ("_vl", "vision", "paligemma", "phi4mm"))


def _resolve_image_path(path: str) -> str:
    image_path = Path(path)
    if image_path.is_absolute():
        return str(image_path)

    candidate = (REPO_ROOT / image_path).resolve()
    if candidate.exists():
        return str(candidate)

    return str(image_path)


def _load_images(image_paths: list[str], limit: int) -> list[Any]:
    from PIL import Image

    if len(image_paths) < limit:
        raise ValueError(f"Expected at least {limit} images, got {len(image_paths)}")
    return [
        Image.open(_resolve_image_path(image_path)).convert("RGB")
        for image_path in image_paths[:limit]
    ]


def _sample_to_vllm_request(
    sample: dict,
    processor,
    limit_mm_images: int,
) -> tuple[dict, str, str]:
    question = sample["conversations"][0]["value"]
    images = _load_images(sample["image_paths"], limit_mm_images)
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": question}]
            ),
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, question, sample["id"]


def _load_generation_config(source: str) -> Any:
    from transformers import GenerationConfig

    try:
        return GenerationConfig.from_pretrained(source, local_files_only=True)
    except OSError:
        return GenerationConfig()


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _build_sampling_params(args, generation_config: Any):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {"max_tokens": args.max_new_tokens}
    if _config_value(generation_config, "do_sample", False):
        kwargs["temperature"] = float(_config_value(generation_config, "temperature", 1.0))
        kwargs["top_p"] = float(_config_value(generation_config, "top_p", 1.0))
        top_k = _config_value(generation_config, "top_k", None)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
    else:
        kwargs["temperature"] = 0.0

    eos_token_id = _config_value(generation_config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        kwargs["stop_token_ids"] = [eos_token_id]
    elif isinstance(eos_token_id, list) and eos_token_id:
        kwargs["stop_token_ids"] = eos_token_id

    return SamplingParams(**kwargs)


def _metric_attr(metrics, *names: str) -> float | None:
    if metrics is None:
        return None

    for name in names:
        if isinstance(metrics, dict):
            value = metrics.get(name)
        else:
            value = getattr(metrics, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    signature = inspect.signature(callable_obj)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return {key: value for key, value in kwargs.items() if value is not None}

    accepted = set(signature.parameters)
    dropped = sorted(key for key in kwargs if key not in accepted and kwargs[key] is not None)
    if dropped:
        print(
            f"[inference_sd_vlm_vllm] ignoring unsupported vLLM args: {', '.join(dropped)}",
            flush=True,
        )
    return {
        key: value
        for key, value in kwargs.items()
        if key in accepted and value is not None
    }


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


def _spec_model_arg(args, method: str) -> str | None:
    if args.speculator_model:
        return args.speculator_model
    if method == "draft_model":
        return args.draft_model
    if method == "eagle":
        return args.eagle_model
    if method == "eagle3":
        return args.eagle3_model
    if method == "medusa":
        return args.medusa_model
    return None


def _resolve_speculator_model(args, method: str) -> str:
    spec_model = _spec_model_arg(args, method)
    if not spec_model:
        raise ValueError(
            f"--{method.replace('_', '-')}-model or --speculator-model is required "
            f"when --speculative-method={method}"
        )

    model_reference, lora_path, _ = _resolve_model_and_lora(spec_model)
    if lora_path is not None:
        if not args.allow_speculator_lora_base_fallback:
            raise ValueError(
                f"{method} speculator path is a LoRA adapter, but vLLM speculative_config "
                "does not accept a separate LoRARequest for speculator weights."
            )
        print(
            f"[inference_sd_vlm_vllm] {method} speculator is a LoRA adapter; "
            "using its base model in speculative_config.",
            flush=True,
        )
    return model_reference


def _build_speculative_config(args) -> dict | None:
    method = args.speculative_method
    if args.enable_speculative_decoding and method == "none":
        method = "ngram"
    if method == "none":
        return None

    config: dict[str, Any] = {
        "method": method,
        "num_speculative_tokens": args.spec_k,
    }

    if method in {"ngram", "ngram_gpu"}:
        config["prompt_lookup_min"] = args.prompt_lookup_min
        config["prompt_lookup_max"] = args.prompt_lookup_max
        return config

    if method == "suffix":
        config.update(
            {
                "suffix_decoding_max_tree_depth": args.suffix_max_tree_depth,
                "suffix_decoding_max_cached_requests": args.suffix_max_cached_requests,
                "suffix_decoding_max_spec_factor": args.suffix_max_spec_factor,
                "suffix_decoding_min_token_prob": args.suffix_min_token_prob,
            }
        )
        return config

    if method in MODEL_BASED_SPEC_METHODS:
        config["model"] = _resolve_speculator_model(args, method)
        if args.draft_tensor_parallel_size is not None:
            config["draft_tensor_parallel_size"] = args.draft_tensor_parallel_size
        if args.draft_max_model_len is not None:
            config["max_model_len"] = args.draft_max_model_len
        if args.draft_attention_backend:
            config["attention_backend"] = args.draft_attention_backend
        if args.draft_quantization:
            config["quantization"] = args.draft_quantization
        if args.parallel_drafting:
            config["parallel_drafting"] = True
        if args.disable_padded_drafter_batch:
            config["disable_padded_drafter_batch"] = True
        if args.enforce_eager is not None:
            config["enforce_eager"] = args.enforce_eager
        return config

    raise ValueError(f"Unsupported speculative method: {method}")


def _build_lora_request(lora_path: str | None):
    if not lora_path:
        return None

    from vllm.lora.request import LoRARequest

    return LoRARequest("drivelm_qwen3vl_target", 1, lora_path)


def _metric_value(metric) -> Any:
    if hasattr(metric, "value"):
        return metric.value
    if hasattr(metric, "values"):
        return metric.values
    return None


def _collect_spec_decode_metrics(llm, spec_k: int) -> dict:
    if not hasattr(llm, "get_metrics"):
        return {}

    metrics = llm.get_metrics()
    num_drafts = 0.0
    num_draft_tokens = 0.0
    num_accepted_tokens = 0.0
    acceptance_counts = [0.0] * spec_k

    for metric in metrics:
        name = getattr(metric, "name", "")
        value = _metric_value(metric)
        if value is None:
            continue

        if name == "vllm:spec_decode_num_drafts":
            num_drafts += float(value)
        elif name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += float(value)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted_tokens += float(value)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos, count in enumerate(list(value)[:spec_k]):
                acceptance_counts[pos] += float(count)

    if num_drafts <= 0:
        return {}

    return {
        "spec_decode_num_drafts": num_drafts,
        "spec_decode_num_draft_tokens": num_draft_tokens,
        "spec_decode_num_accepted_tokens": num_accepted_tokens,
        "spec_decode_mean_acceptance_length": 1.0 + _safe_divide(num_accepted_tokens, num_drafts),
        "spec_decode_acceptance_rate_per_pos": [
            _safe_divide(count, num_drafts) for count in acceptance_counts
        ],
    }


def _set_attention_backend(attn_backend: str) -> None:
    if attn_backend != "auto":
        os.environ["VLLM_ATTENTION_BACKEND"] = attn_backend


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM inference with vLLM speculative decoding")
    parser.add_argument("--target-model", default="outputs/qwen3vl", help="Target model or LoRA adapter directory")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val", help="Dataset path created by load_from_disk")
    parser.add_argument("--output", default="outputs/qwen3vl/infer_results_sd_vlm_vllm.json", help="Output JSON path")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON path for aggregate and per-sample metrics")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per sample")
    parser.add_argument("--start-index", type=int, default=0, help="Start sample index")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional number of samples to run")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of leading samples to exclude from aggregate metrics")
    parser.add_argument("--metrics", action="store_true", help="Print per-sample and aggregate speed metrics")

    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto", help="vLLM dtype")
    parser.add_argument("--num-gpus", type=int, default=1, help="Target model tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=16384, help="Maximum target model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Maximum concurrent sequences for vLLM")
    parser.add_argument("--limit-mm-images", type=int, default=6, help="Number of DriveLM camera images per prompt")
    parser.add_argument(
        "--attn-backend",
        choices=["auto", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "TORCH_SDPA", "XFORMERS"],
        default="auto",
        help="Set VLLM_ATTENTION_BACKEND before importing vLLM",
    )
    parser.add_argument("--use-prefix-caching", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable vLLM prefix caching")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=False, help="Pass through to vLLM")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None, help="Pass through to vLLM when supported")
    parser.add_argument("--max-lora-rank", type=int, default=64, help="Maximum target LoRA rank exposed to vLLM")

    parser.add_argument("--enable-speculative-decoding", action=argparse.BooleanOptionalAction, default=False, help="Compatibility flag; if set without --speculative-method, uses ngram")
    parser.add_argument(
        "--speculative-method",
        choices=["none", "draft_model", "ngram", "ngram_gpu", "suffix", "eagle", "eagle3", "medusa"],
        default="none",
        help="Speculative decoding method. Use none for plain vLLM.",
    )
    parser.add_argument("--spec-k", type=int, default=4, help="Number of speculative draft tokens")
    parser.add_argument("--prompt-lookup-min", type=int, default=2, help="Minimum ngram window")
    parser.add_argument("--prompt-lookup-max", type=int, default=4, help="Maximum ngram window")
    parser.add_argument("--suffix-max-tree-depth", type=int, default=24, help="Maximum suffix tree depth")
    parser.add_argument("--suffix-max-cached-requests", type=int, default=10000, help="Maximum global suffix cache size")
    parser.add_argument("--suffix-max-spec-factor", type=float, default=1.0, help="Maximum suffix speculative factor")
    parser.add_argument("--suffix-min-token-prob", type=float, default=0.1, help="Minimum suffix token probability")

    parser.add_argument("--draft-model", default=None, help="Draft model for --speculative-method=draft_model")
    parser.add_argument("--eagle-model", default=None, help="EAGLE head/model for --speculative-method=eagle")
    parser.add_argument("--eagle3-model", default=None, help="EAGLE3 head/model for --speculative-method=eagle3")
    parser.add_argument("--medusa-model", default=None, help="Medusa heads/model for --speculative-method=medusa")
    parser.add_argument("--speculator-model", default=None, help="Generic override for model-based speculative methods")
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=None, help="Optional tensor parallel size for model-based speculator")
    parser.add_argument("--draft-max-model-len", type=int, default=None, help="Optional max model length for model-based speculator")
    parser.add_argument("--draft-attention-backend", default=None, help="Optional attention backend for model-based speculator")
    parser.add_argument("--draft-quantization", default=None, help="Optional quantization setting for model-based speculator")
    parser.add_argument("--parallel-drafting", action="store_true", help="Enable vLLM parallel drafting when supported")
    parser.add_argument("--disable-padded-drafter-batch", action="store_true", help="Pass through to vLLM EAGLE drafter config")
    parser.add_argument(
        "--allow-speculator-lora-base-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a speculator path is a LoRA adapter, use its base model in speculative_config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _set_attention_backend(args.attn_backend)

    from datasets import load_from_disk
    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import LLM

    target_model, target_lora_path, processor_source = _resolve_model_and_lora(args.target_model)
    speculative_config = _build_speculative_config(args)

    if speculative_config and _looks_multimodal_model(target_model) and speculative_config["method"] in MODEL_BASED_SPEC_METHODS:
        print(
            "[inference_sd_vlm_vllm] warning: model-based speculative methods "
            "may not be supported by your installed vLLM build for multimodal targets; "
            "vLLM will raise an error if unsupported.",
            flush=True,
        )

    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    generation_config = _load_generation_config(processor_source)
    sampling_params = _build_sampling_params(args, generation_config)
    lora_request = _build_lora_request(target_lora_path)

    llm_kwargs = {
        "model": target_model,
        "tokenizer": processor_source,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.num_gpus,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "limit_mm_per_prompt": {"image": args.limit_mm_images},
        "enable_prefix_caching": args.use_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "disable_chunked_mm_input": True,
        "enforce_eager": args.enforce_eager,
        "speculative_config": speculative_config,
        "disable_log_stats": not (args.metrics or args.metrics_output),
        "enable_lora": target_lora_path is not None,
        "max_loras": 1 if target_lora_path else None,
        "max_lora_rank": args.max_lora_rank if target_lora_path else None,
    }
    llm = LLM(**_filter_kwargs(LLM, llm_kwargs))

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
    desc = "vLLM DriveLM Inference"
    if speculative_config:
        desc = f"vLLM {speculative_config['method']} Speculative DriveLM Inference"

    for sample in tqdm(dataset, desc=desc, dynamic_ncols=True):
        request, question, sample_id = _sample_to_vllm_request(
            sample,
            processor,
            args.limit_mm_images,
        )

        started = time.perf_counter()
        outputs = llm.generate(
            [request],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        finished = time.perf_counter()

        request_output = outputs[0]
        answer = request_output.outputs[0].text
        sample_metrics = {"id": sample_id, **_build_sample_metrics(request_output, started, finished)}
        per_sample_metrics.append(sample_metrics)

        results.append({"id": sample_id, "question": question, "answer": answer})
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

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

    summary = _build_metrics_summary(per_sample_metrics[args.warmup_steps:])
    summary.update(_collect_spec_decode_metrics(llm, args.spec_k))
    if args.metrics:
        print(
            f"[metrics] samples={summary['num_samples']} total_tokens={summary['total_output_tokens']} "
            f"total_time={summary['total_wall_time_sec']:.2f}s avg_ttft={summary['avg_ttft_sec']:.2f}s "
            f"avg_latency={summary['avg_latency_sec']:.2f}s avg_prefill_tps={summary['avg_prefill_throughput_tok_per_sec']:.2f} "
            f"avg_decode_tps={summary['avg_decode_throughput_tok_per_sec']:.2f} avg_step_ms={summary['avg_target_step_time_ms']:.2f} "
            f"avg_verify_ms={summary['avg_target_verify_time_ms']:.2f} e2e_tps={summary['end_to_end_throughput_tok_per_sec']:.2f}",
            flush=True,
        )
        if "spec_decode_num_drafts" in summary:
            print(
                f"[metrics] spec_drafts={summary['spec_decode_num_drafts']:.0f} "
                f"spec_draft_tokens={summary['spec_decode_num_draft_tokens']:.0f} "
                f"spec_accepted_tokens={summary['spec_decode_num_accepted_tokens']:.0f} "
                f"mean_acceptance_length={summary['spec_decode_mean_acceptance_length']:.2f}",
                flush=True,
            )

    if args.metrics_output:
        metrics_output_path = Path(args.metrics_output)
        if not metrics_output_path.is_absolute():
            metrics_output_path = REPO_ROOT / metrics_output_path
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_output_path.open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "samples": per_sample_metrics}, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
