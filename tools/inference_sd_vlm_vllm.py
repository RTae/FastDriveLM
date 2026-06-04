import argparse
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


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
    with adapter_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            raise ValueError(f"LoRA adapter is missing base_model_name_or_path: {path / 'adapter_config.json'}")
        base_model = _resolve_maybe_local_reference(base_model)
        return base_model, str(path), base_model

    return str(path), None, str(path)


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


def _resolve_vllm_model_source(model_or_adapter: str, dtype_name: str) -> tuple[str, str]:
    model_reference, lora_path, processor_source = _resolve_model_and_lora(model_or_adapter)
    if lora_path is not None:
        model_reference = _merge_lora_for_vllm(processor_source, lora_path, dtype_name)
        processor_source = model_reference
    return model_reference, processor_source


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


def _is_multimodal_model(model_path: str) -> bool:
    model_type = _load_model_type(model_path)
    if model_type is None:
        return False
    multimodal_markers = ("_vl", "vision", "paligemma", "phi4mm")
    return any(marker in model_type.lower() for marker in multimodal_markers)


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


def _filter_llm_kwargs(llm_cls, kwargs: dict) -> dict:
    signature = inspect.signature(llm_cls)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return {key: value for key, value in kwargs.items() if value is not None}

    accepted = set(signature.parameters)
    dropped = sorted(key for key in kwargs if key not in accepted and kwargs[key] is not None)
    if dropped:
        print(
            f"[inference_sd_vlm_vllm] ignoring unsupported vLLM args: {', '.join(dropped)}",
            flush=True,
        )
    return {key: value for key, value in kwargs.items() if key in accepted and value is not None}


def _build_speculative_config(args) -> dict | None:
    if not args.enable_speculative_decoding:
        return None
    if args.speculative_method == "draft_model":
        if not args.draft_model:
            raise ValueError(
                "--draft-model is required when --enable-speculative-decoding "
                "and --speculative-method=draft_model are set"
            )

        draft_model_reference, _ = _resolve_vllm_model_source(args.draft_model, args.dtype)
        return {
            "model": draft_model_reference,
            "method": "draft_model",
            "num_speculative_tokens": args.spec_k,
        }

    if args.speculative_method == "ngram":
        return {
            "model": "ngram",
            "method": "ngram",
            "num_speculative_tokens": args.spec_k,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
        }

    if args.speculative_method == "suffix":
        return {
            "model": "suffix",
            "method": "suffix",
            "num_speculative_tokens": args.spec_k,
            "suffix_decoding_max_tree_depth": args.suffix_max_tree_depth,
            "suffix_decoding_max_cached_requests": args.suffix_max_cached_requests,
            "suffix_decoding_max_spec_factor": args.suffix_max_spec_factor,
            "suffix_decoding_min_token_prob": args.suffix_min_token_prob,
        }

    raise ValueError(f"Unsupported speculative method: {args.speculative_method}")


def _build_attention_config(args):
    if args.attn_backend == "auto":
        return None

    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    return AttentionConfig(backend=AttentionBackendEnum[args.attn_backend])


def _validate_speculative_support(target_model_reference: str, speculative_config: dict | None) -> None:
    if speculative_config is None:
        return
    method = speculative_config.get("method")
    if method == "draft_model" and _is_multimodal_model(target_model_reference):
        raise RuntimeError(
            "Installed vLLM does not support draft-model speculative decoding for multimodal models yet. "
            "This script can still use --use-prefix-caching or model-free speculative methods such as ngram with Qwen3-VL, "
            "but draft-model speculative decoding requires a text-only model or future vLLM support."
        )
    if method == "suffix":
        if importlib.util.find_spec("arctic_inference") is None:
            raise RuntimeError(
                "Suffix speculative decoding requires arctic-inference==0.1.1 in the current vLLM build."
            )


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


def _build_speculative_config(args, draft_model_reference: str) -> dict:
    config: dict[str, Any] = {
        "method": "draft_model",
        "model": draft_model_reference,
        "num_speculative_tokens": args.spec_k,
        "max_model_len": args.draft_max_model_len or args.max_model_len,
    }

    if args.draft_tensor_parallel_size is not None:
        config["draft_tensor_parallel_size"] = args.draft_tensor_parallel_size
    if args.parallel_drafting:
        config["parallel_drafting"] = True
    if args.enforce_eager is not None:
        config["enforce_eager"] = args.enforce_eager

    return config


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

    return {
        "spec_decode_num_drafts": num_drafts,
        "spec_decode_num_draft_tokens": num_draft_tokens,
        "spec_decode_num_accepted_tokens": num_accepted_tokens,
        "spec_decode_mean_acceptance_length": 1.0 + _safe_divide(num_accepted_tokens, num_drafts),
        "spec_decode_acceptance_rate_per_pos": [
            _safe_divide(count, num_drafts) for count in acceptance_counts
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM inference with vLLM")
    parser.add_argument("--target-model", default="outputs/qwen3vl", help="Target model or adapter directory")
    parser.add_argument("--enable-speculative-decoding", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable vLLM speculative decoding with a draft model")
    parser.add_argument("--speculative-method", choices=["draft_model", "ngram", "suffix"], default="draft_model",
                        help="Speculative decoding method to use when enabled")
    parser.add_argument("--draft-model", default=None, help="Draft model or adapter directory used when speculative decoding is enabled")
    parser.add_argument("--spec-k", type=int, default=4, help="Number of speculative draft tokens when speculative decoding is enabled")
    parser.add_argument("--prompt-lookup-min", type=int, default=5,
                        help="Minimum ngram window for ngram speculative decoding")
    parser.add_argument("--prompt-lookup-max", type=int, default=5,
                        help="Maximum ngram window for ngram speculative decoding")
    parser.add_argument("--suffix-max-tree-depth", type=int, default=24,
                        help="Maximum suffix tree depth for suffix speculative decoding")
    parser.add_argument("--suffix-max-cached-requests", type=int, default=10000,
                        help="Maximum number of cached requests for suffix speculative decoding")
    parser.add_argument("--suffix-max-spec-factor", type=float, default=1.0,
                        help="Maximum speculative factor for suffix speculative decoding")
    parser.add_argument("--suffix-min-token-prob", type=float, default=0.1,
                        help="Minimum token probability for suffix speculative decoding")
    parser.add_argument("--data", default="datasets/DriveLM_nuScenes/split/val", help="Dataset path created by load_from_disk")
    parser.add_argument("--output", default="outputs/qwen3vl/infer_results_sd_vlm_vllm.json", help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens per sample")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=16384, help="Maximum model length")
    parser.add_argument("--start-index", type=int, default=0, help="Start sample index")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional number of samples to run")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of leading samples to exclude from aggregate metrics")
    parser.add_argument("--metrics", action="store_true", help="Print per-sample and aggregate speed metrics")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON path for aggregate and per-sample metrics")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto", help="vLLM dtype")
    parser.add_argument("--attn-backend", choices=["auto", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"],
                        default="auto", help="Explicitly select the vLLM attention backend; auto lets vLLM choose")
    parser.add_argument("--use-prefix-caching", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable or disable vLLM prefix caching; omitted uses the vLLM default")
    parser.add_argument("--max-lora-rank", type=int, default=64, help="Maximum LoRA rank exposed to vLLM")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Maximum concurrent sequences for vLLM")
    parser.add_argument("--limit-mm-images", type=int, default=6, help="Number of DriveLM camera images per prompt")
    parser.add_argument("--parallel-drafting", action="store_true", help="Enable vLLM parallel drafting when supported")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=False, help="Pass through to vLLM")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None, help="Pass through to vLLM when supported")
    parser.add_argument(
        "--allow-draft-lora-base-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If --draft-model is a LoRA adapter, use its base model as the vLLM draft. "
            "vLLM speculative_config has no standard per-draft LoRA request field."
        ),
    )
    return parser.parse_args()


def main():
    _ensure_spawn_start_method()

    args = parse_args()

    model_reference, processor_source = _resolve_vllm_model_source(args.target_model, args.dtype)
    speculative_config = _build_speculative_config(args)

    _validate_vllm_model_support(model_reference)
    _validate_speculative_support(model_reference, speculative_config)
    if speculative_config is not None:
        _validate_vllm_model_support(speculative_config["model"])

    from vllm import LLM

    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    generation_config = _load_generation_config(processor_source)
    sampling_params = _build_sampling_params(args, generation_config)
    attention_config = _build_attention_config(args)

    llm_kwargs = {
        "model": target_model,
        "tokenizer": target_processor_source,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.num_gpus,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "attention_config": attention_config,
        "enable_prefix_caching": args.use_prefix_caching,
        "speculative_config": speculative_config,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": not (args.metrics or args.metrics_output),
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
    for sample in tqdm(dataset, desc="vLLM Speculative DriveLM Inference", dynamic_ncols=True):
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
        with metrics_output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": per_sample_metrics}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
