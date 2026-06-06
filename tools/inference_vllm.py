import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DTYPES = {
    "auto": "auto",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}

VLLM_ATTN_IMPLEMENTATIONS = [
    "auto",
    "eager",
    "sdpa",
    "torch_sdpa",
    "flash_attention_2",
    "flashinfer",
    "xformers",
    "flex_attention",
]

SUPPORTED_COLLATE_FNS = {"drivelm_nus_qwen3vl_collate_fn_val"}


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


def validate_model_path(model_path: str) -> str:
    if not model_path:
        raise ValueError("--model-path is required.")

    path = _resolve_repo_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {path}")

    return str(path)


def has_adapter_artifacts(path: Path) -> bool:
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        return False

    has_weights = (path / "adapter_model.safetensors").exists() or any(
        path.glob("adapter_model*.bin")
    )
    return has_weights


def has_full_model_weights(path: Path) -> bool:
    return (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin.index.json").exists()
    )


def can_load_full_model_from_dir(path: Path) -> bool:
    return (path / "config.json").exists() and has_full_model_weights(path)


def _read_adapter_config(adapter_dir: Path) -> dict:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return {}
    with adapter_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_base_model(args, model_dir: Path) -> str | None:
    if args.base_model:
        return _resolve_maybe_local_reference(args.base_model)

    adapter_config = _read_adapter_config(model_dir)
    base_model = adapter_config.get("base_model_name_or_path")
    if base_model:
        return _resolve_maybe_local_reference(base_model)

    return None


def resolve_trust_remote_code(args, model_reference: str) -> bool:
    if args.trust_remote_code is not None:
        return args.trust_remote_code

    model_reference = str(model_reference).lower()
    return "qwen" in model_reference or "phi" in model_reference


def resolve_model_and_lora(args) -> tuple[str, str | None, str | None]:
    model_path = validate_model_path(args.model_path)
    model_dir = Path(model_path)
    base_model = None
    lora_path = None

    if has_adapter_artifacts(model_dir):
        base_model = resolve_base_model(args, model_dir)
        if not base_model:
            raise ValueError(
                "Detected adapter artifacts but base model is missing. "
                "Provide --base-model or set base_model_name_or_path in adapter_config.json. "
                f"Checked: {model_dir}"
            )
        model_reference = base_model
        lora_path = model_path
    elif can_load_full_model_from_dir(model_dir):
        model_reference = model_path
    elif has_full_model_weights(model_dir):
        raise ValueError(
            "vLLM requires a loadable model directory with config.json. "
            "For checkpoint-only full weights, first export/merge them into a full "
            f"model directory. Checked: {model_dir}"
        )
    else:
        raise ValueError(
            "Unsupported model directory. Expected either a full model directory "
            "(config.json + weights) or a LoRA adapter directory "
            "(adapter_config.json + adapter weights). "
            f"Checked: {model_dir}"
        )

    return model_reference, lora_path, base_model


def load_generation_config(generation_source: str) -> GenerationConfig:
    try:
        return GenerationConfig.from_pretrained(generation_source, local_files_only=True)
    except OSError:
        return GenerationConfig()


def _build_user_content(question: str, images: list[Image.Image]) -> list[dict]:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": question})
    return content


def _resolve_image_path(path: str) -> str:
    image_path = Path(path)
    if image_path.is_absolute():
        return str(image_path)

    repo_candidate = (REPO_ROOT / image_path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    return str(image_path)


def _load_images(image_paths: list[str], limit: int) -> list[Image.Image]:
    if len(image_paths) < limit:
        raise ValueError(f"Expected at least {limit} images, got {len(image_paths)}")
    return [
        Image.open(_resolve_image_path(image_path)).convert("RGB")
        for image_path in image_paths[:limit]
    ]


def sample_to_vllm_request(
    sample: dict,
    processor,
    num_images: int,
) -> tuple[dict, str, str]:
    question = sample["conversations"][0]["value"]
    images = _load_images(sample["image_paths"], num_images)

    messages = [
        {
            "role": "user",
            "content": _build_user_content(question, images),
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return (
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        question,
        sample["id"],
    )


def _get_config_value(config: GenerationConfig, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def build_sampling_params(args, generation_config: GenerationConfig):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {"max_tokens": args.max_new_tokens}

    if _get_config_value(generation_config, "do_sample", False):
        kwargs["temperature"] = float(_get_config_value(generation_config, "temperature", 1.0))
        kwargs["top_p"] = float(_get_config_value(generation_config, "top_p", 1.0))
        top_k = _get_config_value(generation_config, "top_k", None)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
    else:
        kwargs["temperature"] = 0.0

    repetition_penalty = _get_config_value(generation_config, "repetition_penalty", 1.0)
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = float(repetition_penalty)

    eos_token_id = _get_config_value(generation_config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        kwargs["stop_token_ids"] = [eos_token_id]
    elif isinstance(eos_token_id, list) and eos_token_id:
        kwargs["stop_token_ids"] = eos_token_id

    return SamplingParams(**kwargs)


def _set_vllm_attention_backend(args) -> None:
    backend_by_name = {
        "sdpa": "TORCH_SDPA",
        "torch_sdpa": "TORCH_SDPA",
        "flash_attention_2": "FLASH_ATTN",
        "flashinfer": "FLASHINFER",
        "xformers": "XFORMERS",
        "flex_attention": "FLEX_ATTENTION",
    }
    backend = backend_by_name.get(args.attn_implementation)
    if backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = backend

    if args.attn_implementation == "eager" and args.enforce_eager is None:
        args.enforce_eager = True


def _filter_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return {key: value for key, value in kwargs.items() if value is not None}

    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }


def build_llm(args, model_reference: str, lora_path: str | None):
    _set_vllm_attention_backend(args)

    from vllm import LLM

    llm_kwargs = {
        "model": model_reference,
        "tokenizer": args.processor_path or model_reference,
        "trust_remote_code": resolve_trust_remote_code(args, model_reference),
        "dtype": DTYPES[args.dtype],
        "device": args.device,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": args.limit_mm_images},
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "enable_lora": lora_path is not None,
        "max_loras": 1 if lora_path else None,
        "max_lora_rank": args.max_lora_rank if lora_path else None,
    }
    return LLM(**_filter_kwargs(LLM, llm_kwargs))


def build_lora_request(lora_path: str | None):
    if not lora_path:
        return None

    from vllm.lora.request import LoRARequest

    return LoRARequest("drivelm_qwen3vl", 1, lora_path)


def _metric_attr(metrics: Any, *names: str) -> float | None:
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


def _derive_ttft_sec(metrics: Any, started: float, finished: float) -> float:
    first_token_times = [
        _metric_attr(metrics, "first_token_time"),
        _metric_attr(metrics, "first_token_ts"),
    ]
    arrival_times = [
        _metric_attr(metrics, "arrival_time"),
        _metric_attr(metrics, "queued_ts"),
        _metric_attr(metrics, "scheduled_ts"),
    ]

    for first_token_time in first_token_times:
        if first_token_time is None or first_token_time <= 0:
            continue

        if started <= first_token_time <= finished:
            return first_token_time - started

        for arrival_time in arrival_times:
            if arrival_time is None or arrival_time <= 0:
                continue
            candidate = first_token_time - arrival_time
            if 0.0 <= candidate <= finished - started:
                return candidate

    return 0.0


def build_sample_metrics(request_output, started: float, finished: float) -> dict:
    completion = request_output.outputs[0]
    output_tokens = len(completion.token_ids)
    prompt_tokens = len(request_output.prompt_token_ids or [])
    latency_sec = finished - started
    ttft_sec = _derive_ttft_sec(getattr(request_output, "metrics", None), started, finished)
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


def _validate_args(args) -> None:
    if args.collate_fn not in SUPPORTED_COLLATE_FNS:
        supported = ", ".join(sorted(SUPPORTED_COLLATE_FNS))
        raise ValueError(
            "vLLM inference currently supports the Qwen3-VL validation prompt "
            f"format only. Got --collate_fn={args.collate_fn!r}; supported: {supported}"
        )

    if args.model_class not in {"auto", "image-text-to-text"}:
        raise ValueError(
            "vLLM baseline script is intended for image-text-to-text Qwen3-VL. "
            f"Got --model-class={args.model_class!r}."
        )


def main(args):
    _validate_args(args)

    model_reference, lora_path, base_model = resolve_model_and_lora(args)
    processor_source = args.processor_path or base_model or model_reference
    generation_source = args.generation_config_path or base_model or model_reference
    trust_remote_code = resolve_trust_remote_code(args, model_reference)

    processor_kwargs = {"local_files_only": True}
    if trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    # Ensure the tokenizer does not silently truncate multimodal prompts
    # which causes a mismatch between the image token counts in text
    # and the tokenized `input_ids` (see vLLM error about truncation).
    # Best-effort: if a tokenizer is available, increase its max length
    # to at least the configured `--max-model-len` and avoid truncation.
    try:
        tk = getattr(processor, "tokenizer", None)
        if tk is not None:
            # Align tokenizer max length with vLLM context window to avoid
            # accidental truncation inside HF processor calls.
            if hasattr(tk, "model_max_length"):
                try:
                    tk.model_max_length = max(int(getattr(tk, "model_max_length", 0)), int(args.max_model_len))
                except Exception:
                    tk.model_max_length = int(args.max_model_len)

            # Some HF tokenizers carry init kwargs that may enable truncation
            # by default; remove any explicit truncation flag if present.
            if hasattr(tk, "init_kwargs") and isinstance(tk.init_kwargs, dict):
                tk.init_kwargs.pop("truncation", None)

            # If the tokenizer supports a mistral regex fix, enable it.
            if "fix_mistral_regex" in getattr(tk, "__dict__", {}):
                try:
                    tk.fix_mistral_regex = True
                except Exception:
                    pass
    except Exception:
        # Don't fail if we cannot access or modify the tokenizer internals.
        pass
    generation_config = load_generation_config(generation_source)

    llm = build_llm(args, model_reference, lora_path)
    sampling_params = build_sampling_params(args, generation_config)
    lora_request = build_lora_request(lora_path)

    dataset = load_from_disk(args.data)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    data_dict = []
    per_sample_metrics = []
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(dataset, desc="vLLM DriveLM Inference", dynamic_ncols=True):
        request, question, sample_id = sample_to_vllm_request(
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
        sample_metrics = {
            "id": sample_id,
            **build_sample_metrics(request_output, started, finished),
        }
        per_sample_metrics.append(sample_metrics)
        data_dict.append({"id": sample_id, "question": question, "answer": answer})

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)

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


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM Qwen3-VL baseline inference with vLLM")
    parser.add_argument("--data", type=str, default="datasets/DriveLM_nuScenes/split/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_qwen3vl_collate_fn_val")
    parser.add_argument("--output", type=str, default="outputs/qwen3vl/infer_results_vllm.json")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Single local model directory path. Supports full model directories "
            "or LoRA adapter directories with adapter_config.json."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model local path or cached model id. Required for adapter directories if not set in adapter_config.json.",
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help="Optional local processor/tokenizer directory. Defaults to the resolved base model.",
    )
    parser.add_argument(
        "--generation-config-path",
        type=str,
        default=None,
        help="Optional local GenerationConfig directory. Defaults to the resolved base model.",
    )
    parser.add_argument(
        "--model-class",
        choices=["auto", "image-text-to-text", "causal-lm"],
        default="auto",
        help="Kept for CLI compatibility. vLLM baseline currently supports Qwen3-VL image-text-to-text.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES.keys()), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--metrics-output", type=str, default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override trust_remote_code. Defaults to auto for Qwen and Phi models.",
    )
    parser.add_argument("--device", default="cuda", help="Device passed through to vLLM when supported.")
    parser.add_argument(
        "--attn-implementation",
        choices=VLLM_ATTN_IMPLEMENTATIONS,
        default="auto",
        help=(
            "Attention/backend ablation. HF-compatible values are accepted where "
            "vLLM has an equivalent: sdpa->TORCH_SDPA, flash_attention_2->FLASH_ATTN. "
            "Additional vLLM options: flashinfer, xformers, flex_attention."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Kept for CLI compatibility; vLLM manages compilation internally.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of leading samples to exclude from aggregate metrics.",
    )

    # vLLM-specific controls.
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--limit-mm-images", type=int, default=6)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass through to vLLM. Also enabled automatically when --attn-implementation=eager.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
