import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TORCH_DTYPES = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _resolve_maybe_local_reference(reference: str | None) -> str | None:
    if not reference:
        return None

    path = Path(reference)
    if path.is_absolute() and path.exists():
        return str(path)

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    # Fine-tuned adapters often store Hugging Face IDs such as
    # Qwen/Qwen3-VL-8B-Instruct. Prefer the repo-local mirror if present.
    local_model_candidate = (REPO_ROOT / "base_models" / reference).resolve()
    if local_model_candidate.exists():
        return str(local_model_candidate)

    return reference


def _resolve_image_path(path: str) -> str:
    image_path = Path(path)
    if image_path.is_absolute():
        return str(image_path)

    repo_candidate = (REPO_ROOT / image_path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)

    return str(image_path)


def _read_adapter_config(adapter_path: Path) -> dict[str, Any]:
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return {}

    with adapter_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_base_model(args) -> str:
    if args.base_model:
        return str(_resolve_maybe_local_reference(args.base_model))

    adapter_path = _resolve_repo_path(args.adapter_path)
    adapter_config = _read_adapter_config(adapter_path)
    base_model = _resolve_maybe_local_reference(
        adapter_config.get("base_model_name_or_path")
    )
    if base_model:
        return str(base_model)

    default_base = REPO_ROOT / "base_models" / "Qwen" / "Qwen3-VL-8B-Instruct"
    if default_base.exists():
        return str(default_base)

    raise ValueError(
        "Could not resolve the Qwen3-VL base model. Pass --base-model, or set "
        "base_model_name_or_path in outputs/qwen3vl/adapter_config.json. "
        "For your layout this is usually "
        "base_models/Qwen/Qwen3-VL-8B-Instruct."
    )


def validate_adapter_path(adapter_path: str) -> str | None:
    if not adapter_path:
        return None

    path = _resolve_repo_path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Adapter path is not a directory: {path}")

    adapter_config = path / "adapter_config.json"
    has_weights = (path / "adapter_model.safetensors").exists() or any(
        path.glob("adapter_model*.bin")
    )
    if not adapter_config.exists() or not has_weights:
        raise ValueError(
            "Expected a PEFT LoRA adapter directory with adapter_config.json "
            f"and adapter weights. Checked: {path}"
        )

    return str(path)


def _resize_for_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        return img

    width, height = img.size
    if width * height <= max_pixels:
        return img

    scale = (max_pixels / (width * height)) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _load_images(sample: dict[str, Any], num_images: int, max_pixels: int) -> list[Image.Image]:
    image_paths = sample["image_paths"]
    if len(image_paths) < num_images:
        raise ValueError(f"Expected at least {num_images} images, got {len(image_paths)}")

    images = []
    for image_path in image_paths[:num_images]:
        image = Image.open(_resolve_image_path(image_path)).convert("RGB")
        images.append(_resize_for_max_pixels(image, max_pixels))
    return images


def _build_user_content(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return content


def build_calibration_dataset(args, processor):
    dataset = load_from_disk(args.data)
    if args.shuffle_calibration:
        dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(min(args.num_calibration_samples, len(dataset))))

    def preprocess_and_tokenize(sample: dict[str, Any]) -> dict[str, Any]:
        question = sample["conversations"][0]["value"]
        images = _load_images(sample, args.num_images, args.max_pixels_per_image)
        messages = [
            {
                "role": "user",
                "content": _build_user_content(question, images),
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return processor(
            text=[text],
            images=images,
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
        )

    return dataset.map(
        preprocess_and_tokenize,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="Tokenizing DriveLM calibration samples",
    )


def data_collator(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("AWQ calibration expects --batch-size 1 for Qwen3-VL images.")

    return {
        key: torch.tensor(value)
        for key, value in batch[0].items()
        if value is not None
    }


def _model_loader():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError:
        return AutoModelForImageTextToText


def load_merged_model(args, base_model: str, adapter_path: str | None):
    loader = _model_loader()
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    dtype = TORCH_DTYPES[args.dtype]
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = loader.from_pretrained(base_model, **model_kwargs)

    if adapter_path:
        from peft import PeftModel

        print(f"[quantize] merging LoRA adapter: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=True,
        ).merge_and_unload()

    if args.device:
        model.to(args.device)
    model.eval()
    return model


def _supports_kwarg(fn, key: str) -> bool:
    signature = inspect.signature(fn)
    if key in signature.parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def run_awq_oneshot(args, model, processor, dataset, processor_source: str):
    try:
        from llmcompressor import oneshot
    except ImportError:
        from llmcompressor.transformers import oneshot

    try:
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.awq.base import AWQModifier

    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = [
        AWQModifier(
            duo_scaling=args.duo_scaling,
            n_grid=args.awq_grid_size,
        ),
        QuantizationModifier(
            scheme=args.scheme,
            targets=args.targets,
            ignore=args.ignore,
        ),
    ]

    oneshot_kwargs: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "recipe": recipe,
        "max_seq_length": args.max_seq_length,
        "num_calibration_samples": args.num_calibration_samples,
        "batch_size": args.batch_size,
        "data_collator": data_collator,
        "pipeline": args.pipeline,
        "sequential_targets": args.sequential_targets,
    }
    if _supports_kwarg(oneshot, "processor"):
        oneshot_kwargs["processor"] = processor
    elif _supports_kwarg(oneshot, "tokenizer"):
        oneshot_kwargs["tokenizer"] = processor_source
    if _supports_kwarg(oneshot, "trust_remote_code_model"):
        oneshot_kwargs["trust_remote_code_model"] = args.trust_remote_code

    print("[quantize] starting llmcompressor AWQ INT4 oneshot", flush=True)
    oneshot(**oneshot_kwargs)


def save_quantized_model(args, model, processor, base_model: str, adapter_path: str | None):
    output_dir = _resolve_repo_path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quantize] saving compressed model: {output_dir}", flush=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        save_compressed=True,
    )
    processor.save_pretrained(output_dir)

    try:
        generation_config = GenerationConfig.from_pretrained(
            base_model,
            local_files_only=True,
        )
        generation_config.save_pretrained(output_dir)
    except OSError:
        pass

    metadata = {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "scheme": args.scheme,
        "targets": args.targets,
        "ignore": args.ignore,
        "num_calibration_samples": args.num_calibration_samples,
        "max_seq_length": args.max_seq_length,
        "num_images": args.num_images,
        "max_pixels_per_image": args.max_pixels_per_image,
        "pipeline": args.pipeline,
        "sequential_targets": args.sequential_targets,
        "seed": args.seed,
    }
    with (output_dir / "quantization_run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a Qwen3-VL LoRA adapter and quantize it to INT4 AWQ with llmcompressor."
    )
    parser.add_argument(
        "--adapter-path",
        default="outputs/qwen3vl",
        help="PEFT LoRA adapter directory, usually outputs/qwen3vl.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Local Qwen3-VL base model directory. If omitted, the script reads "
            "adapter_config.json and prefers base_models/<HF model id>."
        ),
    )
    parser.add_argument(
        "--data",
        default="datasets/DriveLM_nuScenes/split/val",
        help="DriveLM dataset path created by load_from_disk.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qwen3vl_awq_int4",
        help="Directory where the vLLM-loadable compressed model is saved.",
    )
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument(
        "--max-pixels-per-image",
        type=int,
        default=256 * 28 * 28,
        help="Resize calibration images to this area. Use 0 to disable resizing.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle DriveLM before selecting calibration samples.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(TORCH_DTYPES.keys()),
        default="bfloat16",
        help="Dtype used while loading/merging before quantization.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scheme",
        default="W4A16_ASYM",
        help="llmcompressor quantization scheme. W4A16_ASYM is AWQ INT4 weight-only.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["Linear"],
        help="Module types/names targeted by QuantizationModifier.",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=["re:.*lm_head", "re:.*visual.*"],
        help="Regexes/names to leave unquantized. Default keeps lm_head and visual tower full precision.",
    )
    parser.add_argument(
        "--sequential-targets",
        nargs="+",
        default=["Qwen3VLTextDecoderLayer"],
        help="Decoder layer class names used by llmcompressor sequential calibration.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["basic", "datafree", "sequential", "independent"],
        default="sequential",
        help="llmcompressor calibration pipeline. Sequential reduces peak VRAM during AWQ calibration.",
    )
    parser.add_argument(
        "--duo-scaling",
        choices=["true", "false", "both"],
        default="false",
        help="AWQ duo_scaling setting. Qwen3-VL examples commonly use false.",
    )
    parser.add_argument("--awq-grid-size", type=int, default=20)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    args = parser.parse_args()

    if args.duo_scaling == "true":
        args.duo_scaling = True
    elif args.duo_scaling == "false":
        args.duo_scaling = False

    if args.batch_size != 1:
        raise ValueError("Qwen3-VL calibration with images currently supports --batch-size 1.")

    return args


def main():
    args = parse_args()
    adapter_path = validate_adapter_path(args.adapter_path)
    base_model = resolve_base_model(args)
    processor_source = base_model

    print(f"[quantize] base model: {base_model}", flush=True)
    print(f"[quantize] adapter: {adapter_path}", flush=True)

    processor_kwargs = {"local_files_only": True}
    if args.trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)

    model = load_merged_model(args, base_model, adapter_path)
    dataset = build_calibration_dataset(args, processor)
    run_awq_oneshot(args, model, processor, dataset, processor_source)
    save_quantized_model(args, model, processor, base_model, adapter_path)

    print(f"[quantize] done: {_resolve_repo_path(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
