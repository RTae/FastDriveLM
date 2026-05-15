import argparse
import inspect
import json
from functools import partial
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

from drivevlms.build import build_collate_fn


TORCH_DTYPES = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def is_adapter_directory(path: Path) -> bool:
    if not path.is_dir():
        return False

    has_config = (path / "adapter_config.json").exists()
    has_weights = (path / "adapter_model.safetensors").exists() or any(
        path.glob("adapter_model*.bin")
    )
    return has_config and has_weights


def load_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_adapter_path_from_root(path_str):
    if not path_str:
        return None

    root = Path(path_str)
    if not root.exists():
        raise ValueError(f"Adapter path does not exist: {root}")

    candidates = []
    if root.is_dir():
        candidates.append(root)
        candidates.append(root / "final_model")

        training_info_path = root / "training_info.json"
        if training_info_path.exists():
            try:
                training_info = load_json_file(training_info_path)
                latest_checkpoint = training_info.get("latest_checkpoint")
                if latest_checkpoint:
                    latest_checkpoint = Path(latest_checkpoint)
                    if not latest_checkpoint.is_absolute():
                        latest_checkpoint = root / latest_checkpoint.name
                    candidates.append(latest_checkpoint)
            except json.JSONDecodeError:
                pass

    for candidate in candidates:
        if is_adapter_directory(candidate):
            return str(candidate)

    checked = [str(c) for c in candidates]
    raise ValueError(
        "Could not find a valid adapter directory from --adapter-path. "
        "Expected adapter_config.json and adapter_model.safetensors in one of: "
        f"{checked}."
    )


def resolve_paths(args):
    model_path = args.model_path
    adapter_path = resolve_adapter_path_from_root(args.adapter_path) if args.adapter_path else None

    # If model-path actually points to a LoRA checkpoint/run folder, treat it as adapter input.
    if model_path and not adapter_path:
        try:
            adapter_candidate = resolve_adapter_path_from_root(model_path)
            if adapter_candidate:
                adapter_path = adapter_candidate
                model_path = None
        except ValueError:
            pass

    return model_path, adapter_path


def resolve_base_model(args, adapter_path=None):
    if args.base_model:
        return args.base_model

    if not adapter_path:
        return None

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    adapter_config = load_json_file(adapter_config_path)

    return adapter_config.get("base_model_name_or_path")


def resolve_model_class(args, model_reference):
    if args.model_class != "auto":
        return args.model_class

    model_reference = str(model_reference).lower()
    if "phi-4" in model_reference or "phi4" in model_reference:
        return "causal-lm"

    return "image-text-to-text"


def resolve_trust_remote_code(args, model_reference):
    if args.trust_remote_code is not None:
        return args.trust_remote_code

    model_reference = str(model_reference).lower()
    return "qwen" in model_reference or "phi" in model_reference


def build_validation_collate_fn(collate_fn, processor, device, dtype):
    collate_signature = inspect.signature(collate_fn)
    collate_kwargs = {"processor": processor}

    if "device" in collate_signature.parameters:
        collate_kwargs["device"] = device
    if "dtype" in collate_signature.parameters:
        collate_kwargs["dtype"] = dtype

    return partial(collate_fn, **collate_kwargs)


def load_generation_config(model, generation_source):
    try:
        return GenerationConfig.from_pretrained(generation_source)
    except OSError:
        return GenerationConfig.from_model_config(model.config)


def load_model_and_processor(args):
    model_path, adapter_path = resolve_paths(args)
    base_model = resolve_base_model(args, adapter_path=adapter_path)
    model_reference = model_path or base_model
    if model_reference is None:
        checked_path = None
        if adapter_path:
            checked_path = Path(adapter_path) / "adapter_config.json"
        raise ValueError(
            "Provide --model-path, or pass --adapter-path with adapter_config.json, "
            "or set --base-model explicitly. "
            f"Checked: {checked_path}. If you are using a shell variable like RUN_NAME, "
            "assign it before running python."
        )

    processor_path = args.processor_path or base_model or model_reference
    generation_source = args.generation_config_path or base_model or model_reference
    model_class = resolve_model_class(args, model_reference)
    trust_remote_code = resolve_trust_remote_code(args, model_reference)
    torch_dtype = TORCH_DTYPES[args.dtype]

    processor_kwargs = {}
    if trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)

    model_loader = (
        AutoModelForCausalLM
        if model_class == "causal-lm"
        else AutoModelForImageTextToText
    )
    model_kwargs = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = model_loader.from_pretrained(model_reference, **model_kwargs)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.to(args.device)
    model.eval()

    generation_config = load_generation_config(model, generation_source)
    return model, processor, generation_config

@torch.no_grad() 
def main(args):
    model, processor, generation_config = load_model_and_processor(args)

    # prepare dataset
    collate_fn = build_collate_fn(args.collate_fn)
    val_collate_fn = build_validation_collate_fn(
        collate_fn=collate_fn,
        processor=processor,
        device=args.device,
        dtype=TORCH_DTYPES[args.dtype] or torch.float32,
    )
    dataset = load_from_disk(args.data)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=val_collate_fn,
        num_workers=0,
        shuffle=False,
    )

    def infer(inputs):
        input_len = inputs["input_ids"].shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            generation_config=generation_config
        )
        output = output[:, input_len:]
        results = processor.batch_decode(output, skip_special_tokens=True)
        return results

    def flatten(x):
        return x[0] if isinstance(x, list) else x
    
    data_dict = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader):
            cnt += 1
            inputs, question, ids = batch
            results = infer(inputs.to(args.device))
            data_dict.append(
                {'id': flatten(ids), 'question': flatten(question), 'answer': flatten(results)}
            )

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='DriveLM Inference')
    parser.add_argument("--data", type=str, default="datasets/DriveLM_nuScenes/split/val")
    parser.add_argument("--collate_fn", type=str, default="drivelm_nus_paligemma_collate_fn_val")
    parser.add_argument("--output", type=str, default="datasets/DriveLM_nuScenes/refs/infer_results.json")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a full fine-tuned model directory.")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model name or path. Optional when --adapter-path contains adapter_config.json.")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help=(
                            "Path to LoRA/PEFT weights. Supports direct adapter folders "
                            "(contains adapter_config.json), run output folders (contains final_model), "
                            "or direct checkpoint folders like outputs/.../epoch-3."
                        ))
    parser.add_argument("--processor-path", type=str, default=None,
                        help="Processor/tokenizer source. Defaults to the base model.")
    parser.add_argument("--generation-config-path", type=str, default=None,
                        help="GenerationConfig source. Defaults to the base model or model path.")
    parser.add_argument("--model-class", choices=["auto", "image-text-to-text", "causal-lm"], default="auto",
                        help="Model loader to use. Leave as auto for repo defaults.")
    parser.add_argument("--dtype", choices=sorted(TORCH_DTYPES.keys()), default="auto",
                        help="Optional model dtype override.")
    parser.add_argument("--max-new-tokens", type=int, default=1000,
                        help="Maximum number of tokens to generate per sample.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None,
                        help="Override trust_remote_code. Defaults to auto for Qwen and Phi models.")
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_args())