import argparse
import inspect
from functools import partial
from pathlib import Path
import json

import torch
from datasets import load_from_disk
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


def validate_model_path(model_path):
    if not model_path:
        raise ValueError("--model-path is required.")

    path = Path(model_path)
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
    # Full-model loading requires at least config + model weights in the same directory.
    return (path / "config.json").exists() and has_full_model_weights(path)


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
        return GenerationConfig.from_pretrained(generation_source, local_files_only=True)
    except OSError:
        return GenerationConfig.from_model_config(model.config)


def load_model_and_processor(args):
    model_path = validate_model_path(args.model_path)
    model_dir = Path(model_path)

    if can_load_full_model_from_dir(model_dir):
        model_reference = model_path
    elif has_full_model_weights(model_dir) and not (model_dir / "config.json").exists():
        raise ValueError(
            "Local-only mode requires config.json in --model-path when loading full weights. "
            "Please place config.json in the same directory as pytorch_model.bin/model.safetensors. "
            f"Checked: {model_dir}"
        )
    elif has_adapter_artifacts(model_dir):
        raise ValueError(
            "Detected adapter-only artifacts in --model-path. Local single-path loading cannot merge LoRA adapters "
            "without an external base model. Use a merged/full model directory instead (config.json + model weights + processor files). "
            f"Checked: {model_dir}"
        )
    else:
        raise ValueError(
            "Unsupported model directory. Expected one of: "
            "(1) full model files (config.json + model weights), or "
            "(2) adapter files if you plan to merge outside this script. "
            f"Checked: {model_dir}"
        )

    processor_path = args.processor_path or model_reference
    generation_source = args.generation_config_path or model_reference
    model_class = resolve_model_class(args, model_reference)
    trust_remote_code = resolve_trust_remote_code(args, model_reference)
    torch_dtype = TORCH_DTYPES[args.dtype]

    processor_kwargs = {"local_files_only": True}
    if trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)

    model_loader = (
        AutoModelForCausalLM
        if model_class == "causal-lm"
        else AutoModelForImageTextToText
    )
    model_kwargs = {}
    model_kwargs["local_files_only"] = True
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = model_loader.from_pretrained(model_reference, **model_kwargs)

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
    parser.add_argument("--model-path", type=str, required=True,
                        help=(
                            "Single local model directory path. Must contain full model artifacts: "
                            "config.json + model weights (+ processor files)."
                        ))
    parser.add_argument("--processor-path", type=str, default=None,
                        help="Optional local processor/tokenizer directory. Defaults to --model-path.")
    parser.add_argument("--generation-config-path", type=str, default=None,
                        help="Optional local GenerationConfig directory. Defaults to --model-path.")
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