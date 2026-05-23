import json
from glob import glob
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


def _load_adapter_config(lora_path: str) -> dict:
    config_path = Path(lora_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"LoRA adapter config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_adapter_state(lora_path: str) -> dict[str, torch.Tensor]:
    safetensor_path = Path(lora_path) / "adapter_model.safetensors"
    if safetensor_path.exists():
        state = {}
        with safe_open(str(safetensor_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        return state

    bin_files = sorted(glob(str(Path(lora_path) / "adapter_model*.bin")))
    if bin_files:
        return torch.load(bin_files[0], map_location="cpu")

    raise FileNotFoundError(f"No LoRA adapter weights found in {lora_path}")


def _strip_lora_suffix(key: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _normalize_module_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model."):
        return "model." + name[len("language_model."):]
    return name


def _get_module(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for part in module_name.split("."):
        current = getattr(current, part)
    return current


def _pattern_value(pattern: dict, module_name: str, default):
    if not pattern:
        return default
    best_key = None
    for key in pattern:
        if module_name == key or module_name.endswith(f".{key}"):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    return pattern[best_key] if best_key is not None else default


def _scaling_for_module(adapter_config: dict, module_name: str) -> float:
    rank = _pattern_value(adapter_config.get("rank_pattern", {}), module_name, adapter_config["r"])
    alpha = _pattern_value(adapter_config.get("alpha_pattern", {}), module_name, adapter_config["lora_alpha"])
    if adapter_config.get("use_rslora", False):
        return float(alpha) / (float(rank) ** 0.5)
    return float(alpha) / float(rank)


def _resolve_packed_parameter(weight_name: str, packed_modules_mapping: dict) -> tuple[str, str | int | None]:
    module_name = weight_name[: -len(".weight")]
    for source_name, (packed_name, shard_id) in packed_modules_mapping.items():
        suffix = f".{source_name}"
        if module_name.endswith(suffix):
            return f"{module_name[: -len(source_name)]}{packed_name}.weight", shard_id
    return weight_name, None


def _add_qkv_delta(param: nn.Parameter, module: nn.Module, delta: torch.Tensor, shard_id: str) -> None:
    assert shard_id in {"q", "k", "v"}
    local_delta = delta.chunk(module.tp_size, module.tp_dim)[module.tp_rank]
    if shard_id == "q":
        offset = 0
        shard_size = module.num_heads * module.head_size
    elif shard_id == "k":
        offset = module.num_heads * module.head_size
        shard_size = module.num_kv_heads * module.head_size
    else:
        offset = module.num_heads * module.head_size + module.num_kv_heads * module.head_size
        shard_size = module.num_kv_heads * module.head_size

    target = param.data.narrow(module.tp_dim, offset, shard_size)
    target.add_(local_delta.to(device=target.device, dtype=target.dtype))


def _add_merged_column_delta(param: nn.Parameter, module: nn.Module, delta: torch.Tensor, shard_id: int) -> None:
    local_delta = delta.chunk(module.tp_size, module.tp_dim)[module.tp_rank]
    offset = sum(module.output_sizes[:shard_id]) // module.tp_size
    shard_size = module.output_sizes[shard_id] // module.tp_size
    target = param.data.narrow(module.tp_dim, offset, shard_size)
    target.add_(local_delta.to(device=target.device, dtype=target.dtype))


def _add_unpacked_delta(param: nn.Parameter, module: nn.Module, delta: torch.Tensor) -> None:
    if tuple(param.data.shape) != tuple(delta.shape):
        tp_dim = getattr(module, "tp_dim", None)
        tp_rank = getattr(module, "tp_rank", 0)
        if tp_dim is None:
            raise ValueError(
                f"LoRA delta shape {tuple(delta.shape)} does not match parameter "
                f"shape {tuple(param.data.shape)} and module has no tp_dim"
            )
        shard_size = param.data.size(tp_dim)
        start_idx = tp_rank * shard_size
        delta = delta.narrow(tp_dim, start_idx, shard_size)
    param.data.add_(delta.to(device=param.device, dtype=param.dtype))


def _apply_lora_delta(model: nn.Module, weight_name: str, delta: torch.Tensor, packed_modules_mapping: dict) -> None:
    param_name, shard_id = _resolve_packed_parameter(weight_name, packed_modules_mapping)
    param = model.get_parameter(param_name)
    module = _get_module(model, param_name[: -len(".weight")])

    if isinstance(shard_id, str):
        _add_qkv_delta(param, module, delta, shard_id)
    elif isinstance(shard_id, int):
        _add_merged_column_delta(param, module, delta, shard_id)
    else:
        _add_unpacked_delta(param, module, delta)


def merge_lora_weights(model: nn.Module, lora_path: str, packed_modules_mapping: dict) -> None:
    adapter_config = _load_adapter_config(lora_path)
    if adapter_config.get("fan_in_fan_out", False):
        raise NotImplementedError("fan_in_fan_out LoRA adapters are not supported by the SSD merger")
    if adapter_config.get("use_dora", False):
        raise NotImplementedError("DoRA adapters are not supported by the SSD merger")

    state = _load_adapter_state(lora_path)
    a_suffixes = (".lora_A.weight", ".lora_A.default.weight")
    b_suffixes = (".lora_B.weight", ".lora_B.default.weight")

    lora_a = {}
    lora_b = {}
    for key, value in state.items():
        base_name = _strip_lora_suffix(key, a_suffixes)
        if base_name is not None:
            lora_a[_normalize_module_name(base_name)] = value
            continue
        base_name = _strip_lora_suffix(key, b_suffixes)
        if base_name is not None:
            lora_b[_normalize_module_name(base_name)] = value

    if set(lora_a) != set(lora_b):
        missing_a = sorted(set(lora_b) - set(lora_a))
        missing_b = sorted(set(lora_a) - set(lora_b))
        raise ValueError(f"Mismatched LoRA A/B weights. missing_A={missing_a}, missing_B={missing_b}")

    merged = 0
    for module_name in sorted(lora_a):
        a_weight = lora_a[module_name].to(torch.float32)
        b_weight = lora_b[module_name].to(torch.float32)
        delta = torch.matmul(b_weight, a_weight) * _scaling_for_module(adapter_config, module_name)
        _apply_lora_delta(model, f"{module_name}.weight", delta, packed_modules_mapping)
        merged += 1

    print(f"[lora] merged {merged} LoRA matrices from {lora_path}", flush=True)