import os
import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT


MULTIMODAL_MODEL_TYPES = {"qwen3_vl", "qwen2_5_vl", "qwen2_vl"}


def get_text_config(hf_config: AutoConfig) -> AutoConfig:
    return getattr(hf_config, "text_config", hf_config)


def get_max_position_embeddings(hf_config: AutoConfig) -> int | None:
    text_config = get_text_config(hf_config)
    return getattr(text_config, "max_position_embeddings", getattr(hf_config, "max_position_embeddings", None))


def has_adapter_artifacts(path: Path) -> bool:
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        return False
    has_weights = (path / "adapter_model.safetensors").exists() or any(path.glob("adapter_model*.bin"))
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


def resolve_local_reference(reference: str | None, model_dir: Path) -> str | None:
    if not reference:
        return None
    ref_path = Path(reference)
    if ref_path.is_absolute() and ref_path.exists():
        return str(ref_path)
    candidate_roots = [
        Path.cwd(),
        model_dir,
        model_dir.parent,
        model_dir.parent.parent,
    ]
    for root in candidate_roots:
        candidate = (root / ref_path).resolve()
        if candidate.exists():
            return str(candidate)
    return reference


def resolve_base_model_reference(model_dir: Path, explicit_base_model: str | None = None) -> str | None:
    if explicit_base_model:
        return resolve_local_reference(explicit_base_model, model_dir)

    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        return resolve_local_reference(adapter_config.get("base_model_name_or_path"), model_dir)

    return None


def resolve_model_artifacts(model: str, explicit_base_model: str | None = None) -> tuple[str, str | None, str | None]:
    model_dir = Path(model)
    resolved_model_path = model
    resolved_base_model_path = None
    adapter_path = None

    if has_adapter_artifacts(model_dir):
        adapter_path = str(model_dir)
        resolved_base_model_path = resolve_base_model_reference(model_dir, explicit_base_model)
        if not resolved_base_model_path:
            raise ValueError(
                "Detected adapter artifacts but base model is missing. "
                "Pass base_model=... or set base_model_name_or_path in adapter_config.json."
            )
        resolved_model_path = resolved_base_model_path
    elif can_load_full_model_from_dir(model_dir):
        resolved_model_path = model
    elif has_full_model_weights(model_dir):
        resolved_base_model_path = resolve_base_model_reference(model_dir, explicit_base_model)
        if not resolved_base_model_path:
            raise ValueError(
                "Found full model weights but missing config.json. "
                "Pass base_model=... for config loading."
            )
    return resolved_model_path, resolved_base_model_path, adapter_path

@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    is_multimodal: bool = False
    resolved_model_path: str | None = None
    resolved_base_model_path: str | None = None
    vlm_adapter_path: str | None = None
    resolved_draft_model_path: str | None = None
    resolved_draft_base_model_path: str | None = None
    vlm_draft_adapter_path: str | None = None

    # HF-backed multimodal generation. The custom SSD kernels are text-only today.
    vlm_trust_remote_code: bool = True
    vlm_attn_implementation: str = "sdpa"
    vlm_device_map: str | dict[str, Any] | None = None
    base_model: str | None = None

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)
        self.resolved_model_path = model

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.resolved_model_path, self.resolved_base_model_path, self.vlm_adapter_path = resolve_model_artifacts(
            model, self.base_model)

        config_source = self.resolved_base_model_path or self.resolved_model_path

        self.hf_config = AutoConfig.from_pretrained(config_source, trust_remote_code=self.vlm_trust_remote_code)
        self.is_multimodal = getattr(self.hf_config, "model_type", None) in MULTIMODAL_MODEL_TYPES

        max_position_embeddings = get_max_position_embeddings(self.hf_config)
        if max_position_embeddings is not None:
            self.max_model_len = min(self.max_model_len, max_position_embeddings)

        if self.is_multimodal:
            assert not self.use_eagle, "ERROR: EAGLE is only implemented for text-only Llama in this engine"
            assert not self.draft_async, "ERROR: multimodal Qwen-VL does not support async SSD yet; use speculate=True with a draft model for sync speculative decoding"
            if self.speculate:
                self.resolved_draft_model_path, self.resolved_draft_base_model_path, self.vlm_draft_adapter_path = resolve_model_artifacts(
                    self.draft)
                draft_config_source = self.resolved_draft_base_model_path or self.resolved_draft_model_path
                self.draft_hf_config = AutoConfig.from_pretrained(
                    draft_config_source, trust_remote_code=self.vlm_trust_remote_code)
                draft_model_type = getattr(self.draft_hf_config, "model_type", None)
                assert draft_model_type in MULTIMODAL_MODEL_TYPES, "ERROR: multimodal target requires a multimodal draft model"
                draft_max_position_embeddings = get_max_position_embeddings(self.draft_hf_config)
                if draft_max_position_embeddings is not None:
                    self.max_model_len = min(self.max_model_len, draft_max_position_embeddings)
            return

        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
            draft_max_position_embeddings = get_max_position_embeddings(self.draft_hf_config)
            if draft_max_position_embeddings is not None:
                self.max_model_len = min(self.max_model_len, draft_max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len
