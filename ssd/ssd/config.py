import os
import json
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT


MULTIMODAL_MODEL_TYPES = {"qwen3_vl", "qwen2_5_vl", "qwen2_vl"}


def get_text_config(hf_config: AutoConfig) -> AutoConfig:
    text_config = getattr(hf_config, "text_config", hf_config)
    if text_config is not hf_config and not hasattr(text_config, "tie_word_embeddings"):
        text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    return text_config


def get_max_position_embeddings(hf_config: AutoConfig) -> int | None:
    return getattr(hf_config, "max_position_embeddings", None)


def has_lora_adapter(path: Path) -> bool:
    if not (path / "adapter_config.json").exists():
        return False
    return (path / "adapter_model.safetensors").exists() or any(path.glob("adapter_model*.bin"))


def resolve_local_path(reference: str | None, anchor: Path | None = None) -> str | None:
    if not reference:
        return None
    ref_path = Path(reference)
    if ref_path.is_absolute() and ref_path.exists():
        return str(ref_path)
    roots = [Path.cwd()]
    if anchor is not None:
        roots.extend([anchor, anchor.parent, anchor.parent.parent])
    for root in roots:
        candidate = (root / ref_path).resolve()
        if candidate.exists():
            return str(candidate)
    return reference


def resolve_lora_base_model(lora_path: str) -> str | None:
    adapter_dir = Path(lora_path)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    with adapter_config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    return resolve_local_path(adapter_config.get("base_model_name_or_path"), adapter_dir)


def resolve_model_and_lora_path(model: str, lora_path: str | None) -> tuple[str, str | None]:
    resolved_lora_path = resolve_local_path(lora_path) if lora_path else None
    model_path = Path(model)
    if model_path.exists() and has_lora_adapter(model_path):
        resolved_lora_path = str(model_path)
        base_model = resolve_lora_base_model(resolved_lora_path)
        if not base_model:
            raise ValueError(f"LoRA adapter at {model_path} is missing base_model_name_or_path")
        model = base_model
    return model, resolved_lora_path

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
    is_vlm: bool = False
    lora_path: str | None = None
    draft_lora_path: str | None = None

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
        self.model, self.lora_path = resolve_model_and_lora_path(self.model, self.lora_path)
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        full_hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        self.is_vlm = self.is_vlm or getattr(full_hf_config, "model_type", None) in MULTIMODAL_MODEL_TYPES
        self.hf_config = get_text_config(full_hf_config) if self.is_vlm else full_hf_config
        max_position_embeddings = get_max_position_embeddings(self.hf_config)
        if max_position_embeddings is not None:
            self.max_model_len = min(self.max_model_len, max_position_embeddings) 
        if self.speculate: 
            self.draft, self.draft_lora_path = resolve_model_and_lora_path(self.draft, self.draft_lora_path)
            draft = self.draft
            full_draft_hf_config = AutoConfig.from_pretrained(draft, trust_remote_code=True)
            draft_is_vlm = getattr(full_draft_hf_config, "model_type", None) in MULTIMODAL_MODEL_TYPES
            if self.is_vlm:
                assert draft_is_vlm, "ERROR in Config: VLM target requires VLM draft"
            self.draft_hf_config = get_text_config(full_draft_hf_config) if draft_is_vlm else full_draft_hf_config
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
