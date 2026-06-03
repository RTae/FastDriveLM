import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT


MULTIMODAL_MODEL_TYPES = {"qwen3_vl", "qwen2_5_vl", "qwen2_vl"}


def _get_model_config(model: str, is_vlm: bool):
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if not (is_vlm or getattr(hf_config, "model_type", None) in MULTIMODAL_MODEL_TYPES):
        return hf_config, False

    text_config = getattr(hf_config, "text_config", hf_config)
    for attr in ("vision_config", "vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id"):
        if hasattr(hf_config, attr) and not hasattr(text_config, attr):
            setattr(text_config, attr, getattr(hf_config, attr))
    if not hasattr(text_config, "tie_word_embeddings"):
        text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
    if getattr(text_config, "model_type", None) == "qwen3_vl_text":
        text_config.model_type = "qwen3"
    return text_config, True


def _cap_max_model_len(max_model_len: int, hf_config: AutoConfig) -> int:
    max_position_embeddings = getattr(hf_config, "max_position_embeddings", None)
    return min(max_model_len, max_position_embeddings) if max_position_embeddings else max_model_len

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

    # LoRA adapter paths
    lora_path: str | None = None
    draft_lora_path: str | None = None

    # VLM (Vision Language Model)
    is_vlm: bool = False

    # Ablation study flags
    use_prefix_caching: bool = True
    attn_backend: str = "sparge_sage"   # valid value: "sparge_sage"
    sparge_topk: float = 0.5      # SpargeAttn sparsity ratio for the mixed prefill path.

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

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config, self.is_vlm = _get_model_config(model, self.is_vlm)
        self.max_model_len = _cap_max_model_len(self.max_model_len, self.hf_config)
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config, draft_is_vlm = _get_model_config(draft, self.is_vlm)
            if self.is_vlm:
                assert draft_is_vlm, "ERROR in Config: VLM target requires VLM draft"
            self.max_model_len = _cap_max_model_len(self.max_model_len, self.draft_hf_config)
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

        assert self.attn_backend == "sparge_sage", \
            f"attn_backend must be 'sparge_sage', got {self.attn_backend!r}"
        self.use_prefix_caching = False
        if self.is_vlm:
            self.use_prefix_caching = False
