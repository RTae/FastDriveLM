import torch
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class DriveLMNusQwen3VLDraftConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "/root/autodl-tmp/models/Qwen/Qwen3-VL-2B-Instruct"
    model_preparation: str = "prepare_model_and_processor_qwen3vl"  
    collate_fn_train: str = "drivelm_nus_qwen3vl_collate_fn_train"   
    collate_fn_val: Optional[str] = "drivelm_nus_qwen3vl_collate_fn_val"  

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = "datasets/DriveLM_nuScenes/split/train"      
    dataset_val_name: str = "datasets/DriveLM_nuScenes/split/val"    

    peft_name: Optional[str] = None

    # ── Run / checkpoint ──────────────────────────────────────────────────────
    run_name: str = f"qwen3vl-2b-draft-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir: str = "outputs/qwen3vl_draft/" + f"{run_name}"

    # ── Training hyper-parameters ─────────────────────────────────────────────
    num_train_epochs: int = 3                        
    batch_size_per_gpu: int = 1                      
    gradient_accumulation_steps: int = 8             
    lr: float = 5e-5                                 
    warmup_steps: int = 100
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    # ── LoRA ──────────────────────────────────────────────────────────────────
    use_lora: bool = True
    lora_r: int = 32          # ← smaller than 8B (was 64), draft needs less capacity
    tune_vision_encoder: bool = False

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    dtype = torch.bfloat16
    quantization: bool = False
    use_flash_attention: bool = False

    resume_from_checkpoint: bool = False
    save_lora_adapter_when_checkpointing: bool = True

    save_steps: int = 500
    log_steps: int = 10
    print_steps: int = 10

    find_unused_parameters: bool = False

config = DriveLMNusQwen3VLDraftConfig()
