import torch
from accelerate import PartialState
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from ..registry import register_prepare_model_and_processor


@register_prepare_model_and_processor
def prepare_model_and_processor_qwen3vl(config):
    """
    Prepare a Qwen2.5-VL / Qwen3-VL model and processor for fine-tuning on
    DriveLM-nuScenes.

    Recommended model names:
      - "Qwen/Qwen2.5-VL-7B-Instruct"   (Qwen2.5 VL family)
      - "Qwen/Qwen2.5-VL-3B-Instruct"
      - "Qwen/Qwen3-VL-8B-Instruct"      (Qwen3 VL family)
    """
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=config.dtype,
    )

    if config.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=config.dtype,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = {"": PartialState().local_process_index}

    if config.use_flash_attention:
        assert config.dtype == torch.bfloat16, (
            "Flash attention only supports bfloat16"
        )
        load_kwargs["attn_implementation"] = "flash_attention_2"
        load_kwargs.setdefault(
            "device_map", {"": PartialState().local_process_index}
        )
    else:
        load_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name, **load_kwargs
    )

    if getattr(config, "peft_name", None):
        model = PeftModel.from_pretrained(model, config.peft_name)
        model = model.merge_and_unload()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if config.use_lora:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_r * 2,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Optionally unfreeze the visual encoder as well
        if getattr(config, "tune_vision_encoder", False):
            for param in model.model.visual.parameters():
                param.requires_grad = True
    else:
        # Full fine-tune
        for param in model.parameters():
            param.requires_grad = True

    return model, processor
