# FastDriveLM

## Pre-requisites
- Python 3.12+
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for virtual environment management and dependency installation.

## Installation
1. Install Python dependencies and set up virtual environment
```bash
uv sync
```

## How to enter the virtual environment
```bash
source .venv/bin/activate
```

## Fine-tune model (Optional)

Supported models:

| Model | Config |
|-------|--------|
| PaliGemma | `configs/paligemma/paligemma_drivelm_config.py` |
| Phi-4 Multimodal | `configs/phi4/phi4_drivelm_1xb1-lora_config.py` |
| Qwen2.5-VL / Qwen3-VL | `configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py` |

### Prepare data

1. Download data
```bash
make download_data
```

2. Convert data into fine-tune format
```bash
make create_dataset
```
after running this script, the data will be organized as follows:
```bash
/datasets
└── DriveLM_nuScenes
    ├── nuscenes
    │    └── samples
    ├── QA_dataset_nus
    │    └── v1_1_train_nus.json
    ├── refs
    │    ├── train_cot.json
    │    ├── val_cot.json
    │    └── val_qa_style.json
    └── split
         ├── train/
         └── val/
```

### Run fine-tuning

Single GPU:
```bash
python tools/finetune.py <CONFIG>
```

Multi-GPU (DDP via Accelerate):
```bash
accelerate launch --num_processes=<NUM_GPUS> tools/finetune.py <CONFIG>
```

Examples:
```bash
# PaliGemma — single GPU
python tools/finetune.py configs/paligemma/paligemma_drivelm_config.py

# Phi-4 — single GPU
python tools/finetune.py configs/phi4/phi4_drivelm_1xb1-lora_config.py

# Qwen3-VL — single GPU
python tools/finetune.py configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py

# Qwen3-VL — 2 GPUs
make fine_tune_qwen3vl
```

Checkpoints and logs are saved under `outputs/<model>/<run_name>/`.

To switch to a different Qwen3-VL size, edit `model_name` in the config:
```python
# configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py
model_name: str = "Qwen/Qwen3-VL-8B-Instruct"   # default
# model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
```

### Run inference

Use `--model-path` as the single model directory argument.

`--model-path` supports:

- full model directory (`config.json` + model weights)
- checkpoint-only full weights directory (`pytorch_model.bin`/`model.safetensors` with `--base-model`)
- LoRA adapter directory (`adapter_config.json` + adapter weights)

For adapter and checkpoint-only directories, `--base-model` is optional when base model metadata exists in `adapter_config.json` (or README front matter); otherwise provide `--base-model` explicitly.

Also, use the matching validation collate function for your model:

- `drivelm_nus_paligemma_collate_fn_val`
- `drivelm_nus_qwen3vl_collate_fn_val`
- `drivelm_nus_phi4_collate_fn_val`

```bash
# Qwen2.5-VL / Qwen3-VL LoRA epoch checkpoint (direct path)
OUTPUT_MODEL=outputs/qwen3vl/epoch-3
COLLATE_FN=drivelm_nus_qwen3vl_collate_fn_val
python tools/inference.py \
    --model-path $OUTPUT_MODEL \
    --collate_fn $COLLATE_FN \
    --data datasets/DriveLM_nuScenes/split/val \
    --output $OUTPUT_MODEL/infer_results.json
```

### Evaluate

Run evaluation on the JSON file produced by `tools/inference.py`.

`tools/evaluation.py` now includes a built-in BLEU-1..4, ROUGE-L, and CIDEr scorer, so no extra evaluation package is required.

`--src` should point to your inference output, while `--tgt` should point to the validation reference file.

```bash
# Qwen2.5-VL / Qwen3-VL predictions
RUN_NAME=qwen3vl-2026-05-14_20-38
OUTPUT_MODEL=outputs/qwen3vl/$RUN_NAME/final_model
python tools/evaluation.py \
    --src $OUTPUT_MODEL/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json
```