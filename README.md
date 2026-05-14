# FastDriveLM

## Pre-requisites
- Python 3.12+
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for virtual environment management and dependency installation.

## Installation
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

# Qwen2.5-VL / Qwen3-VL — single GPU
python tools/finetune.py configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py

# Qwen2.5-VL / Qwen3-VL — 2 GPUs
accelerate launch --num_processes=2 tools/finetune.py configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py
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

```bash
python tools/inference.py \
    --data datasets/DriveLM_nuScenes/split/val \
    --collate_fn drivelm_nus_paligemma_collate_fn_val \
    --output datasets/DriveLM_nuScenes/refs/infer_results.json
```

### Evaluate

```bash
python tools/evaluation.py \
    --src datasets/DriveLM_nuScenes/refs/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json
```