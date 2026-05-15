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

For LoRA checkpoints, `--base-model` is optional when `adapter_config.json` is present in the adapter folder; the script will read the base model path from that file automatically.

`--adapter-path` supports:

- a direct adapter folder (for example `outputs/.../epoch-3` or `outputs/.../final_model`)
- a run output folder that contains `final_model` (for example `outputs/qwen3vl/<run_name>`)

Also, use the matching validation collate function for your model:

- `drivelm_nus_paligemma_collate_fn_val`
- `drivelm_nus_qwen3vl_collate_fn_val`
- `drivelm_nus_phi4_collate_fn_val`

```bash
# Qwen2.5-VL / Qwen3-VL LoRA run folder (contains final_model)
RUN_DIR=outputs/qwen3vl/qwen3vl-2026-05-14_20-38
COLLATE_FN=drivelm_nus_qwen3vl_collate_fn_val
python tools/inference.py \
    --adapter-path $RUN_DIR \
    --collate_fn $COLLATE_FN \
    --data datasets/DriveLM_nuScenes/split/val \
    --output $RUN_DIR/infer_results.json

# Qwen2.5-VL / Qwen3-VL LoRA epoch checkpoint (direct path)
EPOCH_DIR=outputs/qwen3vl/epoch-3
python tools/inference.py \
    --adapter-path $EPOCH_DIR \
    --collate_fn $COLLATE_FN \
    --data datasets/DriveLM_nuScenes/split/val \
    --output $EPOCH_DIR/infer_results.json
```

### Evaluate

Run evaluation on the JSON file produced by `tools/inference.py`.

`tools/evaluation.py` now includes a built-in BLEU-1..4, ROUGE-L, and CIDEr scorer, so no extra evaluation package is required.

`--src` should point to your inference output, while `--tgt` should point to the validation reference file.

```bash
# Qwen2.5-VL / Qwen3-VL predictions
RUN_NAME=qwen3vl-2026-05-14_20-38
python tools/evaluation.py \
    --src outputs/qwen3vl/$RUN_NAME/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json

# PaliGemma predictions
RUN_NAME=<run_name>
python tools/evaluation.py \
    --src outputs/paligemma/$RUN_NAME/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json

# Generic form
python tools/evaluation.py \
    --src <OUTPUT_JSON> \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json
```