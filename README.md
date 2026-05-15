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
    --output outputs/qwen3vl/<run_name>/infer_results.json
└── DriveLM_nuScenes
    ├── nuscenes
    │    └── samples
    ├── QA_dataset_nus
    │    └── v1_1_train_nus.json
    ├── refs
    --output outputs/paligemma/<run_name>/infer_results.json
    │    ├── val_cot.json
    │    └── val_qa_style.json
    └── split
         ├── train/
         └── val/
```
    --collate_fn <COLLATE_FN_VAL> \
    --output <OUTPUT_JSON>

Single GPU:
```bash
python tools/finetune.py <CONFIG>
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --output outputs/qwen3vl/<run_name>/infer_results.json

# PaliGemma fine-tuned checkpoint
```

Examples:
```bash
# PaliGemma — single GPU

# Full model checkpoint directory
python tools/inference.py \
    --model-path <MODEL_DIR> \
    --processor-path <BASE_MODEL_OR_PROCESSOR_DIR> \
    --data datasets/DriveLM_nuScenes/split/val \
    --collate_fn <COLLATE_FN_VAL> \
    --output infer_results.json
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

If you do not activate the virtual environment first, replace `python` with `.venv/bin/python`.

For LoRA checkpoints, `--base-model` is optional when `adapter_config.json` is present in `final_model`; the script will read the base model path from that file automatically.

```bash
# Qwen2.5-VL / Qwen3-VL LoRA checkpoint
RUN_NAME=<run_name> python tools/inference.py \
    --adapter-path outputs/qwen3vl/$RUN_NAME/final_model \
    --data datasets/DriveLM_nuScenes/split/val \
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --output outputs/qwen3vl/$RUN_NAME/infer_results.json

# PaliGemma LoRA checkpoint
RUN_NAME=<run_name> python tools/inference.py \
    --adapter-path outputs/paligemma/$RUN_NAME/final_model \
    --data datasets/DriveLM_nuScenes/split/val \
    --collate_fn drivelm_nus_paligemma_collate_fn_val \
    --output outputs/paligemma/$RUN_NAME/infer_results.json

# Full model checkpoint directory
python tools/inference.py \
    --model-path <MODEL_DIR> \
    --processor-path <BASE_MODEL_OR_PROCESSOR_DIR> \
    --data datasets/DriveLM_nuScenes/split/val \
    --collate_fn <COLLATE_FN_VAL> \
    --output <OUTPUT_JSON>
```

Use the matching validation collate function for your model:

- `drivelm_nus_paligemma_collate_fn_val`
- `drivelm_nus_qwen3vl_collate_fn_val`
- `drivelm_nus_phi4_collate_fn_val`

### Evaluate

Run evaluation on the JSON file produced by `tools/inference.py`.

`--src` should point to your inference output, while `--tgt` should point to the validation reference file.

```bash
# Qwen2.5-VL / Qwen3-VL predictions
python tools/evaluation.py \
    --src outputs/qwen3vl/<run_name>/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json

# PaliGemma predictions
python tools/evaluation.py \
    --src outputs/paligemma/<run_name>/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json

# Generic form
python tools/evaluation.py \
    --src <OUTPUT_JSON> \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json
```

Note: `tools/evaluation.py` imports `language_evaluation`. Make sure that dependency is installed in your environment before running evaluation.