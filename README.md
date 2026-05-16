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
| Qwen3-VL | `configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py` |
| Qwen3-VL-Draft | `configs/qwen3/qwen3vl_drivelm-draft_1xb1-lora_config.py` |

| Metric | qwen3vl (baseline) | qwen3vl_draft |
|---|---|---|
| **Final Score** | **0.6443** | **0.6213** |
| Accuracy | 0.9084 | 0.8890 |
| Match Score | 54.51 | 48.997 |
| BLEU-1 | 0.7434 | 0.7701 |
| BLEU-2 | 0.7030 | 0.7255 |
| BLEU-3 | 0.6637 | 0.6824 |
| BLEU-4 | 0.6255 | 0.6404 |
| ROUGE-L | 0.7244 | 0.7240 |
| CIDEr | 0.2938 | 0.2625 |
| test len | 45064 | 48975 |
| ref len | 48829 | 48829 |
| ratio | 0.923 | 1.003 |

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

For adapter and checkpoint-only directories, a base model is required. The script reads it from `adapter_config.json` when available, or you can pass `--base-model` explicitly.

Also, use the matching validation collate function for your model:

- `drivelm_nus_paligemma_collate_fn_val`
- `drivelm_nus_qwen3vl_collate_fn_val`
- `drivelm_nus_phi4_collate_fn_val`

```bash
make inference_qwen3vl OUTPUT_MODEL=outputs/qwen3vl/epoch-3

# or run directly with python

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
python tools/evaluation.py \
    --src ./outputs/qwen3vl_draft/qwen3vl-2b-draft-2026-05-16_15-31/epoch-3/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json
```