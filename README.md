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

## Download pre-trained model weights
Download pre-trained model weights (if not already available):
```bash
make download_models MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
# or
make download_models MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
```

Download fine-tuning model, when you download a model you need to check a adperter_config.json file that the path for the base model is correct, if not, you need to download the base model and put it in the correct path. The base model should be the same as the one used for fine-tuning.
| Model | link |
|-------|------|
| Qwen3-VL-8B (baseline) | [Google Drive](https://drive.google.com/file/d/1oIzpvCMWGj5O8cwTRyc8he5YA6opLozH/view?usp=sharing)
| Qwen3-VL-2B (Draft) | [Google Drive](https://drive.google.com/file/d/1UjANF-QnXxT46p9z9xcm9FjaO55pQhz-/view?usp=sharing) |

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


## Prepare data

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

## Run inference

### Run Qwen3-VL with SSD-style target/draft setup

For the multimodal speculative path in the `ssd/` submodule, use:

- target model: `outputs/qwen3vl`
- draft model: `outputs/qwen3vl_draft`

These are treated as LoRA adapter directories. The SSD multimodal path reads the base model from each `adapter_config.json` and resolves:

- `outputs/qwen3vl` -> `base_models/Qwen/Qwen3-VL-8B-Instruct`
- `outputs/qwen3vl_draft` -> `base_models/Qwen/Qwen3-VL-2B-Instruct`

Run from the repo root so the workspace `.venv` and the `ssd/` package subpath are both visible:

```bash
PYTHONPATH=$PWD/ssd \
SSD_HF_CACHE=$PWD/base_models \
SSD_DATASET_DIR=$PWD/datasets \
.venv/bin/python - <<'PY'
from datasets import load_from_disk
from ssd import LLM, SamplingParams

sample = load_from_disk('datasets/DriveLM_nuScenes/split/val')[0]
question = sample['conversations'][0]['value']
images = sample['image_paths']

llm = LLM(
    'outputs/qwen3vl',
    speculate=True,
    draft='outputs/qwen3vl_draft',
    max_model_len=2048,
)

outputs, _ = llm.generate(
    [{'text': question, 'images': images}],
    [SamplingParams(temperature=0.0, max_new_tokens=80)],
    use_tqdm=False,
)
print(outputs[0]['text'])
PY
```

Notes:

- This uses the VLM target/draft pair you trained.
- `speculate=True` with `draft='outputs/qwen3vl_draft'` uses the target/draft VLM pair.
- `draft_async=True` now selects the first GPU-split VLM SSD path, with target on one GPU and draft on another.
- The current VLM SSD path is still an early implementation: greedy decoding, one request at a time, and Python/HuggingFace-based speculate/verify rather than the full optimized text-only kernel stack.

### Run validation-set inference with the VLM SSD path

Use the dedicated root-level script to run the DriveLM validation split with:

- target: `outputs/qwen3vl`
- draft: `outputs/qwen3vl_draft`

```bash
.venv/bin/python tools/inference_ssd_vlm.py \
    --target-model outputs/qwen3vl \
    --draft-model outputs/qwen3vl_draft \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_ssd.json
```

To print speed metrics during inference and save them as JSON:

```bash
.venv/bin/python tools/inference_ssd_vlm.py \
    --target-model outputs/qwen3vl \
    --draft-model outputs/qwen3vl_draft \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_ssd.json \
    --metrics \
    --metrics-output outputs/qwen3vl/infer_results_ssd.metrics.json
```

Useful options:

- `--max-samples 1` for a smoke test
- `--max-new-tokens 128` to control output length
- `--spec-k 4` to control lookahead
- `--num-gpus 2` for target GPU + draft GPU

Reported metrics follow standard LLM inference conventions:

- `ttft_sec`: time to first token
- `latency_sec`: end-to-end latency per sample
- `decode_throughput_tok_per_sec`: generated tokens divided by post-first-token decode time
- `end_to_end_throughput_tok_per_sec`: generated tokens divided by full request latency
- aggregate summary: average TTFT, average latency, average decode throughput, and total end-to-end throughput across the run

This script uses the current VLM `draft_async` path in the `ssd/` submodule. Right now it supports greedy decoding and processes one request at a time over the dataset.

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

**Also, make sure you have the pre-trained model weights downloaded for inference.**

```bash
make inference_qwen3vl OUTPUT_MODEL=./outputs/Qwen3-VL-8B-Instruct

# or run directly with python

OUTPUT_MODEL=./outputs/Qwen3-VL-8B-Instruct
python tools/inference.py \
    --model-path $OUTPUT_MODEL \
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --data datasets/DriveLM_nuScenes/split/val \
    --output $OUTPUT_MODEL/infer_results.json
```

## Evaluate

Run evaluation on the JSON file produced by `tools/inference.py`.

`tools/evaluation.py` now includes a built-in BLEU-1..4, ROUGE-L, and CIDEr scorer, so no extra evaluation package is required.

`--src` should point to your inference output, while `--tgt` should point to the validation reference file.

**Also, make sure you have the pre-trained model weights downloaded for inference.**

```bash
OUTPUT_MODEL=./outputs/Qwen3-VL-8B-Instruct
python tools/evaluation.py \
    --src $OUTPUT_MODEL/infer_results.json \
    --tgt datasets/DriveLM_nuScenes/refs/val_cot.json
```

## Fine-tune model (Optional)

Supported models:

| Model | Config |
|-------|--------|
| PaliGemma | `configs/paligemma/paligemma_drivelm_config.py` |
| Phi-4 Multimodal | `configs/phi4/phi4_drivelm_1xb1-lora_config.py` |
| Qwen3-VL | `configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py` |
| Qwen3-VL-Draft | `configs/qwen3/qwen3vl_drivelm-draft_1xb1-lora_config.py` |


### Run fine-tuning

1. Download pre-trained model weights (if not already available)
```bash
make download_models MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
# or
make download_models MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
```

2. Run fine-tuning with the config of your choice. Adjust `NUM_GPUS` for multi-GPU training.
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
