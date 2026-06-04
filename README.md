# FastDriveLM

## Pre-requisites
- Python 3.12+
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for virtual environment management and dependency installation.

## Installation
1. Install Python dependencies and set up virtual environment
```bash
uv sync
```

2. Optional: install the local sparse-attention extension if you want the custom speculative VLM path with SpargeAttn prefill

Preferred `uv` path:

```bash
uv sync --extra sparse-attn
```

If you only want the local extension package without a full extra sync, use:

```bash
uv pip install --python .venv/bin/python -e ./spas_sage_attn
```

Notes:

- `spas_sage_attn` is not needed for the plain vLLM workflow.
- `spas_sage_attn` is needed for the custom speculative VLM path documented below.
- `sageattention==1.0.6` is part of the main project dependencies and is used for decode and verify in the custom speculative VLM path.
- Building the sparse-attention extension needs CUDA, `torch`, `ninja`, and enough free disk space for temporary build artifacts.

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

There are two active inference tracks in this repo:

- `tools/inference_sd_vlm_vllm.py`: the vLLM track, for plain vLLM features such as attention backend selection, prefix caching, and supported speculative modes.
- `tools/inference_custom_sd_vlm.py`: the custom speculative VLM track, using SpargeAttn for supported prefill cases and SageAttention for decode and verify.

The custom speculative VLM path is the one that matches the current sparse-attention direction of this repo.

### Runtime environment for vLLM

Use the project virtual environment and export the CUDA/Torch library paths before running either vLLM script.

```bash
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$PWD/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

### vLLM script

`tools/inference_sd_vlm_vllm.py` is the main vLLM entrypoint for this repo.
It runs the vLLM track on the target model.

Optional runtime features:

- `--attn-backend auto|FLASH_ATTN|FLASHINFER|TRITON_ATTN|FLEX_ATTENTION`: explicitly select the vLLM attention backend.
- `--use-prefix-caching` / `--no-use-prefix-caching`: explicitly enable or disable vLLM prefix caching.
- `--enable-speculative-decoding --speculative-method ngram --spec-k <int>`: enable ngram-based speculative decoding.
- `--speculative-method ngram_gpu --spec-k <int>`: enable GPU ngram speculative decoding when supported by the installed vLLM build.
- `--enable-speculative-decoding --speculative-method draft_model --draft-model <path> --spec-k <int>`: enable draft-model speculative decoding when the installed vLLM build and model type support it.
- `--speculative-method eagle --eagle-model <path>` or `--speculative-method eagle3 --eagle3-model <path>`: enable vLLM EAGLE/EAGLE3 speculative decoding with a compatible EAGLE head/model.
- `--speculative-method medusa --medusa-model <path>`: enable vLLM Medusa-head speculative decoding with a compatible Medusa-head model.
- `--enable-speculative-decoding --speculative-method suffix ...`: enable suffix decoding when the extra dependency is installed.

Current status with `vllm 0.19.1`:

- plain vLLM inference works
- prefix caching works
- `ngram` speculative decoding works on Qwen3-VL
- draft-model speculative decoding is not supported for multimodal models such as Qwen3-VL
- EAGLE, EAGLE3, and Medusa are exposed as vLLM pass-through speculative methods, but they require compatible speculator/head checkpoints; they are not the same as the repo's 2B Qwen3-VL LoRA draft adapter
- `suffix` speculative decoding requires `arctic-inference==0.1.1`

In practice, the common workflow is:

- use `tools/inference_sd_vlm_vllm.py` directly for a single vLLM smoke or full run
- use `scripts/run_vllm_report.sh` when you want baseline + vLLM ablation runs + one summary table
- use `tools/inference_custom_sd_vlm.py` when you want the custom speculative VLM runtime with SpargeAttn prefill and SageAttention decode/verify

#### Main smoke test

Plain vLLM smoke test.

```bash
python tools/inference_sd_vlm_vllm.py \
    --target-model outputs/qwen3vl \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_sd_vlm_vllm_base_smoke.json \
    --metrics \
    --metrics-output outputs/qwen3vl/metrics_sd_vlm_vllm_base_smoke.json \
    --max-samples 10 \
    --max-new-tokens 64
```

#### Main full test

Run the same script on the full validation split.

```bash
python tools/inference_sd_vlm_vllm.py \
    --target-model outputs/qwen3vl \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_sd_vlm_vllm.json \
    --metrics \
    --metrics-output outputs/qwen3vl/metrics_sd_vlm_vllm.json \
    --max-new-tokens 128
```

### Ablation runner

If you want one command that runs the baseline plus the vLLM ablation modes and prints a summary table at the end, use the shell wrapper:

```bash
scripts/run_vllm_report.sh
```

#### Smoke test

The defaults already cover the common smoke setup:

```bash
scripts/run_vllm_report.sh \
    --run-label smoke \
    --max-samples 10 \
    --max-new-tokens 64
```

#### Full test

Run the full validation split and write `*_full.json` metrics files:

```bash
scripts/run_vllm_report.sh \
    --full-test \
    --run-label full \
    --max-new-tokens 128
```

This wrapper runs the Hugging Face baseline, then the following vLLM ablation modes into `outputs/qwen3vl/`, and prints a final summary table from the generated metrics files:

- `base`: plain vLLM
- `prefix`: plain vLLM with `--use-prefix-caching`
- `spec`: ngram speculative decoding
- `full`: ngram speculative decoding with `--use-prefix-caching`

You can also run only a subset:

```bash
scripts/run_vllm_report.sh \
    --modes prefix full \
    --run-label smoke \
    --max-samples 10 \
    --max-new-tokens 64
```

`ngram` speculative decoding disables async scheduling in the current vLLM build. That warning is expected.

Draft-model speculative decoding is still blocked for multimodal Qwen3-VL, and suffix decoding needs `arctic-inference==0.1.1` before it can be tested.

### Compare vLLM speculative methods

Use `tools/compare_speculative_decoding.py` on a GPU machine to run several vLLM methods on a small dataset subset with the same sample count, token budget, and output directory, then print one comparison table. This comparison runner is subset-only by default and requires `--max-samples N` with `N > 0` unless you intentionally pass `--allow-full-dataset`.

Model-free comparison:

```bash
python tools/compare_speculative_decoding.py \
    --target-model outputs/qwen3vl \
    --data datasets/DriveLM_nuScenes/split/val \
    --output-dir outputs/qwen3vl/spec_compare \
    --run-label smoke \
    --methods baseline ngram ngram_gpu suffix \
    --max-samples 10 \
    --max-new-tokens 64 \
    --warmup-steps 2 \
    --metrics
```

Comparison including vLLM-provided model/head methods:

```bash
python tools/compare_speculative_decoding.py \
    --target-model outputs/qwen3vl \
    --data datasets/DriveLM_nuScenes/split/val \
    --output-dir outputs/qwen3vl/spec_compare \
    --run-label heads \
    --methods baseline ngram eagle eagle3 medusa \
    --eagle-model /path/to/eagle/head-or-model \
    --eagle3-model /path/to/eagle3/head-or-model \
    --medusa-model /path/to/medusa/head-model \
    --max-samples 10 \
    --max-new-tokens 64 \
    --warmup-steps 2 \
    --metrics
```

If a method is unsupported by your installed vLLM build or incompatible with Qwen3-VL, the comparison runner records the failure and continues by default. Use `--no-continue-on-error` when you want the first failure to stop the run.

### Custom speculative VLM script

`tools/inference_custom_sd_vlm.py` is the custom speculative VLM entrypoint.

Current behavior:

- prefill uses `spas_sage_attn` when the incoming prefill shape matches the supported single-sequence case
- other prefill shapes fall back to `sageattention`
- decode uses `sageattention`
- verify uses `sageattention`

This path is not the async 2-GPU SSD mode. It is a colocated speculative VLM path built from the same codebase, currently run in eager mode.

#### Smoke test

```bash
python tools/inference_custom_sd_vlm.py \
    --target-model outputs/qwen3vl \
    --draft-model outputs/qwen3vl_draft \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_custom_sd_vlm_sparge_sage_smoke.json \
    --metrics \
    --metrics-output outputs/qwen3vl/infer_results_custom_sd_vlm_sparge_sage_smoke.metrics.json \
    --max-samples 10 \
    --max-new-tokens 64
```

#### Full test

```bash
python tools/inference_custom_sd_vlm.py \
    --target-model outputs/qwen3vl \
    --draft-model outputs/qwen3vl_draft \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_custom_sd_vlm_sparge_sage.json \
    --metrics \
    --metrics-output outputs/qwen3vl/infer_results_custom_sd_vlm_sparge_sage.metrics.json \
    --max-new-tokens 128
```

Use this path when your goal is: speculative decoding with VLM plus sparse attention on prefill and SageAttention on decode.

### Baseline script

`tools/inference.py` is the baseline runner.

#### Smoke test

```bash
python tools/inference.py \
    --model-path outputs/qwen3vl \
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_baseline_smoke.json \
    --metrics \
    --metrics-output outputs/qwen3vl/metrics_baseline_smoke.json \
    --max-samples 10 \
    --max-new-tokens 64
```

#### Full test

```bash
python tools/inference.py \
    --model-path outputs/qwen3vl \
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results_baseline_full.json \
    --metrics \
    --metrics-output outputs/qwen3vl/metrics_baseline_full.json \
    --max-new-tokens 128
```

### Compare vLLM against the baseline

The vLLM script and the baseline script write metrics in the same JSON schema, so you can compare the `summary` blocks directly. The example below compares the `full` ablation smoke run against the baseline smoke test.

```bash
python - <<'PY'
import json

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)["summary"]

new_vllm = load("outputs/qwen3vl/metrics_sd_vlm_vllm_full_smoke.json")
baseline = load("outputs/qwen3vl/metrics_baseline_smoke.json")
keys = [k for k in new_vllm if k != "num_samples"]
print(f"{'metric':<45} {'new_vllm':>14} {'baseline':>14}")
print("-" * 79)
for key in keys:
    print(f"{key:<45} {new_vllm[key]:>14.4f} {baseline[key]:>14.4f}")
PY
```

`tools/inference_vllm.py` is still available as a separate pure-vLLM reference script, but the intended comparison here is:

- baseline: `tools/inference.py`
- vLLM path: `tools/inference_sd_vlm_vllm.py`

`tools/inference_sd_vlm_vllm.py` is the maintained vLLM path documented above. `tools/inference_vllm.py` remains a separate reference script, but it is not the main entrypoint described in this README.

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
