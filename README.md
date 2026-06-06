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

3. Install vLLM (using uv add direclty didnt work)
```bash
uv pip install vllm --torch-backend=auto
```

4. Install LLM Compressor for INT4/AWQ quantization
```bash
uv pip install --no-deps llmcompressor==0.10.0.2
```

`llmcompressor` is installed as a quantization-time tool instead of a core
project dependency because the current vLLM lock pins `compressed-tensors`.
Using `--no-deps` keeps the inference environment stable.

Do not use `uv add llmcompressor` here. `uv add` tries to add llmcompressor to
the locked project dependency graph, but current llmcompressor releases conflict
with this repo's `vllm==0.22.0` dependencies. Use the `uv pip install --no-deps`
command above, or run:

```bash
make install_quant_deps
```

The quantization script includes a compatibility shim for the
`compressed_tensors.utils.match._match_name` vs `match_name` API difference seen
when `llmcompressor==0.10.0.2` is installed into the vLLM environment with
`--no-deps`.

Do not use `uv add llmcompressor` here. `uv add` tries to add llmcompressor to
the locked project dependency graph, but current llmcompressor releases conflict
with this repo's `vllm==0.22.0` dependencies. Use the `uv pip install --no-deps`
command above, or run:

```bash
make install_quant_deps
```

The quantization script includes a compatibility shim for the
`compressed_tensors.utils.match._match_name` vs `match_name` API difference seen
when `llmcompressor==0.10.0.2` is installed into the vLLM environment with
`--no-deps`.
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

### Quantize Qwen3-VL with LLM Compressor

The fine-tuned Qwen3-VL checkpoints in this repo are LoRA adapters. For INT4
AWQ inference, first merge the adapter into the local base model, then save a
compressed `compressed-tensors` checkpoint for vLLM:

```bash
python tools/quantize_qwen3vl_awq.py \
    --adapter-path outputs/qwen3vl \
    --base-model base_models/Qwen/Qwen3-VL-8B-Instruct \
    --data datasets/DriveLM_nuScenes/split/val \
    --output-dir outputs/qwen3vl_awq_int4 \
    --num-calibration-samples 256 \
    --max-seq-length 2048 \
    --scheme W4A16_ASYM \
    --awq-mapping-profile qwen3vl
```

If `outputs/qwen3vl/adapter_config.json` already points at the local base model,
`--base-model` can be omitted. If it points at `Qwen/Qwen3-VL-8B-Instruct`, the
script automatically prefers `base_models/Qwen/Qwen3-VL-8B-Instruct` when that
directory exists.

If quantization fails with LoRA shape mismatches like `4096` vs `2048`, the
adapter and base model sizes do not match. For example, the 8B Qwen3-VL adapter
must use `base_models/Qwen/Qwen3-VL-8B-Instruct`; the 2B draft adapter must use
`base_models/Qwen/Qwen3-VL-2B-Instruct`.

The default `--awq-mapping-profile qwen3vl` avoids llmcompressor's default
`v_proj -> o_proj` smoothing pair, which can be incompatible with Qwen3-VL GQA
projection shapes and produces repeated `skipping AWQ ... incompatible balance
layers` warnings. The affected Linear layers are still included in the final
INT4 quantization pass.

Run the quantized model directly with the existing vLLM inference script:

```bash
python tools/inference_vllm.py \
    --model-path outputs/qwen3vl_awq_int4 \
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl_awq_int4/infer_results_vllm.json \
    --metrics \
    --metrics-output outputs/qwen3vl_awq_int4/metrics_vllm.json \
    --warmup-steps 2 \
    --max-samples 20
```

To run unquantized and quantized vLLM inference back-to-back, evaluate both, and
write a compact speed/accuracy comparison:

```bash
python tools/compare_quantized_inference.py \
    --baseline-model outputs/qwen3vl \
    --baseline-base-model base_models/Qwen/Qwen3-VL-8B-Instruct \
    --quantized-model outputs/qwen3vl_awq_int4 \
    --data datasets/DriveLM_nuScenes/split/val \
    --refs datasets/DriveLM_nuScenes/refs/val_cot.json \
    --max-samples 20 \
    --warmup-steps 2
```

Makefile shortcuts:

```bash
make quantize_qwen3vl_awq
make compare_qwen3vl_quant
```

### Performance comparison

Both scripts write metrics in the same JSON schema, so you can compare them directly.

1. Run standard HuggingFace inference and save metrics:

```bash
python tools/inference.py \
    --model-path outputs/qwen3vl \
    --collate_fn drivelm_nus_qwen3vl_collate_fn_val \
    --data datasets/DriveLM_nuScenes/split/val \
    --output outputs/qwen3vl/infer_results.json \
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
