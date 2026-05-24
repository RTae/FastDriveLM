HF_ENDPOINT=https://hf-mirror.com

download_data:
	bash scripts/download_drivelm_nus.sh

create_dataset:
	python scripts/create_drivelm_nus.py datasets/v1_1_train_nus.json

MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
download_model:
	HF_ENDPOINT=$(HF_ENDPOINT) hf download $(MODEL_NAME) --local-dir ./base_models/$(MODEL_NAME)

NUMBER_OF_GPUS := 2
NUM_MACHINES := 1
MIXED_PRECISION := bf16
DYNAMO_BACKEND := no
fine_tune_qwen3vl:
	HF_ENDPOINT=$(HF_ENDPOINT) accelerate launch \
		--num_processes=$(NUMBER_OF_GPUS) \
		--num_machines=$(NUM_MACHINES) \
		--mixed_precision=$(MIXED_PRECISION) \
		--dynamo_backend=$(DYNAMO_BACKEND) \
		tools/finetune.py \
		--log_path qwen3vl.json \
		configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py

fine_tune_qwen3vl_draft:
	HF_ENDPOINT=$(HF_ENDPOINT) accelerate launch \
		--num_processes=$(NUMBER_OF_GPUS) \
		--num_machines=$(NUM_MACHINES) \
		--mixed_precision=$(MIXED_PRECISION) \
		--dynamo_backend=$(DYNAMO_BACKEND) \
		tools/finetune.py \
		--log_path qwen3vl.json \
		configs/qwen3/qwen3vl_drivelm-draft_1xb1-lora_config.py

OUTPUT_MODEL := ./outputs/qwen3vl
inference_qwen3vl:
	python tools/inference.py \
		--model-path $(OUTPUT_MODEL) \
		--collate_fn drivelm_nus_qwen3vl_collate_fn_val \
		--data datasets/DriveLM_nuScenes/split/val \
		--output $(OUTPUT_MODEL)/infer_results.json

ABLATION_OUTPUT_DIR := ./outputs/ablation_ssd_vlm
DRAFT_MODEL := ./outputs/qwen3vl_draft
ablation_ssd_vlm:
	python tools/ablation_ssd_vlm.py \
		--target-model $(OUTPUT_MODEL) \
		--draft-model  $(DRAFT_MODEL) \
		--data         datasets/DriveLM_nuScenes/split/val \
		--output-dir   $(ABLATION_OUTPUT_DIR) \
		--max-samples  50 \
		--warmup-steps 2