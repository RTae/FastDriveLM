HF_ENDPOINT=https://hf-mirror.com

download_data:
	bash scripts/download_drivelm_nus.sh

create_dataset:
	python scripts/create_drivelm_nus.py datasets/v1_1_train_nus.json

MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
download_model:
	HF_ENDPOINT=$(HF_ENDPOINT) hf download $(MODEL_NAME) --local-dir /root/autodl-tmp/models/$(MODEL_NAME)

NUMBER_OF_GPUS := 2
NUM_MACHINES := 1
MIXED_PRECISION := bf16
DYNAMO_BACKEND := no
CONFIG := configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py
fine_tune:
	HF_ENDPOINT=$(HF_ENDPOINT) accelerate launch --num_processes=$(NUMBER_OF_GPUS) --num_machines=$(NUM_MACHINES) --mixed_precision=$(MIXED_PRECISION) --dynamo_backend=$(DYNAMO_BACKEND) tools/finetune.py $(CONFIG)