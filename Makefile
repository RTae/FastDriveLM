HF_ENDPOINT=https://hf-mirror.com

download_data:
	bash scripts/download_drivelm_nus.sh

create_dataset:
	python scripts/create_drivelm_nus.py datasets/v1_1_train_nus.json

NUMBER_OF_GPUS := 2
CONFIG := configs/qwen3/qwen3vl_drivelm_1xb1-lora_config.py
fine_tune:
	HF_ENDPOINT=$(HF_ENDPOINT) accelerate launch --num_processes=$(NUMBER_OF_GPUS) tools/finetune.py $(CONFIG)