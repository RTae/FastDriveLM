# FastDriveLM

## Installation
```bash
uv sync
```

## How to develop
```bash
source .venv/bin/activate
```

## Fine-tune model
### Prepare data

1. Download data
```bash
bash scripts/download_drivelm_nus.sh
```

2. Convert data into fine-tune format
```bash
python scripts/create_drivelm_nus.py datasets/v1_1_train_nus.json
```