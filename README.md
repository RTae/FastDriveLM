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