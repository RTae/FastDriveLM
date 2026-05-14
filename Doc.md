# Introduction
DriveVLMS is a framework for fine-tuning and inference of VLMs for autonomous driving.

It currently supports the following VLMs from huggingface and datasets availably.

| VLMs | Dataset |
|------------------------|---------|
| [paligemma](https://huggingface.co/google/paligemma-3b-pt-224) | [DriveLM-nuScenes](https://github.com/OpenDriveLab/DriveLM) |
| [phi4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | [DriveLM-nuScenes](https://github.com/OpenDriveLab/DriveLM) |


Developers can use this framework to easily perform model inference and customize development, including fine-tuning new models on new datasets.

# How to Use
## finetune
```bash
python tools/finetune.py ${CONFIG}
# finetune phi4
## 1 GPU 
python tools/finetune.py configs/phi4/phi4_vision_text_drivelm_1xb1-lora_config.py
## DDP
accelerate launch --num_processes=2 tools/finetune.py configs/phi4/phi4_vision_text_drivelm_1xb1-lora_config.py 
# finetune paligemma
## 1 GPU
python tools/finetune.py configs/paligemma/paligemma_drivelm_config.py 
## DDP
accelerate launch --num_processes=${NUM_GPUS} tools/finetune.py configs/paligemma/paligemma_drivelm_config.py 
```
accelerate launch --num_processes=3 tools/finetune.py configs/paligemma/paligemma_drivelm_config.py
Parameter Explanation:
- ${CONFIG}: Specifies the training configuration and model details for the model to be fine-tuned.
- ${NUM_GPUS}: The number of GPUs used in DDP Training.

## Demo
Load the finetuned model and run inference on samples to evaluate its performance.

```bash
# phi4
python demos/DriveLM_demo_phi4.py
# paligemma
python demos/DriveLM_demo_paligemma.py
```
# Add new models
If you intend to fine-tune a new model using the DriveLM-nuScenes dataset, it is recommended to convert the dataset into the format used in Hugging Face's fine-tuning examples. This approach minimizes unexpected errors, as your coding efforts will be primarily focused on data processing. For guidance, you can refer to the DriveLM project or the official [DriveLM website](https://github.com/OpenDriveLab/DriveLM).

Otherwise, If you intend to develop a new fine-tuning process using this project, you need to add and successfully register the following custom modules.

## step 1. add new preparation file
Add new preparation file in `src/drivelms/preparation` folder.

Since different models on Hugging Face have varying loading methods, our framework supports a customizable model preparation module to accommodate new models.On the Hugging Face model homepage, the specific method for loading the model is typically provided

For example, paligemma is loads in this way.
```bash
model_id = "google/paligemma-3b-pt-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)
```
https://huggingface.co/google/paligemma-3b-pt-224

while phi4 model has different way.
```bash
# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"
# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    # if you do not use Ampere or later GPUs, change attention to "eager"
    _attn_implementation='flash_attention_2',
).eval()
```
https://huggingface.co/microsoft/Phi-4-multimodal-instruct


The model preparation stage alse contains the lora configuration.
We utilize the LoRA (Low-Rank Adaptation) method from Hugging Face's PEFT (Parameter-Efficient Fine-Tuning) library to efficiently fine-tune our model. 
To learn more about `peft` and `lora`, please visit https://hugging-face.cn/docs/peft/index

## step 2. custom collate_fn function
Add new collate_fn function in `src/drivevlms/collate_fn`folder.

​In PyTorch's DataLoader, the `collate_fn`specifies how individual data samples are combined into batches. This function is crucial for ensuring that the data fed into the model matches its expected input format. ​
It's essential that the data preparation aligns with the model's default input requirements. For instance, when working with the PaliGemma model, labels are provided as suffixes, so no special processing is needed. Conversely, the Phi4 model requires specific label formatting, necessitating additional transformations. Therefore, when integrating new models, particular attention must be given to label handling to ensure compatibility.

## step 3. add new config file
Add new config file to specify your `model_name`,`model_preparation`,`collate_fn_train` implementation, as you can see in `configs/phi4/phi4_vision_text_drivelm_1xb1-lora_config.py`


We employ a registry mechanism to manage various module implementations. Therefore, it's necessary to register custom modules using the registry.
# More information
1. In order to facilitate the deployment of LLMs on chips, the source code for `PaliGemma` and `phi4` in the transformers library has been modified. Therefore, the files in the directory `src/drivevlms/models` are the modified model files, which we use for finetuning.

# Version 0.2 update

1. update dataset pixel coordinate system transform

    Resize locations from pixel coordinate system [1600 x 900] to target
    ie. paligemma [224 x 224], phi4 [448 x 448].
2. update system prompts

    Tell the VLM the input six images with their locations. For example, one prompt of paligemma now is 

    ```bash
    ['You are an autonomous driving labeler. You have access to six camera images (front<image>, front-right<image>, front-left<image>, back<image>, back-right<image>, back-left<image>).\nWhat are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.']
    ```

3. update inference and evaluation
    inference with whole val dataset
    ```bash
    python tools/inference.py --output "data/DriveLM_nuScenes/refs/infer_results_21-49.json"
    ```

    calculate metrics for inference results.
    ```bash
    python tools/evaluation.py --src "data/DriveLM_nuScenes/refs/infer_results_21-49.json" --tgt "data/DriveLM_nuScenes/refs/val_cot.json"
    ```
    Prepare the dependencies for language evalution, please refer to https://github.com/OpenDriveLab/DriveLM/tree/main/challenge to see how to install the https://github.com/bckim92/language-evaluation. 