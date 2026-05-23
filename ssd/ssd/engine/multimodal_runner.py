from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch
from PIL import Image
from tqdm.auto import tqdm
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from ssd.config import Config, get_text_config
from ssd.sampling_params import SamplingParams


MultimodalPrompt = str | Mapping[str, Any]


def _torch_dtype_from_config(config: Config) -> torch.dtype | None:
    text_config = get_text_config(config.hf_config)
    dtype = (
        getattr(config.hf_config, "torch_dtype", None)
        or getattr(text_config, "torch_dtype", None)
        or getattr(config.hf_config, "dtype", None)
        or getattr(text_config, "dtype", None)
    )
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype, None)
    return None


def _load_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if str(image).startswith(("http://", "https://")):
            raise ValueError("Remote image URLs are not supported by SSD VLM prompts; pass a local path or PIL image.")
        return Image.open(image_path).convert("RGB")
    if isinstance(image, Mapping) and "image" in image:
        return _load_image(image["image"])
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def _extract_images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "image":
                continue
            image_value = item.get("image")
            if image_value is not None:
                images.append(_load_image(image_value))
    return images


def _build_messages(question: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


class MultimodalRunner:
    def __init__(self, config: Config, metrics: dict[str, Any]):
        self.config = config
        self.metrics = metrics
        self.dtype = _torch_dtype_from_config(config)
        self.processor = AutoProcessor.from_pretrained(
            config.resolved_base_model_path or config.resolved_model_path or config.model,
            trust_remote_code=config.vlm_trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": config.vlm_trust_remote_code,
            "attn_implementation": config.vlm_attn_implementation,
        }
        if self.dtype is not None:
            load_kwargs["dtype"] = self.dtype

        device_map = config.vlm_device_map
        if device_map is None and config.num_gpus > 1:
            device_map = "auto"
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.model = self._load_model(
            config.resolved_model_path or config.model,
            config.vlm_adapter_path,
            load_kwargs,
        )
        if device_map is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()

        self.assistant_model = None
        if config.speculate:
            self.assistant_model = self._load_model(
                config.resolved_draft_model_path or config.draft,
                config.vlm_draft_adapter_path,
                load_kwargs,
            )
            self.assistant_model.to(self.device)
            self.assistant_model.eval()

    def _load_model(self, model_reference: str, adapter_path: str | None, load_kwargs: dict[str, Any]):
        model = AutoModelForImageTextToText.from_pretrained(model_reference, **load_kwargs)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
            model = model.merge_and_unload()
        return model

    def _normalize_prompt(self, prompt: MultimodalPrompt) -> tuple[str, list[Image.Image] | None]:
        if isinstance(prompt, str):
            return prompt, None

        prompt_dict = dict(prompt)
        explicit_images = prompt_dict.get("images", prompt_dict.get("image", None))
        images = None
        if explicit_images is not None:
            if not isinstance(explicit_images, list):
                explicit_images = [explicit_images]
            images = [_load_image(image) for image in explicit_images]

        if "messages" in prompt_dict:
            messages = deepcopy(prompt_dict["messages"])
            if images is None:
                images = _extract_images_from_messages(messages)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=prompt_dict.get("add_generation_prompt", True),
            )
            return text, images or None

        text = prompt_dict.get("text", prompt_dict.get("prompt", prompt_dict.get("question", None)))
        if text is None:
            raise ValueError("Multimodal prompts must be a string or a dict with 'messages', 'text', 'prompt', or 'question'.")

        if images:
            messages = _build_messages(str(text), images)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=prompt_dict.get("add_generation_prompt", True),
            )
        return str(text), images

    def _prepare_inputs(self, prompt: MultimodalPrompt):
        text, images = self._normalize_prompt(prompt)
        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)

        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            if torch.is_floating_point(value) and self.dtype is not None:
                inputs[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                inputs[key] = value.to(device=self.device)
        return inputs

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[MultimodalPrompt],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        stream_callback=None,
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(prompts) == len(sampling_params), "prompts and sampling_params must have the same length"

        progress = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        outputs = []

        for seq_id, (prompt, params) in enumerate(zip(prompts, sampling_params)):
            inputs = self._prepare_inputs(prompt)
            input_len = inputs["input_ids"].shape[1]

            generation_kwargs = {
                "max_new_tokens": params.max_new_tokens,
                "do_sample": params.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            if not params.ignore_eos:
                generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            if params.temperature > 0:
                generation_kwargs["temperature"] = params.temperature
            if self.assistant_model is not None:
                generation_kwargs["assistant_model"] = self.assistant_model

            started = perf_counter()
            generated = self.model.generate(**inputs, **generation_kwargs)
            elapsed = perf_counter() - started
            new_token_ids = generated[0, input_len:].detach().cpu().tolist()

            self.metrics["prefill_total_tokens"] += input_len
            self.metrics["decode_total_tokens"] += len(new_token_ids)
            self.metrics["target_step_times"].append(elapsed)
            if elapsed > 0:
                self.metrics["prefill_total_time"] += elapsed
                self.metrics["decode_total_time"] += elapsed

            if stream_callback and new_token_ids:
                stream_callback(seq_id, new_token_ids)

            outputs.append({
                "text": self.tokenizer.decode(new_token_ids, skip_special_tokens=True),
                "token_ids": new_token_ids,
            })
            if progress:
                progress.update(1)

        if progress:
            progress.close()
        return outputs, self.metrics