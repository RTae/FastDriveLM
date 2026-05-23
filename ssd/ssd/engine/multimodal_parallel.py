from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from ssd.config import Config, get_text_config
from ssd.sampling_params import SamplingParams
from ssd.utils.verify import verify


MultimodalPrompt = str | Mapping[str, Any]


def _torch_dtype_from_config(config: Config, hf_config) -> torch.dtype | None:
    text_config = get_text_config(hf_config)
    dtype = (
        getattr(hf_config, "torch_dtype", None)
        or getattr(text_config, "torch_dtype", None)
        or getattr(hf_config, "dtype", None)
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
        return Image.open(Path(image)).convert("RGB")
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


def _normalize_prompt(processor, prompt: MultimodalPrompt) -> tuple[str, list[Image.Image] | None]:
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
        messages = prompt_dict["messages"]
        if images is None:
            images = _extract_images_from_messages(messages)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=prompt_dict.get("add_generation_prompt", True),
        )
        return text, images or None

    text = prompt_dict.get("text", prompt_dict.get("prompt", prompt_dict.get("question", None)))
    if text is None:
        raise ValueError("Multimodal prompts must include 'text', 'prompt', 'question', or 'messages'.")

    if images:
        text = processor.apply_chat_template(
            _build_messages(str(text), images),
            tokenize=False,
            add_generation_prompt=prompt_dict.get("add_generation_prompt", True),
        )
    return str(text), images


def _select_device(index: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{index}")


@dataclass
class WorkerState:
    prompt: MultimodalPrompt | None = None
    accepted_token_ids: list[int] | None = None


class MultimodalWorker:
    def __init__(self, model_reference: str, adapter_path: str | None, trust_remote_code: bool, device_index: int):
        self.model_reference = model_reference
        self.adapter_path = adapter_path
        self.trust_remote_code = trust_remote_code
        self.device_index = device_index
        self.state = WorkerState(prompt=None, accepted_token_ids=[])
        self.device = _select_device(device_index)

        self.processor = AutoProcessor.from_pretrained(model_reference, trust_remote_code=trust_remote_code)
        self.hf_config = AutoConfig.from_pretrained(model_reference, trust_remote_code=trust_remote_code)
        self.dtype = _torch_dtype_from_config(Config(model_reference), self.hf_config)
        load_kwargs = {"trust_remote_code": trust_remote_code, "device_map": None}
        if self.dtype is not None:
            load_kwargs["dtype"] = self.dtype
        self.model = AutoModelForImageTextToText.from_pretrained(model_reference, **load_kwargs)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=True)
            self.model = self.model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

    def _prepare_inputs(self, prompt: MultimodalPrompt, extra_token_ids: list[int] | None = None):
        text, images = _normalize_prompt(self.processor, prompt)
        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        if extra_token_ids:
            extra = torch.tensor([extra_token_ids], dtype=inputs["input_ids"].dtype)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
            if "attention_mask" in inputs:
                attn_extra = torch.ones((1, len(extra_token_ids)), dtype=inputs["attention_mask"].dtype)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], attn_extra], dim=1)
            if "mm_token_type_ids" in inputs:
                mm_extra = torch.zeros((1, len(extra_token_ids)), dtype=inputs["mm_token_type_ids"].dtype)
                inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], mm_extra], dim=1)
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value) and self.dtype is not None:
                    inputs[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    inputs[key] = value.to(device=self.device)
        return inputs

    @torch.inference_mode()
    def prefill(self, prompt: MultimodalPrompt):
        self.state.prompt = prompt
        self.state.accepted_token_ids = []
        inputs = self._prepare_inputs(prompt)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        next_token = int(logits.argmax(dim=-1).item())
        return {"next_token": next_token}

    @torch.inference_mode()
    def speculate(self, recovery_token: int, lookahead: int, temperature: float = 0.0):
        assert self.state.prompt is not None
        prefix = list(self.state.accepted_token_ids or []) + [recovery_token]
        spec_tokens: list[int] = []
        logits_q = []
        for _ in range(lookahead):
            inputs = self._prepare_inputs(self.state.prompt, prefix + spec_tokens)
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            logits_q.append(logits[0].detach().cpu())
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs[0], 1).item())
            else:
                next_token = int(logits.argmax(dim=-1).item())
            spec_tokens.append(next_token)
        return {
            "speculations": [recovery_token] + spec_tokens,
            "logits_q": torch.stack(logits_q, dim=0),
        }

    @torch.inference_mode()
    def verify(self, speculations: list[int]):
        assert self.state.prompt is not None
        inputs = self._prepare_inputs(self.state.prompt, list(self.state.accepted_token_ids or []) + speculations)
        outputs = self.model(**inputs)
        k_plus_1 = len(speculations)
        logits = outputs.logits[0, -k_plus_1:, :].detach().cpu()
        return {"logits_p": logits}

    def accept(self, accepted_suffix: list[int]):
        if self.state.accepted_token_ids is None:
            self.state.accepted_token_ids = []
        self.state.accepted_token_ids.extend(accepted_suffix)
        return {"accepted_len": len(self.state.accepted_token_ids)}

    def reset(self):
        self.state = WorkerState(prompt=None, accepted_token_ids=[])
        return {"ok": True}


class MultimodalParallelRunner:
    def __init__(self, config: Config, metrics: dict[str, Any]):
        self.config = config
        self.metrics = metrics
        assert config.speculate, "MultimodalParallelRunner is only for speculative VLM execution"
        self.processor = AutoProcessor.from_pretrained(
            config.resolved_base_model_path or config.resolved_model_path or config.model,
            trust_remote_code=config.vlm_trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.target = MultimodalWorker(
            config.resolved_model_path or config.model,
            config.vlm_adapter_path,
            config.vlm_trust_remote_code,
            0,
        )
        self.draft = MultimodalWorker(
            config.resolved_draft_model_path or config.draft,
            config.vlm_draft_adapter_path,
            config.vlm_trust_remote_code,
            1 if torch.cuda.device_count() > 1 else 0,
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        atexit.register(self.close)

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    @torch.inference_mode()
    def generate(self, prompts: list[MultimodalPrompt], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool = True, stream_callback=None):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(prompts) == len(sampling_params), "prompts and sampling_params must have the same length"
        assert len(prompts) == 1, "The first VLM SSD implementation currently supports batch_size=1"

        prompt = prompts[0]
        params = sampling_params[0]
        assert params.temperature == 0.0, "The first VLM SSD implementation currently supports greedy decoding only"

        target_prefill_future = self.executor.submit(self.target.prefill, prompt)
        draft_prefill_future = self.executor.submit(self.draft.prefill, prompt)
        target_prefill = target_prefill_future.result()
        draft_prefill_future.result()
        recovery_token = target_prefill["next_token"]
        accepted: list[int] = []

        while len(accepted) < params.max_new_tokens:
            speculate_result = self.draft.speculate(
                recovery_token=recovery_token,
                lookahead=self.config.speculate_k,
                temperature=params.draft_temperature or params.temperature,
            )
            verify_result = self.target.verify(speculations=speculate_result["speculations"])

            logits_p = verify_result["logits_p"].unsqueeze(0)
            logits_q = speculate_result["logits_q"].unsqueeze(0)
            speculations = torch.tensor([speculate_result["speculations"]], dtype=torch.int64)
            temperatures_target = torch.tensor([params.temperature], dtype=torch.float32)
            temperatures_draft = torch.tensor([params.draft_temperature or params.temperature], dtype=torch.float32)
            new_suffixes, recovery_tokens = verify(
                logits_p=logits_p,
                logits_q=logits_q,
                speculations=speculations,
                temperatures_target=temperatures_target,
                temperatures_draft=temperatures_draft,
                cache_hits=None,
            )
            accepted_suffix = new_suffixes[0]
            recovery_token = recovery_tokens[0]
            self.target.accept(accepted_suffix=accepted_suffix)
            self.draft.accept(accepted_suffix=accepted_suffix)
            accepted.extend(accepted_suffix)
            if stream_callback and accepted_suffix:
                stream_callback(0, accepted_suffix)
            if not params.ignore_eos and self.eos_token_id is not None and recovery_token == self.eos_token_id:
                break

        output_token_ids = accepted[:params.max_new_tokens]
        outputs = [{
            "text": self.tokenizer.decode(output_token_ids, skip_special_tokens=True),
            "token_ids": output_token_ids,
        }]
        return outputs, self.metrics