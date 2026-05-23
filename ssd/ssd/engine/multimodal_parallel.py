from __future__ import annotations

import atexit
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
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
    accepted_token_ids: list[int] = field(default_factory=list)
    prompt_inputs: dict[str, Any] | None = None
    prompt_length: int = 0
    committed_cache: Any | None = None


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

    def _prepare_inputs(self, prompt: MultimodalPrompt):
        text, images = _normalize_prompt(self.processor, prompt)
        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value) and self.dtype is not None:
                    inputs[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    inputs[key] = value.to(device=self.device)
        return inputs

    def _prepare_prompt_state(self, prompt: MultimodalPrompt):
        inputs = self._prepare_inputs(prompt)
        self.state.prompt = prompt
        self.state.accepted_token_ids = []
        self.state.prompt_inputs = inputs
        self.state.prompt_length = int(inputs["input_ids"].shape[1])
        self.state.committed_cache = None
        return inputs

    def _build_attention_mask(self, total_length: int):
        assert self.state.prompt_inputs is not None
        attention_mask = self.state.prompt_inputs.get("attention_mask")
        if attention_mask is None:
            return None
        base_length = int(attention_mask.shape[1])
        if total_length == base_length:
            return attention_mask
        extra_length = total_length - base_length
        extra_mask = torch.ones((1, extra_length), dtype=attention_mask.dtype, device=self.device)
        return torch.cat([attention_mask, extra_mask], dim=1)

    def _build_mm_token_type_ids(self, total_length: int):
        assert self.state.prompt_inputs is not None
        mm_token_type_ids = self.state.prompt_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            return None
        base_length = int(mm_token_type_ids.shape[1])
        if total_length == base_length:
            return mm_token_type_ids
        extra_length = total_length - base_length
        extra_mm = torch.zeros((1, extra_length), dtype=mm_token_type_ids.dtype, device=self.device)
        return torch.cat([mm_token_type_ids, extra_mm], dim=1)

    def _cached_forward(self, token_ids: list[int], past_key_values, total_prefix_length: int, logits_to_keep: int):
        assert self.state.prompt_inputs is not None
        input_dtype = self.state.prompt_inputs["input_ids"].dtype
        input_ids = torch.tensor([token_ids], dtype=input_dtype, device=self.device)
        total_length = total_prefix_length + len(token_ids)
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=self._build_attention_mask(total_length),
            mm_token_type_ids=self._build_mm_token_type_ids(total_length),
            image_grid_thw=self.state.prompt_inputs.get("image_grid_thw"),
            video_grid_thw=self.state.prompt_inputs.get("video_grid_thw"),
            use_cache=True,
            is_first_iteration=False,
        )
        return self.model(**model_inputs, logits_to_keep=logits_to_keep)

    def measure_prompt_tokens(self, prompt: MultimodalPrompt) -> int:
        if self.state.prompt == prompt and self.state.prompt_inputs is not None:
            return self.state.prompt_length
        inputs = self._prepare_inputs(prompt)
        return int(inputs["input_ids"].shape[1])

    @torch.inference_mode()
    def prefill(self, prompt: MultimodalPrompt):
        inputs = self._prepare_prompt_state(prompt)
        outputs = self.model(**inputs, use_cache=True, logits_to_keep=1)
        self.state.committed_cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = int(logits.argmax(dim=-1).item())
        return {"next_token": next_token}

    @torch.inference_mode()
    def speculate(self, recovery_token: int, lookahead: int, temperature: float = 0.0):
        assert self.state.prompt is not None
        spec_tokens: list[int] = []
        logits_q = []
        branch_cache = deepcopy(self.state.committed_cache)
        prefix_length = self.state.prompt_length + len(self.state.accepted_token_ids)
        next_input_token = recovery_token
        for _ in range(lookahead):
            outputs = self._cached_forward(
                token_ids=[next_input_token],
                past_key_values=branch_cache,
                total_prefix_length=prefix_length,
                logits_to_keep=1,
            )
            branch_cache = outputs.past_key_values
            prefix_length += 1
            logits = outputs.logits[:, -1, :]
            logits_q.append(logits[0].detach().cpu())
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs[0], 1).item())
            else:
                next_token = int(logits.argmax(dim=-1).item())
            spec_tokens.append(next_token)
            next_input_token = next_token
        return {
            "speculations": [recovery_token] + spec_tokens,
            "logits_q": torch.stack(logits_q, dim=0),
        }

    @torch.inference_mode()
    def verify(self, speculations: list[int]):
        assert self.state.prompt is not None
        outputs = self._cached_forward(
            token_ids=speculations,
            past_key_values=deepcopy(self.state.committed_cache),
            total_prefix_length=self.state.prompt_length + len(self.state.accepted_token_ids),
            logits_to_keep=len(speculations),
        )
        logits = outputs.logits[0].detach().cpu()
        return {"logits_p": logits}

    def accept(self, accepted_suffix: list[int]):
        if not accepted_suffix:
            return {"accepted_len": len(self.state.accepted_token_ids)}
        outputs = self._cached_forward(
            token_ids=accepted_suffix,
            past_key_values=self.state.committed_cache,
            total_prefix_length=self.state.prompt_length + len(self.state.accepted_token_ids),
            logits_to_keep=1,
        )
        self.state.committed_cache = outputs.past_key_values
        self.state.accepted_token_ids.extend(accepted_suffix)
        return {"accepted_len": len(self.state.accepted_token_ids)}

    def reset(self):
        self.state = WorkerState(prompt=None)
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

        prompt_tokens = self.target.measure_prompt_tokens(prompt)

        prefill_started = perf_counter()
        target_prefill_future = self.executor.submit(self.target.prefill, prompt)
        draft_prefill_future = self.executor.submit(self.draft.prefill, prompt)
        target_prefill = target_prefill_future.result()
        draft_prefill_future.result()
        prefill_elapsed = perf_counter() - prefill_started

        self.metrics["prefill_total_tokens"] += prompt_tokens
        self.metrics["prefill_total_time"] += prefill_elapsed

        recovery_token = target_prefill["next_token"]
        accepted: list[int] = []

        while len(accepted) < params.max_new_tokens:
            step_started = perf_counter()
            speculate_result = self.draft.speculate(
                recovery_token=recovery_token,
                lookahead=self.config.speculate_k,
                temperature=params.draft_temperature or params.temperature,
            )
            verify_started = perf_counter()
            verify_result = self.target.verify(speculations=speculate_result["speculations"])
            verify_elapsed = perf_counter() - verify_started

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
            step_elapsed = perf_counter() - step_started

            self.metrics["decode_total_tokens"] += len(accepted_suffix)
            self.metrics["decode_total_time"] += step_elapsed
            self.metrics["target_step_times"].append(step_elapsed)
            self.metrics["target_verify_times"].append(verify_elapsed)

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