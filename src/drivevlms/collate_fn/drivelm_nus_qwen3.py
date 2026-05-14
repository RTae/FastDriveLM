import copy

import torch
from PIL import Image

from ..registry import register_collate_fn

_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192

# Limit each image to ~256 patches (28×28 px each) so 6 cameras stay well under
# _MAX_TRAINING_LENGTH: 6 × 256 = 1536 image tokens + text tokens ≈ 2–3k total.
_MAX_PIXELS_PER_IMAGE = 256 * 28 * 28  # ≈ 200 704 pixels


def _resize_for_qwen(img: Image.Image) -> Image.Image:
    """Resize image to fit within _MAX_PIXELS_PER_IMAGE (aspect ratio preserved)."""
    w, h = img.size
    if w * h <= _MAX_PIXELS_PER_IMAGE:
        return img
    scale = (_MAX_PIXELS_PER_IMAGE / (w * h)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

# Camera order must match the six image_paths slots in the dataset
_CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _build_user_content(question: str, images: list) -> list:
    """
    Build the 'content' list for the user turn.
    Six images are inserted first, then the text instruction.
    """
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": question})
    return content


def _apply_label_mask(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """
    Return a labels tensor where prompt tokens are masked with _IGNORE_INDEX.
    Only the answer tokens (from prompt_len onward) contribute to the loss.
    """
    labels = copy.deepcopy(input_ids)
    labels[:, :prompt_len] = _IGNORE_INDEX
    return labels


@register_collate_fn
def drivelm_nus_qwen3vl_collate_fn_train(examples, processor, dtype):
    """
    Collate function for training Qwen2.5-VL / Qwen3-VL on DriveLM-nuScenes.

    The full conversation (prompt + answer) is tokenised once; labels are built
    by masking the prompt part with _IGNORE_INDEX so that the loss is computed
    only on the answer tokens.
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    mm_token_type_ids_list = []
    pixel_values_list = []
    image_grid_thw_list = []

    for example in examples:
        question = example["conversations"][0]["value"]
        answer = example["conversations"][1]["value"]

        images = [
            _resize_for_qwen(Image.open(example["image_paths"][i]).convert("RGB"))
            for i in range(6)
        ]

        # ── Full conversation (prompt + answer) ──────────────────────────────
        messages_full = [
            {
                "role": "user",
                "content": _build_user_content(question, images),
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        text_full = processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        # ── Prompt only (to determine where answer tokens start) ─────────────
        messages_prompt = [
            {
                "role": "user",
                "content": _build_user_content(question, images),
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        # Tokenise both; use the full encoding for pixel_values / image_grid_thw
        enc_full = processor(
            text=[text_full],
            images=images,
            return_tensors="pt",
            padding=False,
        )
        enc_prompt = processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt",
            padding=False,
        )

        input_ids = enc_full.input_ids  # (1, seq_len)
        prompt_len = enc_prompt.input_ids.shape[1]

        labels = _apply_label_mask(input_ids, prompt_len)

        # mm_token_type_ids is required by Qwen3-VL to distinguish text/image/video tokens
        mm_ids = getattr(enc_full, "mm_token_type_ids", None)

        # Truncate to max training length
        if input_ids.shape[1] > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if mm_ids is not None:
                mm_ids = mm_ids[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                labels[:, -1] = processor.tokenizer.eos_token_id

        input_ids_list.append(input_ids.squeeze(0))
        labels_list.append(labels.squeeze(0))
        attention_mask_list.append(
            torch.ones(input_ids.shape[1], dtype=torch.long)
        )
        if mm_ids is not None:
            mm_token_type_ids_list.append(mm_ids.squeeze(0))

        # Extract pixel values and image metadata
        if hasattr(enc_full, "pixel_values") and enc_full.pixel_values is not None:
            pixel_values_list.append(enc_full.pixel_values)
        if hasattr(enc_full, "image_grid_thw") and enc_full.image_grid_thw is not None:
            image_grid_thw_list.append(enc_full.image_grid_thw)

    # Pad sequences to the longest in the batch
    max_len = max(t.shape[0] for t in input_ids_list)
    pad_id = processor.tokenizer.pad_token_id or 0

    def pad_1d(t, fill):
        pad = torch.full((max_len - t.shape[0],), fill, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    input_ids = torch.stack([pad_1d(t, pad_id) for t in input_ids_list])
    labels = torch.stack([pad_1d(t, _IGNORE_INDEX) for t in labels_list])
    attention_mask = torch.stack([pad_1d(t, 0) for t in attention_mask_list])

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    if mm_token_type_ids_list:
        result["mm_token_type_ids"] = torch.stack(
            [pad_1d(t, 0) for t in mm_token_type_ids_list]
        )

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0).to(dtype)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

    return result


@register_collate_fn
def drivelm_nus_qwen3vl_collate_fn_val(examples, processor, device):
    """
    Collate function for validation / inference with Qwen2.5-VL / Qwen3-VL.
    Returns tokenised prompt tensors alongside the raw questions and sample ids.
    """
    ids = [example["id"] for example in examples]
    questions = [example["conversations"][0]["value"] for example in examples]

    # Currently only batch_size=1 is fully supported for multi-image inference
    assert len(examples) == 1, (
        "drivelm_nus_qwen3vl_collate_fn_val currently supports batch_size=1"
    )

    example = examples[0]
    question = example["conversations"][0]["value"]
    images = [
        Image.open(example["image_paths"][i]).convert("RGB") for i in range(6)
    ]

    messages = [
        {
            "role": "user",
            "content": _build_user_content(question, images),
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    tokens = processor(
        text=[text_prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )

    return tokens, questions, ids
