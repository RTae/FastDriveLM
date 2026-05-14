import argparse
import copy
import json
import os
import random
import re

from datasets import Dataset


QA_CATEGORIES = ("perception", "prediction", "planning", "behavior")
STATUS_QUESTION_PREFIX = "what is the moving status of object"
PLANNING_QUESTION_PATTERNS = (
    "what actions could the ego vehicle take",
    "lead to a collision",
    "safe actions",
)
TAG_MULTIPLE_CHOICE = (0,)
TAG_PLANNING = (1,)
TAG_IMPORTANCE = (2,)
TAG_GRAPH = (3,)
OPTION_LABELS = ("A", "B", "C", "D")
MOVING_STATUS_CHOICES = [
    "Going ahead.",
    "Turn right.",
    "Turn left.",
    "Stopped.",
    "Back up.",
    "Reverse parking.",
    "Drive backward.",
]
EGO_VEHICLE_BEHAVIOR_CHOICES = [
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving very fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving with normal speed.",
    "The ego vehicle is steering to the left. The ego vehicle is driving fast.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving fast.",
    "The ego vehicle is going straight. The ego vehicle is driving slowly.",
    "The ego vehicle is going straight. The ego vehicle is driving with normal speed.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving slowly.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast.",
    "The ego vehicle is steering to the right. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the right. The ego vehicle is driving very fast.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving very fast.",
    "The ego vehicle is going straight. The ego vehicle is not moving.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving with normal speed.",
    "The ego vehicle is steering to the right. The ego vehicle is driving slowly.",
    "The ego vehicle is steering to the right. The ego vehicle is driving with normal speed.",
    "The ego vehicle is going straight. The ego vehicle is driving very fast.",
    "The ego vehicle is going straight. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving slowly.",
]
COORD_PATTERN = re.compile(r"<([^,]+),([^,]+),([0-9.]+),([0-9.]+)>")
REFS_DIR = os.path.join("datasets", "DriveLM_nuScenes", "refs")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def iter_key_frames(scene_file):
    for scene_id, scene_data in scene_file.items():
        for frame_id, frame_data in scene_data["key_frames"].items():
            yield scene_id, frame_id, frame_data


def build_frame_entry(image_paths):
    return {
        "QA": {category: [] for category in QA_CATEGORIES},
        "image_paths": image_paths,
    }


def tagged_copy(qa, tag):
    qa_with_tag = dict(qa)
    qa_with_tag["tag"] = list(tag)
    return qa_with_tag


def add_first_matching_qa(source_qas, target_qas, predicate, tag):
    for qa in source_qas:
        if predicate(qa):
            target_qas.append(tagged_copy(qa, tag))
            return


def all_terms_in_text(terms, text):
    lowered_text = text.lower()
    return all(term.lower() in lowered_text for term in terms)


def contains_yes_or_no(text):
    lowered_text = text.lower()
    return "yes" in lowered_text or "no" in lowered_text


def extract_object_classes(frame_data_infos):
    return [
        obj_data["Visual_description"].split(".")[0]
        for obj_data in frame_data_infos.values()
    ]


def append_planning_questions(planning_qas, target_qas):
    found_patterns = set()

    for qa in planning_qas:
        question = qa["Q"].lower()

        for pattern in PLANNING_QUESTION_PATTERNS:
            if pattern in question and pattern not in found_patterns:
                target_qas.append(tagged_copy(qa, TAG_PLANNING))
                found_patterns.add(pattern)

        if len(found_patterns) == len(PLANNING_QUESTION_PATTERNS):
            return


def build_multiple_choice_qa(question, answer, candidates):
    options_pool = list(candidates)
    if answer not in options_pool:
        raise ValueError(f"Answer {answer!r} not found in choice pool.")

    options_pool.remove(answer)
    if len(options_pool) < 3:
        raise ValueError("Multiple-choice generation requires at least 4 candidates.")

    choices = random.sample(options_pool, 3)
    choices.append(answer)
    random.shuffle(choices)

    prompt = (
        f"{question} Please select the correct answer from the following options: "
        f"A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}"
    )
    return {"Q": prompt, "A": OPTION_LABELS[choices.index(answer)]}


def extract_data(root_path):
    train_file = load_json(root_path)
    test_data = {}

    for scene_id, frame_id, frame_data in iter_key_frames(train_file):
        frame_data_infos = frame_data["key_object_infos"]
        frame_data_qa = frame_data["QA"]

        scene_entry = test_data.setdefault(scene_id, {"key_frames": {}})
        frame_entry = build_frame_entry(frame_data["image_paths"])
        scene_entry["key_frames"][frame_id] = frame_entry

        object_classes = extract_object_classes(frame_data_infos)
        object_locations = list(frame_data_infos)
        qa_output = frame_entry["QA"]

        add_first_matching_qa(
            frame_data_qa["perception"],
            qa_output["perception"],
            lambda qa: all_terms_in_text(object_classes, qa["A"]),
            TAG_IMPORTANCE,
        )
        add_first_matching_qa(
            frame_data_qa["perception"],
            qa_output["perception"],
            lambda qa: STATUS_QUESTION_PREFIX in qa["Q"].lower(),
            TAG_MULTIPLE_CHOICE,
        )
        add_first_matching_qa(
            frame_data_qa["prediction"],
            qa_output["prediction"],
            lambda qa: all_terms_in_text(object_locations, qa["A"]),
            TAG_GRAPH,
        )
        add_first_matching_qa(
            frame_data_qa["prediction"],
            qa_output["prediction"],
            lambda qa: contains_yes_or_no(qa["A"]),
            TAG_MULTIPLE_CHOICE,
        )
        append_planning_questions(frame_data_qa["planning"], qa_output["planning"])
        qa_output["behavior"] = [
            tagged_copy(qa, TAG_MULTIPLE_CHOICE)
            for qa in frame_data_qa["behavior"]
        ]

    return test_data


def rule_based1(question, answer):
    return build_multiple_choice_qa(question, answer, MOVING_STATUS_CHOICES)


def rule_based2(question, answer):
    return build_multiple_choice_qa(question, answer, EGO_VEHICLE_BEHAVIOR_CHOICES)


def loop_test(test_file):
    for scene_id, frame_id, frame_data in iter_key_frames(test_file):
        frame_data_qa = frame_data["QA"]
        frame_entry = build_frame_entry(frame_data["image_paths"])

        frame_entry["QA"]["prediction"] = [dict(qa) for qa in frame_data_qa["prediction"]]
        frame_entry["QA"]["planning"] = [dict(qa) for qa in frame_data_qa["planning"]]

        for qa in frame_data_qa["perception"]:
            if STATUS_QUESTION_PREFIX in qa["Q"].lower():
                frame_entry["QA"]["perception"].append(
                    {
                        **qa,
                        **rule_based1(qa["Q"], qa["A"]),
                    }
                )
                continue

            frame_entry["QA"]["perception"].append(dict(qa))

        for qa in frame_data_qa["behavior"]:
            frame_entry["QA"]["behavior"].append(
                {
                    **qa,
                    **rule_based2(qa["Q"], qa["A"]),
                }
            )

        test_file[scene_id]["key_frames"][frame_id] = frame_entry

    return test_file


def normalize_image_paths(image_paths):
    return [
        image_path.replace("..", "data/DriveLM_nuScenes")
        for image_path in image_paths.values()
    ]


def split_by_key_frame(test_file, train_ratio=0.8, seed=42):
    frame_datas = []

    for scene_id, frame_id, frame_data in iter_key_frames(test_file):
        frame_entry = copy.deepcopy(frame_data)
        frame_entry["image_paths"] = normalize_image_paths(frame_data["image_paths"])
        frame_entry["scene_id"] = scene_id
        frame_entry["frame_id"] = frame_id
        frame_datas.append(frame_entry)

    rng = random.Random(seed)
    rng.shuffle(frame_datas)

    split_idx = int(len(frame_datas) * train_ratio)
    train_frames = frame_datas[:split_idx]
    val_frames = frame_datas[split_idx:]
    return train_frames, val_frames


def convert2vlm(test_file):
    output = []
    for frame_data in test_file:
        frame_data_qa = frame_data["QA"]
        qa_pairs = (
            frame_data_qa["perception"]
            + frame_data_qa["prediction"]
            + frame_data_qa["planning"]
            + frame_data_qa["behavior"]
        )

        for idx, qa in enumerate(qa_pairs):
            output.append(
                {
                    "id": f"{frame_data['scene_id']}_{frame_data['frame_id']}_{idx}",
                    "image": frame_data["image_paths"],
                    "conversations": [
                        {"from": "human", "value": qa["Q"]},
                        {"from": "gpt", "value": qa["A"]},
                    ],
                }
            )

    return output


def convert_to_hf_dataset(json_data):
    hf_data = [
        {
            "id": item["id"],
            "image_paths": item["image"],
            "conversations": item["conversations"],
        }
        for item in json_data
    ]
    return Dataset.from_list(hf_data)


def rescale_coords(x, y, orig_size=(1600, 900), target_size=(224, 224)):
    new_x = x / orig_size[0] * target_size[0]
    new_y = y / orig_size[1] * target_size[1]
    return round(new_x, 2), round(new_y, 2)


def replace_coords_in_text(text, target_size=(224, 224)):
    def repl(match):
        name, cam, x, y = match.groups()
        x_new, y_new = rescale_coords(float(x), float(y), target_size=target_size)
        return f"<{name},{cam},{x_new},{y_new}>"

    return COORD_PATTERN.sub(repl, text)


def convert_coors_system(data, target_size=(224, 224)):
    for item in data:
        for qa_list in item.get("QA", {}).values():
            for qa_item in qa_list:
                if isinstance(qa_item.get("Q"), str):
                    qa_item["Q"] = replace_coords_in_text(
                        qa_item["Q"],
                        target_size=target_size,
                    )
                if isinstance(qa_item.get("A"), str):
                    qa_item["A"] = replace_coords_in_text(
                        qa_item["A"],
                        target_size=target_size,
                    )
    return data


def create_drivelm_nus(args):
    test_data = extract_data(args.src)
    rule_data = loop_test(test_data)
    train_frames, val_frames = split_by_key_frame(rule_data)

    target_size = (args.resize_tgt, args.resize_tgt)
    train_frames = convert_coors_system(train_frames, target_size=target_size)
    val_frames = convert_coors_system(val_frames, target_size=target_size)

    os.makedirs(REFS_DIR, exist_ok=True)
    save_json(os.path.join(REFS_DIR, "train_cot.json"), train_frames)
    save_json(os.path.join(REFS_DIR, "val_cot.json"), val_frames)

    output_train = convert2vlm(train_frames)
    output_val = convert2vlm(val_frames)
    save_json(os.path.join(REFS_DIR, "val_qa_style.json"), output_val)

    train_dataset = convert_to_hf_dataset(output_train)
    val_dataset = convert_to_hf_dataset(output_val)
    train_dataset.save_to_disk(args.train_data)
    val_dataset.save_to_disk(args.val_data)
    print("finished...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        type=str,
        default=None,
        help="the json file download from DriveLM-nuScenes repo website",
    )
    parser.add_argument(
        "--resize_tgt",
        type=int,
        default=224,
        help="the pixel coordinate system to be transformed to",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="datasets/DriveLM_nuScenes/split/train/",
        help="the huggingface Dataset style data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="datasets/DriveLM_nuScenes/split/val/",
        help="the huggingface Dataset style data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_drivelm_nus(args)


