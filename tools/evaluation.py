import re
import argparse
import json
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def normalize_prediction_answer(answer):
    if isinstance(answer, list):
        return answer[0] if answer else ""
    if answer is None:
        return ""
    return answer


class CocoEvaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

    def run_evaluation(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        gts = {index: [reference] for index, reference in enumerate(references)}
        res = {index: [prediction] for index, prediction in enumerate(predictions)}

        results = {}
        for scorer, method in self.scorers:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for metric_name, metric_score in zip(method, score):
                    results[metric_name] = metric_score
            else:
                results[method] = score
        return results


class evaluation_suit():
    def __init__(self):
        self.language_eval = CocoEvaluator()
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            scores.append(1.0 if answer == GT else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def eval_language(self):
        answer = self.language["answer"]
        GT = self.language["GT"]
        results_gen = self.language_eval.run_evaluation(answer, GT)
        return {f"val/{k}": v for k, v in results_gen.items()}

    def eval_match(self):
        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score * 100)
        return sum(outs1) / len(outs1) if outs1 else 0.0

    def eval_graph(self, question):
        # FIX: guard against set_graph never being called
        if not hasattr(self, 'graph'):
            return False
        question_nums = re.findall(r'\d+\.\d+', question)
        question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        """
        answer: [[1.,2.], [2., 3.]]
        GT: [[1., 2.], [2., 3.]]
        """
        answer_nums = re.findall(r'\d+\.\d+', answer)
        GT_nums = re.findall(r'\d+\.\d+', GT)

        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]

        answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0

        for pred in answer_nums:
            closest_distance = float('inf')
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 16:
                true_positives += 1
                matched_out.append(closest_gt)
                GT_nums = np.delete(GT_nums, closest_id, axis=0)
            else:
                false_positives += 1

        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)

        return matched_out, F1

    def set_graph(self, answer, GT):
        if isinstance(answer, list):
            answer = answer[0]
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        # FIX: normalize answer here to handle list inputs safely
        answer = normalize_prediction_answer(answer)
        if 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
        if 1 in tag:
            pass  # GPT eval not implemented
        if 2 in tag:
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
        if 3 in tag:
            self.match["match"]["GT"].append(GT)
            self.match["match"]["answer"].append(answer)
            self.match["GPT"].append((answer, GT))

    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["language"] = self.eval_language()
        scores["match"] = self.eval_match()
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--src', type=str, default="datasets/DriveLM_nuScenes/refs/infer_results.json", help='path to prediction file')
    parser.add_argument('--tgt', type=str, default="datasets/DriveLM_nuScenes/refs/val_cot.json", help='path to test file')
    args = parser.parse_args()

    with open(args.src, 'r') as f:
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}

    with open(args.tgt, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
    for frame_data in test_file:
        first_flag = True
        frame_data_qa = frame_data['QA']
        for i, qa in enumerate(frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]):
            question = qa['Q']
            GT = qa['A']
            tag = qa['tag']
            idx = frame_data['scene_id'] + "_" + frame_data['frame_id'] + "_" + str(i)

            if idx not in pred_file.keys():
                break

            predict = normalize_prediction_answer(pred_file[idx].get("answer"))

            if first_flag:
                first_flag = False
                evaluation.set_graph(predict, GT)
                evaluation.forward(tag, predict, GT)
            else:
                if evaluation.eval_graph(question):
                    evaluation.forward(tag, predict, GT)

    output = evaluation.evaluation()
    print("accuracy score: ", output["accuracy"])
    print("match score:    ", output["match"])
    print("language score: ", output["language"])

    # Normalize to 0-1 and combine scores
    scores = []

    # language
    score = 0
    for idx, key in enumerate(output["language"].keys()):
        if idx < 4:
            score += output["language"][key] / 4. / 3.
        elif idx == 4:
            score += output["language"][key] / 3.
        else:
            score += output["language"][key] / 10. / 3.
    scores.append(score)

    # match
    scores.append(output["match"] / 100.)

    # accuracy
    scores.append(output["accuracy"])

    # FIX: print combined score instead of just accuracy
    final_score = sum(scores) / len(scores)
    print(f"final combined score: {final_score:.4f}")