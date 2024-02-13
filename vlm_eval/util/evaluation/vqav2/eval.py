"""
eval.py

Refactored version of the VQA and VQAEval scripts from the Salesforce/LAVIS & VQA-v2 official repositories; see below
for original LICENSE.
"""
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# === Core Function for Running the Official VQAv2 Evaluation ===


def run_vqa_evaluation(questions_json: Path, annotations_json: Path, results_json: Path) -> Dict[str, float]:
    vqa_index = VQAIndex(questions_json, annotations_json)
    vqa_result = vqa_index.load_result_file(result_file=results_json, question_file=questions_json)

    # Create VQA Evaluation Runner from both Index
    vqa_scorer = VQAEvaluator(vqa_index, vqa_result)
    vqa_scorer.evaluate()

    # Create Metrics Dictionary
    metrics = {
        "accuracy": vqa_scorer.accuracy["overall"],
        "accuracy-by-answer-type": vqa_scorer.accuracy["answer_type"],
        "accuracy-by-question-type": vqa_scorer.accuracy["question_type"],
    }

    return metrics


# === Code from LAVIS / PyCoCoTools ===
#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in LAVIS or https://opensource.org/licenses/BSD-3-Clause

__author__ = "aagrawal"
__version__ = "0.9"


# === Constant Mappings ===
CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = ["a", "an", "the"]
PUNCTUATION = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

# === Regular Expressions ===
RE_PERIOD_STRIP, RE_COMMA_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)"), re.compile(r"(\d)(,)(\d)")


# === Processing Helper Functions ===
def process_punctuation(text: str) -> str:
    processed = text
    for punkt in PUNCTUATION:
        if (punkt + " " in text) or (" " + punkt in text) or (re.search(RE_COMMA_STRIP, text) is not None):
            processed = processed.replace(punkt, "")
        else:
            processed = processed.replace(punkt, " ")
    return RE_PERIOD_STRIP.sub("", processed, re.UNICODE)


def process_digits_articles_contractions(text: str) -> str:
    filtered = [DIGIT_MAP.get(word, word) for word in text.lower().split() if word not in ARTICLES]
    filtered = [CONTRACTIONS.get(word, word) for word in filtered]
    return " ".join(filtered)


# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQAIndex             - Indexing structure that loads VQA annotation file and formats questions/answers.
#  get_question_ids     - Get question ids that satisfy given filter conditions.
#  get_image_ids        - Get image ids that satisfy given filter conditions.
#  get_annotations      - Get annotations for the specified question ids.
#  display_qa           - Display the specified questions and answers.
#  load_result_file     - Load result file, creating and populating a separate VQAIndex.


class VQAIndex:
    def __init__(self, question_file: Optional[Path] = None, annotation_file: Optional[Path] = None) -> None:
        """
        VQA helper class for reading, indexing, and visualizing questions and answers.

        :param annotation_file: Path to VQAv2-formatted annotation JSON file.
        :param question_file: Path to VQAv2-formatted question JSON file.
        """
        self.questions, self.annotations, self.qid2ann, self.qid2question, self.img2ann = {}, {}, {}, {}, {}
        if annotation_file is not None and question_file is not None:
            with open(question_file, "r") as q_file, open(annotation_file, "r") as ann_file:
                self.questions = json.load(q_file)
                self.annotations = json.load(ann_file)

            self.create_index()

    def create_index(self) -> None:
        """Build index datastructures from loaded question and annotation files."""
        self.qid2ann = {ann["question_id"]: ann for ann in self.annotations["annotations"]}
        self.qid2question = {question["question"]: question for question in self.questions["questions"]}
        self.img2ann = {ann["image_id"]: [] for ann in self.annotations["annotations"]}
        for ann in self.annotations["annotations"]:
            self.img2ann[ann["image_id"]].append(ann)

    def get_question_ids(
        self,
        image_ids: Optional[List[int]] = None,
        question_types: Optional[List[int]] = None,
        answer_types: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Get the list of unique question IDs that satisfy given filter conditions; default behavior returns all IDs.

        :param image_ids: List of image IDs to retrieve questions for.
        :param question_types: List of question types to retrieve questions for.
        :param answer_types: List of answer types to retrieve questions for.

        :return: List of question IDs that satisfy the specified filter conditions.
        """
        if image_ids is None and question_types is None and answer_types is None:
            return list(self.qid2ann.keys())

        # Otherwise, process filter conditions
        annotations = (
            [self.img2ann[image_id] for image_id in self.img2ann if image_id in image_ids]
            if image_ids is not None
            else self.annotations["annotations"]
        )
        annotations = (
            [ann for ann in annotations if ann["question_type"] in question_types]
            if question_types is not None
            else annotations
        )
        annotations = (
            [ann for ann in annotations if ann["answer_type"] in answer_types]
            if answer_types is not None
            else annotations
        )
        return [ann["question_id"] for ann in annotations]

    def get_image_ids(
        self,
        question_ids: Optional[List[int]] = None,
        question_types: Optional[List[int]] = None,
        answer_types: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Get the list of unique image IDs that satisfy given filter conditions; default behavior returns all IDs.

        :param question_ids: List of question IDs to retrieve images for.
        :param question_types: List of question types to retrieve images for.
        :param answer_types: List of answer types to retrieve images for.

        :return: List of image IDs that satisfy the specified filter conditions.
        """
        if question_ids is None and question_types is None and answer_types is None:
            return list(self.img2ann.keys())

        # Otherwise, process filter conditions
        annotations = (
            [self.qid2ann[qid] for qid in self.qid2ann if qid in question_ids]
            if question_ids is not None
            else self.annotations["annotations"]
        )
        annotations = (
            [ann for ann in annotations if ann["question_type"] in question_types]
            if question_types is not None
            else annotations
        )
        annotations = (
            [ann for ann in annotations if ann["answer_type"] in answer_types]
            if answer_types is not None
            else annotations
        )
        return [ann["image_id"] for ann in annotations]

    def get_annotations(self, qids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Load annotations for the specified question IDs. Default (qids = None) returns empty list.

        :param qids: List of question IDs to retrieve annotations for.

        :return: List of annotation dictionaries for the specified question IDs
        """
        return [self.qid2ann[qid] for qid in qids] if qids is not None else []

    def display_qa(self, annotations: List[Dict[str, Any]]) -> None:
        """Display the specified annotations (questions & answers)."""
        for annotation in annotations:
            print(f"Question: {self.qid2question[annotation['question_id']]} -- Answers:")
            for answer in annotation["answers"]:
                print(f"\t{answer['answer_id']}) {answer['answer']}")

    def load_result_file(self, result_file: Path, question_file: Path) -> "VQAIndex":
        """Create a separate VQAIndex populated from a file with predicted answers (results)."""
        result_index = VQAIndex()
        with open(question_file, "r") as q_file:
            result_index.questions = json.load(q_file)

        # Copy Annotations from the Current Index Instance
        result_index.annotations["task_type"] = deepcopy(self.questions["task_type"])
        result_index.annotations["data_type"] = deepcopy(self.questions["data_type"])
        result_index.annotations["data_subtype"] = deepcopy(self.questions["data_subtype"])
        result_index.annotations["license"] = deepcopy(self.questions["license"])

        with open(result_file, "r") as r_file:
            result_annotations = json.load(r_file)
            assert isinstance(result_annotations, list), f"Results in `{result_file}` should be a list!"

        # Parse out Result Question IDs & Validate
        result_qids = [r_ann["question_id"] for r_ann in result_annotations]
        assert set(result_qids) == set(self.get_question_ids()), (
            f"Results in `{result_file}` do not correspond to the currently loaded questions. "
            "Either the provided results do not have predictions for all question IDs in the annotation file, "
            "or there is at least one question ID that does not belong to the question IDs in the annotation file!"
        )

        # Populate Annotations
        for r_ann in result_annotations:
            qid = r_ann["question_id"]
            if result_index.annotations["task_type"] == "Multiple Choice":
                assert (
                    r_ann["answer"] in self.qid2question[qid]["multiple_choices"]
                ), "Predict answer is not in multiple choice set!"

            # Transfer over metadata from current index...
            gt_ann = self.qid2ann[qid]
            for key in ["image_id", "question_type", "answer_type"]:
                r_ann[key] = gt_ann[key]

        # Finalize & Build Index Datastructures
        result_index.annotations["annotations"] = result_annotations
        result_index.create_index()

        return result_index


# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
class VQAEvaluator:
    def __init__(self, gt_vqa_index: Optional[VQAIndex] = None, result_vqa_index: Optional[VQAIndex] = None) -> None:
        """
        Evaluation helper class for taking two pre-computed VQA Indices, and returning stratified results (accuracy,
        accuracy per question type, etc.).

        :param gt_vqa_index: Ground-truth VQA Index instance, built from the ground-truth answers/annotations.
        :param result_vqa_index: Result VQA Index instance, built from the predicted answers/annotations.
        """
        self.gt_vqa_index, self.result_vqa_index = gt_vqa_index, result_vqa_index
        self.eval_qa, self.eval_question_type, self.eval_answer_type = {}, {}, {}
        self.accuracy = {}

        # Populate Questions (if possible)
        if self.gt_vqa_index is not None:
            self.params = {"question_id": gt_vqa_index.get_question_ids()}

    def evaluate(self, question_ids: Optional[List[int]] = None) -> None:
        """Core Evaluation --> iterates through (specified) questions, computing accuracies per split."""
        question_ids = self.params["question_id"] if question_ids is None else question_ids
        gt_qid2ann, result_qid2ann = self.gt_vqa_index.qid2ann, self.result_vqa_index.qid2ann

        # Compute Accuracy
        for qid in tqdm(question_ids, desc="[*] Computing Accuracy", leave=False):
            result_answer = result_qid2ann[qid]["answer"]
            result_answer = result_answer.replace("\n", " ").replace("\t", " ").strip()
            result_answer = process_punctuation(result_answer)
            result_answer = process_digits_articles_contractions(result_answer)

            # Process Answers in the Ground-Truth Index
            for answer in gt_qid2ann[qid]["answers"]:
                answer["answer"] = process_punctuation(answer["answer"])

            # Consistency Handling for Multiple Answers in Ground-Truth Annotations
            gt_accuracies = []
            for answer in gt_qid2ann[qid]["answers"]:
                matching_answers = [
                    other
                    for other in gt_qid2ann[qid]["answers"]
                    if (other != answer) and (other["answer"] == result_answer)
                ]
                gt_accuracies.append(min(1, len(matching_answers) / 3))

            # Handle Accuracies per Type
            question_type, answer_type = gt_qid2ann[qid]["question_type"], gt_qid2ann[qid]["answer_type"]
            gt_accuracy = sum(gt_accuracies) / len(gt_accuracies)

            # Add to Trackers
            self.eval_qa[qid] = gt_accuracy
            self.eval_question_type.setdefault(question_type, {})[qid] = gt_accuracy
            self.eval_answer_type.setdefault(answer_type, {})[qid] = gt_accuracy

        # Finalize Accuracies
        self.finalize_accuracies()

    def finalize_accuracies(self, n_places: int = 2) -> None:
        """Create readable accuracy dictionary with overall/split metrics."""
        self.accuracy["overall"] = round(100 * (sum(self.eval_qa.values()) / len(self.eval_qa)), n_places)
        self.accuracy["question_type"] = {
            question_type: round(
                100
                * (sum(self.eval_question_type[question_type].values()) / len(self.eval_question_type[question_type])),
                n_places,
            )
            for question_type in self.eval_question_type
        }
        self.accuracy["answer_type"] = {
            answer_type: round(
                100 * (sum(self.eval_answer_type[answer_type].values()) / len(self.eval_answer_type[answer_type])),
                n_places,
            )
            for answer_type in self.eval_answer_type
        }
