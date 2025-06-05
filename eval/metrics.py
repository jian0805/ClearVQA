import json
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from .vqa_tools import TextCleaner, VQA
from .vqaEval import VQAEval, VQATool


def compute_vqa_score(results):
    """
    results: list of dict
    gt_answers: dict
    """
    vqa_tool = VQATool()
    acc_dict = dict()
    
    print("Computing VQA Score...")
    for res in tqdm(results):
        q_id = res["question_id"]
        pred = res["answer"]
        pred = vqa_tool.processPunctuation(pred)
        pred = vqa_tool.processDigitArticle(pred)

        gt_answers = res["gt_answers"]
        for ans in gt_answers:
            ans = vqa_tool.processPunctuation(ans)
            ans = vqa_tool.processDigitArticle(ans)
        
        num_match = sum([pred == gt for gt in gt_answers])
        vqa_acc = min(1.0, num_match / 3.0)

        acc_dict[q_id] = vqa_acc
        
    accuracy = np.mean(list(acc_dict.values())) * 100
    return EasyDict({"VQA": accuracy, "evalQA": acc_dict})


def compute_exact_match(results):
    cleaner = TextCleaner()
    exact_match_list = []

    acc_dict = dict()
    print("Computing Exact Match")
    for result in tqdm(results):
        question_id = result["question_id"]
        pred_answer = cleaner.clean_texts([result["answer"]])[0]
        answers = result["gt_answers"]
        answers = cleaner.clean_texts(answers)
        exact_match = 1 if pred_answer in answers else 0
        exact_match_list.append(exact_match)
        acc_dict[question_id] = exact_match
    
    em = np.mean(np.array(exact_match_list)) * 100
    return EasyDict({"exact_match": em, "evalQA": acc_dict})