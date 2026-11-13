"""SQuAD v2 style Exact Match and F1 (pure Python, no downloads needed)."""

import re
import string
from collections import Counter
from typing import Dict, List


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    # If either side is empty (unanswerable), F1 is 1 only when both are empty.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_squad_v2(predictions: List[str], gold_answers: List[List[str]]) -> Dict[str, float]:
    """gold_answers[i] is the list of acceptable answers; empty list = unanswerable.

    Returns overall EM/F1 plus HasAns F1 (answerable questions) and NoAns
    accuracy (did the model correctly abstain on unanswerable ones).
    """
    em_all, f1_all, f1_has, no_ans_correct, n_has, n_no = [], [], [], 0, 0, 0
    for pred, golds in zip(predictions, gold_answers):
        if not golds:  # unanswerable question: correct answer is the empty string
            n_no += 1
            correct = float(normalize_answer(pred) == "")
            no_ans_correct += correct
            em_all.append(correct)
            f1_all.append(correct)
        else:
            n_has += 1
            em = max(exact_match(pred, g) for g in golds)
            f1 = max(f1_score(pred, g) for g in golds)
            em_all.append(em)
            f1_all.append(f1)
            f1_has.append(f1)
    n = max(len(predictions), 1)
    return {
        "exact_match": round(100 * sum(em_all) / n, 2),
        "f1": round(100 * sum(f1_all) / n, 2),
        "has_ans_f1": round(100 * sum(f1_has) / max(n_has, 1), 2),
        "no_ans_accuracy": round(100 * no_ans_correct / max(n_no, 1), 2),
    }
