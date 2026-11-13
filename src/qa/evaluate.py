"""Evaluate a fine-tuned QA model on SQuAD 2.0 validation questions.

Uses the Hugging Face question-answering pipeline for span extraction and
post-processing, then scores with src/qa/metrics.py (EM, F1, HasAns F1,
NoAns accuracy).

    python -m src.qa.evaluate --model distilbert-base-uncased
"""

import argparse
from typing import Dict, List, Tuple

from src.qa.metrics import score_squad_v2


def evaluate_model(model_dir: str, dataset, batch_size: int = 16) -> Tuple[Dict[str, float], List[dict]]:
    """Return (metrics, a few example predictions for error analysis)."""
    import torch
    from transformers import pipeline

    qa = pipeline("question-answering", model=model_dir, tokenizer=model_dir,
                  device=0 if torch.cuda.is_available() else -1)
    questions = [q.strip() for q in dataset["question"]]
    contexts = list(dataset["context"])
    inputs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]

    raw = qa(inputs, batch_size=batch_size, handle_impossible_answer=True)
    if isinstance(raw, dict):  # a single input returns a dict
        raw = [raw]
    predictions = [r["answer"] for r in raw]
    golds = [list(a["text"]) for a in dataset["answers"]]  # empty list = unanswerable

    metrics = score_squad_v2(predictions, golds)
    examples = [{"question": questions[i], "gold": golds[i][:1], "prediction": predictions[i]}
                for i in range(min(10, len(predictions)))]
    return metrics, examples


def main():
    from src.data_loading import load_config, load_qa
    from src.utils import output_dir, save_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    data = load_qa(cfg, args.max_samples)
    metrics, examples = evaluate_model(str(output_dir(cfg, "qa", args.model)), data["validation"])
    print(metrics)
    save_results(cfg, "qa", args.model, {"model": args.model, "metrics": metrics, "examples": examples})


if __name__ == "__main__":
    main()
