"""Evaluation for the summarization module: ROUGE-1/2/L and BERTScore.

Generates summaries for the test set with beam search, then scores them against
the reference abstracts. Used by train.py after fine-tuning, and can also be run
alone on a saved model:

    python -m src.summarization.evaluate --model facebook/bart-base
"""

import argparse
from typing import Dict, List

import torch
from tqdm import tqdm


def generate_summaries(model, tokenizer, articles: List[str], prefix: str,
                       max_input_tokens: int, gen_cfg: dict, batch_size: int = 8) -> List[str]:
    """Generate one summary per article with beam search."""
    device = next(model.parameters()).device
    model.eval()
    outputs: List[str] = []
    for i in tqdm(range(0, len(articles), batch_size), desc="Generating"):
        batch = [prefix + a for a in articles[i:i + batch_size]]
        enc = tokenizer(batch, max_length=max_input_tokens, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        enc.pop("token_type_ids", None)  # seq2seq models do not use segment ids
        with torch.no_grad():
            ids = model.generate(**enc, num_beams=gen_cfg["num_beams"],
                                 max_new_tokens=gen_cfg["max_new_tokens"], early_stopping=True)
        outputs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return [o.strip() for o in outputs]


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-1/2/L F-measures and BERTScore F1 (all reported as 0-100)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    metrics = {k: round(100 * v / n, 2) for k, v in totals.items()}

    try:
        from bert_score import score as bert_score
        # distilbert keeps BERTScore fast enough for free Colab GPUs.
        _, _, f1 = bert_score(predictions, references, model_type="distilbert-base-uncased",
                              lang="en", verbose=False)
        metrics["bertscore_f1"] = round(100 * f1.mean().item(), 2)
    except Exception as err:  # keep ROUGE even if BERTScore cannot download its model
        print(f"BERTScore skipped: {err}")
        metrics["bertscore_f1"] = None
    return metrics


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from src.data_loading import load_config, load_summarization
    from src.utils import output_dir, prefix_for, save_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    c = cfg["summarization"]
    path = output_dir(cfg, "summarization", args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
    test = load_summarization(cfg, args.max_samples)["test"]
    preds = generate_summaries(model, tokenizer, test["article"], prefix_for(args.model),
                               c["max_input_tokens"], c["generation"])
    metrics = compute_metrics(preds, test["abstract"])
    print(metrics)
    save_results(cfg, "summarization", args.model, {"model": args.model, "metrics": metrics})


if __name__ == "__main__":
    main()
