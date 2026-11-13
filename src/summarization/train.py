"""Fine-tune a sequence-to-sequence model (BART or T5) to write paper abstracts.

Usage (run once per model so the two variants can be compared):

    python -m src.summarization.train --model facebook/bart-base
    python -m src.summarization.train --model t5-small
    python -m src.summarization.train --model t5-small --max-samples 200 --epochs 1   # quick test

Outputs: the fine-tuned model in outputs/summarization/<model>/ and a metrics
file in results/summarization__<model>.json (ROUGE-1/2/L and BERTScore).
"""

import argparse
import time

import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from src.data_loading import load_config, load_summarization
from src.summarization.evaluate import compute_metrics, generate_summaries
from src.utils import build_training_args, has_gpu, output_dir, prefix_for, save_results, set_seed


def build_tokenize_fn(tokenizer, prefix: str, max_in: int, max_out: int):
    def tokenize(batch):
        model_inputs = tokenizer([prefix + a for a in batch["article"]],
                                 max_length=max_in, truncation=True)
        labels = tokenizer(text_target=batch["abstract"], max_length=max_out, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. facebook/bart-base or t5-small")
    parser.add_argument("--max-samples", type=int, default=None, help="use fewer examples for a quick test")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    c, t = cfg["summarization"], cfg["training"]
    set_seed(cfg["seed"])
    epochs = args.epochs or t["summarization_epochs"]

    print(f"GPU available: {has_gpu()}")
    data = load_summarization(cfg, args.max_samples)
    print({k: len(v) for k, v in data.items()})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    prefix = prefix_for(args.model)

    tokenize = build_tokenize_fn(tokenizer, prefix, c["max_input_tokens"], c["max_target_tokens"])
    train_tok = data["train"].map(tokenize, batched=True, remove_columns=data["train"].column_names)

    out_dir = output_dir(cfg, "summarization", args.model)
    training_args = build_training_args(
        Seq2SeqTrainingArguments,
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=t["learning_rate"] if "t5" not in args.model else 3e-4,  # T5 likes a larger LR
        per_device_train_batch_size=t["summarization_batch_size"],
        gradient_accumulation_steps=t["summarization_grad_accum"],
        num_train_epochs=epochs,
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        fp16=has_gpu() and "t5" not in args.model,  # T5 can overflow in fp16
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_tok,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))

    start = time.time()
    trainer.train()
    train_minutes = round((time.time() - start) / 60, 1)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Evaluate on the held-out test split.
    model = trainer.model
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    test = data["test"]
    preds = generate_summaries(model, tokenizer, test["article"], prefix,
                               c["max_input_tokens"], c["generation"])
    metrics = compute_metrics(preds, test["abstract"])
    print(metrics)

    # Keep a few outputs for the error-analysis section of the report.
    examples = [{"reference": test["abstract"][i], "prediction": preds[i]}
                for i in range(min(5, len(preds)))]
    save_results(cfg, "summarization", args.model, {
        "model": args.model,
        "metrics": metrics,
        "train_examples": len(data["train"]),
        "test_examples": len(test),
        "epochs": epochs,
        "train_minutes": train_minutes,
        "examples": examples,
    })


if __name__ == "__main__":
    main()
