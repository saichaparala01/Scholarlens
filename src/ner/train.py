"""Fine-tune a transformer for Named Entity Recognition (token classification).

Usage:
    python -m src.ner.train --model allenai/scibert_scivocab_uncased
    python -m src.ner.train --model bert-base-uncased
    python -m src.ner.train --model bert-base-uncased --max-samples 200 --epochs 1   # quick test

Outputs: model in outputs/ner/<model>/ and metrics in results/ner__<model>.json
(entity-level precision, recall, F1 and token accuracy, plus per-type scores).
"""

import argparse
import time

import numpy as np
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

from src.data_loading import load_config, load_ner
from src.ner.metrics import entity_scores
from src.utils import build_training_args, has_gpu, output_dir, save_results, set_seed


def tokenize_and_align(examples, tokenizer, max_len: int):
    """Tokenize pre-split words and align word-level tags to sub-word tokens.

    A word may become several sub-word tokens. Only the first sub-word keeps the
    tag; the rest get -100 so the loss ignores them.
    """
    tokenized = tokenizer(examples["tokens"], truncation=True, max_length=max_len,
                          is_split_into_words=True)
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels, previous = [], None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous:
                labels.append(tags[word_id])
            else:
                labels.append(-100)
            previous = word_id
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized


def decode_predictions(logits, label_ids, label_names):
    """Turn model output back into tag-name sequences, skipping ignored positions."""
    preds = np.argmax(logits, axis=-1)
    true_tags, pred_tags = [], []
    for pred_row, label_row in zip(preds, label_ids):
        t_seq, p_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                t_seq.append(label_names[l])
                p_seq.append(label_names[p])
        true_tags.append(t_seq)
        pred_tags.append(p_seq)
    return true_tags, pred_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    c, t = cfg["ner"], cfg["training"]
    set_seed(cfg["seed"])
    epochs = args.epochs or t["ner_epochs"]

    print(f"GPU available: {has_gpu()}")
    data = load_ner(cfg, args.max_samples)
    print({k: len(v) for k, v in data.items()})
    label_names = data["train"].features["ner_tags"].feature.names
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label_names), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)  # fresh classification head for this tag set

    def prep(split):
        return data[split].map(lambda ex: tokenize_and_align(ex, tokenizer, c["max_seq_len"]),
                               batched=True, remove_columns=data[split].column_names)
    train_tok, test_tok = prep("train"), prep("test")

    out_dir = output_dir(cfg, "ner", args.model)
    training_args = build_training_args(
        TrainingArguments,
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["ner_batch_size"],
        per_device_eval_batch_size=t["ner_batch_size"],
        num_train_epochs=epochs,
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        fp16=has_gpu(),
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_tok,
                      data_collator=DataCollatorForTokenClassification(tokenizer))

    start = time.time()
    trainer.train()
    train_minutes = round((time.time() - start) / 60, 1)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    output = trainer.predict(test_tok)
    true_tags, pred_tags = decode_predictions(output.predictions, output.label_ids, label_names)
    metrics = entity_scores(true_tags, pred_tags)
    print({k: v for k, v in metrics.items() if k != "per_type"})

    save_results(cfg, "ner", args.model, {
        "model": args.model,
        "metrics": {k: v for k, v in metrics.items() if k != "per_type"},
        "per_type": metrics["per_type"],
        "tags": label_names,
        "train_sentences": len(data["train"]),
        "test_sentences": len(data["test"]),
        "epochs": epochs,
        "train_minutes": train_minutes,
    })


if __name__ == "__main__":
    main()
