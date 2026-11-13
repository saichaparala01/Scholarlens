"""Fine-tune an extractive QA model (DistilBERT or RoBERTa) on SQuAD 2.0.

SQuAD 2.0 includes unanswerable questions, so the model must also learn to say
"no answer". That matters for ScholarLens: a question about something that is
not in the paper should not get an invented answer.

Usage:
    python -m src.qa.train --model distilbert-base-uncased
    python -m src.qa.train --model roberta-base
    python -m src.qa.train --model distilbert-base-uncased --max-samples 200 --epochs 1   # quick test

Outputs: model in outputs/qa/<model>/ and metrics in results/qa__<model>.json
(Exact Match, F1, HasAns F1, NoAns accuracy).
"""

import argparse
import time

from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

from src.data_loading import load_config, load_qa
from src.qa.evaluate import evaluate_model
from src.utils import build_training_args, has_gpu, output_dir, save_results, set_seed


def prepare_train_features(examples, tokenizer, max_len: int, stride: int):
    """Tokenize question+context and convert answer character spans to token spans.

    Long contexts are split into overlapping windows (doc_stride). A window that
    does not contain the answer is labelled with the [CLS] position, which is how
    the model learns to predict "no answer".
    """
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions, examples["context"], truncation="only_second", max_length=max_len,
        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions, end_positions = [], []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        answers = examples["answers"][sample_map[i]]

        if len(answers["answer_start"]) == 0:  # unanswerable question
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start = 0
        while sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(input_ids) - 1
        while sequence_ids[token_end] != 1:
            token_end -= 1

        if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
            # The answer is outside this window.
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)
            while offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. distilbert-base-uncased or roberta-base")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    c, t = cfg["qa"], cfg["training"]
    set_seed(cfg["seed"])
    epochs = args.epochs or t["qa_epochs"]

    print(f"GPU available: {has_gpu()}")
    data = load_qa(cfg, args.max_samples)
    print({k: len(v) for k, v in data.items()})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    train_tok = data["train"].map(
        lambda ex: prepare_train_features(ex, tokenizer, c["max_seq_len"], c["doc_stride"]),
        batched=True, remove_columns=data["train"].column_names,
    )
    print(f"Training windows: {len(train_tok)}")

    out_dir = output_dir(cfg, "qa", args.model)
    training_args = build_training_args(
        TrainingArguments,
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=t["learning_rate"] if "roberta" not in args.model else 3e-5,
        per_device_train_batch_size=t["qa_batch_size"],
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
                      data_collator=default_data_collator)

    start = time.time()
    trainer.train()
    train_minutes = round((time.time() - start) / 60, 1)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics, examples = evaluate_model(str(out_dir), data["validation"])
    print(metrics)
    save_results(cfg, "qa", args.model, {
        "model": args.model,
        "metrics": metrics,
        "train_questions": len(data["train"]),
        "train_windows": len(train_tok),
        "eval_questions": len(data["validation"]),
        "epochs": epochs,
        "train_minutes": train_minutes,
        "examples": examples,
    })


if __name__ == "__main__":
    main()
