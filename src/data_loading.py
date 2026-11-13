"""Dataset loaders for the three ScholarLens tasks.

Each loader reads sizes and names from configs/config.yaml, cleans the data
with src/preprocessing.py, subsamples reproducibly and returns a Hugging Face
DatasetDict ready for tokenization. Run this file directly for a quick smoke
test of any task:

    python -m src.data_loading --task summarization --max-samples 200
"""

import argparse
from pathlib import Path

import yaml
from datasets import DatasetDict, load_dataset

from src.preprocessing import clean_text, is_valid_pair

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _subsample(ds, n: int, seed: int):
    """Shuffle with a fixed seed and keep at most n rows."""
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


# ---------------------------------------------------------------- Summarization
def load_summarization(cfg: dict, max_samples: int = None) -> DatasetDict:
    """arXiv papers (article) paired with their abstracts (abstract)."""
    c, seed = cfg["summarization"], cfg["seed"]
    raw = load_dataset(c["dataset"])  # splits: train / validation / test

    def clean(example):
        return {
            "article": clean_text(example["article"]),
            "abstract": clean_text(example["abstract"]),
        }

    sizes = {"train": c["train_samples"], "validation": c["val_samples"], "test": c["test_samples"]}
    out = {}
    for split, n in sizes.items():
        n = min(n, max_samples) if max_samples else n
        # Oversample a little because the quality filter removes some rows.
        ds = _subsample(raw[split], int(n * 1.2), seed).map(clean)
        ds = ds.filter(lambda ex: is_valid_pair(ex["article"], ex["abstract"]))
        out[split] = ds.select(range(min(n, len(ds))))
    return DatasetDict(out)


# --------------------------------------------------------------------------- QA
def load_qa(cfg: dict, max_samples: int = None) -> DatasetDict:
    """SQuAD 2.0 (includes unanswerable questions, which makes QA harder)."""
    c, seed = cfg["qa"], cfg["seed"]
    raw = load_dataset(c["dataset"])  # splits: train / validation

    def clean(example):
        # Only the question is cleaned: answer start offsets index into the
        # context, so editing the context would corrupt the labels.
        return {"question": example["question"].strip()}

    n_train = min(c["train_samples"], max_samples) if max_samples else c["train_samples"]
    n_val = min(c["val_samples"], max_samples) if max_samples else c["val_samples"]
    return DatasetDict({
        "train": _subsample(raw["train"], n_train, seed).map(clean),
        "validation": _subsample(raw["validation"], n_val, seed).map(clean),
    })


# -------------------------------------------------------------------------- NER
def load_ner(cfg: dict, max_samples: int = None) -> DatasetDict:
    """Scientific NER. Falls back to CoNLL-2003 if SciERC cannot be loaded."""
    c, seed = cfg["ner"], cfg["seed"]
    try:
        raw = load_dataset(c["dataset"])
        name = c["dataset"]
    except Exception as err:  # dataset scripts/hub availability can vary
        print(f"Could not load {c['dataset']} ({err}); falling back to {c['fallback_dataset']}")
        raw = load_dataset(c["fallback_dataset"])
        name = c["fallback_dataset"]
    print(f"NER dataset in use: {name}")

    out = {}
    for split in raw.keys():
        ds = raw[split]
        if max_samples:
            ds = _subsample(ds, max_samples, seed)
        out[split] = ds
    return DatasetDict(out)


LOADERS = {"summarization": load_summarization, "qa": load_qa, "ner": load_ner}


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a ScholarLens dataset loader")
    parser.add_argument("--task", choices=LOADERS.keys(), required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    data = LOADERS[args.task](cfg, args.max_samples)
    for split, ds in data.items():
        print(f"{split}: {len(ds)} examples, columns = {ds.column_names}")
    first_split = next(iter(data))
    print("\nSample row:\n", data[first_split][0])


if __name__ == "__main__":
    main()
