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
from datasets import ClassLabel, DatasetDict, Sequence, load_dataset

from src.preprocessing import clean_text, is_valid_pair

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_hub(name: str):
    """load_dataset with fallbacks for script-based datasets and newer `datasets` versions.

    1) plain load  2) trust_remote_code=True (script datasets on datasets<4)
    3) the Hub's auto-converted parquet branch (works without any script)
    """
    last_err = None
    for extra in ({}, {"trust_remote_code": True}, {"revision": "refs/convert/parquet"}):
        try:
            return load_dataset(name, **extra)
        except Exception as err:  # try the next strategy
            last_err = err
    raise last_err


def _subsample(ds, n: int, seed: int):
    """Shuffle with a fixed seed and keep at most n rows."""
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


# ---------------------------------------------------------------- Summarization
def load_summarization(cfg: dict, max_samples: int = None) -> DatasetDict:
    """Scientific papers (article) paired with their abstracts (abstract)."""
    c, seed = cfg["summarization"], cfg["seed"]
    raw, used = None, None
    for name in c["datasets"]:
        try:
            candidate = _load_hub(name)
        except Exception as err:
            print(f"[summarization] could not load {name}: {str(err)[:120]}")
            continue
        cols = candidate[next(iter(candidate.keys()))].column_names
        if "article" in cols and "abstract" in cols:
            raw, used = candidate, name
            break
        print(f"[summarization] {name} has columns {cols}; need article + abstract. Skipping.")
    if raw is None:
        raise RuntimeError("No summarization dataset in configs/config.yaml could be loaded.")
    print(f"Summarization dataset in use: {used}")

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
        ds = _subsample(raw[split], int(n * 1.2) + 1, seed).map(clean)
        ds = ds.filter(lambda ex: is_valid_pair(ex["article"], ex["abstract"]))
        out[split] = ds.select(range(min(n, len(ds))))
    return DatasetDict(out)


# --------------------------------------------------------------------------- QA
def load_qa(cfg: dict, max_samples: int = None) -> DatasetDict:
    """SQuAD 2.0 (includes unanswerable questions, which makes QA harder)."""
    c, seed = cfg["qa"], cfg["seed"]
    raw = _load_hub(c["dataset"])  # splits: train / validation

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
def _ner_label_names(first_split, tag_col: str, raw) -> list:
    """Return the ordered tag names, from ClassLabel metadata or the data itself."""
    feature = first_split.features[tag_col]
    inner = getattr(feature, "feature", None)
    if inner is not None and hasattr(inner, "names"):
        return list(inner.names)
    seen = set()
    for split in raw.values():
        for tags in split[tag_col]:
            seen.update(tags)
    return sorted(seen, key=str)


def _pick_split(raw, *names):
    for n in names:
        if n in raw:
            return raw[n]
    return None


def load_ner(cfg: dict, max_samples: int = None) -> DatasetDict:
    """Token-classification data with columns `tokens` and integer `ner_tags`.

    Tries every dataset listed in config.yaml in order and uses the first one
    that loads and has token + tag columns. The returned splits are always
    train / validation / test, and label names live on the `ner_tags` feature:
        ds["train"].features["ner_tags"].feature.names
    """
    c, seed = cfg["ner"], cfg["seed"]
    for name in c["datasets"]:
        try:
            raw = _load_hub(name)
        except Exception as err:  # hub availability / script datasets can fail
            print(f"[NER] could not load {name}: {str(err)[:120]}")
            continue
        first = raw[next(iter(raw.keys()))]
        token_col = "tokens" if "tokens" in first.column_names else None
        tag_col = next((t for t in ("ner_tags", "labels", "tags") if t in first.column_names), None)
        if token_col is None or tag_col is None:
            print(f"[NER] {name} has columns {first.column_names}; need tokens + ner_tags. Skipping.")
            continue

        labels = _ner_label_names(first, tag_col, raw)
        label2id = {str(l): i for i, l in enumerate(labels)}

        def normalise(example):
            tags = example[tag_col]
            ids = [t if isinstance(t, int) else label2id[str(t)] for t in tags]
            return {"tokens": list(example[token_col]), "ner_tags": ids}

        splits = {}
        for split_name, ds in raw.items():
            ds = ds.map(normalise, remove_columns=ds.column_names)
            splits[split_name] = ds.cast_column("ner_tags", Sequence(ClassLabel(names=[str(l) for l in labels])))

        train = _pick_split(splits, "train")
        val = _pick_split(splits, "validation", "valid", "dev")
        test = _pick_split(splits, "test")
        if val is None:  # carve a validation set out of train
            parts = train.train_test_split(test_size=0.1, seed=seed)
            train, val = parts["train"], parts["test"]
        if test is None:
            test = val

        out = {"train": train, "validation": val, "test": test}
        if max_samples:
            out = {k: _subsample(v, max_samples, seed) for k, v in out.items()}
        print(f"NER dataset in use: {name} | {len(labels)} tags: {labels}")
        return DatasetDict(out)
    raise RuntimeError("No NER dataset in configs/config.yaml could be loaded.")


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
