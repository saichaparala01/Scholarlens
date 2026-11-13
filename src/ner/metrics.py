"""Entity-level precision / recall / F1 for BIO-tagged sequences (pure Python).

An entity counts as correct only if BOTH its type and its exact span match the
gold annotation (the standard CoNLL evaluation). This replaces the seqeval
package so there is nothing extra to install.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

Entity = Tuple[str, int, int]  # (type, start_index, end_index)


def extract_entities(tags: List[str]) -> Set[Entity]:
    """Convert a tag sequence such as [B-Task, I-Task, O] into entity spans."""
    entities: Set[Entity] = set()
    start, etype = None, None
    for i, tag in enumerate(list(tags) + ["O"]):  # sentinel closes the last entity
        if tag == "O":
            prefix, typ = "O", None
        elif len(tag) > 2 and tag[1] == "-" and tag[0] in "BI":
            prefix, typ = tag[0], tag[2:]
        else:  # untagged scheme such as "Task": treat as a continuing entity
            prefix, typ = "I", tag
        if prefix == "O" or prefix == "B" or typ != etype:
            if etype is not None:
                entities.add((etype, start, i - 1))
                etype = None
            if prefix != "O":
                start, etype = i, typ
    return entities


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(100 * precision, 2), "recall": round(100 * recall, 2), "f1": round(100 * f1, 2)}


def entity_scores(true_tags: List[List[str]], pred_tags: List[List[str]]) -> dict:
    """Overall and per-type scores plus token accuracy."""
    counts = defaultdict(lambda: [0, 0, 0])  # type -> [tp, fp, fn]
    correct_tokens = total_tokens = 0
    for gold_seq, pred_seq in zip(true_tags, pred_tags):
        gold, pred = extract_entities(gold_seq), extract_entities(pred_seq)
        for ent in gold & pred:
            counts[ent[0]][0] += 1
        for ent in pred - gold:
            counts[ent[0]][1] += 1
        for ent in gold - pred:
            counts[ent[0]][2] += 1
        correct_tokens += sum(g == p for g, p in zip(gold_seq, pred_seq))
        total_tokens += len(gold_seq)

    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    overall = _prf(tp, fp, fn)
    overall["token_accuracy"] = round(100 * correct_tokens / max(total_tokens, 1), 2)
    overall["per_type"] = {t: {**_prf(*c), "support": c[0] + c[2]} for t, c in sorted(counts.items())}
    return overall
