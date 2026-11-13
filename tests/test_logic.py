"""Offline unit tests: no internet, no GPU, no model downloads.

Run with:  python -m pytest tests -q
A tiny local tokenizer stands in for BERT/RoBERTa so the label-alignment code
(the part most likely to contain bugs) can be checked quickly.
"""

import pytest

from src.preprocessing import clean_text, is_valid_pair, split_into_passages, split_sentences
from src.qa.metrics import f1_score, normalize_answer, score_squad_v2

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    corpus = ["BERT is a language model from Google", "What is BERT", "models use transformers",
              "unanswerable question about nothing", "named entity recognition of methods"] * 20
    tok.train_from_iterator(corpus, trainers.WordPieceTrainer(
        vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]")


# ----------------------------------------------------------------- preprocessing
def test_clean_text_removes_noise():
    raw = "Deep models [1, 2] help (Smith et al., 2019). See Figure 3 at https://x.org and $x^2$. @xmath1"
    cleaned = clean_text(raw)
    for bad in ("[1, 2]", "http", "@xmath", "Figure 3", "Smith"):
        assert bad not in cleaned


def test_sentence_split_protects_abbreviations():
    sents = split_sentences("We follow Devlin et al. closely, e.g. using BERT. It works well.")
    assert len(sents) == 2 and "et al." in sents[0]


def test_passages_overlap_and_cover_text():
    text = " ".join(f"Sentence number {i} is about neural networks." for i in range(60))
    passages = split_into_passages(text, max_words=50)
    assert len(passages) > 3
    assert "Sentence number 0" in passages[0] and "Sentence number 59" in passages[-1]


def test_pair_filter():
    assert is_valid_pair("word " * 200, "word " * 40)
    assert not is_valid_pair("word " * 50, "word " * 40)


# ------------------------------------------------------------------- QA metrics
def test_qa_metrics():
    assert normalize_answer("The Transformer!") == "transformer"
    assert f1_score("the big model", "big model") == pytest.approx(1.0)
    scores = score_squad_v2(["BERT", "", "wrong"], [["BERT"], [], ["right"]])
    assert scores["exact_match"] == pytest.approx(66.67, abs=0.01)
    assert scores["no_ans_accuracy"] == 100.0


# ------------------------------------------------------- QA span alignment
def test_qa_span_labels_point_at_answer(tiny_tokenizer):
    from src.qa.train import prepare_train_features
    context = "BERT is a language model from Google"
    examples = {
        "question": ["What is BERT", "What is nothing"],
        "context": [context, context],
        "answers": [{"text": ["language model"], "answer_start": [10]},
                    {"text": [], "answer_start": []}],
    }
    feats = prepare_train_features(examples, tiny_tokenizer, max_len=64, stride=16)
    ids = feats["input_ids"][0]
    s, e = feats["start_positions"][0], feats["end_positions"][0]
    assert tiny_tokenizer.decode(ids[s:e + 1]).replace(" ##", "") == "language model"
    cls = ids.index(tiny_tokenizer.cls_token_id)
    assert feats["start_positions"][1] == cls and feats["end_positions"][1] == cls


# ------------------------------------------------------------------ NER alignment
def test_ner_alignment_labels_first_subword_only(tiny_tokenizer):
    from src.ner.train import tokenize_and_align
    batch = {"tokens": [["BERT", "models", "transformers"]], "ner_tags": [[1, 0, 2]]}
    out = tokenize_and_align(batch, tiny_tokenizer, max_len=32)
    labels = out["labels"][0]
    assert labels[0] == -100 and labels[-1] == -100            # [CLS] and [SEP]
    assert [l for l in labels if l != -100] == [1, 0, 2]       # one label per word, in order


def test_ner_decode_and_entity_metrics():
    import numpy as np
    from src.ner.metrics import entity_scores, extract_entities
    from src.ner.train import decode_predictions
    names = ["O", "B-Method", "I-Method"]
    logits = np.zeros((1, 4, 3)); logits[0, 0, 1] = 5; logits[0, 1, 2] = 5; logits[0, 2, 0] = 5
    label_ids = np.array([[1, 2, 0, -100]])
    true_tags, pred_tags = decode_predictions(logits, label_ids, names)
    assert true_tags == pred_tags == [["B-Method", "I-Method", "O"]]
    assert entity_scores(true_tags, pred_tags)["f1"] == 100.0
    # one wrong span out of two gold entities -> precision 50, recall 50
    gold = [["B-Task", "I-Task", "O", "B-Method"]]
    pred = [["B-Task", "O", "O", "B-Method"]]
    scores = entity_scores(gold, pred)
    assert scores["precision"] == 50.0 and scores["recall"] == 50.0
    assert extract_entities(["B-A", "I-A", "B-A"]) == {("A", 0, 1), ("A", 2, 2)}


# -------------------------------------------------------------- summarization input
def test_summarization_tokenize_fn(tiny_tokenizer):
    from src.summarization.train import build_tokenize_fn
    fn = build_tokenize_fn(tiny_tokenizer, "summarize: ", max_in=8, max_out=5)
    out = fn({"article": ["BERT is a language model from Google " * 5], "abstract": ["BERT is a language model"]})
    assert len(out["input_ids"][0]) <= 8 and len(out["labels"][0]) <= 5


# ------------------------------------------------------------------- comparison
def test_compare_table():
    from src.evaluation.compare import markdown_table
    table = markdown_table([{"model": "a", "metrics": {"f1": 1.0}, "train_minutes": 3}])
    assert "| a | 1.0 | 3 |" in table
