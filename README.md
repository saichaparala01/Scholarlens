# ScholarLens: An NLP Assistant for Research Papers

## Team
| Name | Student ID | Module |
|------|-----------|--------|
| Member 1 | TODO | Data preprocessing + Summarization |
| Member 2 | TODO | Semantic retrieval + Question Answering |
| Member 3 | TODO | Named Entity Recognition + Evaluation |
| Member 4 | TODO | Demo app + Architecture diagrams + Repo |

Course: TODO | Instructor: TODO | Semester: TODO

## Introduction
ScholarLens is a multi-task NLP pipeline that helps readers digest scientific papers.
Given a paper, it (1) generates an abstract-style summary, (2) extracts key scientific
entities such as methods, datasets, tasks and metrics, and (3) answers natural-language
questions about the paper using semantic retrieval followed by extractive QA.

Each module fine-tunes at least two transformer variants so we can compare them with
multiple metrics (ROUGE, BERTScore, F1, Exact Match, entity-level precision/recall/F1).

## Pipeline
```
Paper text -> cleaning -> section/sentence split
                 |-> Summarization (BART vs T5)        -> summary
                 |-> NER (SciBERT vs BERT-base)        -> entities
                 |-> Retrieval (sentence-transformers) -> passages
                          -> QA (DistilBERT vs RoBERTa) -> answer
```

## Repository layout
```
scholarlens/
  configs/config.yaml        # model names, dataset sizes, hyperparameters
  src/
    preprocessing.py         # shared text cleaning and sentence splitting
    data_loading.py          # dataset loaders for all three tasks
    summarization/           # Member 1
    qa/                      # Member 2
    ner/                     # Member 3
    evaluation/              # metrics scripts
  app/                       # Gradio demo (Member 4)
  notebooks/                 # Colab training notebooks
  results/                   # metrics tables, plots (generated)
  docs/                      # architecture diagrams, report
```

## Setup
```bash
pip install -r requirements.txt
python -m src.data_loading --task summarization --max-samples 2000   # quick smoke test
```

## Status
- [x] Repo scaffold, preprocessing, data loaders
- [ ] Summarization fine-tuning (BART vs T5)
- [ ] QA fine-tuning (DistilBERT vs RoBERTa)
- [ ] NER fine-tuning (SciBERT vs BERT)
- [ ] Evaluation and comparison tables
- [ ] Demo app
- [ ] Report (IEEE format) and presentation
