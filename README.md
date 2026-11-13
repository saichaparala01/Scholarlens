# ScholarLens: An NLP Assistant for Research Papers

## Team
| Name | Student ID | Module |
|------|-----------|--------|
| Member 1 | TODO | Data preprocessing + Summarization (BART vs T5) + Architecture diagrams |
| Member 2 | TODO | Semantic retrieval + Question Answering (DistilBERT vs RoBERTa) + Demo app |
| Member 3 | TODO | Named Entity Recognition (SciBERT vs BERT) + Evaluation + Repo |

Course: TODO | Instructor: TODO | Semester: TODO

## Introduction
ScholarLens is a multi-task NLP system that helps readers digest scientific papers.
Given a paper, it (1) generates an abstract-style summary, (2) extracts key scientific
entities such as methods, tasks and datasets, and (3) answers natural-language questions
about the paper using semantic retrieval followed by extractive question answering.

Each task fine-tunes two transformer variants so they can be compared with multiple
metrics: ROUGE-1/2/L and BERTScore (summarization), Exact Match and F1 (QA), and
entity-level precision, recall and F1 (NER). Everything uses free tools: Hugging Face
Transformers, PyTorch, Google Colab, FastAPI and plain HTML.

## Pipeline
```
Paper text -> cleaning -> sentence / passage split
                 |-> Summarization (BART vs T5)          -> summary
                 |-> NER (SciBERT vs BERT)               -> entities
                 |-> Retrieval (sentence-transformers)   -> top passages
                          -> QA (DistilBERT vs RoBERTa)  -> answer + evidence
```

## Repository layout
See [docs/SETUP.md](docs/SETUP.md) for the full layout and step-by-step instructions.

```
configs/config.yaml     models, dataset sizes, hyperparameters (edit here, not in scripts)
src/                    preprocessing, data loading, training + evaluation per task, pipeline
backend/main.py         FastAPI server (REST API)
frontend/index.html     web interface
app/gradio_demo.py      optional Gradio demo
notebooks/              Colab notebook that trains everything on a free GPU
results/                metric files and comparison tables
docs/                   architecture diagrams and setup guide
tests/                  offline unit tests
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests -q                         # offline sanity check
python -m src.data_loading --task qa --max-samples 200   # data loads?

# Train (use Colab, GPU needed). One command per model:
python -m src.summarization.train --model facebook/bart-base
python -m src.summarization.train --model t5-small
python -m src.qa.train --model distilbert-base-uncased
python -m src.qa.train --model roberta-base
python -m src.ner.train --model allenai/scibert_scivocab_uncased
python -m src.ner.train --model bert-base-uncased
python -m src.evaluation.compare                  # builds the comparison tables

# Run the app
uvicorn backend.main:app --port 8000              # open http://localhost:8000
```

## Datasets
| Task | Dataset | Notes |
|------|---------|-------|
| Summarization | ccdv/arxiv-summarization | scientific papers paired with their abstracts |
| QA | SQuAD 2.0 | 130k+ questions, includes unanswerable ones |
| NER | first available of the datasets in `configs/config.yaml` | the script prints which one it used |

## Status
- [x] Repo, preprocessing, data loaders, training and evaluation scripts
- [x] Pipeline, backend, frontend, Colab notebook
- [ ] Summarization trained and evaluated (BART vs T5)
- [ ] QA trained and evaluated (DistilBERT vs RoBERTa)
- [ ] NER trained and evaluated (SciBERT vs BERT)
- [ ] Comparison tables and diagrams
- [ ] Report (IEEE format) and presentation
