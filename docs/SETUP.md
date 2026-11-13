# ScholarLens: setup and run guide (all free tools)

| Need | Free tool |
|------|-----------|
| Code editor with AI help | [Cursor](https://cursor.com) (Hobby/free plan) or VS Code |
| GPU for training | Google Colab free tier (T4 GPU) |
| Models and datasets | Hugging Face Hub (free account optional) |
| Training / NLP | PyTorch, Hugging Face Transformers, sentence-transformers |
| Backend | FastAPI + Uvicorn |
| Frontend | Plain HTML/CSS/JS (no build step), optional Gradio demo |
| Code hosting | GitHub |
| Diagrams | draw.io (diagrams.net) |
| Plagiarism check | Your school's tool, or free checkers |

## Project layout

```
configs/config.yaml        all settings: models, dataset sizes, hyperparameters
src/preprocessing.py       text cleaning, sentence + passage splitting
src/data_loading.py        dataset loaders (summarization, QA, NER)
src/summarization/         train.py, evaluate.py        (BART vs T5)
src/qa/                    retriever.py, train.py, evaluate.py, metrics.py   (DistilBERT vs RoBERTa)
src/ner/                   train.py                      (SciBERT vs BERT)
src/evaluation/compare.py  builds the report's comparison tables and charts
src/pipeline.py            the inference pipeline used by the backend and demo
backend/main.py            FastAPI server
frontend/index.html        the web interface
app/gradio_demo.py         optional Gradio demo (good for Colab)
notebooks/ScholarLens_Colab.ipynb   train everything on a free GPU
tests/test_logic.py        offline unit tests
```

## Part A: open the project in Cursor (everyone)

1. Install Cursor from cursor.com and sign in with the free plan.
2. Unzip the project, then **File > Open Folder** and choose the `scholarlens` folder.
3. Open the built-in terminal (**Ctrl+`**).
4. Create an environment and install the libraries (Python 3.10 or newer):
   ```bash
   python -m venv .venv
   # Windows:   .venv\Scripts\activate
   # Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Check the project works without any GPU or downloads:
   ```bash
   python -m pytest tests -q
   ```
6. Push to GitHub from Cursor's **Source Control** tab (left sidebar), or upload on github.com.

You do **not** train on your laptop. Training needs a GPU, so it happens in Colab (Part B).

## Part B: train on Colab (each member does their own module)

1. Push the project to GitHub, then open `notebooks/ScholarLens_Colab.ipynb` in Colab
   (github.com > the notebook file > "Open in Colab", or upload it at colab.research.google.com).
2. **Runtime > Change runtime type > T4 GPU.**
3. In the first cell, set `REPO_URL` to your repo. Run Section 0 and 1.
4. Run Section 2 (200-example test). Fix any error before the long run.
5. Run your module's section:

   | Member | Section | Commands (inside the notebook) |
   |--------|---------|--------------------------------|
   | 1 | 3A | `src.summarization.train` with `facebook/bart-base`, then `t5-small` |
   | 2 | 3B | `src.qa.train` with `distilbert-base-uncased`, then `roberta-base` |
   | 3 | 3C | `src.ner.train` with `allenai/scibert_scivocab_uncased`, then `bert-base-uncased` |

6. Each run saves a metrics file to `results/` (on your Drive). Download them (Section 5) and
   upload them to the GitHub repo's `results/` folder.
7. Member 3 runs `python -m src.evaluation.compare` once all six files exist. It writes
   `results/comparison.md` and chart images in `docs/` for the report.

If Colab runs out of time or memory, lower `train_samples` in `configs/config.yaml`
(for example summarization 5000, QA 10000). Do not drop a model: the two-model
comparison is a requirement.

## Part C: run the web app (backend + frontend)

1. Put your trained model folders in `outputs/` (Colab saved them to Drive: copy
   `outputs/summarization`, `outputs/qa`, `outputs/ner` into the project), and set the winners in
   `configs/config.yaml` under `serving:`.
2. Start the server from the project root:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
3. Open http://localhost:8000. Paste a paper (or upload a PDF), then use the Summary, Entities and Ask tabs.
   API docs are at http://localhost:8000/docs.
4. Before your models are trained, `serving.allow_fallback: true` lets the app use public checkpoints
   so you can check the interface. **Set it to `false` for the final demo** so the app uses *your* models.

Alternative for a live demo from Colab: run Section 4 of the notebook, which opens a public Gradio link.

## Part D: before you submit

- [ ] README has team names, IDs and the short introduction
- [ ] `results/` contains the six metric files and `comparison.md`
- [ ] `docs/` contains your architecture diagrams
- [ ] Report written from your real numbers, rewritten in your own words, plagiarism-checked
- [ ] Every member can explain the code in their own module
