"""ScholarLens inference pipeline: one object that runs all NLP tasks on a paper.

    from src.pipeline import ScholarLens
    lens = ScholarLens()
    lens.summarize(paper_text)               # abstract-style summary
    lens.entities(paper_text)                # methods / tasks / datasets ...
    lens.ask("What dataset was used?", paper_text)   # retrieval + QA

Models load lazily (on first use) from outputs/<task>/<model>/ where the
training scripts saved them. If a fine-tuned model is missing and
`serving.allow_fallback` is true in config.yaml, a public checkpoint is used so
the demo still runs; `lens.status()` reports which is which.
"""

import hashlib
from collections import defaultdict
from typing import Dict, List

from src.data_loading import load_config
from src.preprocessing import clean_text, split_into_passages, split_sentences
from src.utils import model_path, prefix_for


class ScholarLens:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        self._models: Dict[str, object] = {}
        self._sources: Dict[str, str] = {}
        self._retriever = None
        self._index_key = None

    # ------------------------------------------------------------ model loading
    def _device(self) -> int:
        import torch
        return 0 if torch.cuda.is_available() else -1

    def _resolve(self, task: str):
        """Return (path_or_hub_id, is_fine_tuned) for a task."""
        serving = self.cfg["serving"]
        local = model_path(self.cfg, task, serving[task])
        if local.exists() and any(local.iterdir()):
            return str(local), True
        if serving.get("allow_fallback"):
            print(f"[ScholarLens] No fine-tuned {task} model at {local}. "
                  f"Using public fallback {serving['fallback'][task]} (demo only).")
            return serving["fallback"][task], False
        raise FileNotFoundError(
            f"No fine-tuned {task} model found at {local}. Train it first "
            f"(see README) or set serving.allow_fallback: true in configs/config.yaml.")

    def _load(self, task: str):
        if task in self._models:
            return self._models[task]
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, pipeline)
        path, fine_tuned = self._resolve(task)
        self._sources[task] = "fine-tuned" if fine_tuned else "public fallback"
        if task == "summarization":
            import torch
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
            model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self._models[task] = (tok, model)
        elif task == "ner":
            self._models[task] = pipeline("token-classification", model=path, tokenizer=path,
                                          aggregation_strategy="simple", device=self._device())
        elif task == "qa":
            self._models[task] = pipeline("question-answering", model=path, tokenizer=path,
                                          device=self._device())
        else:
            raise ValueError(task)
        return self._models[task]

    def status(self) -> Dict[str, str]:
        """Which models are loaded and whether they are fine-tuned or fallbacks."""
        return {t: self._sources.get(t, "not loaded yet") for t in ("summarization", "ner", "qa")}

    # ------------------------------------------------------------ summarization
    def _summarize_chunk(self, text: str) -> str:
        import torch
        tok, model = self._load("summarization")
        c = self.cfg["summarization"]
        name = self.cfg["serving"]["summarization"]
        enc = tok(prefix_for(name) + text, max_length=c["max_input_tokens"], truncation=True,
                  return_tensors="pt").to(model.device)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            ids = model.generate(**enc, num_beams=c["generation"]["num_beams"],
                                 max_new_tokens=c["generation"]["max_new_tokens"], early_stopping=True)
        return tok.decode(ids[0], skip_special_tokens=True).strip()

    def summarize(self, text: str, chunk_words: int = 500, max_chunks: int = 6) -> Dict[str, object]:
        """Summarize a long paper: summarize each chunk, then summarize the summaries."""
        words = clean_text(text).split()
        if len(words) < 30:
            return {"summary": "", "chunks": 0, "note": "Text is too short to summarize."}
        chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)][:max_chunks]
        partial = [self._summarize_chunk(c) for c in chunks]
        summary = partial[0]
        if len(partial) > 1:
            combined = " ".join(partial)
            summary = self._summarize_chunk(combined) if len(combined.split()) > 80 else combined
        return {"summary": summary, "chunks": len(chunks), "chunk_summaries": partial}

    # --------------------------------------------------------------------- NER
    def entities(self, text: str, window_words: int = 150, max_windows: int = 40) -> List[dict]:
        """Extract entities window by window; merge duplicates and count mentions."""
        ner = self._load("ner")
        sentences = split_sentences(clean_text(text))
        windows, current = [], []
        for sent in sentences:
            current.append(sent)
            if sum(len(s.split()) for s in current) >= window_words:
                windows.append(" ".join(current))
                current = []
        if current:
            windows.append(" ".join(current))

        found = defaultdict(lambda: {"count": 0, "score": 0.0})
        for window in windows[:max_windows]:
            for ent in ner(window):
                key = (ent["word"].strip(), ent["entity_group"])
                found[key]["count"] += 1
                found[key]["score"] = max(found[key]["score"], float(ent["score"]))
        results = [{"text": w, "label": l, "count": v["count"], "score": round(v["score"], 3)}
                   for (w, l), v in found.items() if w]
        return sorted(results, key=lambda r: (-r["count"], -r["score"]))

    # ------------------------------------------------------ retrieval + answering
    def _get_retriever(self, passages: List[str], key: str):
        if self._retriever is None:
            from src.qa.retriever import Retriever
            self._retriever = Retriever(self.cfg["qa"]["retriever"])
        if key != self._index_key:  # re-embed only when the paper changes
            self._retriever.index(passages)
            self._index_key = key
        return self._retriever

    def ask(self, question: str, text: str) -> Dict[str, object]:
        """Retrieve the most relevant passages, then extract an answer from them."""
        passages = split_into_passages(text)
        if not passages:
            return {"answer": "", "confidence": 0.0, "evidence": "", "note": "No text to search."}
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        retriever = self._get_retriever(passages, key)
        hits = retriever.search(question, top_k=self.cfg["qa"]["top_k_passages"])

        qa = self._load("qa")
        best = {"answer": "", "confidence": 0.0, "evidence": ""}
        for passage, similarity in hits:
            out = qa(question=question, context=passage, handle_impossible_answer=True)
            if out["answer"].strip() and out["score"] > best["confidence"]:
                best = {"answer": out["answer"].strip(), "confidence": round(float(out["score"]), 3),
                        "evidence": passage, "similarity": round(similarity, 3)}
        if not best["answer"]:
            best["note"] = "The paper does not seem to answer this question."
            best["evidence"] = hits[0][0] if hits else ""
        best["retrieved"] = [{"passage": p, "similarity": round(s, 3)} for p, s in hits]
        return best
