"""Semantic retrieval: find the passages of a paper most relevant to a question.

Passages and questions are embedded with a sentence-transformers model and
ranked by cosine similarity. This is the "semantic similarity" NLP task of the
project, and it feeds the QA model so it only reads the best few passages
instead of the whole paper.
"""

from typing import List, Tuple

import numpy as np


class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.passages: List[str] = []
        self.embeddings = None

    def index(self, passages: List[str]) -> None:
        """Embed all passages of one paper (normalised, so dot product = cosine)."""
        self.passages = passages
        if passages:
            self.embeddings = self.model.encode(passages, normalize_embeddings=True,
                                                show_progress_bar=False)
        else:
            self.embeddings = np.zeros((0, 1))

    def search(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top_k (passage, similarity) pairs, best first."""
        if not self.passages:
            return []
        q = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:top_k]
        return [(self.passages[i], float(scores[i])) for i in order]
