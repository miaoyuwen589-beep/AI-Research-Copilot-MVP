"""Small, offline retrieval layer for the Apple filing research page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_DOCUMENT_PATH = Path(__file__).parent / "data" / "apple_filing_excerpts.json"


def load_documents(path: str | Path = DEFAULT_DOCUMENT_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        documents = json.load(handle)
    if not documents:
        raise ValueError("The filing collection is empty")
    for document in documents:
        if not {"document_id", "source", "text"}.issubset(document):
            raise ValueError("Each filing excerpt needs document_id, source and text")
    return documents


class FilingRetriever:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(document["text"] for document in documents)

    def search(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Please enter a question about the Apple filings")
        query = self.vectorizer.transform([cleaned])
        scores = (self.matrix @ query.T).toarray().ravel()
        order = np.argsort(-scores)[: min(top_k, len(self.documents))]
        return [
            {**self.documents[int(index)], "score": float(scores[int(index)])}
            for index in order
        ]
