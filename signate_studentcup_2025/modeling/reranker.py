"""Cross-encoder reranking implementation.

This module provides cross-encoder models for reranking retrieval results.
Cross-encoders score query-document pairs jointly, providing more accurate
ranking than bi-encoders while being computationally more expensive.

Supported models:
- Ruri rerankers (Japanese-specialized, SOTA on JMTEB)
- mGTE multilingual rerankers
- Any SentenceTransformer CrossEncoder model
"""

from abc import ABC, abstractmethod

from loguru import logger
import numpy as np


class CrossEncoderBackend(ABC):
    """Abstract base class for cross-encoder rerankers."""

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> np.ndarray:
        """
        Score query-document pairs.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            Array of scores (same length as documents)
        """
        pass


class SentenceTransformerReranker(CrossEncoderBackend):
    """
    SentenceTransformer CrossEncoder-based reranker.

    Supports any cross-encoder model from the SentenceTransformer ecosystem.
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-large",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize SentenceTransformer reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("cpu" or "cuda")
            batch_size: Batch size for inference
        """
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading CrossEncoder model: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        self.model_name = model_name
        logger.info(f"CrossEncoder loaded. Device: {device}")

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        """
        Score query-document pairs using cross-encoder.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            Array of scores (higher is better)
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Predict scores
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )

        return np.array(scores, dtype=np.float32)


class RuriReranker(SentenceTransformerReranker):
    """
    Ruri reranker - Japanese-specialized cross-encoder.

    Ruri is the top-performing Japanese embedding model on JMTEB benchmark.
    This reranker provides significant accuracy gains for Japanese text.

    Available models:
    - large: cl-nagoya/ruri-reranker-large (1024 dim, best accuracy)
    - small: cl-nagoya/ruri-reranker-small (768 dim, faster)
    """

    def __init__(
        self,
        model_size: str = "large",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize Ruri reranker.

        Args:
            model_size: Model size ("large" or "small")
            device: Device to use ("cpu" or "cuda")
            batch_size: Batch size for inference
        """
        model_map = {
            "large": "cl-nagoya/ruri-reranker-large",
            "small": "cl-nagoya/ruri-reranker-small",
        }

        if model_size not in model_map:
            raise ValueError(
                f"Unknown model size: {model_size}. Choose from {list(model_map.keys())}"
            )

        model_name = model_map[model_size]
        super().__init__(model_name=model_name, device=device, batch_size=batch_size)
        logger.info(f"Initialized RuriReranker ({model_size})")


def rerank_candidates(
    query: str,
    candidates: list[tuple[int, float, str]],
    reranker: CrossEncoderBackend,
) -> list[tuple[int, float]]:
    """
    Rerank candidates using cross-encoder.

    Args:
        query: Query text
        candidates: List of (doc_id, bi_score, text) tuples
        reranker: Cross-encoder reranker instance

    Returns:
        List of (doc_id, rerank_score) tuples sorted by rerank_score (descending)
    """
    if not candidates:
        return []

    # Extract document IDs and texts
    doc_ids = [c[0] for c in candidates]
    doc_texts = [c[2] for c in candidates]

    # Score with cross-encoder
    rerank_scores = reranker.score(query, doc_texts)

    # Sort by rerank score (descending)
    reranked = sorted(zip(doc_ids, rerank_scores), key=lambda x: x[1], reverse=True)

    return list(reranked)
