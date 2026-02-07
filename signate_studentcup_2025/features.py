from abc import ABC, abstractmethod
from pathlib import Path

import faiss
from loguru import logger
import numpy as np
from tqdm import tqdm
import typer

from signate_studentcup_2025.config import (
    OpenRouterConfig,
    OutputConfig,
)
from signate_studentcup_2025.weave import weave_op_decorator, weave_op_decorator_configured

app = typer.Typer()


# === 埋め込みバックエンド ===


class EmbeddingBackend(ABC):
    """埋め込みバックエンドの抽象基底クラス"""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """テキストを埋め込みベクトルに変換"""
        pass

    @abstractmethod
    def get_dim(self) -> int:
        """埋め込みベクトルの次元数を取得"""
        pass


class OpenRouterEmbeddingBackend(EmbeddingBackend):
    """OpenRouter APIによる埋め込み生成"""

    # バックエンド名
    backend_name = "openrouter"

    def __init__(self, api_key: str, model: str = None, dim: int = None):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model or OpenRouterConfig.EMBEDDING_MODEL
        self._dim = dim or OpenRouterConfig.EMBEDDING_DIM

    @weave_op_decorator_configured("TRACE_EMBEDDINGS")
    def encode(self, texts: list[str]) -> np.ndarray:
        """バッチ処理でAPI呼び出し"""
        # OpenAI Embeddings APIは最大2048テキストまでバッチ可能
        batch_size = 100
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="埋め込み生成中"):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def get_dim(self) -> int:
        return self._dim


# Local embedding model backend


class SentenceTransformerBackend(EmbeddingBackend):
    """
    SentenceTransformer based embedding backend for local models.

    Supports models like:
    - intfloat/multilingual-e5-large (1024 dim)
    - intfloat/multilingual-e5-base (768 dim)
    - ruri-large (Japanese specialized)
    """

    backend_name = "sentence-transformer"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cpu",
        batch_size: int = 32,
        prefix_query: str = "query:",
        prefix_passage: str = "passage:",
    ):
        """
        Initialize SentenceTransformer backend.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("cpu" or "cuda")
            batch_size: Batch size for encoding
            prefix_query: Prefix for query texts (E5 models require "query:" prefix)
            prefix_passage: Prefix for corpus/passages (E5 models require "passage:" prefix)
        """
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.prefix_query = prefix_query
        self.prefix_passage = prefix_passage
        logger.info(f"Model loaded. Dimension: {self._dim}, Device: {device}")

    @weave_op_decorator_configured("TRACE_EMBEDDINGS")
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            is_query: If True, add query prefix (for E5 models)

        Returns:
            Embedding array of shape (len(texts), dim)
        """
        # Add prefix for E5 models if specified
        prefix = self.prefix_query if is_query else self.prefix_passage
        if prefix and (self.prefix_query or self.prefix_passage):
            texts = [f"{prefix} {text}" for text in texts]

        # Encode with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # E5 models require L2 normalization
        )

        return embeddings.astype(np.float32)

    def get_dim(self) -> int:
        return self._dim


class EmbeddingModel:
    """Unified embedding model interface."""

    def __init__(
        self,
        backend: str = "openrouter",
        model_name: str | None = None,
        device: str = "cpu",
    ):
        """
        Initialize embedding model.

        Args:
            backend: Backend type ("openrouter" or "sentence-transformer")
            model_name: Model name (optional, uses config default if not specified)
            device: Device for local models ("cpu" or "cuda")
        """
        if backend == "openrouter":
            if not OpenRouterConfig.API_KEY:
                raise ValueError("OPENROUTER_API_KEYが設定されていません")
            self.backend = OpenRouterEmbeddingBackend(
                api_key=OpenRouterConfig.API_KEY, model=OpenRouterConfig.EMBEDDING_MODEL
            )
        elif backend == "sentence-transformer":
            # Use provided model_name or default to e5-large
            if model_name is None:
                model_name = "intfloat/multilingual-e5-large"
            self.backend = SentenceTransformerBackend(
                model_name=model_name,
                device=device,
            )
        else:
            raise ValueError(f"未知のバックエンド: {backend}")

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings
            is_query: If True, use query prefix (for E5 models with sentence-transformer backend)

        Returns:
            Embedding array
        """
        # For backends that don't support is_query parameter, we need to handle it
        if hasattr(self.backend, "encode"):
            import inspect

            sig = inspect.signature(self.backend.encode)
            if "is_query" in sig.parameters:
                return self.backend.encode(texts, is_query=is_query)
            else:
                return self.backend.encode(texts)
        return self.backend.encode(texts)

    def get_dim(self) -> int:
        return self.backend.get_dim()


# === FAISSインデックス構築 ===


@weave_op_decorator
def build_faiss_index(
    corpus_texts: list[str],
    model: EmbeddingModel,
    save_path: Path | None = None,
    wandb_run=None,
) -> tuple[faiss.Index, np.ndarray]:
    """
    FAISSインデックスを構築

    Args:
        corpus_texts: コーパステキストのリスト
        model: 埋め込みモデル
        save_path: インデックス保存先（オプション）
        wandb_run: Wandb Runオブジェクト（オプション）

    Returns:
        (FAISSインデックス, 埋め込み配列)
    """
    # 埋め込み生成 (corpus uses is_query=False for E5 "passage:" prefix)
    logger.info(f"コーパス埋め込み生成: {len(corpus_texts)}件")
    embeddings = model.encode(corpus_texts, is_query=False)

    # 正規化（コサイン類似度用）
    faiss.normalize_L2(embeddings)

    # インデックス構築
    index = faiss.IndexFlatIP(model.get_dim())
    index.add(embeddings)

    logger.info(f"FAISSインデックス構築完了: {index.ntotal}ベクトル")

    # 保存
    if save_path:
        import pickle

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"index": index, "embeddings": embeddings}, f)
        logger.info(f"インデックスを保存: {save_path}")

    # Wandb Artifactとしてログ
    if wandb_run and save_path:
        from datetime import datetime
        import json

        import wandb as wandb_module

        from signate_studentcup_2025.config import ArtifactsConfig

        if ArtifactsConfig.LOG_FAISS_INDEX:
            # バックエンド名を取得（backendはEmbeddingBackendオブジェクト）
            if hasattr(model, "backend") and hasattr(model.backend, "backend_name"):
                backend_name = model.backend.backend_name
            else:
                backend_name = "unknown"

            # モデル名を取得
            if hasattr(model, "backend") and hasattr(model.backend, "model_name"):
                model_name = model.backend.model_name
            elif hasattr(model, "backend") and hasattr(model.backend, "model"):
                # For OpenRouter backend where model is a string
                model_name = str(model.backend.model)
            else:
                model_name = backend_name

            artifact = wandb_module.Artifact(
                name=f"faiss-index-{backend_name}",
                type="faiss_index",
                metadata={
                    "backend": backend_name,
                    "embedding_model": model_name,
                    "dimension": model.get_dim(),
                    "num_vectors": len(corpus_texts),
                    "index_type": "IndexFlatIP",
                    "created_at": datetime.now().isoformat(),
                },
            )

            # インデックスファイルを追加
            artifact.add_file(str(save_path))

            # メタデータJSONを追加
            metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "corpus_size": len(corpus_texts),
                        "embedding_dim": model.get_dim(),
                        "backend": backend_name,
                        "model": model_name,
                    },
                    f,
                )
            artifact.add_file(str(metadata_path))

            wandb_run.log_artifact(artifact)
            logger.info(f"Logged FAISS index artifact: {artifact.name}")

            # 一時メタデータファイルを削除
            metadata_path.unlink(missing_ok=True)

    return index, embeddings


@weave_op_decorator_configured("TRACE_RETRIEVAL")
def retrieve_top_k(
    query_text: str,
    index: faiss.Index,
    model: EmbeddingModel,
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    上位K件の類似ドキュメントを検索

    Args:
        query_text: クエリテキスト
        index: FAISSインデックス
        model: 埋め込みモデル
        k: 取得件数

    Returns:
        [(ドキュメントID, スコア), ...] のリスト
    """
    # クエリ埋め込み (is_query=True for E5 models)
    query_emb = model.encode([query_text], is_query=True)
    faiss.normalize_L2(query_emb)

    # 検索
    scores, ids = index.search(query_emb, k)

    return list(zip(ids[0], scores[0]))


@app.command()
def main(
    input_path: Path = typer.Option(None, help="入力TSVパス"),
    output_path: Path = typer.Option(None, help="出力インデックスパス"),
):
    """
    特徴量生成スクリプト（FAISSインデックス構築）
    """
    logger.info("特徴量生成中...")

    # デフォルトパス設定
    if output_path is None:
        output_path = OutputConfig.INTERIM_DIR / "faiss_index.pkl"

    # TODO: データ読み込みとインデックス構築の実装
    logger.info("特徴量生成完了")


if __name__ == "__main__":
    app()
