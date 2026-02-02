from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

from signate_studentcup_2025.config import (
    PROCESSED_DATA_DIR,
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


# TODO: ローカル埋め込みモデルバックエンドの実装
"""
TODO: SentenceTransformerBackendクラスを追加して以下のモデルをサポート:
- intfloat/multilingual-e5-base
- sonoisa/sentence-distilbert-base-ja

実装手順:
1. `uv add sentence-transformers` でライブラリ追加
2. SentenceTransformerBackendクラスを実装
3. EmbeddingModel.__init__() で "sentence-transformer" を選択可能にする
4. EmbeddingBackendConfigでモデル名を指定可能にする

class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self._dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32)

    def get_dim(self) -> int:
        return self._dim
"""


class EmbeddingModel:
    """統一された埋め込みモデルインターフェース"""

    def __init__(self, backend: str = "openrouter"):
        if backend == "openrouter":
            if not OpenRouterConfig.API_KEY:
                raise ValueError("OPENROUTER_API_KEYが設定されていません")
            self.backend = OpenRouterEmbeddingBackend(
                api_key=OpenRouterConfig.API_KEY,
                model=OpenRouterConfig.EMBEDDING_MODEL
            )
        elif backend == "sentence-transformer":
            # 後で実装
            raise NotImplementedError("Sentence-Transformerバックエンドは未実装です")
        else:
            raise ValueError(f"未知のバックエンド: {backend}")

    def encode(self, texts: list[str]) -> np.ndarray:
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
    # 埋め込み生成
    logger.info(f"コーパス埋め込み生成: {len(corpus_texts)}件")
    embeddings = model.encode(corpus_texts)

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
        from signate_studentcup_2025.config import ArtifactsConfig
        import wandb as wandb_module
        from datetime import datetime
        import json

        if ArtifactsConfig.LOG_FAISS_INDEX:
            # バックエンド名を取得（backendはEmbeddingBackendオブジェクト）
            if hasattr(model, 'backend') and hasattr(model.backend, 'backend_name'):
                backend_name = model.backend.backend_name
            else:
                backend_name = "unknown"

            # モデル名を取得
            if hasattr(model, 'backend') and hasattr(model.backend, 'model'):
                model_name = model.backend.model
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
                }
            )

            # インデックスファイルを追加
            artifact.add_file(str(save_path))

            # メタデータJSONを追加
            metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "corpus_size": len(corpus_texts),
                    "embedding_dim": model.get_dim(),
                    "backend": backend_name,
                    "model": model_name,
                }, f)
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
    # クエリ埋め込み
    query_emb = model.encode([query_text])
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
