"""特徴量モジュールのテスト（GPUなし、APIモック対応）"""

from unittest.mock import patch, Mock
import numpy as np
import faiss
from signate_studentcup_2025.features import (
    EmbeddingModel,
    OpenRouterEmbeddingBackend,
    build_faiss_index,
    retrieve_top_k,
)


def test_openrouter_embedding_backend_with_mock(mock_openai_client):
    """OpenRouter埋め込みバックエンドのテスト（モック使用）"""
    backend = OpenRouterEmbeddingBackend(
        api_key="test-key",
        model="test-model",
        dim=1536,
    )

    # OpenAIクライアントをモックに置き換え
    backend.client = mock_openai_client

    # エンコード実行（モックは10件を返す）
    texts = ["テスト文章" for _ in range(10)]
    embeddings = backend.encode(texts)

    # 検証
    assert embeddings.shape == (10, 1536)
    assert embeddings.dtype == np.float32


def test_build_faiss_index_with_mock():
    """FAISSインデックス構築のテスト"""
    # モック埋め込みモデル
    mock_model = Mock()
    mock_model.get_dim.return_value = 768
    mock_model.encode.return_value = np.random.rand(10, 768).astype(np.float32)

    # インデックス構築
    corpus = ["テスト"] * 10
    index, embeddings = build_faiss_index(corpus, mock_model, save_path=None)

    # 検証
    assert index.ntotal == 10
    assert embeddings.shape == (10, 768)


def test_retrieve_top_k():
    """Top-K検索のテスト"""
    # モック埋め込みモデル
    mock_model = Mock()
    dim = 768
    mock_model.get_dim.return_value = dim

    # FAISSインデックス作成
    index = faiss.IndexFlatIP(dim)
    corpus_embeddings = np.random.rand(10, dim).astype(np.float32)
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)

    # モックのencodeメソッドを設定
    mock_model.encode.return_value = np.random.rand(1, dim).astype(np.float32)

    # 検索実行
    results = retrieve_top_k("クエリ", index, mock_model, k=5)

    # 検証
    assert len(results) == 5
    assert all(isinstance(doc_id, (int, np.integer)) for doc_id, _ in results)
    assert all(isinstance(score, (float, np.floating)) for _, score in results)
