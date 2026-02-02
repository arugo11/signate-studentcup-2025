"""テスト設定とフィクスチャ"""

from unittest.mock import Mock
import pytest
import polars as pl
import numpy as np


@pytest.fixture
def sample_base_df():
    """サンプルベース作品DataFrame"""
    return pl.DataFrame({
        "id": [1, 2, 3],
        "category": ["映画", "アニメ", "漫画"],
        "title": ["作品A", "作品B", "作品C"],
        "story": ["あらすじA", "あらすじB", "あらすじC"]
    })


@pytest.fixture
def sample_fiction_df():
    """サンプル架空作品DataFrame"""
    return pl.DataFrame({
        "id": [1, 2],
        "story": ["混合あらすじ1", "混合あらすじ2"],
        "id_a": [1, 2],
        "id_b": [2, 3],
        "title_a": ["作品A", "作品B"],
        "title_b": ["作品B", "作品C"],
    })


@pytest.fixture
def mock_openai_client():
    """OpenAIクライアントのモック"""
    mock_client = Mock()

    # 埋め込みAPIのモック
    mock_embedding_response = Mock()
    mock_embedding_response.data = [
        Mock(embedding=np.random.rand(1536).tolist()) for _ in range(10)
    ]
    mock_client.embeddings.create.return_value = mock_embedding_response

    # チャットAPIのモック
    mock_chat_response = Mock()
    mock_chat_response.choices = [
        Mock(message=Mock(content='{"work_a_id": 1, "work_b_id": 2}'))
    ]
    mock_client.chat.completions.create.return_value = mock_chat_response

    return mock_client


@pytest.fixture
def sample_api_key():
    """テスト用ダミーAPIキー"""
    return "test-api-key-12345"
