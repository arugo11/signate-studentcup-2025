"""データセットモジュールのテスト"""

from pathlib import Path
import polars as pl
from signate_studentcup_2025.dataset import (
    load_base_stories,
    load_fiction_data,
    prepare_corpus,
    validate_submission_format,
)


def test_prepare_corpus(sample_base_df):
    """コーパス準備のテスト"""
    corpus = prepare_corpus(sample_base_df)

    assert len(corpus) == 3
    assert "タイトル: 作品A" in corpus[0]
    assert "あらすじ: あらすじA" in corpus[0]


def test_validate_submission_format():
    """提出形式バリデーションのテスト"""
    # 正常な形式
    valid_df = pl.DataFrame({
        "id": range(1, 341),
        "a": [1] * 340,
        "b": [2] * 340,
    })
    assert validate_submission_format(valid_df) is True

    # 異常な形式（カラム不正）
    invalid_df = pl.DataFrame({
        "id": range(1, 341),
        "wrong_column": [1] * 340,
        "b": [2] * 340,
    })
    assert validate_submission_format(invalid_df) is False

    # 異常な形式（行数不正）
    invalid_count_df = pl.DataFrame({
        "id": range(1, 10),
        "a": [1] * 9,
        "b": [2] * 9,
    })
    assert validate_submission_format(invalid_count_df) is False
