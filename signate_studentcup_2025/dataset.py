from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from tqdm import tqdm
import typer

from signate_studentcup_2025.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, OutputConfig, DataConfig

app = typer.Typer()


def load_base_stories(path: Path | None = None) -> pl.DataFrame:
    """
    ベース作品データ（50作品）をTSVから読み込み

    Args:
        path: ファイルパス（未指定はYAMLのデフォルト）

    Returns:
        DataFrame with columns: [id, category, title, story]
    """
    if path is None:
        path = DataConfig.BASE_STORIES_PATH

    df = pl.read_csv(path, separator="\t")
    logger.info(f"ベース作品数: {len(df)}件")
    logger.debug(f"カラム: {df.columns}")

    return df


def load_fiction_data(path: Path | None = None) -> pl.DataFrame:
    """
    架空作品データをTSVから読み込み

    Args:
        path: ファイルパス（未指定はYAMLのデフォルト）

    Returns:
        practice: DataFrame with columns: [id, id_a, id_b, title_a, title_b, story]
        test: DataFrame with columns: [id, story]
    """
    if path is None:
        # デフォルトはtestデータ
        path = DataConfig.FICTION_STORIES_TEST_PATH

    df = pl.read_csv(path, separator="\t")
    logger.info(f"架空作品数: {len(df)}件")
    logger.debug(f"カラム: {df.columns}")

    return df


def prepare_corpus(base_df: pl.DataFrame) -> list[str]:
    """
    コーパステキストを生成（タイトル + あらすじ）

    Args:
        base_df: ベース作品DataFrame

    Returns:
        コーパステキストのリスト
    """
    corpus = [
        f"タイトル: {row['title']}\nあらすじ: {row['story']}"
        for row in base_df.iter_rows(named=True)
    ]

    logger.info(f"コーパスサイズ: {len(corpus)}件")
    return corpus


def validate_submission_format(predictions: pl.DataFrame, sample_path: Path | None = None) -> bool:
    """
    予測結果の形式をバリデーション

    Args:
        predictions: 予測結果DataFrame (id, a, b)
        sample_path: サンプル提出ファイルパス

    Returns:
        バリデーション結果
    """
    logger.info("提出形式をバリデーション...")

    # カラムチェック
    if predictions.columns != ["id", "a", "b"]:
        logger.error(f"カラムが不正です: {predictions.columns}")
        return False

    # 行数チェック（340件）
    if len(predictions) != 340:
        logger.error(f"行数が不正です: {len(predictions)}件（期待: 340件）")
        return False

    # 値の範囲チェック（IDは1-50）
    for col in ["a", "b"]:
        min_val = predictions[col].min()
        max_val = predictions[col].max()

        if min_val < 1 or max_val > 50:
            logger.error(f"{col}の値が範囲外です: min={min_val}, max={max_val}（期待: 1-50）")
            return False

    logger.success("バリデーション通過")
    return True


@app.command()
def main(
    input_path: Path = typer.Option(None, help="入力TSVパス"),
    output_path: Path = typer.Option(None, help="出力CSVパス"),
):
    """
    データ前処理スクリプト
    """
    logger.info("データセットを処理中...")

    # デフォルトパス設定
    if input_path is None:
        input_path = DataConfig.BASE_STORIES_PATH
    if output_path is None:
        output_path = OutputConfig.SUBMISSION_DIR / "processed_base_stories.csv"

    # データ読み込み
    base_df = load_base_stories(input_path)

    # コーパス作成
    corpus = prepare_corpus(base_df)

    # 保存
    corpus_df = pl.DataFrame({"text": corpus})
    corpus_df.write_csv(output_path)
    logger.success(f"処理完了: {output_path}")


if __name__ == "__main__":
    app()
