from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from tqdm import tqdm
import typer
import wandb

from signate_studentcup_2025.config import (
    MODELS_DIR,
    OpenRouterConfig,
    OutputConfig,
    RetrievalConfig,
    WandbConfig,
    DataConfig,
)
from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data, prepare_corpus
from signate_studentcup_2025.features import EmbeddingModel, build_faiss_index, retrieve_top_k

app = typer.Typer()


# === 予測器 ===

class Predictor(ABC):
    """予測器の抽象基底クラス"""

    @abstractmethod
    def predict(self, query_text: str) -> tuple[int, int]:
        """クエリから2作品IDを予測"""
        pass


class OpenRouterRetrievalPredictor(Predictor):
    """OpenRouter埋め込み + FAISSによる検索ベース予測"""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or RetrievalConfig.TOP_K
        self.model = EmbeddingModel(backend="openrouter")
        self.index = None
        self.base_df = None
        self.interim_dir = OutputConfig.INTERIM_DIR

    def fit(self, base_df: pl.DataFrame):
        """ベース作品からインデックス構築"""
        self.base_df = base_df
        corpus = prepare_corpus(base_df)

        # インデックス構築
        index_path = self.interim_dir / "faiss_index_openrouter.pkl"
        self.index, _ = build_faiss_index(
            corpus,
            self.model,
            save_path=index_path,
        )

    def predict(self, query_text: str) -> tuple[int, int]:
        """上位2件を予測"""
        # Top-K検索
        results = retrieve_top_k(query_text, self.index, self.model, k=self.top_k)

        # FAISSインデックスのIDをbase_dfの実際のIDに変換
        top_2_ids = sorted([self.base_df[int(doc_id), 'id'] for doc_id, _ in results[:2]])
        return (top_2_ids[0], top_2_ids[1])


class OpenRouterDirectPredictor(Predictor):
    """OpenRouter LLMによる直接予測"""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OpenRouterConfig.API_KEY,
        )
        self.model = model
        self.base_df = None

    def fit(self, base_df: pl.DataFrame):
        """ベース作品情報を保持"""
        self.base_df = base_df

    def predict(self, query_text: str) -> tuple[int, int]:
        """LLMに直接2作品を予測させる"""
        # 作品リストを作成（簡略化）
        works_list = "\n".join([
            f"ID {row['id']}: {row['title']}"
            for row in self.base_df.iter_rows(named=True)
        ])

        # プロンプト構築
        prompt = f"""以下は映画・アニメ・漫画のあらすじです。このあらすじは、2つの既存作品の世界観やテーマを組み合わせて作られています。

## 作品リスト
{works_list}

## 問題のあらすじ
{query_text}

## タスク
このあらすじの元ネタとなった2作品のIDを特定してください。

## 出力形式
以下のJSON形式で出力してください：
{{
  "work_a_id": 1,
  "work_b_id": 2
}}
"""

        # API呼び出し
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは映画・アニメ・漫画の専門家です。あらすじから元ネタ作品を特定するのが得意です。",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        # 結果をパース
        import json

        result = json.loads(response.choices[0].message.content)

        # ソートして順序を固定
        ids = sorted([result["work_a_id"], result["work_b_id"]])
        return (ids[0], ids[1])


# TODO: ローカル埋め込みモデル予測器の実装
"""
TODO: DenseRetrievalPredictorクラスを追加してOpenRouterRetrievalPredictorと同様のインターフェースで使用可能にする。

実装手順:
1. `uv add sentence-transformers` でライブラリ追加
2. DenseRetrievalPredictorクラスを実装（OpenRouterRetrievalPredictorと同じ構造）
3. CLIで --approach dense を選択可能にする

class DenseRetrievalPredictor(Predictor):
    def __init__(self, model_name: str = "e5-base", top_k: int = 10):
        self.model = EmbeddingModel(backend="sentence-transformer", model_name=model_name)
        ...
"""


@app.command()
def predict(
    approach: str = typer.Option("retrieval", help="retrieval | direct"),
    model: str = typer.Option(None, help="OpenRouterモデル名（未指定はYAMLのデフォルト）"),
    top_k: int = typer.Option(None, help="Top-K検索のK値（未指定はYAMLのデフォルト）"),
    output_path: Path = typer.Option(None, help="出力CSVパス"),
):
    """
    推論を実行し提出ファイルを生成

    Args:
        approach: アプローチ（retrieval | direct）
        model: 使用するモデル（retrievalでは埋め込み、directではチャットモデル）
        top_k: Top-K検索のK値
        output_path: 出力ファイルパス（未指定はYAMLのデフォルト）
    """
    # デフォルト値をYAMLから取得
    if model is None:
        model = OpenRouterConfig.DEFAULT_CHAT_MODEL
    if top_k is None:
        top_k = RetrievalConfig.TOP_K
    if output_path is None:
        output_path = OutputConfig.SUBMISSION_DIR / "submission.csv"

    # Wandb初期化（最小限実装）
    if WandbConfig.ENABLED:
        wandb.init(
            entity=WandbConfig.ENTITY,
            project=WandbConfig.PROJECT,
            job_type="predict",
            config={
                "approach": approach,
                "model": model,
                "top_k": top_k,
                "test_samples": 340,
            },
            mode=WandbConfig.MODE,
        )

    # データ読み込み（パスはYAMLから取得）
    base_df = load_base_stories(DataConfig.BASE_STORIES_PATH)
    test_df = load_fiction_data(DataConfig.FICTION_STORIES_TEST_PATH)

    # 予測器初期化
    if approach == "retrieval":
        predictor = OpenRouterRetrievalPredictor(top_k=top_k)
    elif approach == "direct":
        predictor = OpenRouterDirectPredictor(model=model)
    else:
        raise ValueError(f"未知のアプローチ: {approach}")

    # 学習（インデックス構築）
    logger.info("インデックス構築中...")
    predictor.fit(base_df)

    # 推論実行
    logger.info(f"推論実行中: {len(test_df)}件")
    results = []
    for row in tqdm(test_df.iter_rows(named=True), total=len(test_df)):
        pred_ids = predictor.predict(row["story"])
        results.append({"id": row["id"], "a": pred_ids[0], "b": pred_ids[1]})

    # Polars DataFrameで保存
    result_df = pl.DataFrame(results)
    result_df.write_csv(output_path, include_header=False)
    logger.success(f"提出ファイルを保存しました: {output_path}")

    # Wandbにログ（最小限実装）
    if WandbConfig.ENABLED:
        wandb.log({"num_predictions": len(results)})
        # TODO: 将来的に追加するWandb機能
        """
        TODO: Artifactsで提出ファイルを保存
        artifact = wandb.Artifact(name=f"submission_{approach}", type="submission")
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)

        TODO: 推論結果の要約をログ
        wandb.log({
            "prediction_distribution": wandb.Histogram(pred_ids),
            "unique_predictions": len(set(results)),
        })
        """
        wandb.finish()


if __name__ == "__main__":
    app()
