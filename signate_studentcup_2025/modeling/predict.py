from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import polars as pl
from tqdm import tqdm
import typer
import wandb

from signate_studentcup_2025.config import (
    ArtifactsConfig,
    DataConfig,
    OpenRouterConfig,
    OutputConfig,
    RerankingConfig,
    RetrievalConfig,
    WandbConfig,
    load_prompt,
)
from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data, prepare_corpus
from signate_studentcup_2025.features import EmbeddingModel, build_faiss_index, retrieve_top_k
from signate_studentcup_2025.weave import weave_op_decorator, weave_op_decorator_configured

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

    @weave_op_decorator
    def fit(self, base_df: pl.DataFrame, wandb_run=None):
        """ベース作品からインデックス構築

        Args:
            base_df: ベース作品DataFrame
            wandb_run: Wandb Runオブジェクト（オプション）
        """
        self.base_df = base_df
        corpus = prepare_corpus(base_df)

        # インデックス構築
        index_path = self.interim_dir / "faiss_index_openrouter.pkl"
        self.index, _ = build_faiss_index(
            corpus,
            self.model,
            save_path=index_path,
            wandb_run=wandb_run,
        )

    @weave_op_decorator_configured("TRACE_PREDICTIONS")
    def predict(self, query_text: str) -> tuple[int, int]:
        """上位2件を予測"""
        # Top-K検索
        results = retrieve_top_k(query_text, self.index, self.model, k=self.top_k)

        # FAISSインデックスのIDをbase_dfの実際のIDに変換
        top_2_ids = sorted([self.base_df[int(doc_id), "id"] for doc_id, _ in results[:2]])
        return (top_2_ids[0], top_2_ids[1])


class OpenRouterDirectPredictor(Predictor):
    """OpenRouter LLMによる直接予測"""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        prompt_name: str = "base",
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OpenRouterConfig.API_KEY,
        )
        self.model = model
        self.prompt_name = prompt_name
        self.prompt_config = load_prompt(prompt_name)
        self.base_df = None

        logger.info(f"Initialized OpenRouterDirectPredictor with prompt: {prompt_name}")

    def fit(self, base_df: pl.DataFrame):
        """ベース作品情報を保持"""
        self.base_df = base_df

    @weave_op_decorator_configured("TRACE_PREDICTIONS")
    def predict(self, query_text: str) -> tuple[int, int]:
        """LLMに直接2作品を予測させる"""
        # 作品リストを作成（簡略化）
        works_list = "\n".join(
            [f"ID {row['id']}: {row['title']}" for row in self.base_df.iter_rows(named=True)]
        )

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


# Local embedding model predictor


class DenseRetrievalPredictor(Predictor):
    """
    Local SentenceTransformer based retrieval predictor.

    Uses local embedding models (E5, Ruri, etc.) with FAISS index.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        top_k: int = None,
        device: str = "cpu",
    ):
        """
        Initialize DenseRetrievalPredictor.

        Args:
            model_name: HuggingFace model name (e.g., "intfloat/multilingual-e5-large")
            top_k: Number of top results to retrieve
            device: Device to use ("cpu" or "cuda")
        """
        self.top_k = top_k or RetrievalConfig.TOP_K
        self.model = EmbeddingModel(
            backend="sentence-transformer",
            model_name=model_name,
            device=device,
        )
        self.index = None
        self.base_df = None
        self.interim_dir = OutputConfig.INTERIM_DIR
        self.model_name = model_name

        logger.info(f"Initialized DenseRetrievalPredictor with model: {model_name}")

    @weave_op_decorator
    def fit(self, base_df: pl.DataFrame, wandb_run=None):
        """
        Build FAISS index from base stories.

        Args:
            base_df: Base stories DataFrame
            wandb_run: Wandb Run object (optional)
        """
        self.base_df = base_df
        corpus = prepare_corpus(base_df)

        # Build index with model-specific filename
        safe_model_name = self.model_name.replace("/", "_")
        index_path = self.interim_dir / f"faiss_index_{safe_model_name}.pkl"
        self.index, _ = build_faiss_index(
            corpus,
            self.model,
            save_path=index_path,
            wandb_run=wandb_run,
        )

    @weave_op_decorator_configured("TRACE_PREDICTIONS")
    def predict(self, query_text: str) -> tuple[int, int]:
        """
        Predict top 2 work IDs.

        Args:
            query_text: Fiction story text

        Returns:
            Tuple of (work_a_id, work_b_id) sorted
        """
        # Top-K retrieval
        results = retrieve_top_k(query_text, self.index, self.model, k=self.top_k)

        # Convert FAISS index IDs to actual work IDs from base_df
        top_2_ids = sorted([self.base_df[int(doc_id), "id"] for doc_id, _ in results[:2]])
        return (top_2_ids[0], top_2_ids[1])


class RerankingPredictor(Predictor):
    """
    Two-stage predictor: Bi-encoder retrieval + Cross-encoder reranking.

    Uses a bi-encoder for fast initial retrieval, then reranks the top-K candidates
    using a cross-encoder for more accurate ranking. This approach combines the speed
    of bi-encoders with the accuracy of cross-encoders.

    Recommended for:
    - Japanese text (use Ruri reranker)
    - When accuracy is more important than speed
    - Small candidate sets (< 100 documents)
    """

    def __init__(
        self,
        embedding_model: str = "intfloat/multilingual-e5-large",
        reranker_model: str = "cl-nagoya/ruri-reranker-large",
        retrieval_k: int = None,
        device: str = "cpu",
    ):
        """
        Initialize RerankingPredictor.

        Args:
            embedding_model: Bi-encoder model name (for initial retrieval)
            reranker_model: Cross-encoder model name (for reranking)
            retrieval_k: Number of candidates to retrieve for reranking
            device: Device to use ("cpu" or "cuda")
        """
        from signate_studentcup_2025.modeling.reranker import (
            RuriReranker,
            SentenceTransformerReranker,
        )

        self.retrieval_k = retrieval_k or RerankingConfig.RETRIEVAL_K
        self.device = device

        # Bi-encoder for retrieval
        self.bi_encoder = EmbeddingModel(
            backend="sentence-transformer",
            model_name=embedding_model,
            device=device,
        )
        self.index = None
        self.base_df = None
        self.interim_dir = OutputConfig.INTERIM_DIR
        self.embedding_model = embedding_model

        # Cross-encoder for reranking
        if "ruri" in reranker_model.lower():
            size = "large" if "large" in reranker_model else "small"
            self.reranker = RuriReranker(model_size=size, device=device)
        else:
            self.reranker = SentenceTransformerReranker(model_name=reranker_model, device=device)

        self.reranker_model = reranker_model
        logger.info(f"Initialized RerankingPredictor: {embedding_model} + {reranker_model}")

    def fit(self, base_df: pl.DataFrame, wandb_run=None):
        """
        Build FAISS index from base stories.

        Args:
            base_df: Base stories DataFrame
            wandb_run: Wandb Run object (optional)
        """
        self.base_df = base_df
        corpus = prepare_corpus(base_df)

        # Build index with model-specific filename
        safe_model_name = self.embedding_model.replace("/", "_")
        index_path = self.interim_dir / f"faiss_index_{safe_model_name}.pkl"
        self.index, _ = build_faiss_index(
            corpus,
            self.bi_encoder,
            save_path=index_path,
            wandb_run=wandb_run,
        )

    def predict(self, query_text: str) -> tuple[int, int]:
        """
        Two-stage prediction: bi-encoder retrieval + cross-encoder reranking.

        Args:
            query_text: Fiction story text

        Returns:
            Tuple of (work_a_id, work_b_id) sorted
        """
        from signate_studentcup_2025.modeling.reranker import rerank_candidates

        # Stage 1: Bi-encoder retrieval (top-K)
        results = retrieve_top_k(query_text, self.index, self.bi_encoder, k=self.retrieval_k)

        # Prepare candidates with text (doc_id, bi_score, text)
        candidates = [
            (self.base_df[int(doc_id), "id"], score, self.base_df[int(doc_id), "story"])
            for doc_id, score in results
        ]

        # Stage 2: Cross-encoder reranking
        reranked = rerank_candidates(query_text, candidates, self.reranker)

        # Take top-2 from reranked list
        top_2_ids = sorted([reranked[0][0], reranked[1][0]])
        return (top_2_ids[0], top_2_ids[1])


@app.command()
def predict(
    approach: str = typer.Option("retrieval", help="retrieval | dense | reranking | direct"),
    model: str = typer.Option(
        None,
        help="Model name (e.g., 'intfloat/multilingual-e5-large' for dense/reranking, or OpenRouter model for retrieval/direct)",
    ),
    reranker_model: str = typer.Option(None, help="Reranker model for 'reranking' approach"),
    retrieval_k: int = typer.Option(None, help="Top-K candidates for reranking"),
    top_k: int = typer.Option(None, help="Top-K retrieval K value"),
    device: str = typer.Option("cpu", help="Device for local models (cpu | cuda)"),
    output_path: Path = typer.Option(None, help="Output CSV path"),
):
    """
    推論を実行し提出ファイルを生成

    Args:
        approach: アプローチ（retrieval | dense | reranking | direct）
            - retrieval: OpenRouter API embeddings
            - dense: Local SentenceTransformer models (E5, Ruri, etc.)
            - reranking: Bi-encoder retrieval + Cross-encoder reranking
            - direct: OpenRouter LLM direct prediction
        model: 使用するモデル名
        reranker_model: Rerankingアプローチ用のリランカーモデル
        retrieval_k: Rerankingアプローチの検索K値
        top_k: Top-K検索のK値
        device: デバイス（ローカルモデル用）
        output_path: 出力ファイルパス
    """
    from datetime import datetime

    # デフォルト値をYAMLから取得
    if top_k is None:
        top_k = RetrievalConfig.TOP_K
    if output_path is None:
        output_path = OutputConfig.SUBMISSION_DIR / "submission.csv"

    # Set default model based on approach
    if model is None:
        if approach == "dense":
            model = "intfloat/multilingual-e5-large"
        elif approach == "reranking":
            model = "intfloat/multilingual-e5-large"
        elif approach == "direct":
            model = OpenRouterConfig.DEFAULT_CHAT_MODEL
        else:  # retrieval
            model = OpenRouterConfig.EMBEDDING_MODEL

    # Set reranking-specific defaults
    if approach == "reranking":
        if reranker_model is None:
            reranker_model = RerankingConfig.MODEL
        if retrieval_k is None:
            retrieval_k = RerankingConfig.RETRIEVAL_K

    # Wandb初期化
    run = None
    if WandbConfig.ENABLED:
        run = wandb.init(
            entity=WandbConfig.ENTITY,
            project=WandbConfig.PROJECT,
            job_type="predict",
            config={
                "approach": approach,
                "model": model,
                "top_k": top_k,
                "reranker_model": reranker_model if approach == "reranking" else None,
                "retrieval_k": retrieval_k if approach == "reranking" else None,
                "device": device,
                "test_samples": 340,
            },
            mode=WandbConfig.MODE,
        )

        # Weave初期化
        from signate_studentcup_2025.weave import init_weave

        init_weave(
            entity=WandbConfig.ENTITY,
            project=WandbConfig.PROJECT,
            enabled=True,
        )

    # データ読み込み（パスはYAMLから取得）
    base_df = load_base_stories(DataConfig.BASE_STORIES_PATH)
    test_df = load_fiction_data(DataConfig.FICTION_STORIES_TEST_PATH)

    # データセットをArtifactとしてログ
    if run and ArtifactsConfig.LOG_DATASETS:
        from signate_studentcup_2025.config import log_dataset_as_artifact

        log_dataset_as_artifact(
            base_df,
            artifact_name="base-stories",
            artifact_type="dataset",
            wandb_run=run,
            metadata={"source": "raw/base_stories.tsv"},
        )

        log_dataset_as_artifact(
            test_df,
            artifact_name="fiction-test",
            artifact_type="dataset",
            wandb_run=run,
            metadata={"source": "raw/fiction_stories_test.tsv"},
        )

    # 予測器初期化
    if approach == "retrieval":
        predictor = OpenRouterRetrievalPredictor(top_k=top_k)
    elif approach == "dense":
        predictor = DenseRetrievalPredictor(
            model_name=model,
            top_k=top_k,
            device=device,
        )
    elif approach == "reranking":
        predictor = RerankingPredictor(
            embedding_model=model,
            reranker_model=reranker_model,
            retrieval_k=retrieval_k,
            device=device,
        )
    elif approach == "direct":
        predictor = OpenRouterDirectPredictor(model=model)
    else:
        raise ValueError(f"未知のアプローチ: {approach}")

    # 学習（インデックス構築）
    logger.info("インデックス構築中...")
    if approach in ("retrieval", "dense", "reranking") and run:
        predictor.fit(base_df, wandb_run=run)
    else:
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

    # 提出ファイルをArtifactとしてログ
    if run and ArtifactsConfig.LOG_SUBMISSIONS:
        submission_artifact = wandb.Artifact(
            name=f"submission-{approach}",
            type="submission",
            metadata={
                "approach": approach,
                "model": model,
                "top_k": top_k,
                "reranker_model": reranker_model if approach == "reranking" else None,
                "retrieval_k": retrieval_k if approach == "reranking" else None,
                "device": device,
                "num_predictions": len(results),
                "timestamp": datetime.now().isoformat(),
            },
        )
        submission_artifact.add_file(str(output_path))
        run.log_artifact(submission_artifact)
        submission_artifact.wait()
        run.log_artifact(submission_artifact, aliases=["latest"])
        logger.info(f"Logged submission artifact: {submission_artifact.name}")

    # メトリクスログ
    if run:
        unique_predictions = len(set((r["a"], r["b"]) for r in results))
        run.log(
            {
                "num_predictions": len(results),
                "unique_predictions": unique_predictions,
                "unique_ratio": unique_predictions / len(results),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    app()
