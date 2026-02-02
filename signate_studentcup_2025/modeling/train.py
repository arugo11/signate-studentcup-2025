from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import wandb
from datetime import datetime

from signate_studentcup_2025.config import (
    DataConfig,
    OutputConfig,
    WandbConfig,
    WeaveConfig,
    DashboardConfig,
    ArtifactsConfig,
)
from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data, prepare_corpus
from signate_studentcup_2025.modeling.predict import OpenRouterRetrievalPredictor
from signate_studentcup_2025.modeling.evaluate import compute_metrics
from signate_studentcup_2025.modeling.analysis import (
    ErrorAnalyzer,
    PredictionAnalyzer,
    EmbeddingVisualizer,
)

app = typer.Typer()

@app.command()
def train(
    approach: str = typer.Option("retrieval", help="Prediction approach"),
    top_k: int = typer.Option(10, help="Top-K for retrieval"),
    eval_on_practice: bool = typer.Option(True, help="Evaluate on practice data"),
):
    """
    Build index and optionally evaluate (training pipeline for this competition).

    For retrieval-based systems, "training" = building FAISS index.
    """
    # Wandb初期化
    run = None
    if WandbConfig.ENABLED:
        run = wandb.init(
            entity=WandbConfig.ENTITY,
            project=WandbConfig.PROJECT,
            job_type="train",
            config={
                "approach": approach,
                "top_k": top_k,
                "eval_on_practice": eval_on_practice,
            },
            mode=WandbConfig.MODE,
        )

        # Weave初期化
        if WeaveConfig.ENABLED:
            from signate_studentcup_2025.weave import init_weave
            init_weave(
                entity=WandbConfig.ENTITY,
                project=WandbConfig.PROJECT,
                enabled=WeaveConfig.ENABLED,
            )

    # データ読み込み
    logger.info("Loading data...")
    base_df = load_base_stories(DataConfig.BASE_STORIES_PATH)

    if eval_on_practice:
        practice_df = load_fiction_data(DataConfig.FICTION_STORIES_PRACTICE_PATH)

    # 予測器初期化
    predictor = OpenRouterRetrievalPredictor(top_k=top_k)

    # 学習（インデックス構築）
    logger.info("Building FAISS index (training)...")
    if run:
        predictor.fit(base_df, wandb_run=run)
    else:
        predictor.fit(base_df)

    # メトリクスログ
    if run:
        run.log({
            "training/index_size": predictor.index.ntotal,
            "training/embedding_dim": predictor.model.get_dim(),
            "training/corpus_size": len(base_df),
        })

    # 評価（オプション）
    if eval_on_practice:
        logger.info("Evaluating on practice data...")
        predictions = []
        ground_truth = []

        for row in tqdm(practice_df.iter_rows(named=True), total=len(practice_df)):
            pred_ids = predictor.predict(row["story"])
            true_ids = tuple(sorted([row["id_a"], row["id_b"]]))
            predictions.append(pred_ids)
            ground_truth.append(true_ids)

        # 指標計算
        metrics = compute_metrics(predictions, ground_truth)

        logger.info("=== Training Results ===")
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")

        # エラー分析
        if run and DashboardConfig.ERROR_ANALYSIS:
            error_analyzer = ErrorAnalyzer(
                predictions=predictions,
                ground_truth=ground_truth,
                queries_df=practice_df,
                base_df=base_df,
            )
            error_analyzer.log_to_wandb(run)

        # 分布分析
        if run and DashboardConfig.DISTRIBUTION_PLOTS:
            pred_analyzer = PredictionAnalyzer(predictions=predictions)
            pred_analyzer.log_distribution_plots(run)

        # 埋め込み可視化
        if run and DashboardConfig.EMBEDDING_VIZ:
            logger.info("Generating embedding visualizations...")

            # ベース作品の埋め込みを取得
            base_embeddings = predictor.model.encode(
                prepare_corpus(base_df).to_list()
            )

            # クエリ（practiceデータ）の埋め込みを取得
            query_texts = practice_df["story"].to_list()
            query_embeddings = predictor.model.encode(query_texts)

            # 結合して可視化
            import numpy as np
            all_embeddings = np.vstack([base_embeddings, query_embeddings])
            labels = (
                [f"Base_{row['id']}" for row in base_df.iter_rows(named=True)] +
                [f"Query_{i}" for i in range(len(query_embeddings))]
            )

            # Visualizer初期化
            visualizer = EmbeddingVisualizer(all_embeddings, labels=labels)

            # UMAPで2D投影
            projection = visualizer.create_2d_projection(method="umap")

            # 散布図をログ
            visualizer.log_scatter_plot(projection, run, title="UMAP Projection of Embeddings")

            # クラスタ分析
            visualizer.log_cluster_analysis(projection, run)

            logger.info("Embedding visualizations logged to Wandb")

        # メトリクスログ
        if run:
            run.log(metrics)

    # 終了
    if run:
        wandb.finish()

    logger.success("Training complete!")
    logger.info(f"Index saved to: {OutputConfig.INTERIM_DIR}/faiss_index_openrouter.pkl")

if __name__ == "__main__":
    app()
