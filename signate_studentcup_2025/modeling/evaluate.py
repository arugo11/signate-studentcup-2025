from pathlib import Path

from loguru import logger
import numpy as np
import polars as pl
from tqdm import tqdm
import typer
import wandb

from signate_studentcup_2025.config import (
    ArtifactsConfig,
    DashboardConfig,
    DataConfig,
    EvaluationConfig,
    OpenRouterConfig,
    RerankingConfig,
    RetrievalConfig,
    WandbConfig,
)
from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data, prepare_corpus
from signate_studentcup_2025.modeling.analysis import (
    EmbeddingVisualizer,
    ErrorAnalyzer,
    PredictionAnalyzer,
)
from signate_studentcup_2025.modeling.predict import (
    DenseRetrievalPredictor,
    OpenRouterDirectPredictor,
    OpenRouterRetrievalPredictor,
    RerankingPredictor,
)

app = typer.Typer()


# === 評価指標 ===


def compute_metrics(
    predictions: list[tuple[int, int]],
    ground_truth: list[tuple[int, int]],
    k_values: list[int] = [1, 5, 10, 20],
) -> dict[str, float]:
    """
    評価指標を計算

    Args:
        predictions: 予測結果 [(id_a, id_b), ...]
        ground_truth: 正解 [(id_a, id_b), ...]
        k_values: Hit Rate@K, NDCG@K のK値リスト

    Returns:
        指標を含む辞書
    """
    metrics = {}

    # Accuracy（コンペ指標）
    exact_matches = sum(1 for p, g in zip(predictions, ground_truth) if set(p) == set(g))
    metrics["accuracy"] = exact_matches / len(predictions)

    # Hit Rate@K（少なくとも1件が正解）
    for k in k_values:
        hit_count = 0
        for pred, true_set in zip(predictions, [set(g) for g in ground_truth]):
            if len(set(pred[: min(k, len(pred))]) & true_set) > 0:
                hit_count += 1
        metrics[f"hit_rate@{k}"] = hit_count / len(predictions)

    # MRR（最初の正解の順位の逆数）
    reciprocal_ranks = []
    for pred, true_set in zip(predictions, [set(g) for g in ground_truth]):
        for rank, doc_id in enumerate(pred, start=1):
            if doc_id in true_set:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    metrics["mrr"] = np.mean(reciprocal_ranks)

    # NDCG@K
    for k in k_values:
        ndcg_scores = []
        for pred, true_set in zip(predictions, [set(g) for g in ground_truth]):
            dcg = sum(
                (1 if doc_id in true_set else 0) / np.log2(rank + 1)
                for rank, doc_id in enumerate(pred[:k], start=1)
            )
            ideal_dcg = sum(1 / np.log2(rank + 1) for rank in range(1, min(len(true_set), k) + 1))
            ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)

    return metrics


@app.command()
def evaluate(
    approach: str = typer.Option("retrieval", help="retrieval | dense | reranking | direct"),
    model: str = typer.Option(
        None, help="Model name (e.g., 'intfloat/multilingual-e5-large' for dense/reranking)"
    ),
    reranker_model: str = typer.Option(None, help="Reranker model for 'reranking' approach"),
    retrieval_k: int = typer.Option(None, help="Top-K candidates for reranking"),
    top_k: int = typer.Option(None, help="Top-K検索のK値（未指定はYAMLのデフォルト）"),
    device: str = typer.Option("cpu", help="Device for local models (cpu | cuda)"),
):
    """
    practiceデータ（20件）で評価
    """
    import tempfile

    # デフォルト値をYAMLから取得
    if top_k is None:
        top_k = RetrievalConfig.TOP_K

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
            job_type="evaluate",
            config={
                "approach": approach,
                "model": model,
                "top_k": top_k,
                "reranker_model": reranker_model if approach == "reranking" else None,
                "retrieval_k": retrieval_k if approach == "reranking" else None,
                "device": device,
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
    practice_df = load_fiction_data(DataConfig.FICTION_STORIES_PRACTICE_PATH)

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
            practice_df,
            artifact_name="fiction-practice",
            artifact_type="dataset",
            wandb_run=run,
            metadata={"source": "raw/fiction_stories_practice.tsv"},
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
    logger.info(f"推論実行中: {len(practice_df)}件")
    predictions = []
    ground_truth = []

    for row in tqdm(practice_df.iter_rows(named=True), total=len(practice_df)):
        pred_ids = predictor.predict(row["story"])
        true_ids = tuple(sorted([row["id_a"], row["id_b"]]))

        predictions.append(pred_ids)
        ground_truth.append(true_ids)

    # 指標計算（k_valuesはYAMLから取得）
    metrics = compute_metrics(predictions, ground_truth, k_values=EvaluationConfig.K_VALUES)

    # エラー分析（DashboardConfigで制御）
    if run and DashboardConfig.ERROR_ANALYSIS:
        logger.info("Running error analysis...")
        error_analyzer = ErrorAnalyzer(
            predictions=predictions,
            ground_truth=ground_truth,
            queries_df=practice_df,
            base_df=base_df,
        )
        error_analyzer.log_to_wandb(run)

    # 予測分布分析（DashboardConfigで制御）
    if run and DashboardConfig.DISTRIBUTION_PLOTS:
        logger.info("Analyzing prediction distribution...")
        pred_analyzer = PredictionAnalyzer(predictions=predictions)
        pred_analyzer.log_distribution_plots(run)

    # 埋め込み可視化（DashboardConfigで制御）
    if run and DashboardConfig.EMBEDDING_VIZ:
        logger.info("Generating embedding visualizations...")

        # ベース作品の埋め込みを取得
        base_embeddings = predictor.model.encode(prepare_corpus(base_df).to_list())

        # クエリ（practiceデータ）の埋め込みを取得
        query_texts = practice_df["story"].to_list()
        query_embeddings = predictor.model.encode(query_texts)

        # 結合して可視化
        all_embeddings = np.vstack([base_embeddings, query_embeddings])
        labels = [f"Base_{row['id']}" for row in base_df.iter_rows(named=True)] + [
            f"Query_{i}" for i in range(len(query_embeddings))
        ]

        # Visualizer初期化
        visualizer = EmbeddingVisualizer(all_embeddings, labels=labels)

        # UMAPで2D投影
        projection = visualizer.create_2d_projection(method="umap")

        # 散布図をログ
        visualizer.log_scatter_plot(projection, run, title="UMAP Projection of Embeddings")

        # クラスタ分析
        visualizer.log_cluster_analysis(projection, run)

        logger.info("Embedding visualizations logged to Wandb")

    # 結果表示
    logger.info("=== 評価結果 ===")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    for k in EvaluationConfig.K_VALUES:
        if f"hit_rate@{k}" in metrics:
            logger.info(f"Hit Rate@{k}: {metrics[f'hit_rate@{k}']:.3f}")
    logger.info(f"MRR: {metrics['mrr']:.3f}")

    # 評価結果をArtifactとしてログ
    if run and ArtifactsConfig.LOG_EVALUATIONS:
        # 詳細結果のDataFrameを作成
        results_df = practice_df.with_columns(
            predicted_a=pl.Series([p[0] for p in predictions]),
            predicted_b=pl.Series([p[1] for p in predictions]),
            correct=pl.Series([set(p) == set(g) for p, g in zip(predictions, ground_truth)]),
        )

        # Artifactを作成
        eval_artifact = wandb.Artifact(
            name=f"evaluation-{approach}",
            type="evaluation",
            metadata={
                "approach": approach,
                "model": model,
                "top_k": top_k,
                "reranker_model": reranker_model if approach == "reranking" else None,
                "retrieval_k": retrieval_k if approach == "reranking" else None,
                "accuracy": metrics["accuracy"],
                **{k: v for k, v in metrics.items() if k != "accuracy"},
            },
        )

        # 詳細結果CSVを追加
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            results_df.write_csv(f.name)
            eval_artifact.add_file(f.name, name="detailed_results.csv")
            temp_path = f.name

        run.log_artifact(eval_artifact)
        eval_artifact.wait()  # アップロード完了を待機
        run.log_artifact(eval_artifact, aliases=["latest"])
        logger.info(f"Logged evaluation artifact: {eval_artifact.name}")

        # 一時ファイルを削除
        Path(temp_path).unlink(missing_ok=True)

    # メトリクスをログ
    if run:
        run.log(metrics)
        wandb.finish()


@app.command()
def sweep(
    sweep_config: str = typer.Option("config/sweeps.yaml", help="Path to sweep config YAML"),
    count: int = typer.Option(100, help="Number of trials to run"),
):
    """
    Launch hyperparameter sweep.

    Example:
        python signate_studentcup_2025/modeling/evaluate.py sweep \\
            --sweep-config config/sweeps_retrieval.yaml --count 50
    """

    if not WandbConfig.ENABLED:
        logger.error("Wandb is not enabled. Please set WANDB_API_KEY environment variable.")
        raise typer.Exit(1)

    # Convert to absolute path
    config_path = Path(sweep_config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / sweep_config

    if not config_path.exists():
        logger.error(f"Sweep config not found: {config_path}")
        raise typer.Exit(1)

    # Import here to avoid issues
    import wandb as wandb_module
    import yaml

    # Load sweep config
    with open(config_path, encoding="utf-8") as f:
        sweep_config_yaml = yaml.safe_load(f)

    # Update program path in config
    sweep_config_yaml["program"] = "signate_studentcup_2025/modeling/sweep_wrapper.py"

    # Initialize sweep
    sweep_id = wandb_module.sweep(
        sweep=sweep_config_yaml,
        entity=WandbConfig.ENTITY,
        project=WandbConfig.PROJECT,
    )

    logger.info(f"Launched sweep: {sweep_id}")
    logger.info(f"Starting wandb agent for {count} trials...")

    # Start agent
    from signate_studentcup_2025.modeling.sweep_wrapper import evaluate_sweep_trial

    wandb_module.agent(
        sweep_id,
        function=evaluate_sweep_trial,
        count=count,
        entity=WandbConfig.ENTITY,
        project=WandbConfig.PROJECT,
    )

    logger.success("Sweep complete!")


if __name__ == "__main__":
    app()
