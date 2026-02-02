from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm
import typer
import wandb

from signate_studentcup_2025.config import (
    DataConfig,
    EvaluationConfig,
    OpenRouterConfig,
    OutputConfig,
    RetrievalConfig,
    WandbConfig,
)
from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data
from signate_studentcup_2025.modeling.predict import (
    OpenRouterDirectPredictor,
    OpenRouterRetrievalPredictor,
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
    exact_matches = sum(
        1 for p, g in zip(predictions, ground_truth) if set(p) == set(g)
    )
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
            ideal_dcg = sum(
                1 / np.log2(rank + 1) for rank in range(1, min(len(true_set), k) + 1)
            )
            ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)

    return metrics


@app.command()
def evaluate(
    approach: str = typer.Option("retrieval", help="retrieval | direct"),
    model: str = typer.Option(None, help="OpenRouterモデル名（未指定はYAMLのデフォルト）"),
    top_k: int = typer.Option(None, help="Top-K検索のK値（未指定はYAMLのデフォルト）"),
):
    """
    practiceデータ（20件）で評価
    """
    # デフォルト値をYAMLから取得
    if model is None:
        model = OpenRouterConfig.DEFAULT_CHAT_MODEL
    if top_k is None:
        top_k = RetrievalConfig.TOP_K

    # Wandb初期化（最小限実装）
    if WandbConfig.ENABLED:
        wandb.init(
            entity=WandbConfig.ENTITY,
            project=WandbConfig.PROJECT,
            job_type="evaluate",
            config={
                "approach": approach,
                "model": model,
                "top_k": top_k,
            },
            mode=WandbConfig.MODE,
        )

    # データ読み込み（パスはYAMLから取得）
    base_df = load_base_stories(DataConfig.BASE_STORIES_PATH)
    practice_df = load_fiction_data(DataConfig.FICTION_STORIES_PRACTICE_PATH)

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

    # 結果表示
    logger.info("=== 評価結果 ===")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    for k in EvaluationConfig.K_VALUES:
        if f"hit_rate@{k}" in metrics:
            logger.info(f"Hit Rate@{k}: {metrics[f'hit_rate@{k}']:.3f}")
    logger.info(f"MRR: {metrics['mrr']:.3f}")

    # Wandbにログ（最小限実装）
    if WandbConfig.ENABLED:
        wandb.log(metrics)
        wandb.finish()
        # TODO: 将来的に追加するWandb機能
        """
        TODO: Artifactsで評価結果を保存・共有
        wandb.log_artifact(
            artifact_or_path="evaluation_results.json",
            type="evaluation",
            metadata={"approach": approach, "model": model}
        )

        TODO: Weaveで評価プロセスをトレース
        @weave.op()
        def evaluate_with_trace():
            ...
        """


if __name__ == "__main__":
    app()
