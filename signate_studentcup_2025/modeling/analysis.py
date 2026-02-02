"""Error analysis and visualization module for evaluation insights.

This module provides tools for analyzing prediction errors, visualizing
distributions, and understanding model behavior.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap


class ErrorAnalyzer:
    """Analyzer for misclassified samples and error patterns."""

    def __init__(
        self,
        predictions: list[tuple[int, int]],
        ground_truth: list[tuple[int, int]],
        queries_df: pl.DataFrame,
        base_df: pl.DataFrame,
    ):
        """
        Initialize error analyzer.

        Args:
            predictions: List of predicted pairs [(id_a, id_b), ...]
            ground_truth: List of true pairs [(id_a, id_b), ...]
            queries_df: Query DataFrame with story text
            base_df: Base stories DataFrame with metadata
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.queries_df = queries_df
        self.base_df = base_df

        # Compute correctness
        self.correctness = [
            set(p) == set(g) for p, g in zip(predictions, ground_truth)
        ]

        self.num_correct = sum(self.correctness)
        self.num_errors = len(self.correctness) - self.num_correct

    def create_error_dataframe(self) -> pl.DataFrame:
        """
        Create detailed DataFrame with error information.

        Returns:
            DataFrame with columns: query_id, story, predicted_a, predicted_b,
                                   true_a, true_b, correct, partial_match
        """
        data = []
        for i, (pred, true, query_row) in enumerate(zip(
            self.predictions,
            self.ground_truth,
            self.queries_df.iter_rows(named=True)
        )):
            pred_set = set(pred)
            true_set = set(true)

            # Check for partial match (exactly one correct ID)
            intersection = pred_set & true_set
            partial_match = len(intersection) == 1

            data.append({
                "query_id": i,  # インデックスをIDとして使用
                "story": query_row["story"],
                "predicted_a": pred[0],
                "predicted_b": pred[1],
                "true_a": true[0],
                "true_b": true[1],
                "correct": self.correctness[i],
                "partial_match": partial_match,
            })

        return pl.DataFrame(data)

    def get_misclassified_samples(self) -> pl.DataFrame:
        """
        Get only misclassified samples.

        Returns:
            DataFrame with error samples only
        """
        error_df = self.create_error_dataframe()
        return error_df.filter(~pl.col("correct"))

    def analyze_error_patterns(self) -> dict[str, Any]:
        """
        Analyze common error patterns.

        Returns:
            Dictionary with error statistics
        """
        error_df = self.get_misclassified_samples()

        patterns = {
            "total_errors": self.num_errors,
            "error_rate": self.num_errors / len(self.predictions),
            "partial_matches": error_df.filter(pl.col("partial_match")).height,
            "complete_misses": error_df.filter(~pl.col("partial_match")).height,
        }

        # Most frequently mispredicted IDs
        all_predicted_ids = []
        all_true_ids = []

        for pred, true in zip(self.predictions, self.ground_truth):
            if set(pred) != set(true):
                all_predicted_ids.extend(pred)
                all_true_ids.extend(true)

        if all_predicted_ids:
            # Convert to Polars Series for value_counts
            pred_counts = pl.Series(all_predicted_ids).value_counts().sort("count", descending=True)
            true_counts = pl.Series(all_true_ids).value_counts().sort("count", descending=True)

            patterns["most_common_predicted_ids"] = pred_counts.head(10).to_dicts()
            patterns["most_common_true_ids"] = true_counts.head(10).to_dicts()

        return patterns

    def log_to_wandb(self, wandb_run, artifact_name: str = "error-analysis-practice"):
        """
        Log error analysis to Wandb as artifact.

        Args:
            wandb_run: Wandb Run object
            artifact_name: Name for the artifact
        """
        import tempfile
        import wandb as wandb_module

        # Create error DataFrame
        error_df = self.create_error_dataframe()

        # Analyze patterns
        patterns = self.analyze_error_patterns()

        # Create artifact
        artifact = wandb_module.Artifact(
            name=artifact_name,
            type="error_analysis",
            metadata=patterns
        )

        # Save detailed error CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            error_df.write_csv(f.name)
            artifact.add_file(f.name, name="detailed_errors.csv")
            temp_path = f.name

        # Save misclassified only
        misclassified_df = self.get_misclassified_samples()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            misclassified_df.write_csv(f.name)
            artifact.add_file(f.name, name="misclassified_samples.csv")
            temp_path2 = f.name

        # Log patterns as metrics
        if wandb_run:
            wandb_run.log({
                "error_analysis/total_errors": patterns["total_errors"],
                "error_analysis/error_rate": patterns["error_rate"],
                "error_analysis/partial_matches": patterns["partial_matches"],
                "error_analysis/complete_misses": patterns["complete_misses"],
            })

        wandb_run.log_artifact(artifact)
        logger.info(f"Logged error analysis artifact: {artifact.name}")

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        Path(temp_path2).unlink(missing_ok=True)

        return artifact


class PredictionAnalyzer:
    """Analyzer for prediction distribution patterns."""

    def __init__(self, predictions: list[tuple[int, int]]):
        """
        Initialize prediction analyzer.

        Args:
            predictions: List of predicted pairs [(id_a, id_b), ...]
        """
        self.predictions = predictions

    def analyze_prediction_distribution(self) -> dict[str, Any]:
        """
        Analyze distribution of predictions.

        Returns:
            Dictionary with distribution statistics
        """
        # Convert to sorted tuples for consistency
        sorted_predictions = [tuple(sorted(p)) for p in self.predictions]

        # Count unique predictions
        unique_predictions = set(sorted_predictions)
        unique_count = len(unique_predictions)
        total_count = len(sorted_predictions)

        # Calculate entropy (diversity measure)
        from collections import Counter
        pred_counts = Counter(sorted_predictions)
        probs = [count / total_count for count in pred_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)

        # Max possible entropy (all predictions unique)
        max_entropy = np.log(total_count) if total_count > 0 else 0

        return {
            "total_predictions": total_count,
            "unique_predictions": unique_count,
            "unique_ratio": unique_count / total_count if total_count > 0 else 0,
            "entropy": entropy,
            "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
            "most_common_prediction": pred_counts.most_common(1)[0] if pred_counts else None,
        }

    def log_distribution_plots(self, wandb_run):
        """
        Create and log distribution plots to Wandb.

        Args:
            wandb_run: Wandb Run object
        """
        import wandb as wandb_module
        import io

        # Analyze distribution
        stats = self.analyze_prediction_distribution()

        # Log stats as metrics
        wandb_run.log({
            "distribution/unique_predictions": stats["unique_predictions"],
            "distribution/unique_ratio": stats["unique_ratio"],
            "distribution/entropy": stats["entropy"],
            "distribution/normalized_entropy": stats["normalized_entropy"],
        })

        # Create histogram
        sorted_predictions = [tuple(sorted(p)) for p in self.predictions]
        from collections import Counter
        pred_counts = Counter(sorted_predictions)

        # Get top 20 most common predictions
        top_predictions = pred_counts.most_common(20)

        if top_predictions:
            fig, ax = plt.subplots(figsize=(12, 6))

            labels = [str(p[0]) for p in top_predictions]
            counts = [p[1] for p in top_predictions]

            ax.bar(range(len(labels)), counts)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlabel('Predicted Pair (sorted IDs)')
            ax.set_ylabel('Frequency')
            ax.set_title('Top 20 Most Common Predictions')
            plt.tight_layout()

            # Save to Wandb
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp, format='png', dpi=100)
                tmp_path = tmp.name

            wandb_run.log({
                "distribution/top_predictions_histogram": wandb_module.Image(tmp_path)
            })
            plt.close(fig)

            # 一時ファイルを削除
            import os
            os.unlink(tmp_path)

            logger.info("Logged distribution plots to Wandb")


class EmbeddingVisualizer:
    """Visualizer for embedding space using dimensionality reduction."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: list[str] | None = None,
    ):
        """
        Initialize embedding visualizer.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Optional labels for each embedding
        """
        self.embeddings = embeddings
        self.labels = labels or [f"Sample_{i}" for i in range(len(embeddings))]

    def create_2d_projection(
        self,
        method: str = "umap",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Create 2D projection of embeddings.

        Args:
            method: Reduction method ("umap", "tsne", "pca")
            n_components: Number of components (usually 2 for 2D)
            random_state: Random seed
            **kwargs: Additional arguments for the reduction method

        Returns:
            Array of shape (n_samples, n_components)
        """
        logger.info(f"Creating {method.upper()} projection...")

        if method == "umap":
            # Default UMAP parameters
            n_neighbors = kwargs.get("n_neighbors", 15)
            min_dist = kwargs.get("min_dist", 0.1)

            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state,
            )
            projection = reducer.fit_transform(self.embeddings)

        elif method == "tsne":
            perplexity = kwargs.get("perplexity", min(30, len(self.embeddings) - 1))

            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                method='barnes_hut' if len(self.embeddings) > 1000 else 'exact',
            )
            projection = reducer.fit_transform(self.embeddings)

        elif method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(
                n_components=n_components,
                random_state=random_state,
            )
            projection = reducer.fit_transform(self.embeddings)

        else:
            raise ValueError(f"Unknown reduction method: {method}")

        logger.info(f"Projection complete: {projection.shape}")
        return projection

    def log_scatter_plot(
        self,
        projection: np.ndarray,
        wandb_run,
        title: str = "Embedding Projection",
    ):
        """
        Create and log scatter plot to Wandb.

        Args:
            projection: 2D projection array
            wandb_run: Wandb Run object
            title: Plot title
        """
        import wandb as wandb_module
        import io

        fig, ax = plt.subplots(figsize=(10, 10))

        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            alpha=0.6,
            s=50,
            c=range(len(projection)),
            cmap='viridis',
        )

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Sample Index')

        plt.tight_layout()

        # Save to Wandb
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        wandb_run.log({
            "embeddings/scatter_plot": wandb_module.Image(buf)
        })
        plt.close(fig)

        logger.info("Logged embedding scatter plot to Wandb")

    def log_cluster_analysis(
        self,
        projection: np.ndarray,
        wandb_run,
        n_clusters: int | None = None,
    ):
        """
        Perform cluster analysis and log metrics.

        Args:
            projection: 2D projection array
            wandb_run: Wandb Run object
            n_clusters: Number of clusters (default: sqrt(n_samples))
        """
        from sklearn.cluster import KMeans

        if n_clusters is None:
            n_clusters = int(np.sqrt(len(projection)))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(projection)

        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(projection, cluster_labels)
        else:
            silhouette = 0.0

        # Log metrics
        wandb_run.log({
            "embeddings/n_clusters": n_clusters,
            "embeddings/silhouette_score": silhouette,
        })

        logger.info(f"Cluster analysis: {n_clusters} clusters, silhouette={silhouette:.3f}")
