"""Sweep analysis tools for hyperparameter optimization results.

This module provides utilities for analyzing Wandb sweep results,
extracting insights, and exporting data.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
import typer
import wandb

app = typer.Typer()


def analyze_sweep(
    sweep_path: str,
    entity: str | None = None,
    project: str | None = None,
) -> pd.DataFrame:
    """
    Fetch and analyze sweep results.

    Args:
        sweep_path: Sweep path (e.g., "entity/project/sweep_id")
        entity: Wandb entity (overrides sweep_path)
        project: Wandb project (overrides sweep_path)

    Returns:
        DataFrame with sweep results
    """
    # Parse sweep path
    parts = sweep_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid sweep path: {sweep_path}. Expected format: entity/project/sweep_id")

    sweep_entity = entity or parts[0]
    sweep_project = project or parts[1]
    sweep_id = parts[2]

    logger.info(f"Fetching sweep: {sweep_entity}/{sweep_project}/{sweep_id}")

    # Get sweep
    api = wandb.Api()
    sweep = api.sweep(f"{sweep_entity}/{sweep_project}/{sweep_id}")

    # Get all runs
    runs = []
    for run in sweep.runs:
        if run.state == "finished":
            run_data = {
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
            }

            # Add config parameters
            for key, value in run.config.items():
                if key not in ["_wandb", "runtime"]:
                    run_data[key] = value

            # Add summary metrics
            for key, value in run.summary.items():
                if isinstance(value, (int, float)):
                    run_data[key] = value

            runs.append(run_data)

    df = pd.DataFrame(runs)

    # Sort by accuracy (descending)
    if "accuracy" in df.columns:
        df = df.sort_values("accuracy", ascending=False)

    logger.info(f"Fetched {len(df)} finished runs")
    return df


def print_top_configurations(df: pd.DataFrame, top_n: int = 10):
    """
    Print top N configurations from sweep results.

    Args:
        df: Sweep results DataFrame
        top_n: Number of top configurations to print
    """
    if "accuracy" not in df.columns:
        logger.warning("Accuracy column not found in results")
        return

    top_configs = df.head(top_n)

    logger.info(f"\n=== Top {top_n} Configurations ===")
    for idx, row in top_configs.iterrows():
        logger.info(f"\nRank {idx + 1}: Accuracy = {row['accuracy']:.3f}")

        # Print key parameters
        params = ["approach", "top_k", "model", "faiss_index_type"]
        for param in params:
            if param in row:
                logger.info(f"  {param}: {row[param]}")


def calculate_parameter_importance(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate parameter importance based on correlation with accuracy.

    Args:
        df: Sweep results DataFrame

    Returns:
        Dictionary with parameter importance scores
    """
    if "accuracy" not in df.columns:
        logger.warning("Cannot calculate importance: accuracy column missing")
        return {}

    importance = {}

    # Get numeric parameters
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Remove non-parameter columns
    exclude_cols = ["accuracy", "run_id", "state"]
    param_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Calculate correlation with accuracy
    for param in param_cols:
        corr = df[[param, "accuracy"]].corr().iloc[0, 1]
        if not pd.isna(corr):
            importance[param] = abs(corr)

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return importance


def export_sweep_results(
    df: pd.DataFrame,
    output_path: Path,
):
    """
    Export sweep results to CSV.

    Args:
        df: Sweep results DataFrame
        output_path: Output CSV path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.success(f"Exported sweep results to: {output_path}")


@app.command()
def analyze_sweep_command(
    sweep_path: str = typer.Argument(..., help="Sweep path (e.g., entity/project/sweep_id)"),
    top_n: int = typer.Option(10, help="Number of top configurations to display"),
    output_path: str = typer.Option(None, help="Output CSV path for results"),
):
    """
    Analyze sweep results and print insights.

    Example:
        python signate_studentcup_2025/modeling/sweep_analysis.py analyze-sweep \\
            argo11/studentcup-2025/ABC123 --top_n 10 \\
            --output-path reports/sweep_results.csv
    """
    # Fetch sweep results
    df = analyze_sweep(sweep_path)

    # Print top configurations
    print_top_configurations(df, top_n=top_n)

    # Calculate parameter importance
    logger.info("\n=== Parameter Importance (correlation with accuracy) ===")
    importance = calculate_parameter_importance(df)
    for param, score in importance.items():
        logger.info(f"  {param}: {score:.3f}")

    # Export to CSV
    if output_path:
        export_sweep_results(df, Path(output_path))


@app.command()
def compare_sweeps(
    sweep_paths: list[str] = typer.Argument(..., help="Multiple sweep paths to compare"),
):
    """
    Compare multiple sweeps side-by-side.

    Example:
        python signate_studentcup_2025/modeling/sweep_analysis.py compare-sweeps \\
            argo11/studentcup-2025/sweep1 argo11/studentcup-2025/sweep2
    """
    results = {}

    for sweep_path in sweep_paths:
        try:
            df = analyze_sweep(sweep_path)
            sweep_id = sweep_path.split("/")[-1]
            results[sweep_id] = df

            logger.info(f"\n=== Sweep: {sweep_id} ===")
            logger.info(f"Total runs: {len(df)}")
            if "accuracy" in df.columns:
                logger.info(f"Best accuracy: {df['accuracy'].max():.3f}")
                logger.info(f"Mean accuracy: {df['accuracy'].mean():.3f}")
                logger.info(f"Std accuracy: {df['accuracy'].std():.3f}")

        except Exception as e:
            logger.error(f"Failed to analyze sweep {sweep_path}: {e}")

    # Summary comparison
    logger.info("\n=== Summary Comparison ===")
    for sweep_id, df in results.items():
        if "accuracy" in df.columns:
            logger.info(f"{sweep_id}: best={df['accuracy'].max():.3f}, "
                       f"mean={df['accuracy'].mean():.3f}, "
                       f"runs={len(df)}")


if __name__ == "__main__":
    app()
