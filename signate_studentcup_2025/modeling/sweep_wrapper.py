"""Sweep wrapper for hyperparameter optimization with Wandb.

This module provides utilities for launching and managing Wandb sweeps.
"""

from pathlib import Path

from loguru import logger
import typer
import wandb

from signate_studentcup_2025.config import (
    DataConfig,
    EvaluationConfig,
    WandbConfig,
)

app = typer.Typer()


def launch_sweep(
    sweep_config_path: Path,
    entity: str,
    project: str,
    count: int = 100,
) -> str:
    """
    Launch a new sweep or resume existing one.

    Args:
        sweep_config_path: Path to sweep YAML config
        entity: Wandb entity
        project: Wandb project name
        count: Number of trials to run

    Returns:
        Sweep ID
    """
    import yaml

    # Load sweep config
    with open(sweep_config_path, encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=entity,
        project=project,
    )

    logger.info(f"Launched sweep: {sweep_id}")
    return sweep_id


def evaluate_sweep_trial(
    approach: str = "retrieval",
    top_k: int = 10,
    model: str = "openai/gpt-4o-mini",
    device: str = "cpu",
):
    """
    Evaluate a single sweep trial.

    This function is called by wandb.agent() for each trial.

    Args:
        approach: Prediction approach ("retrieval", "dense", or "direct")
        top_k: Top-K for retrieval
        model: Model name
        device: Device for local models (cpu | cuda)
    """
    # Import here to avoid circular dependencies
    from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data
    from signate_studentcup_2025.modeling.analysis import ErrorAnalyzer, PredictionAnalyzer
    from signate_studentcup_2025.modeling.evaluate import compute_metrics
    from signate_studentcup_2025.modeling.predict import (
        DenseRetrievalPredictor,
        OpenRouterDirectPredictor,
        OpenRouterRetrievalPredictor,
    )

    # Initialize wandb run (handled by agent)
    run = wandb.run

    if run is None:
        logger.error("Wandb run not initialized. This function should be called by wandb.agent()")
        return

    # Log configuration
    run.config.update(
        {
            "approach": approach,
            "top_k": top_k,
            "model": model,
            "device": device,
        }
    )

    logger.info(
        f"Running sweep trial: approach={approach}, top_k={top_k}, model={model}, device={device}"
    )

    # Load data
    base_df = load_base_stories(DataConfig.BASE_STORIES_PATH)
    practice_df = load_fiction_data(DataConfig.FICTION_STORIES_PRACTICE_PATH)

    # Initialize predictor
    if approach == "retrieval":
        predictor = OpenRouterRetrievalPredictor(top_k=top_k)
    elif approach == "dense":
        predictor = DenseRetrievalPredictor(
            model_name=model,
            top_k=top_k,
            device=device,
        )
    elif approach == "direct":
        predictor = OpenRouterDirectPredictor(model=model)
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Fit predictor
    logger.info("Fitting predictor...")
    predictor.fit(base_df, wandb_run=run)

    # Run predictions
    logger.info(f"Running predictions: {len(practice_df)} samples")
    predictions = []
    ground_truth = []

    from tqdm import tqdm

    for row in tqdm(practice_df.iter_rows(named=True), total=len(practice_df)):
        pred_ids = predictor.predict(row["story"])
        true_ids = tuple(sorted([row["id_a"], row["id_b"]]))

        predictions.append(pred_ids)
        ground_truth.append(true_ids)

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth, k_values=EvaluationConfig.K_VALUES)

    # Log primary metric
    run.log({"accuracy": metrics["accuracy"]})

    # Log additional metrics
    for key, value in metrics.items():
        if key != "accuracy":
            run.log({key: value})

    # Run error analysis
    error_analyzer = ErrorAnalyzer(
        predictions=predictions,
        ground_truth=ground_truth,
        queries_df=practice_df,
        base_df=base_df,
    )
    error_analyzer.log_to_wandb(run)

    # Run distribution analysis
    pred_analyzer = PredictionAnalyzer(predictions=predictions)
    pred_analyzer.log_distribution_plots(run)

    logger.info(f"Trial complete: accuracy={metrics['accuracy']:.3f}")


@app.command()
def launch_sweep_command(
    sweep_config: str = typer.Option("config/sweeps.yaml", help="Path to sweep config YAML"),
    count: int = typer.Option(100, help="Number of trials to run"),
):
    """
    Launch a hyperparameter sweep.

    Example:
        python signate_studentcup_2025/modeling/sweep_wrapper.py launch-sweep \\
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

    # Launch sweep
    sweep_id = launch_sweep(
        sweep_config_path=config_path,
        entity=WandbConfig.ENTITY,
        project=WandbConfig.PROJECT,
        count=count,
    )

    # Start agent
    logger.info(f"Starting wandb agent for {count} trials...")
    wandb.agent(
        sweep_id,
        function=evaluate_sweep_trial,
        count=count,
        entity=WandbConfig.ENTITY,
        project=WandbConfig.PROJECT,
    )

    logger.info("Sweep complete!")


@app.command()
def evaluate_sweep_trial_command(
    approach: str = typer.Option("retrieval", help="Prediction approach"),
    top_k: int = typer.Option(10, help="Top-K for retrieval"),
    model: str = typer.Option("openai/gpt-4o-mini", help="Model name"),
    device: str = typer.Option("cpu", help="Device for local models (cpu | cuda)"),
):
    """
    Evaluate a single sweep trial (for testing).

    This allows manual testing of a trial configuration without
    running through the full sweep agent.

    Example:
        python signate_studentcup_2025/modeling/sweep_wrapper.py evaluate-sweep-trial \\
            --approach retrieval --top_k 20
    """
    if not WandbConfig.ENABLED:
        logger.error("Wandb is not enabled. Please set WANDB_API_KEY environment variable.")
        raise typer.Exit(1)

    # Initialize wandb run
    wandb.init(
        entity=WandbConfig.ENTITY,
        project=WandbConfig.PROJECT,
        job_type="sweep_trial",
        config={
            "approach": approach,
            "top_k": top_k,
            "model": model,
            "device": device,
        },
        mode=WandbConfig.MODE,
    )

    # Initialize Weave
    from signate_studentcup_2025.weave import init_weave

    init_weave(
        entity=WandbConfig.ENTITY,
        project=WandbConfig.PROJECT,
        enabled=True,
    )

    # Run trial
    try:
        evaluate_sweep_trial(
            approach=approach,
            top_k=top_k,
            model=model,
            device=device,
        )
        logger.success("Trial complete")
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    app()
