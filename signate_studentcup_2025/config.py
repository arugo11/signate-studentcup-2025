import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
import yaml

# パス
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# 環境変数読み込み（.envをプロジェクトルートから読み込む）
load_dotenv(PROJ_ROOT / ".env")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configディレクトリ
CONFIG_DIR = PROJ_ROOT / "signate_studentcup_2025" / "config"

# Loguru with tqdm
try:
    from tqdm import tqdm

    # ハンドラーが存在する場合のみ削除
    try:
        logger.remove(0)
    except ValueError:
        # ハンドラーが存在しない場合は無視
        pass
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# === YAML設定読み込み ===


def load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """YAMLファイルを読み込み"""
    if not yaml_path.exists():
        logger.warning(f"設定ファイルが見つかりません: {yaml_path}")
        return {}

    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# デフォルト設定
_DEFAULT_CONFIG = load_yaml_config(CONFIG_DIR / "default.yaml")

# OpenRouterモデル設定
_OPENROUTER_CONFIG = load_yaml_config(CONFIG_DIR / "models" / "openrouter.yaml")

# ローカルモデル設定（後で使用）
_LOCAL_MODEL_CONFIG = load_yaml_config(CONFIG_DIR / "models" / "local.yaml")


# === 環境変数 + YAMLから設定を構築 ===


class WandbConfig:
    """Wandb設定（YAML + 環境変数）"""

    ENABLED = os.getenv("WANDB_API_KEY") is not None
    API_KEY = os.getenv("WANDB_API_KEY")
    ENTITY = os.getenv("WANDB_ENTITY", _DEFAULT_CONFIG.get("wandb", {}).get("entity", "argo11"))
    PROJECT = _DEFAULT_CONFIG.get("wandb", {}).get("project", "studentcup-2025")
    MODE = _DEFAULT_CONFIG.get("wandb", {}).get("mode", "online")

    if not ENABLED:
        MODE = "disabled"


class OpenRouterConfig:
    """OpenRouter API設定（YAML + 環境変数）"""

    ENABLED = os.getenv("OPENROUTER_API_KEY") is not None
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    BASE_URL = "https://openrouter.ai/api/v1"

    # 埋め込みモデル（YAMLから読み込み）
    EMBEDDING_MODEL = _OPENROUTER_CONFIG.get("embedding", {}).get(
        "model", "openai/text-embedding-3-small"
    )
    EMBEDDING_DIM = _OPENROUTER_CONFIG.get("embedding", {}).get("dim", 1536)
    EMBEDDING_MAX_LENGTH = _OPENROUTER_CONFIG.get("embedding", {}).get("max_length", 8191)

    # チャットモデル（YAMLから読み込み）
    CHAT_MODELS = {
        key: value["id"]
        for key, value in _OPENROUTER_CONFIG.get("chat", {}).get("models", {}).items()
    }
    DEFAULT_CHAT_MODEL = (
        _OPENROUTER_CONFIG.get("chat", {})
        .get("models", {})
        .get(_OPENROUTER_CONFIG.get("chat", {}).get("default_model", "gpt-4o-mini"), {})
        .get("id", "openai/gpt-4o-mini")
    )


class RetrievalConfig:
    """検索設定（YAMLから読み込み）"""

    TOP_K = _DEFAULT_CONFIG.get("retrieval", {}).get("top_k", 10)
    FAISS_INDEX_TYPE = _DEFAULT_CONFIG.get("retrieval", {}).get("faiss_index_type", "IndexFlatIP")


class EvaluationConfig:
    """評価設定（YAMLから読み込み）"""

    METRICS = _DEFAULT_CONFIG.get("evaluation", {}).get(
        "metrics", ["accuracy", "hit_rate", "mrr", "ndcg"]
    )
    K_VALUES = _DEFAULT_CONFIG.get("evaluation", {}).get("k_values", [1, 5, 10, 20])


class DataConfig:
    """データパス設定（YAMLから読み込み）"""

    BASE_STORIES_PATH = Path(
        _DEFAULT_CONFIG.get("data", {}).get("base_stories_path", "data/raw/base_stories.tsv")
    )
    FICTION_STORIES_PRACTICE_PATH = Path(
        _DEFAULT_CONFIG.get("data", {}).get(
            "fiction_stories_practice_path", "data/raw/fiction_stories_practice.tsv"
        )
    )
    FICTION_STORIES_TEST_PATH = Path(
        _DEFAULT_CONFIG.get("data", {}).get(
            "fiction_stories_test_path", "data/raw/fiction_stories_test.tsv"
        )
    )


class OutputConfig:
    """出力設定（YAMLから読み込み）"""

    SUBMISSION_DIR = Path(
        _DEFAULT_CONFIG.get("output", {}).get("submission_dir", "data/processed")
    )
    INTERIM_DIR = Path(_DEFAULT_CONFIG.get("output", {}).get("interim_dir", "data/interim"))


class ArtifactsConfig:
    """Artifacts管理設定（YAMLから読み込み）"""

    LOG_FAISS_INDEX = _DEFAULT_CONFIG.get("artifacts", {}).get("log_faiss_index", True)
    LOG_DATASETS = _DEFAULT_CONFIG.get("artifacts", {}).get("log_datasets", True)
    LOG_SUBMISSIONS = _DEFAULT_CONFIG.get("artifacts", {}).get("log_submissions", True)
    LOG_EVALUATIONS = _DEFAULT_CONFIG.get("artifacts", {}).get("log_evaluations", True)


class WeaveConfig:
    """Weaveトレース設定（YAMLから読み込み）"""

    ENABLED = _DEFAULT_CONFIG.get("weave", {}).get("enabled", True)
    TRACE_EMBEDDINGS = _DEFAULT_CONFIG.get("weave", {}).get("trace_embeddings", True)
    TRACE_RETRIEVAL = _DEFAULT_CONFIG.get("weave", {}).get("trace_retrieval", True)
    TRACE_PREDICTIONS = _DEFAULT_CONFIG.get("weave", {}).get("trace_predictions", True)
    TRACE_RERANKING = _DEFAULT_CONFIG.get("weave", {}).get("trace_reranking", True)


class DashboardConfig:
    """ダッシュボード設定（YAMLから読み込み）"""

    ERROR_ANALYSIS = _DEFAULT_CONFIG.get("dashboard", {}).get("error_analysis", True)
    EMBEDDING_VIZ = _DEFAULT_CONFIG.get("dashboard", {}).get("embedding_viz", False)
    DISTRIBUTION_PLOTS = _DEFAULT_CONFIG.get("dashboard", {}).get("distribution_plots", True)


class SweepsConfig:
    """Sweeps設定（YAMLから読み込み）"""

    MAX_TRIALS = _DEFAULT_CONFIG.get("sweeps", {}).get("max_trials", 100)
    OPTIMIZATION_METRIC = _DEFAULT_CONFIG.get("sweeps", {}).get("optimization_metric", "accuracy")


class RerankingConfig:
    """Reranking設定（YAMLから読み込み）"""

    ENABLED = _DEFAULT_CONFIG.get("reranking", {}).get("enabled", True)
    MODEL = _DEFAULT_CONFIG.get("reranking", {}).get("model", "cl-nagoya/ruri-reranker-large")
    RETRIEVAL_K = _DEFAULT_CONFIG.get("reranking", {}).get("retrieval_k", 20)
    BATCH_SIZE = _DEFAULT_CONFIG.get("reranking", {}).get("batch_size", 32)
    DEVICE = _DEFAULT_CONFIG.get("reranking", {}).get("device", "cpu")


# === Artifactsヘルパー関数 ===


def log_dataset_as_artifact(
    df,
    artifact_name: str,
    artifact_type: str,
    wandb_run,
    metadata: dict | None = None,
):
    """
    DataFrameをWandb Artifactとしてログ

    Args:
        df: Polars DataFrame
        artifact_name: Artifact名（例: "base-stories"）
        artifact_type: Artifactタイプ（例: "dataset"）
        wandb_run: Wandb Runオブジェクト
        metadata: 追加メタデータ

    Returns:
        wandb.Artifact: ログされたArtifact
    """
    import tempfile

    import wandb as wandb_module

    artifact = wandb_module.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata={
            "num_rows": len(df),
            "columns": df.columns,
            **(metadata or {}),
        },
    )

    # 一時ファイルに保存してartifactに追加
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.write_csv(f.name)
        artifact.add_file(f.name, name=f"{artifact_name}.csv")
        temp_path = f.name

    wandb_run.log_artifact(artifact)
    logger.info(f"Logged dataset artifact: {artifact.name}")

    # 一時ファイルを削除
    Path(temp_path).unlink(missing_ok=True)

    return artifact


"""
Wandb機能実装状況:

実装済み:
✓ Artifacts管理
  - FAISSインデックスの保存（build_faiss_indexで自動ログ）
  - データセットの保存（log_dataset_as_artifact関数）
  - 提出ファイルのバージョニング（predict.pyで実装）
  - 評価結果の保存（evaluate.pyで実装）

✓ Sweeps（ハイパーパラメータ探索）
  - sweep_wrapper.py: wandb.sweep() + wandb.agent()
  - sweep_analysis.py: 結果分析と比較
  - 設定ファイル: config/sweeps*.yaml

✓ Weave（トレース・デバッグ）
  - init_weave(): Weave初期化
  - weave_op_decorator: @weave.op()デコレータ
  - weave_op_decorator_configured: 設定によるOn/Off制御
  - WeaveConfigによる制御（TRACE_EMBEDDINGS, TRACE_RETRIEVAL, TRACE_PREDICTIONS）

✓ ダッシュボード機能
  - ErrorAnalyzer: 誤分類分析
  - PredictionAnalyzer: 予測分布分析
  - EmbeddingVisualizer: 埋め込み空間可視化
  - DashboardConfigによるOn/Off制御（ERROR_ANALYSIS, DISTRIBUTION_PLOTS, EMBEDDING_VIZ）

参考: https://docs.wandb.ai/guides
"""
