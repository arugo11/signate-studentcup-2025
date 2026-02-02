from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os
from typing import Any
import yaml

# 環境変数読み込み
load_dotenv()

# パス
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

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
    EMBEDDING_MODEL = _OPENROUTER_CONFIG.get("embedding", {}).get("model", "openai/text-embedding-3-small")
    EMBEDDING_DIM = _OPENROUTER_CONFIG.get("embedding", {}).get("dim", 1536)
    EMBEDDING_MAX_LENGTH = _OPENROUTER_CONFIG.get("embedding", {}).get("max_length", 8191)

    # チャットモデル（YAMLから読み込み）
    CHAT_MODELS = {
        key: value["id"]
        for key, value in _OPENROUTER_CONFIG.get("chat", {}).get("models", {}).items()
    }
    DEFAULT_CHAT_MODEL = _OPENROUTER_CONFIG.get("chat", {}).get("models", {}).get(
        _OPENROUTER_CONFIG.get("chat", {}).get("default_model", "gpt-4o-mini"), {}
    ).get("id", "openai/gpt-4o-mini")


class RetrievalConfig:
    """検索設定（YAMLから読み込み）"""
    TOP_K = _DEFAULT_CONFIG.get("retrieval", {}).get("top_k", 10)
    FAISS_INDEX_TYPE = _DEFAULT_CONFIG.get("retrieval", {}).get("faiss_index_type", "IndexFlatIP")


class EvaluationConfig:
    """評価設定（YAMLから読み込み）"""
    METRICS = _DEFAULT_CONFIG.get("evaluation", {}).get("metrics", ["accuracy", "hit_rate", "mrr", "ndcg"])
    K_VALUES = _DEFAULT_CONFIG.get("evaluation", {}).get("k_values", [1, 5, 10, 20])


class DataConfig:
    """データパス設定（YAMLから読み込み）"""
    BASE_STORIES_PATH = Path(_DEFAULT_CONFIG.get("data", {}).get("base_stories_path", "data/raw/base_stories.tsv"))
    FICTION_STORIES_PRACTICE_PATH = Path(_DEFAULT_CONFIG.get("data", {}).get("fiction_stories_practice_path", "data/raw/fiction_stories_practice.tsv"))
    FICTION_STORIES_TEST_PATH = Path(_DEFAULT_CONFIG.get("data", {}).get("fiction_stories_test_path", "data/raw/fiction_stories_test.tsv"))


class OutputConfig:
    """出力設定（YAMLから読み込み）"""
    SUBMISSION_DIR = Path(_DEFAULT_CONFIG.get("output", {}).get("submission_dir", "data/processed"))
    INTERIM_DIR = Path(_DEFAULT_CONFIG.get("output", {}).get("interim_dir", "data/interim"))


# TODO: 将来的に追加するWandb機能
"""
TODO: Wandbの高度な機能を実装

現在の実装: 基本的なログ（metrics, config）のみ

将来的に追加する機能:
1. Artifacts管理
   - FAISSインデックスの保存・共有
   - 埋め込みモデルのキャッシュ
   - 提出ファイルのバージョニング
   実装方法: wandb.log_artifact(), wandb.use_artifact()

2. Sweeps（ハイパーパラメータ探索）
   - top_k, model_nameの最適化
   - 複数アプローチの比較
   実装方法: wandb.sweep() + wandb.agent()

3. Weave（トレース・デバッグ）
   - 検索・再ランキングのステップをトレース
   - エラー解析を容易に
   実装方法: @weave.op() デコレータ

4. Modelの保存・ロード
   - 学習済みモデルの登録
   - 推論時のモデルバージョン管理
   実装方法: wandb.Artifact()

参考: https://docs.wandb.ai/guides
"""
