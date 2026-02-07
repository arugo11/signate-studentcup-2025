````md
# CLAUDE.md

このファイルは, このリポジトリ内のコードを扱う際に Claude Code (claude.ai/code) へガイダンスを提供します.

## Project Overview

これは **Signate Student Cup 2025** の機械学習コンペ提出物です. タスクは, LLM が生成した混合作品のあらすじが, どの2つの元作品(映画, アニメ, 漫画)を組み合わせて作られたかを予測することです. 50個の元作品の要素を混ぜて生成された340個の架空あらすじが与えられ, モデルは正しい元作品ペアを特定する必要があります.
また,推論速度は評価対象にないため長時間の処理を行って,精度を高めることも有効であることが大事です.

**Competition URL**: https://user.competition.signate.jp/ja/competition/detail/?competition=385dcbba17b645f3ac10f827dfba03f6

## Environment and Dependencies

**重要: パッケージ管理コマンドは常に `uv` を使用すること.**
- `pip`, `pip install`, `python -m pip` を直接使わないこと
- 新しい依存関係の追加は `uv add <package>` を使う
- pyproject.toml から依存関係を同期するには `uv sync` を使う
- uv 環境内でスクリプトを実行するには `uv run <script>` を使う

**Package Manager**: `uv` (モダンな Python パッケージマネージャ)
- **Python Version**: 3.10
- **Virtual Environment**: `.venv/` に配置

### Setup Commands

```bash
# Create virtual environment
make create_environment

# Install dependencies
make requirements
# or: uv sync

# Activate environment (if needed manually)
# Windows: .\\.venv\\Scripts\\activate
# Unix/macOS: source ./.venv/bin/activate
````

### Environment Variables

必須(`.env` ファイルに設定):

* `OPENROUTER_API_KEY` - 埋め込み生成と LLM 推論に必要

任意:

* `WANDB_API_KEY` - 実験トラッキング, アーティファクト, sweep のため
* `WANDB_ENTITY` - Wandb entity (team/username)

`.example.env` を `.env` にコピーして設定してください.

## Common Commands

```bash
# Process/transform data
python signate_studentcup_2025/dataset.py --input-path data/raw/base_stories.tsv --output-path data/processed/processed_base_stories.csv

# Evaluate on practice data
python signate_studentcup_2025/modeling/evaluate.py --approach retrieval --top-k 10
python signate_studentcup_2025/modeling/evaluate.py --approach direct --model openai/gpt-4o-mini

# Run inference/prediction (generate submission)
python signate_studentcup_2025/modeling/predict.py --approach retrieval --top-k 10 --output-path data/processed/submission.csv

# Hyperparameter sweep
python signate_studentcup_2025/modeling/evaluate.py sweep --sweep-config config/sweeps_retrieval.yaml --count 50

# Code quality checks
make lint           # Check code style (ruff format --check + ruff check)
make format         # Format code (ruff check --fix + ruff format)
make clean          # Delete .pyc files and __pycache__ directories

# Run tests
pytest
```

## Project Architecture

このプロジェクトは **Cookiecutter Data Science (CCDS)** のテンプレート構造に従います.

```
signate_studentcup_2025/          # Main source package
├── config.py                     # Centralized configuration (paths, YAML loading, Wandb/Weave setup)
├── config/                       # YAML configuration files
│   ├── default.yaml              # Default settings (paths, metrics, sweeps)
│   ├── models/                   # Model configurations
│   │   ├── openrouter.yaml       # OpenRouter embedding/chat models
│   │   └── local.yaml            # Local model configurations
│   ├── prompts/                  # Prompt templates for direct LLM inference
│   └── sweeps*.yaml              # Hyperparameter sweep configurations
├── dataset.py                    # Data processing pipeline (load_base_stories, load_fiction_data)
├── features.py                   # Feature engineering (EmbeddingModel, FAISS index, retrieval)
├── weave.py                      # Weave tracing utilities (decorators, initialization)
├── modeling/
│   ├── predict.py                # Inference predictors (RetrievalPredictor, DirectPredictor)
│   ├── evaluate.py               # Evaluation metrics, sweep launcher
│   ├── analysis.py               # Error analysis, visualization utilities
│   ├── sweep_wrapper.py          # Wandb sweep execution wrapper
│   └── sweep_analysis.py         # Sweep result analysis
└── plots.py                      # Visualization utilities

data/
├── raw/                          # Original competition data (TSV files)
├── interim/                      # Intermediate data (FAISS indices)
├── processed/                    # Final datasets and submission files
└── external/                     # Third-party data sources

models/                           # Trained models, checkpoints
reports/                          # Analysis outputs, figures
notebooks/                        # Jupyter notebooks for exploration
references/                       # Documentation, task descriptions
tests/                           # Test suite (pytest)
```

## Configuration System

設定はすべて `config.py` と `signate_studentcup_2025/config/` 配下の YAML ファイルに集約されています.

### Config Classes (in `config.py`)

* `WandbConfig` - Wandb 連携(entity, project, mode)
* `OpenRouterConfig` - API keys, model names, embedding settings
* `RetrievalConfig` - Top-K, FAISS index type
* `EvaluationConfig` - Metrics, hit rate/NDCG の K 値
* `DataConfig` - 全データファイルのパス
* `OutputConfig` - 出力ディレクトリ
* `ArtifactsConfig` - Wandb にログする対象
* `WeaveConfig` - トレーシング制御(embeddings, retrieval, predictions)
* `DashboardConfig` - 可視化トグル(error analysis, embedding viz)
* `SweepsConfig` - ハイパーパラメータ sweep 設定

### YAML Configuration Files

ハードコードではなく, これらのファイルを編集してください.

* `default.yaml` - メイン設定(paths, metrics, wandb/weave settings)
* `models/openrouter.yaml` - 埋め込みモデルと chat model 設定
* `sweeps.yaml` / `sweeps_retrieval.yaml` / `sweeps_direct.yaml` - ハイパーパラメータ探索空間
* `prompts/*.yaml` - direct LLM 推論用のプロンプトテンプレート

## Prediction Approaches

### 1. Retrieval Approach (`--approach retrieval`)

OpenRouter embeddings + FAISS 類似検索を使用します.

* OpenRouter embedding API でクエリテキストを埋め込み
* 元作品の埋め込みで作成した FAISS index を検索
* 最も類似した上位2作品を返す

主要クラス: `modeling/predict.py` の `OpenRouterRetrievalPredictor`

### 2. Direct Approach (`--approach direct`)

LLM の chat completion を直接使用します.

* クエリ全文 + 作品リストを OpenRouter chat model に送信
* モデルが2つの作品 ID を含む JSON を出力
* `--prompt-name` フラグでプロンプトテンプレートを選択可能

主要クラス: `modeling/predict.py` の `OpenRouterDirectPredictor`

### TODO: Local Embedding Backend

ローカル SentenceTransformer backend は `features.py` に部分実装されています(TODO コメント参照). 完成させるには:

1. `sentence-transformers` 依存関係を追加
2. `SentenceTransformerBackend` クラスを実装
3. `--approach dense` の CLI オプションを追加

## Evaluation Metrics

`modeling/evaluate.py` に実装されています.

* **Accuracy** - コンペ指標(両方の ID が一致する必要あり)
* **Hit Rate@K** - top-K の中に少なくとも1つ正解 ID が含まれる
* **MRR** - Mean Reciprocal Rank
* **NDCG@K** - Normalized Discounted Cumulative Gain

## Wandb Integration Features

このプロジェクトは包括的な Wandb 連携を持ちます.

### Artifacts

* FAISS indices(index 構築時に自動ログ)
* Datasets(base stories, fiction data)
* Submissions(予測ファイルのバージョニング)
* Evaluations(メタデータ付き詳細結果)

### Sweeps

* `evaluate.py sweep` によるハイパーパラメータ最適化
* `config/sweeps*.yaml` による設定ベースの sweep 定義
* `modeling/sweep_analysis.py` による sweep 分析

### Weave Tracing

* Decorators: `@weave_op_decorator`, `@weave_op_decorator_configured`
* embeddings, retrieval, predictions のトレーシングを設定可能
* `default.yaml` の `WeaveConfig` で制御

### Analysis & Visualization

* `ErrorAnalyzer` - 誤分類分析
* `PredictionAnalyzer` - 予測分布プロット
* `EmbeddingVisualizer` - 埋め込み空間の UMAP 投影
* `default.yaml` の `DashboardConfig` で制御

## Code Patterns

### CLI Scripts

実行可能スクリプトはすべて CLI インターフェースに **Typer** を使います.

```python
app = typer.Typer()

@app.command()
def main(arg_path: Path = DEFAULT_PATH):
    logger.info("Processing...")
    # ... code ...
    logger.success("Complete.")

if __name__ == "__main__":
    app()
```

### Abstract Base Classes

* `features.py` の `EmbeddingBackend` - 埋め込みモデル backend のため
* `modeling/predict.py` の `Predictor` - 予測アプローチのため

### Weave Decorators

```python
# Always trace (if Weave enabled)
from signate_studentcup_2025.weave import weave_op_decorator

@weave_op_decorator
def my_function(x):
    return x * 2

# Trace only if config flag is True
from signate_studentcup_2025.weave import weave_op_decorator_configured

@weave_op_decorator_configured("TRACE_EMBEDDINGS")
def encode(texts):
    return embeddings
```

### Data Loading

```python
from signate_studentcup_2025.dataset import load_base_stories, load_fiction_data, prepare_corpus

# Load base stories (50 source works)
base_df = load_base_stories(DataConfig.BASE_STORIES_PATH)

# Load fiction data (practice or test)
fiction_df = load_fiction_data(DataConfig.FICTION_STORIES_TEST_PATH)

# Prepare corpus for embedding
corpus = prepare_corpus(base_df)  # Returns polars Series with formatted text
```

### Logging

**Loguru** を tqdm 連携で使用します. logger は `config.py` で設定され, tqdm の進捗バーを壊さずに出力するようになっています.

### Linting

フォーマットと lint に **ruff** を使用します.

* Line length: 99 characters
* isort 有効で import 並び替え
* `config/__init__.py` が動的 import を必要とするため, E402 (module level imports not at top of file) はグローバルで無視

変更を commit する前に `make lint` または `uv run ruff check` でスタイルを確認してください.

## Competition Data Files

`data/raw/` にあります.

* **base_stories.tsv** - 50元作品. columns: `id`, `category`, `title`, `story`
* **fiction_stories_practice.tsv** - practice データ. columns: `id`, `id_a`, `id_b`, `title_a`, `title_b`, `story`
* **fiction_stories_test.tsv** - 340個の test あらすじ. columns: `id`, `story`

### Submission Format

header なし CSV. columns: `id`, `a`, `b`

* ID はソート必須(a < b)
* 出力先デフォルトは `data/processed/submission.csv`

## Evaluation Metric

**Accuracy**: 2つの元作品 ID が両方一致する必要があります(順序は不問). `{predicted_a, predicted_b}` が `{actual_a, actual_b}` と等しい場合のみ正解です. 部分一致(片方だけ正解)は不正解として扱われます.

## Language Notes

* コンペのドキュメントは **Japanese** です(`references/task.md` 参照)
* データファイル(あらすじ, タイトル)は **Japanese** です
* 保守性のため, コードコメントと変数名は **English** にするべきです
* コードベース内に Japanese のコメントや変数名が一部あります. これらは段階的に English へ移行してください

```
```
