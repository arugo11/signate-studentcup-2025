# signate-studentcup-2025

SIGNATE Student Cup 2025 向けのベースライン実装です。
OpenRouter の埋め込み + FAISS 検索、または LLM 直接推論で、架空作品の元ネタ2作品IDを推定します。

コンペページ:
https://user.competition.signate.jp/ja/competition/detail/?competition=385dcbba17b645f3ac10f827dfba03f6

## できること
- retrieval: OpenRouter 埋め込み + FAISS で類似検索して2作品IDを推定
- direct: OpenRouter チャットモデルで2作品IDを直接推定
- practice データで評価（accuracy / hit_rate / mrr / ndcg）
- 提出形式のバリデーション

## セットアップ
- Python 3.10
- 依存関係の導入:
  - `uv venv --python 3.10`
  - `uv sync`（または `make requirements`）
- 環境変数（`.env`）:
  - `OPENROUTER_API_KEY` は必須
  - `WANDB_API_KEY` / `WANDB_ENTITY` は任意

`.example.env` を `.env` にコピーして設定してください。

## データ配置
`signate_studentcup_2025/config/default.yaml` のデフォルト:
- `data/raw/base_stories.tsv`
- `data/raw/fiction_stories_practice.tsv`
- `data/raw/fiction_stories_test.tsv`

想定カラム:
- base: `id, category, title, story`
- practice: `id, id_a, id_b, title_a, title_b, story`
- test: `id, story`

## 使い方
### 評価（practice）
```
python signate_studentcup_2025/modeling/evaluate.py --approach retrieval --top-k 10
python signate_studentcup_2025/modeling/evaluate.py --approach direct --model openai/gpt-4o-mini
```

### 提出ファイル生成（test）
```
python signate_studentcup_2025/modeling/predict.py --approach retrieval --top-k 10 --output-path data/processed/submission.csv
```
`submission.csv` はヘッダーなしで `id,a,b` 順に出力されます。

### コーパス作成（任意）
```
python signate_studentcup_2025/dataset.py --input-path data/raw/base_stories.tsv --output-path data/processed/processed_base_stories.csv
```

## 設定
- `signate_studentcup_2025/config/default.yaml`
  - データパス / top_k / 評価K / 出力先
- `signate_studentcup_2025/config/models/openrouter.yaml`
  - OpenRouter の埋め込み・チャットモデル設定
- `signate_studentcup_2025/config/models/local.yaml`
  - ローカル埋め込み（将来用）

## 出力と中間ファイル
- 提出ファイル: `data/processed/submission.csv`
- FAISS インデックス: `data/interim/faiss_index_openrouter.pkl`

## テスト
```
pytest
```

## プロジェクト構成（主要部分）
```
signate_studentcup_2025/
  config.py
  config/
    default.yaml
    models/openrouter.yaml
  dataset.py
  features.py
  modeling/
    predict.py
    evaluate.py
```
