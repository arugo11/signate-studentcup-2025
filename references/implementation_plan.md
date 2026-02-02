# 実装計画（Transformer + RAG + Wandb/Weave）

## 1. 目的とゴール
- 架空あらすじから元ネタ2作品IDを推定する。
- 手法は「検索（Dense Retrieval）→再ランキング（Transformer）」のRAG構成。
- Wandb/Weaveで実験管理・トレース・評価指標を可視化し、再現性を担保。

## 2. 入出力仕様
### 入力
- 作品データ: data/raw/base_stories.tsv
- 学習用架空作品: data/raw/fiction_stories_practice.tsv
- テスト架空作品: data/raw/fiction_stories_test.tsv

### 出力
- 提出ファイル: data/processed/submission.csv
  - 形式: 「id, a, b」(ヘッダーなし)
  - idは架空作品ID、a/bは予測された元作品ID
  - a/bの順序は問わない

### 前提条件
- Wandbはオンライン実行（事前に`wandb login`）
- Python 3.10系
- GPUがある場合は推論/学習の高速化に使用


## 3. 全体構成
### 3.1 パイプライン
1) 前処理
- base_stories.tsvから作品コーパスを作成
- タイトル + あらすじを統合したテキストを生成
 - e5系埋め込みのために`query:` / `passage:`プレフィックスを付与

2) 検索インデックス構築（Dense）
- SentenceTransformerで埋め込み
- FAISS IndexFlatIP（正規化ベクトル）

3) 再ランキング学習
- fiction_stories_practice.tsvから「query（架空あらすじ） × base作品」ペアを作成
- 正例: id_a / id_b
- 負例: その他からn件サンプリング
- Cross-Encoder（BERT日本語）で二値分類

4) 推論
- 架空あらすじの埋め込みでTop-K検索
- 再ランキングで上位2件を選択
- submission.csv生成


## 4. 実装タスク詳細
### 4.1 データ生成
- ファイル: signate_studentcup_2025/dataset.py
- 実装内容:
  - build_rerank_dataset
  - 乱数seed固定
  - Wandbログ: rows数 / n_negatives / seed
  - 乱数seedはPython/NumPy/Torchに同一値を設定

### 4.2 ベクトルインデックス構築
- ファイル: signate_studentcup_2025/features.py
- 実装内容:
  - build_faiss_index
  - Embedモデル: intfloat/multilingual-e5-base
  - Wandbログ: corpus_size / embedding_dim
  - Artifacts: base_corpus.csv と base_faiss.index を登録

### 4.3 再ランキング学習
- ファイル: signate_studentcup_2025/modeling/train.py
- 実装内容:
  - train_reranker
  - 学習ログ: loss / eval_loss / eval_accuracy
  - Wandb Trainer統合
  - Artifacts: 学習済みモデルディレクトリを登録

### 4.4 推論
- ファイル: signate_studentcup_2025/modeling/predict.py
- 実装内容:
  - predict
  - Weaveトレース対象: embed_query / retrieve / rerank
  - Wandbログ: pred_rows / top_k
  - 推論結果ファイルのArtifacts登録


## 5. Wandb / Weave 組み込み方針
### 5.1 共通初期化
- entity: argo11
- project: studentcup-2025
- job_typeで各処理の意味を区別

### 5.2 Weaveトレーシング
- 検索・再ランキングを明示的に@weave.opでラップ
- Trace Treeを確認し、Retrieval失敗とRerank失敗の切り分けを可能化

### 5.3 Artifacts
- Vector index
- Corpus
- Rerankerモデル
- 推論結果


## 6. 評価計画（Hit Rate / MRR / NDCG）
### 6.1 指標定義
- Hit Rate@K（Recall@K）
  - practiceの正解2作品が、検索上位K件に含まれている割合
- MRR
  - 正解のうち最上位の順位の逆数の平均
- NDCG
  - 上位に正解が集まるほど高いランキング指標

### 6.1.1 正解判定のルール
- 2作品の集合一致を正解とする（順序不問）
- practiceデータのid_a/id_bを集合化して評価

### 6.2 実装の位置
- 追加スクリプト: signate_studentcup_2025/modeling/evaluate.py
  - practiceデータで「検索のみ」と「検索+再ランキング」の両方を評価
  - wandb.logで各K（例: 5/10/20）の指標を記録
  - Weaveで検索→再ランキングの各ステップをトレース

### 6.3 ログ設計
- wandb.log
  - hit_rate@5, hit_rate@10, hit_rate@20
  - mrr
  - ndcg@5, ndcg@10, ndcg@20
  - retrieval_latency_ms, rerank_latency_ms（任意）


## 7. Artifacts登録
### 7.1 登録対象
- base_corpus.csv
- base_faiss.index
- train_rerank.csv
- rerankerモデルディレクトリ
- submission.csv

### 7.2 付与するメタデータ
- 埋め込みモデル名
- top_k / n_negatives / max_length
- git commit hash（任意）

### 7.2.1 命名規則
- base_corpus:{YYYYMMDD}-{short_hash}
- faiss_index:{YYYYMMDD}-{short_hash}
- reranker:{YYYYMMDD}-{short_hash}
- submission:{YYYYMMDD}-{short_hash}

### 7.3 連携方針
- build_faiss_indexの完了時にindex/corpusをArtifact登録
- train_rerankerの完了時にモデルArtifact登録
- predictの完了時にsubmissionをArtifact登録

## 8. ハイパーパラメータ探索（Sweeps）
### 8.1 対象パラメータ
- top_k: 5, 10, 20
- n_negatives: 4, 8, 16
- max_length: 128, 256, 384

### 8.2 Sweep設計
- 目的指標: hit_rate@10 または mrr
- Sweep方式: grid または bayes
- 実行単位: evaluate.pyを1試行としてwandb agentで実行

### 8.3 成果物
- sweeps.yaml
- best runのconfigを次の学習/推論に反映


## 9. 具体的な進行手順
1. build_rerank_dataset
2. build_faiss_index
3. train_reranker
4. evaluate（practice）
5. predict（test）


## 10. 追加改善案
- 検索の多様性確保: MMR
- ハイブリッド検索: BM25 + Dense
- Rerankモデルを日本語CrossEncoder強化
- 長文分割: chunk_size導入


## 11. リスクと対策
- 検索Recall不足
  - top_k増加
  - 埋め込みモデル変更
- 再ランキング過学習
  - n_negatives増加
  - early stopping
- 実験再現性不足
  - Artifacts化
  - seed固定


## 12. 次のステップ
- Hit Rate / MRR / NDCG 計測機能の追加
- Sweeps設定の作成
- Artifacts登録の実装
