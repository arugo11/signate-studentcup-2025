# データ拡張＋RAG再ランク実装計画

## 目的
- 架空あらすじ生成（合成データ）とRAG再ランクを統合し、最終のペア一致精度を最大化する。
- 合成データは再ランク学習の補助に限定し、分布ずれのリスクを管理する。

## 前提
- 既存ベース作品: 50件
- 架空作品推定タスク（2作品の順不同）
- 既存の検索ベース推論と評価実装が存在
  - 予測: [signate_studentcup_2025/modeling/predict.py](signate_studentcup_2025/modeling/predict.py)
  - 学習: [signate_studentcup_2025/modeling/train.py](signate_studentcup_2025/modeling/train.py)

## 1. 合成データ生成方針
### 1.1 テンプレート
- 6テンプレート（T0, T1, T2, T3, T5, T6）を使用
- 詳細は [references/expand-dataset.md](references/expand-dataset.md) を採用

### 1.2 生成規模
- 50C2=1,225 × 6テンプレ = 7,350件
- トーン/視点追加（30%サンプル）= 約+2,200件
- 合計 9,500〜10,000件

### 1.3 出力制約
- 320〜420字、1段落、改行禁止
- 固有名詞禁止（人名・地名・作品名・組織名など）
- A/B双方から最低2要素を融合

## 2. 品質担保
### 2.1 自動フィルタ
- 文字数・段落数チェック
- 固有名詞らしき語（中黒/長カタカナ/英字列）検出で再生成
- 入力文との20文字以上の連続一致で再生成
- 埋め込み類似度: `min(simA, simB)` が閾値未満なら再生成

### 2.2 重複排除
- MinHash or Embedding Clusteringで同一文を排除

## 3. モデル設計
### 3.1 Bi-encoder（検索）
- 目的: Recall@k最大化
- 正例: (story, base A) & (story, base B)
- 負例: 他48作品

### 3.2 Cross-encoder（再ランク）
- 入力: [query story] + [candidate base story]
- 出力: 関連スコア
- 50件全件再ランク可能（A100）

### 3.3 直接分類（補助）
- 50クラスmulti-label BCE
- 合成データ比率は10〜30%に制限

## 4. 評価指標
- Pair Recall@k
- Label Recall@k
- MRR
- NDCG@k
- Set-F1

## 5. 実験スケジュール（48h）
### Day 1
1. 合成データ生成（OpenRouter）
2. 自動フィルタ＋再生成
3. 分布・重複・長さチェック

### Day 2
4. Bi-encoder学習
5. Cross-encoder学習
6. practiceで評価（学習には使用しない）
7. テスト推論

## 6. 実装タスク一覧
- 合成生成スクリプト作成
- 品質フィルタ実装
- Bi-encoder学習パイプライン
- Cross-encoder学習パイプライン
- 評価指標の追加（Recall@k, MRR, NDCG@k, Set-F1）
- 予測パイプラインへの統合

## 7. リスクと対策
- **分布ずれ**: 合成データの比率を抑制し、実データに近い制約を厳守
- **過適応**: 生成テンプレの多様化と再ランク重視で緩和
- **固有名詞混入**: 自動検出＋再生成で対応

---