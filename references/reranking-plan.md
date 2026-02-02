# RAG再ランク 実装計画

## 目的
- 検索で候補を絞り、Cross-encoderで再ランクして最終のペア一致精度を最大化する。
- 合成データは補助として利用（詳細は [references/expand-dataset-plan.md](references/expand-dataset-plan.md)）。

## 前提
- 既存の検索ベース推論と評価実装が存在
  - 予測: [signate_studentcup_2025/modeling/predict.py](signate_studentcup_2025/modeling/predict.py)
  - 学習: [signate_studentcup_2025/modeling/train.py](signate_studentcup_2025/modeling/train.py)

## 1. モデル設計
### 1.1 Bi-encoder（検索）
- 目的: Recall@k最大化
- 正例: (story, base A) & (story, base B)
- 負例: 他48作品

### 1.2 Cross-encoder（再ランク）
- 入力: [query story] + [candidate base story]
- 出力: 関連スコア
- 50件全件再ランク可能（A100）

### 1.3 直接分類（補助）
- 50クラスmulti-label BCE
- 合成データ比率は10〜30%に制限

## 2. 評価指標
- Pair Recall@k
- Label Recall@k
- MRR
- NDCG@k
- Set-F1

## 3. 実験スケジュール（48h）
### Day 1
1. 合成データ生成（OpenRouter）
2. 自動フィルタ＋再生成
3. 分布・重複・長さチェック

### Day 2
4. Bi-encoder学習
5. Cross-encoder学習
6. practiceで評価（学習には使用しない）
7. テスト推論

## 4. 実装タスク一覧（再ランク側）
- Bi-encoder学習パイプライン
- Cross-encoder学習パイプライン
- 評価指標の追加（Recall@k, MRR, NDCG@k, Set-F1）
- 予測パイプラインへの統合

## 5. リスクと対策
- **過適応**: 生成テンプレの多様化と再ランク重視で緩和
