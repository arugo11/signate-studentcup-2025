# 精度向上のための戦略（論文・過去コンペ根拠付き）

本ファイルは、ベースラインから精度を上げるための代表的戦略を、論文・過去コンペ（TREC/MS MARCO など）の知見に基づいて整理したものです。各戦略は「狙い」「実装イメージ」「注意点」を中心に、適用の具体性が伝わるように記述しています。

---

## 1. 文埋め込み（bi-encoder）による候補検索の強化
**狙い**: 架空あらすじと元作品の意味的近さを、文埋め込みの距離で直接測る。

**根拠**: SBERT は文埋め込みを Siamese/Triplet で学習し高速な類似検索を可能にした手法で、文レベルの意味類似に強いと報告されています。SimCSE はドロップアウトを使った単純な対照学習で高精度な文埋め込みを実現。E5 は弱教師あり対照学習で広範な検索タスクに転移し、BEIR で BM25 を上回る例を示しています。 [R1][R2][R3]

**実装イメージ**:
- 架空あらすじをクエリ、50本の元作品あらすじをドキュメントとして埋め込み。
- コサイン類似度で上位k件を候補化。
- 後段の再ランキングやペア推定の候補として使う。

**注意点**:
- 日本語向け埋め込みモデル（多言語モデル含む）を選ぶ。
- 文章長が長い場合は段落分割や要約でノイズを抑える。

---

## 2. 対照学習（dual-encoder）で検索器をタスク特化
**狙い**: 「架空あらすじ↔元作品あらすじ」の対応関係に合わせて埋め込みを微調整し、近傍検索の精度を上げる。

**根拠**: DPR は dual-encoder を対照学習で訓練し、BM25 を上回る検索精度を示しています。 [R4]

**実装イメージ**:
- 正例: (架空あらすじ, 正解の元作品)
- 負例: 同バッチ内の他作品（in-batch negatives）
- 損失: InfoNCE / 多クラス分類

**注意点**:
- 正解が2作品なので、正例を「2本とも正解」として扱うか、2本を個別に正例にする設計が必要。

---

## 3. 二段階検索（候補生成→BERT再ランキング）
**狙い**: まず広く候補を集め、その後にクロスエンコーダで精密に順位付けする。

**根拠**: BERT を用いた再ランキングは MS MARCO/TREC の文脈で強力な性能向上が報告されています。TREC Deep Learning Track でも深層学習によるランク付けが従来法を上回ったとされています。 [R5][R12][R13]

**実装イメージ**:
- 第1段: BM25 or bi-encoder で上位k件
- 第2段: クロスエンコーダ（BERT系）で(架空あらすじ, 元作品あらすじ)の関連度を再スコア

**注意点**:
- 全組み合わせは 340×50=17,000 程度なので、計算資源があれば全件再ランキングも可能。

---

## 4. Late Interaction（ColBERT）で高精度かつ効率的な照合
**狙い**: 単一ベクトルの粗い一致ではなく、トークン単位の細粒度一致を保ちつつ高速化する。

**根拠**: ColBERT は BERT の表現をクエリ/文書で独立に計算し、late interaction により細粒度一致を保ったまま高速化できることを示しています。 [R6]

**実装イメージ**:
- 元作品側のトークン埋め込みを事前計算しインデックス化。
- 架空あらすじのトークン埋め込みとの MaxSim でスコア化。

**注意点**:
- 学習・推論コストが増えるので、候補生成や再ランキングのどちらに使うか設計が必要。

---

## 5. 疎ベクトル検索（BM25/SPLADE）で語彙一致を強化
**狙い**: 固有表現や特徴的フレーズの一致を確実に拾う。

**根拠**: BM25 は古典的だが依然強いベースラインとして位置づけられており、BEIR でも頑健なベースラインと報告されています。SPLADE は疎表現を学習し、語彙一致と意味拡張を両立する神経疎モデルとして提案されています。 [R7][R8][R9]

**実装イメージ**:
- BM25で上位候補を収集し、再ランキングに渡す。
- 余力があれば SPLADE 系の疎表現モデルを導入し、語彙一致を強化。

**注意点**:
- 架空あらすじに含まれる「世界観・テーマ語」を拾えるため、密ベクトルだけより補完的。

---

## 6. クエリ/文書拡張（doc2query/GAR）で語彙ギャップを埋める
**狙い**: LLMや生成モデルで「含まれていないが関連する語」を補う。

**根拠**: doc2query は文書から想定クエリを生成して文書拡張し、検索性能を向上させる方法として提案。GAR はクエリ拡張を生成で行い、BM25やDPRとの併用効果を示しています。 [R10][R11]

**実装イメージ**:
- 元作品あらすじに対して「連想される問い」や「代表的なキーワード」を生成し拡張。
- 架空あらすじ側を要約・書き換えして拡張し、スコア融合。

**注意点**:
- 生成文がノイズになることがあるため、拡張文数やフィルタリング設計が重要。

---

## 7. 疎+密のハイブリッド融合（スコア融合/ランク融合）
**狙い**: 語彙一致（疎）と意味一致（密）の長所を同時に活かす。

**根拠**: BEIR では re-ranking や late-interaction が平均的に強いが計算コストが高い一方、疎・密は補完関係にあると示唆されています。GAR は BM25 と DPR の併用による改善を示しています。 [R8][R11]

**実装イメージ**:
- BM25 スコア + 埋め込み類似度の線形融合。
- あるいは BM25候補 + dense再ランキング + 再度BM25でtie-break。

**注意点**:
- 融合係数はCVで最適化（小規模でも有効）。

---

## 8. 「2作品=集合」出力を意識した順序不変モデリング
**狙い**: 正解が「順不同の2作品」である点を、モデル設計に反映する。

**根拠**: Deep Sets は集合入力/出力に対する順序不変性を理論的に定式化し、Set Transformer は注意機構で集合内相互作用を表現できます。 [R14][R15]

**実装イメージ**:
- 上位候補群の埋め込みを集合として入力し、2要素集合を出力する。
- あるいは「ペア候補」を列挙し、順序不変のスコアで順位付け。

**注意点**:
- 50作品と小規模なので、ペア候補（50C2=1225）を全列挙し再ランキングする設計も現実的。

---

# 参考文献
- [R1] Reimers, N. & Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* arXiv:1908.10084. https://arxiv.org/abs/1908.10084
- [R2] Gao, T. et al. *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* arXiv:2104.08821. https://arxiv.org/abs/2104.08821
- [R3] Wang, L. et al. *Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5).* arXiv:2212.03533. https://arxiv.org/abs/2212.03533
- [R4] Karpukhin, V. et al. *Dense Passage Retrieval for Open-Domain Question Answering.* arXiv:2004.04906. https://arxiv.org/abs/2004.04906
- [R5] Nogueira, R. & Cho, K. *Passage Re-ranking with BERT.* arXiv:1901.04085. https://arxiv.org/abs/1901.04085
- [R6] Khattab, O. & Zaharia, M. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* arXiv:2004.12832. https://arxiv.org/abs/2004.12832
- [R7] Robertson, S. & Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR (2009). https://doi.org/10.1561/1500000019
- [R8] Thakur, N. et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models.* arXiv:2104.08663. https://arxiv.org/abs/2104.08663
- [R9] Formal, T. et al. *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.* arXiv:2107.05720. https://arxiv.org/abs/2107.05720
- [R10] Nogueira, R. et al. *Document Expansion by Query Prediction (doc2query).* arXiv:1904.08375. https://arxiv.org/abs/1904.08375
- [R11] Mao, Y. et al. *Generation-Augmented Retrieval for Open-domain Question Answering.* arXiv:2009.08553. https://arxiv.org/abs/2009.08553
- [R12] Voorhees, E. et al. *Overview of the TREC 2019 Deep Learning Track.* NIST (2020). https://www.nist.gov/publications/overview-trec-2019-deep-learning-track
- [R13] Bajaj, P. et al. *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.* arXiv:1611.09268. https://arxiv.org/abs/1611.09268
- [R14] Zaheer, M. et al. *Deep Sets.* arXiv:1703.06114. https://arxiv.org/abs/1703.06114
- [R15] Lee, J. et al. *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* arXiv:1810.00825. https://arxiv.org/abs/1810.00825
