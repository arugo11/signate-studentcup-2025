# **コンペティション環境におけるRAGシステム構築のためのWandbモニタリング・ベストプラクティスおよび評価戦略に関する包括的調査報告書**

## **1\. 序論：不確実性の高いRAGシステムにおける可視化の戦略的意義**

現代のデータサイエンスコンペティションにおいて、Retrieval-Augmented Generation（RAG）システムは、大規模言語モデル（LLM）の幻覚（Hallucination）を抑制し、ドメイン固有の知識を正確に提供するための標準的なアーキテクチャとしての地位を確立している。しかし、RAGシステムは「検索（Retrieval）」と「生成（Generation）」という二つの確率的なプロセスが連鎖する複合システムであり、その挙動は極めて複雑かつブラックボックス化しやすい特性を持つ。コンペティションという極限まで精度と効率を追求する環境下において、単にシステムを実装するだけでは勝利には不十分である。システム内部の挙動を詳細に監視（モニタリング）し、ボトルネックを特定し、実験サイクルを高速に回すための強固なMLOps基盤が不可欠となる 1。

Weights & Biases（Wandb）は、この課題に対して包括的なソリューションを提供するプラットフォームである。特に、近年のLLM開発に特化した**Wandb Weave**、ハイパーパラメータ最適化のための**Sweeps**、そしてデータとモデルのバージョン管理を行う**Artifacts**を有機的に結合させることで、RAGシステムの開発プロセスを科学的かつ再現可能なものへと昇華させることが可能となる。

本報告書では、コンペティションでの勝利を目的としたRAGシステムの構築において、Wandbのベストプラクティスに基づき、何を、どのように、なぜ監視すべきかを網羅的に論じる。検索精度の定量的評価から、LLM-as-a-Judgeを用いた生成品質の自動採点、さらには推論コストとレイテンシのトレードオフ分析に至るまで、実践的なモニタリング戦略を詳述する。

## ---

**2\. RAGパイプラインの包括的トレーシングとWeaveの統合**

RAGシステムのデバッグが困難である主たる要因は、最終的な回答の誤りが、検索の失敗に起因するのか、それとも生成の推論ミスに起因するのかの切り分けが直感的には不可能である点にある。この問題を解決するための第一歩は、システム全体の実行フローを微細に記録する「トレーシング（Tracing）」の実装である 3。

### **2.1 実行グラフの完全な可視化とWeave Opの適用**

Wandb Weaveは、関数デコレータ @weave.op() を用いることで、Pythonプログラム内の任意の関数の入力と出力を自動的にキャプチャし、クラウド上のダッシュボードで「トレースツリー（Trace Tree）」として可視化する機能を提供する。コンペティション用RAGにおいては、以下の主要コンポーネントを明示的にトレース対象とすべきである 3。

| コンポーネント | 監視対象データ（入力） | 監視対象データ（出力） | 監視の目的と分析視点 |
| :---- | :---- | :---- | :---- |
| **Query Transformation** | ユーザーの元のクエリ | 書き換え・拡張されたクエリ | クエリ拡張が本来の意図を歪めていないか、または検索に有効なキーワードが付加されているかの確認 1。 |
| **Retrieval (Retriever)** | クエリ（Embeddingベクトル含む） | 取得されたドキュメントチャンク群 | 正解ドキュメントが含まれているか（Hit Rate）、無関係なノイズが含まれていないかの確認。メタデータの検証。 |
| **Reranking (Cross-Encoder)** | 取得されたチャンク群とクエリ | 再ランク付けされたチャンクとスコア | リランカーが正解ドキュメントの順位を適切に押し上げているか（MRRの改善度）の定量的評価 1。 |
| **Prompt Construction** | システムプロンプト、取得コンテキスト | LLMへ入力される最終プロンプト | コンテキストウィンドウ制限による情報の切り捨て（Truncation）が発生していないかの確認。 |
| **Generation (LLM)** | 最終プロンプト | 生成された回答テキスト | 回答の正確性、フォーマット遵守（JSON等）、幻覚の有無の確認。 |

これらのコンポーネントを個別にトレースすることで、例えば「回答が誤っている」という事象が発生した際、Weaveのタイムライン上で即座に「検索段階では正解が含まれていたが、リランキングで順位が下がり、LLMへのコンテキストから漏れた」といった詳細な原因特定が可能となる。

### **2.2 LangChainおよびLlamaIndexとのシームレスな統合**

多くのコンペティション参加者は、RAG構築のフレームワークとしてLangChainやLlamaIndexを採用している。Wandb Weaveはこれらのフレームワークに対する「自動パッチング（Auto-patching）」機能を有しており、最小限のコード変更で高度なトレーシングを実現できる。

LangChainを使用する場合、WEAVE\_TRACE\_LANGCHAIN="true" 環境変数を設定し、weave.init() を呼び出すだけで、Chain内部の各ステップが自動的に記録される 5。しかし、ベストプラクティスとしては、自動トレースに依存しすぎず、コンペ特有の前処理やカスタムロジック部分には手動で @weave.op() を付与し、意図した粒度でのログ収集を行うことが推奨される。

LlamaIndexの場合、global\_handler にWeaveを設定することで、インデックス構築からクエリエンジン、さらにはエージェントの推論ステップまでを一貫して追跡可能である 6。特に、インデックス構築時のドキュメント解析（Parsing）結果や、チャンク化（Chunking）の様子をArtifactsとして保存することは、実験の再現性を担保する上で極めて重要である。

## ---

**3\. 検索フェーズ（Retrieval Phase）における定量的監視指標**

RAGシステムの性能の上限は、検索フェーズの質によって決定される。検索漏れ（Recall不足）は、どれほど高性能なLLMを用いても回復不可能な致命的エラーとなるため、このフェーズの監視は最優先事項である。Wandbのベストプラクティスでは、以下の指標を用いて検索性能を多角的に評価する 7。

### **3.1 適合率と再現率に基づく指標**

コンペティションにおいて、検索システムの性能を測定するために必須となる指標は以下の通りである。これらはWandbのカスタムScorerとして実装し、実験ごとに自動計算されるようパイプラインに組み込むべきである 8。

| 指標 (Metric) | 定義と計算ロジック | コンペティションにおける解釈と監視のポイント |
| :---- | :---- | :---- |
| **Hit Rate (Recall@K)** | 上位K個の取得チャンク内に、正解（Ground Truth）となる情報が含まれている割合。 | K=5, 10, 20などの複数の閾値で監視する。コンペではLLMに入力可能なトークン数に限りがあるため、現実的なK（例: K=5）でのHit Rateが実質的な性能キャップとなる 11。 |
| **MRR (Mean Reciprocal Rank)** | 正解ドキュメントが検索結果の何番目に位置しているか（順位の逆数）の平均値。 | システムがどれだけ「自信を持って」正解を上位に提示できているかを示す。Hit Rateが高くてもMRRが低い場合、LLMが「Lost in the Middle」現象を起こすリスクが高まる 8。 |
| **NDCG (Normalized Discounted Cumulative Gain)** | 順位を考慮した利得の累積値。 | 複数の関連ドキュメントが必要なクエリにおいて、それら全てを上位に集める能力を評価する。単一の正解だけでなく、網羅的な情報収集が求められるタスクで重要となる。 |

### **3.2 埋め込み空間（Embedding Space）の可視化と分析**

数値指標だけでは見えない検索の質的特性を理解するために、Embeddingの分布を可視化することは極めて有効である。Wandbでは、取得されたドキュメントとクエリのベクトルを低次元（2Dまたは3D）に射影し、散布図としてプロットすることが推奨される 13。

この可視化により、以下の洞察が得られる：

* **クラスタリングの妥当性**: 意味的に近いドキュメントが正しく近くに配置されているか。
* **クエリとドキュメントの乖離**: クエリの分布とドキュメントの分布が離れている場合、ドメイン適応（Domain Adaptation）やファインチューニングが必要であることを示唆する。
* **外れ値の特定**: 検索スコアは高いが、意味的には無関係な「幻覚の元」となるドキュメントが存在しないかを視覚的に確認できる。

### **3.3 ハイブリッド検索とリランキングの寄与度監視**

近年、ベクトル検索（Dense Retrieval）とキーワード検索（Sparse Retrieval/BM25）を組み合わせるハイブリッド検索が標準的になりつつある 1。Wandbのダッシュボードでは、それぞれの検索手法がどの程度正解ドキュメントの取得に寄与しているかを個別にトラッキングすべきである。

また、リランカー（Reranker）を導入する場合、リランク前後でのMRRの変化（Lift）を監視することが重要である。リランカーは計算コストが高いため、MRRの改善幅が推論時間の増加に見合うものであるかを常に評価し、コスト対効果のバランスを見極める必要がある 1。

## ---

**4\. 生成フェーズ（Generation Phase）における品質監視とLLM-as-a-Judge**

検索されたコンテキストが適切であっても、LLMがそれを正しく解釈し、論理的に正しい回答を生成できるとは限らない。生成フェーズの監視には、従来のNLP指標（BLEUやROUGE）だけでは不十分であり、より意味的な評価が求められる。ここでは、「LLM-as-a-Judge（審査員としてのLLM）」を用いた自動評価パイプラインの構築がベストプラクティスとなる 15。

### **4.1 Ragasメトリクスによる多次元評価**

RAGシステムの評価フレームワークとして広く採用されているRagas（Retrieval Augmented Generation Assessment）の指標をWandb Weaveに統合し、以下の観点から生成品質をスコアリングする 18。

| 評価指標 | 評価内容 | 監視の目的 | Wandbでの実装アプローチ |
| :---- | :---- | :---- | :---- |
| **Faithfulness (忠実性)** | 回答が取得されたコンテキストのみに基づいているか。 | 外部知識による幻覚（Hallucination）の検知。コンペでは「与えられた資料のみから回答せよ」という制約が多いため重要。 | ContextEntityRecallScorer等をWeave Scorerとしてラップし、回答とコンテキストの矛盾を検知 20。 |
| **Answer Relevancy (回答関連性)** | 回答がクエリの意図に対して直接的かつ適切か。 | 冗長な回答や、質問をはぐらかすような回答の排除。 | LLMジャッジにクエリと回答を入力し、関連度を0-1でスコアリングさせる。 |
| **Context Precision** | 取得されたコンテキストのうち、実際に回答生成に寄与した割合。 | ノイズ情報の混入度合いの測定。これが低いとLLMの推論能力が低下する。 | 正解回答の生成に必要な文が、コンテキストの上位に含まれているかを評価。 |

### **4.2 LLM-as-a-Judgeの実装と運用**

コンペティションではテストデータの正解（Ground Truth）が公開されていないことが一般的であるため、参加者は自身で検証用データセット（Validation Set）を作成し、それに対する評価を行う必要がある。Wandb Weaveを用いることで、GPT-4などの高性能モデルを「審査員」として定義し、開発中のモデル（Student Model）の出力を自動採点させるパイプラインを構築できる 17。

具体的な実装としては、weave.Scorer クラスを継承し、審査用プロンプトを内包したカスタム評価ロジックを作成する。

Python

import weave
from weave import Scorer

class CompetitionAccuracyScorer(Scorer):
    @weave.op()
    async def score(self, target: str, prediction: str) \-\> dict:
        \# コンペ固有の評価基準をプロンプトに埋め込む
        prompt \= f"""
        あなたはコンペティションの厳格な審査員です。
        以下の正解と予測を比較し、意味的に一致しているかを判定してください。
        数値の誤差やフォーマット違反には厳しく採点してください。

        正解: {target}
        予測: {prediction}

        判定 (Yes/No):
        理由:
        """
        \# LLMジャッジを呼び出し（コードは抽象化）
        result \= await call\_llm\_judge(prompt)
        return {"is\_correct": result.verdict \== "Yes", "reasoning": result.reason}

このように、単なるスコアだけでなく、\*\*判定の理由（Reasoning）\*\*もWandbにログとして残すことで、「なぜスコアが低かったのか」を定性的に分析することが可能になる。これはモデルの弱点を発見し、プロンプトエンジニアリングや検索ロジックの改善に直結する貴重なインサイトとなる 17。

## ---

**5\. ハイパーパラメータ最適化とWandb Sweepsの活用**

RAGシステムは、チャンクサイズ、オーバーラップ量、検索数（Top-K）、生成時のTemperatureなど、相互に依存する多数のハイパーパラメータによって構成されている。これらを手動で調整することは非効率であり、局所最適解に陥るリスクが高い。Wandb Sweepsを活用し、これらのパラメータを体系的に探索・最適化することが、コンペティションで上位に入賞するための重要な戦略となる 1。

### **5.1 RAG特有の探索空間とパラメータ設定**

RAGシステムの性能を最大化するために探索すべき主要なハイパーパラメータと、その設定指針は以下の通りである。

| パラメータ領域 | パラメータ名 | 探索範囲の例 | 影響とトレードオフの考察 |
| :---- | :---- | :---- | :---- |
| **Chunking** | chunk\_size | 128, 256, 512, 1024 | 小さいと文脈が分断され、大きいとノイズが増える。Embeddingモデルの入力制限との兼ね合いも重要 24。 |
| **Chunking** | chunk\_overlap | 0, 50, 100, 200 | チャンク境界での情報損失を防ぐために必要だが、重複が多すぎると冗長になる。 |
| **Retrieval** | top\_k | 3, 5, 10, 20, 50 | Kを増やせばRecall（Hit Rate）は向上するが、PrecisionとLLMの処理速度・コストが悪化する。Sweepで最適なバランス点を見つける必要がある 8。 |
| **Retrieval** | search\_type | Similarity, MMR, Hybrid | 単純な類似度検索か、多様性を重視するMMR（Maximal Marginal Relevance）か。クエリの性質に依存する。 |
| **Generation** | temperature | 0.0 \- 0.7 | RAGでは事実性を重視するため低めが推奨されるが、表現力が求められるタスクでは調整が必要。 |

### **5.2 Parallel Coordinates Plotによる相関分析**

Sweepsの結果分析において、最も強力なツールとなるのが\*\*Parallel Coordinates Plot（並行座標プロット）\*\*である。このチャートを使用することで、複数のハイパーパラメータと最終的な評価メトリクス（例：Hit RateやOverall Accuracy）との間の複雑な相関関係を視覚的に把握できる 23。

例えば、「Chunk Sizeが512、かつTop-Kが10前後の時に、精度が最も高くなる傾向がある」といった多変量間のパターンを直感的に発見できる。また、**Parameter Importance**機能を用いることで、どのパラメータが結果に最も大きな影響を与えているか（感度分析）を定量的に知り、調整の優先順位を決定することができる 23。

## ---

**6\. システム効率とコストの監視：トレードオフの最適化**

コンペティションによっては、精度だけでなく、推論速度（Latency）や計算リソースの使用効率が評価対象となる場合がある。また、APIコスト（特にOpenAI等の商用モデルを使用する場合）は、実験回数を制約する現実的な要因となる。Wandbを用いてこれらの「非機能要件」を監視し、精度とのトレードオフを最適化する戦略が求められる 7。

### **6.1 レイテンシの分解とボトルネック特定**

Weaveのトレース機能は、各処理ステップの実行時間をタイムラインとして記録する。これにより、システム全体のレイテンシ（Total Latency）を以下のコンポーネントに分解し、どこがボトルネックになっているかを特定できる。

* **Retrieval Latency**: ベクトルデータベースの検索にかかる時間。インデックスのサイズや検索アルゴリズム（HNSW vs Flat）に依存する。
* **Reranking Latency**: Cross-Encoderによる再ランク付けの時間。精度向上には寄与するが、計算コストが高いため、ここの遅延が許容範囲内かを常に監視する必要がある。
* **Generation Latency (TTFT & Total)**: LLMが最初のトークンを出力するまでの時間（Time To First Token）と、生成完了までの総時間。プロンプト長（入力トークン数）と相関するため、不要なコンテキスト削減の指針となる。

### **6.2 トークン使用量とコストの可視化**

RAGシステムは、検索された大量のドキュメントをプロンプトに挿入するため、トークン消費量が肥大化しやすい。Wandbでは、llm\_usage や cost を自動的に追跡し、実験ごとのコストを可視化できる 29。

* **Prompt Tokens vs Completion Tokens**: 入力トークン（検索結果）と出力トークン（回答）の比率を監視する。入力トークンが過大であるにもかかわらず精度が向上していない場合、情報の密度（Information Density）が低いことを示唆しており、チャンク戦略の見直しが必要となる。
* **モデル別のROI分析**: GPT-4、GPT-3.5-Turbo、Claude 3 Haikuなど、異なるモデルを使用した場合の「精度向上率」と「コスト増加率」をWandbの散布図（Scatter Plot）で比較し、コンペの制約内で最も投資対効果（ROI）が高いモデル選定を行う。

## ---

**7\. アーティファクトによるデータとモデルの完全なバージョン管理**

RAGシステムの実験において、コードの変更だけでなく、データの変更（ドキュメントの追加、前処理の変更、Embeddingモデルの変更）が結果に大きな影響を与える。再現性を担保するためには、Wandb Artifactsを用いてこれらの中間生成物を厳密にバージョン管理する必要がある 30。

### **7.1 Vector Indexのアーティファクト化**

Embeddingモデルを変更したり、チャンクサイズを変更したりするたびに、ベクトルインデックス（Vector Store）の中身は完全に変化する。このインデックス自体をWandb Artifactsとして保存し、実験（Run）と紐付けることで、以下のメリットが得られる。

* **完全な再現性**: 過去の実験結果（良いスコアが出た時）と同じ状態の検索インデックスを即座に復元し、検証できる。
* **リネージ（Lineage）の追跡**: どの生データ（Raw Data）から、どのスクリプト（Preprocessing Script）を経て、現在のインデックスが生成されたかという依存関係をグラフとして可視化できる。

### **7.2 評価データセット（Golden Set）のバージョン管理**

評価に使用するQAペア（Golden Dataset）も、コンペの進行とともに修正・拡張されていくものである。Wandb Artifactsでデータセットをバージョン管理し、run.use\_artifact() を用いて評価スクリプト内で動的にロードすることで、「スコアの向上がモデルの改善によるものか、評価データの変更によるものか」を明確に区別することができる 8。

## ---

**8\. 結論**

コンペティションにおけるRAGシステムの構築は、探索空間が広く、かつ構成要素間の相互作用が複雑な多次元の最適化問題である。この問題に対して、「勘や経験」に頼るのではなく、Wandbのエコシステム（Weave, Sweeps, Artifacts）を駆使して「データとメトリクス」に基づいた意思決定を行うことが、勝利への最短経路となる。

本報告書で詳述した通り、検索精度の徹底的な監視（Hit Rate/MRR）、生成品質の自動評価（LLM-as-a-Judge/Ragas）、ハイパーパラメータの科学的探索、そしてリソース効率の最適化を統合的に実施することで、参加者はRAGシステムのブラックボックスを解明し、競争力のある堅牢なモデルを構築することが可能となる。モニタリングは単なる「確認作業」ではなく、モデルの性能限界を突破するための能動的な「エンジニアリングプロセス」として位置づけられるべきである。

#### **引用文献**

1. RAG techniques: From naive to advanced \- Weights & Biases \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/site/articles/rag-techniques/](https://wandb.ai/site/articles/rag-techniques/)
2. What is retrieval augmented generation? \- Weights & Biases \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/site/articles/what-is-retrieval-augmented-generation/](https://wandb.ai/site/articles/what-is-retrieval-augmented-generation/)
3. Evaluate RAG applications \- Weights & Biases Documentation, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/weave/tutorial-rag](https://docs.wandb.ai/weave/tutorial-rag)
4. Scoring Overview \- Weights & Biases Documentation, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/weave/guides/evaluation/scorers](https://docs.wandb.ai/weave/guides/evaluation/scorers)
5. LangChain \- Weights & Biases Documentation, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/weave/guides/integrations/langchain](https://docs.wandb.ai/weave/guides/integrations/langchain)
6. LlamaIndex \- Weights & Biases Documentation, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/weave/guides/integrations/llamaindex](https://docs.wandb.ai/weave/guides/integrations/llamaindex)
7. LLM evaluation: Metrics, frameworks, and best practices | genai-research \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA)
8. OCR-powered document summarization in banking with W\&B Weave and LlamaIndex | solution-accelerator-mrm-eval – Weights & Biases \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/wandb-smle/solution-accelerator-mrm-eval/reports/OCR-powered-document-summarization-in-banking-with-W-B-Weave-and-LlamaIndex--VmlldzoxNDg4MjAzNQ](https://wandb.ai/wandb-smle/solution-accelerator-mrm-eval/reports/OCR-powered-document-summarization-in-banking-with-W-B-Weave-and-LlamaIndex--VmlldzoxNDg4MjAzNQ)
9. Precision and recall at K in ranking and recommendations \- Evidently AI, 2月 1, 2026にアクセス、 [https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)
10. Retrieval Metrics Tutorial: Recall@k and MRR Explained | by rajnish khatri | Medium, 2月 1, 2026にアクセス、 [https://medium.com/@rajnish\_khatri/retrieval-metrics-tutorial-recall-k-and-mrr-explained-d2f12afb9c89](https://medium.com/@rajnish_khatri/retrieval-metrics-tutorial-recall-k-and-mrr-explained-d2f12afb9c89)
11. 10 metrics to evaluate recommender and ranking systems \- Evidently AI, 2月 1, 2026にアクセス、 [https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)
12. How to evaluate an LLM Part 3: LLMs evaluating LLMs | wandbot-eval – Weights & Biases, 2月 1, 2026にアクセス、 [https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-3-LLMs-evaluating-LLMs--Vmlldzo1NzEzMDcz](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-3-LLMs-evaluating-LLMs--Vmlldzo1NzEzMDcz)
13. Step for developing and evaluating RAG application with W\&B | rag-hands-on – Weights & Biases \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/wandb-japan/rag-hands-on/reports/Step-for-developing-and-evaluating-RAG-application-with-W-B--Vmlldzo1NzU4OTAx](https://wandb.ai/wandb-japan/rag-hands-on/reports/Step-for-developing-and-evaluating-RAG-application-with-W-B--Vmlldzo1NzU4OTAx)
14. Evaluate using local scorers \- Weights & Biases Documentation, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/weave/guides/evaluation/weave\_local\_scorers](https://docs.wandb.ai/weave/guides/evaluation/weave_local_scorers)
15. Streamline GenAI workflows with W\&B Weave \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/site/weave/](https://wandb.ai/site/weave/)
16. Evaluate your RAG pipeline using LLM as a Judge with custom dataset creation (Part 2), 2月 1, 2026にアクセス、 [https://wandb.ai/ai-team-articles/evals/reports/Evaluate-your-RAG-pipeline-using-LLM-as-a-Judge-with-custom-dataset-creation-Part-2---VmlldzoxNTIwNjI2MQ](https://wandb.ai/ai-team-articles/evals/reports/Evaluate-your-RAG-pipeline-using-LLM-as-a-Judge-with-custom-dataset-creation-Part-2---VmlldzoxNTIwNjI2MQ)
17. Tutorial: Implementing LLM as a Judge for evaluation | judgebench \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/byyoung3/judgebench/reports/Tutorial-Implementing-LLM-as-a-Judge-for-evaluation--VmlldzoxNTQ5OTk1OA](https://wandb.ai/byyoung3/judgebench/reports/Tutorial-Implementing-LLM-as-a-Judge-for-evaluation--VmlldzoxNTQ5OTk1OA)
18. How to evaluate a Langchain RAG system with RAGAs | ML\_NEWS3 – Weights & Biases, 2月 1, 2026にアクセス、 [https://wandb.ai/byyoung3/ML\_NEWS3/reports/How-to-evaluate-a-Langchain-RAG-system-with-RAGAs--Vmlldzo5NzU1NDYx](https://wandb.ai/byyoung3/ML_NEWS3/reports/How-to-evaluate-a-Langchain-RAG-system-with-RAGAs--Vmlldzo5NzU1NDYx)
19. Building and evaluating a RAG system with DSPy and W\&B Weave | ML\_NEWS3 \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/byyoung3/ML\_NEWS3/reports/Building-and-evaluating-a-RAG-system-with-DSPy-and-W-B-Weave---Vmlldzo5OTE0MzM4](https://wandb.ai/byyoung3/ML_NEWS3/reports/Building-and-evaluating-a-RAG-system-with-DSPy-and-W-B-Weave---Vmlldzo5OTE0MzM4)
20. Use builtin scorers \- Weights & Biases Documentation, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/weave/guides/evaluation/builtin\_scorers](https://docs.wandb.ai/weave/guides/evaluation/builtin_scorers)
21. Run LLM evaluations right in the W\&B Weave UI | product-announcements-fc \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/wandb\_fc/product-announcements-fc/reports/Run-LLM-evaluations-right-in-the-W-B-Weave-UI--VmlldzoxNDM1MzUzNw?galleryTag=generative-modeling](https://wandb.ai/wandb_fc/product-announcements-fc/reports/Run-LLM-evaluations-right-in-the-W-B-Weave-UI--VmlldzoxNDM1MzUzNw?galleryTag=generative-modeling)
22. Tune hyperparameters with sweeps \- Weights & Biases Documentation \- Wandb, 2月 1, 2026にアクセス、 [https://docs.wandb.ai/models/tutorials/sweeps](https://docs.wandb.ai/models/tutorials/sweeps)
23. Hyperparameter optimization with W\&B | sweep-demo – Weights & Biases \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/example-team/sweep-demo/reports/Hyperparameter-optimization-with-W-B--Vmlldzo1NTYwOQ](https://wandb.ai/example-team/sweep-demo/reports/Hyperparameter-optimization-with-W-B--Vmlldzo1NTYwOQ)
24. Optimizing RAG Chunk Size: Your Definitive Guide to Better Retrieval Accuracy, 2月 1, 2026にアクセス、 [https://machinelearningplus.com/gen-ai/optimizing-rag-chunk-size-your-definitive-guide-to-better-retrieval-accuracy/](https://machinelearningplus.com/gen-ai/optimizing-rag-chunk-size-your-definitive-guide-to-better-retrieval-accuracy/)
25. Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex, 2月 1, 2026にアクセス、 [https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)
26. What are your RAG parameters e.g. top k, chunk size, chunk overlap? : r/LangChain \- Reddit, 2月 1, 2026にアクセス、 [https://www.reddit.com/r/LangChain/comments/1evtu7d/what\_are\_your\_rag\_parameters\_eg\_top\_k\_chunk\_size/](https://www.reddit.com/r/LangChain/comments/1evtu7d/what_are_your_rag_parameters_eg_top_k_chunk_size/)
27. Enhanced hyperparameter optimization with W\&B Sweeps \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/site/sweeps/](https://wandb.ai/site/sweeps/)
28. Production-ready LLM evaluation guide \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/ai-team-articles/llm-evaluation/reports/Production-ready-LLM-evaluation-guide--VmlldzoxNTI5MjA2NA](https://wandb.ai/ai-team-articles/llm-evaluation/reports/Production-ready-LLM-evaluation-guide--VmlldzoxNTI5MjA2NA)
29. A guide to LLM debugging, tracing, and monitoring | genai-research – Weights & Biases, 2月 1, 2026にアクセス、 [https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ)
30. Creating a customer support chatbot using Claude 3, Llamaindex and W\&B Weave \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/sauravmaheshkar/chatbot-claude3-llamaindex-weave/reports/Creating-a-customer-support-chatbot-using-Claude-3-Llamaindex-and-W-B-Weave--Vmlldzo4NDY3NDAx](https://wandb.ai/sauravmaheshkar/chatbot-claude3-llamaindex-weave/reports/Creating-a-customer-support-chatbot-using-Claude-3-Llamaindex-and-W-B-Weave--Vmlldzo4NDY3NDAx)
31. Building Advanced Query Engine and Evaluation with LlamaIndex and W\&B | llama-index-report – Weights & Biases \- Wandb, 2月 1, 2026にアクセス、 [https://wandb.ai/ayush-thakur/llama-index-report/reports/Building-Advanced-Query-Engine-and-Evaluation-with-LlamaIndex-and-W-B--Vmlldzo0OTIzMjMy](https://wandb.ai/ayush-thakur/llama-index-report/reports/Building-Advanced-Query-Engine-and-Evaluation-with-LlamaIndex-and-W-B--Vmlldzo0OTIzMjMy)