# signate-studentcup-2025 documentation!

## Description

https://user.competition.signate.jp/ja/competition/detail/?competition=385dcbba17b645f3ac10f827dfba03f6

## Commands

The Makefile contains the central entry points for common tasks related to this project.

## RAG System Architecture

```mermaid
flowchart TB
    subgraph Data["Data Pipeline"]
        BS["base_stories.tsv<br/>(50 base works)"]
        FS["fiction_stories_test.tsv<br/>(340 test samples)"]
        CORPUS["prepare_corpus()<br/>Corpus preparation"]
    end

    subgraph Embedding["Embedding Generation"]
        EMB["EmbeddingModel<br/>(OpenRouter API)"]
        VEC["Embedding vectors<br/>(dim=3072)"]
    end

    subgraph Index["FAISS Index"]
        NORM["L2 Normalization<br/>(for cosine similarity)"]
        IDX["IndexFlatIP<br/>(Inner product index)"]
        SAVE["Save index<br/>(faiss_index.pkl)"]
    end

    subgraph Retrieval["Retrieval Prediction"]
        QUERY["Query synopsis"]
        QEMB["Query embedding"]
        SEARCH["FAISS.search()<br/>Top-K retrieval"]
        PRED["Predictor.predict()<br/>Return top-2"]
    end

    subgraph Direct["Direct Prediction (Alternative)"]
        LLM["OpenRouter LLM<br/>(gpt-4o-mini, etc.)"]
        LPROMPT["Prompt engineering"]
        LOUT["JSON output"]
    end

    subgraph Wandb["Experiment Tracking"]
        ARTIFACTS["Artifacts<br/>(index/datasets/submission)"]
        RUNS["Runs & Metrics"]
    end

    BS --> CORPUS
    FS --> QUERY

    CORPUS --> EMB
    EMB --> VEC
    VEC --> NORM
    NORM --> IDX
    IDX --> SAVE
    SAVE -.-> ARTIFACTS

    QUERY --> QEMB
    QEMB --> SEARCH
    IDX --> SEARCH
    SEARCH --> PRED
    PRED --> SUB["submission.csv"]
    SUB -.-> ARTIFACTS

    QUERY --> LLM
    LLM --> LPROMPT
    LPROMPT --> LOUT
    LOUT --> SUB

    ARTIFACTS --> RUNS

    style Embedding fill:#e1f5ff
    style Index fill:#ffe1f5
    style Retrieval fill:#f5ffe1
    style Direct fill:#fff5e1
    style Wandb fill:#e1e1ff
```

### System Components

| Component | Description | Location |
|-----------|-------------|----------|
| **EmbeddingModel** | Unified interface for text embeddings using OpenRouter API | `features.py:96-117` |
| **build_faiss_index()** | Constructs FAISS vector index with L2 normalization | `features.py:122-206` |
| **retrieve_top_k()** | Performs top-K similarity search | `features.py:209-234` |
| **OpenRouterRetrievalPredictor** | Retrieval-based predictor using FAISS | `predict.py:37-73` |
| **OpenRouterDirectPredictor** | Direct LLM-based predictor | `predict.py:76-141` |

### Data Flow

1. **Training Phase** (`Predictor.fit()`):
   - Load `base_stories.tsv` (50 works)
   - Generate embeddings via OpenRouter API
   - Build FAISS IndexFlatIP with L2 normalization
   - Save index + log to W&B Artifacts

2. **Prediction Phase** (`Predictor.predict()`):
   - **Retrieval**: Embed query → FAISS search → Return top-2 IDs
   - **Direct**: LLM analyzes query with work list → Returns 2 IDs

3. **Output**: `submission.csv` with columns `[id, a, b]`

