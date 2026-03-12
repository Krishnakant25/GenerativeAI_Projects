# 🔍 Hybrid Search with Pinecone + LangChain

A production-grade hybrid retrieval pipeline combining **BM25 sparse encoding** and **dense semantic embeddings** in a single Pinecone Serverless index — the retrieval backbone used in modern RAG systems.

---

## 🏗️ Architecture

```
Query
  │
  ├──► BM25 Encoder (sparse vector)  ──────────┐
  │                                             ▼
  └──► all-MiniLM-L6-v2 (dense vector) ──► Pinecone dotproduct
                                               │
                                         Alpha fusion
                                         (0=BM25, 1=Dense)
                                               │
                                          Top-K Results
```

| Component | Tool |
|---|---|
| Dense Embeddings | `all-MiniLM-L6-v2` (HuggingFace, 384-dim) |
| Sparse Encoding | BM25 (`pinecone-text`) |
| Vector Database | Pinecone Serverless (AWS us-east-1, dotproduct) |
| Retriever | `PineconeHybridSearchRetriever` (LangChain) |
| Python | 3.11.9 |

---

## 📊 Benchmark Results

Evaluated across **10 labelled queries** and **5 alpha configurations**:

| Alpha | Mode | MRR@3 | Hit Rate@3 | Avg Latency | P95 Latency |
|---|---|---|---|---|---|
| 0.0 | Pure BM25 | **0.783** | 90.0% | 252 ms | 280 ms |
| 0.25 | BM25-leaning hybrid | 0.717 | **100%** | 243 ms | 247 ms |
| 0.5 | Balanced hybrid | 0.717 | **100%** | **242 ms** | 245 ms |
| 0.75 | Semantic-leaning | 0.700 | **100%** | 243 ms | 248 ms |
| 1.0 | Pure Dense | 0.700 | **100%** | 242 ms | 246 ms |

**Key finding:** BM25 achieves highest precision (MRR 0.783) but misses semantically phrased queries. Hybrid fusion (alpha ≥ 0.25) achieves perfect recall (100% Hit Rate@3) while maintaining competitive latency under 250 ms.

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/your-username/hybrid-search-pinecone.git
cd hybrid-search-pinecone
```

### 2. Create a Python 3.11 virtual environment
```bash
py -3.11 -m venv venv311
venv311\Scripts\activate        # Windows
# source venv311/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 5. Run the notebook
Open `hybrid_search_updated.ipynb` in VS Code or Jupyter and run all cells top to bottom.

---

## 🔑 Environment Variables

Create a `.env` file (never commit this):

```
PINECONE_API_KEY=your_pinecone_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Get your keys:
- Pinecone → https://app.pinecone.io
- HuggingFace → https://huggingface.co/settings/tokens

---

## 📁 Project Structure

```
hybrid-search-pinecone/
│
├── hybrid_search_updated.ipynb   # Main notebook (pipeline + benchmarks)
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── .gitignore                    # Excludes .env, venv, cache
└── README.md
```

> `bm25_values.json` is excluded from version control — it is auto-generated when you run the notebook.

---

## 📖 Notebook Sections

| Section | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Load environment variables securely |
| 3 | Initialize Pinecone Serverless index |
| 4 | Load HuggingFace dense embedding model |
| 5 | Fit & serialize BM25 encoder |
| 6 | Build `PineconeHybridSearchRetriever` |
| 7 | Index documents (dense + sparse upsert) |
| 8 | Run hybrid queries |
| 9 | Alpha tuning — keyword vs semantic balance |
| 10 | Extend index with custom documents |
| 11 | Cleanup (delete index) |
| 12 | **Production benchmark** — MRR@3, Hit Rate@3, latency |

---

## 💡 Why Hybrid Search?

| Scenario | Best Mode |
|---|---|
| "New York trip" (exact keyword) | BM25 (alpha=0.0) |
| "last place I visited" (semantic) | Dense (alpha=1.0) |
| Mixed real-world queries | **Hybrid (alpha=0.25–0.5)** ✅ |

Pure keyword search fails on paraphrased or intent-based queries. Pure semantic search can miss exact entity matches. Hybrid combines both for robust production retrieval.

---

## 🛠️ Tech Stack

`Python 3.11` · `LangChain` · `Pinecone Serverless` · `HuggingFace` · `sentence-transformers` · `pinecone-text` · `python-dotenv`

---

## 📜 License

MIT
