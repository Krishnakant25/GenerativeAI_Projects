# 💹 Financial Knowledge Hybrid Search Engine

A hybrid retrieval system over **2,195 real financial document chunks** from Wikipedia and ArXiv, combining BM25 sparse encoding and dense semantic embeddings on Pinecone Serverless — the retrieval backbone powering modern RAG systems in financial intelligence applications.

🔗 **Live Demo:** [Coming soon — Streamlit Cloud]

---

## 🎯 Strategic Objectives

> Designed to address the growing demand for **intelligent, context-aware financial information retrieval** in FinTech, investment research, and enterprise knowledge management.

**1. Accelerate Financial Research & Decision-Making**
Replace manual document trawling with sub-300ms intelligent retrieval across thousands of financial sources — enabling analysts to surface relevant insights in seconds rather than hours, directly reducing research overhead costs.

**2. Maximize Retrieval Accuracy Across Query Types**
By combining keyword precision (BM25) with semantic understanding (dense embeddings), the system achieves **86.7% Hit Rate@5** and **MRR@5 of 0.73** — ensuring both exact financial term lookups and intent-based queries return high-quality results, reducing information gaps in critical decisions.

**3. Deliver a Scalable, Cost-Efficient Knowledge Base**
Built on Pinecone Serverless with automated Wikipedia and ArXiv ingestion pipelines, the system scales from hundreds to millions of documents without infrastructure overhead — providing a low-cost foundation for enterprise financial Q&A and RAG applications.

**4. Enable Transparent, Tunable Retrieval for Domain Experts**
The alpha fusion parameter gives financial teams direct control over retrieval behavior — tuning between keyword precision and semantic recall based on query context — improving end-user trust and adoption in analyst workflows.

---

## 🌍 Real-World Scenario

> **User:** Priya, a Junior Investment Analyst at a mid-sized asset management firm.
>
> **Pain Point:** Priya spends 2–3 hours each morning manually searching through financial documents, research papers, and market reference materials to prepare briefings on topics like portfolio risk, derivatives exposure, or macroeconomic indicators. Traditional keyword search tools return noisy, irrelevant results for conceptual queries like *"how does investor psychology affect market volatility"* — forcing her to rephrase queries repeatedly or scan documents manually.
>
> **With this system:** Priya opens the Streamlit app, types her query in plain English, and receives the top 5 most relevant financial document chunks in under 300ms — ranked by a hybrid model that understands both the exact terminology and the underlying intent. She switches the alpha slider from 0.5 to 1.0 for a broader semantic sweep when exploring new concepts, or back to 0.0 for precise term lookups when referencing specific financial models. What previously took hours now takes seconds — freeing her to focus on analysis and client recommendations rather than information retrieval.

---

## 📊 Benchmark Results

Evaluated on **15 hard financial queries** across **3 difficulty categories** and **5 alpha configurations**:

| Alpha | Mode | MRR@5 | Hit Rate@5 | Avg Latency |
|---|---|---|---|---|
| 0.0 | Pure BM25 | 0.639 | 80.0% | 252 ms |
| 0.25 | BM25-leaning Hybrid | 0.717 | 100% | 243 ms |
| **0.5** | **Balanced Hybrid** | **0.730** | **86.7%** | **264 ms** |
| 0.75 | Semantic-leaning | 0.700 | 100% | 243 ms |
| 1.0 | Pure Semantic | 0.616 | 86.7% | 242 ms |

**Hybrid fusion (alpha=0.5) outperformed pure BM25 by +14.2% and pure semantic by +18.6% MRR** across 3 query difficulty categories: semantic-hard, BM25-hard, and hybrid-critical.

---

## 🏗️ Architecture

```
Query
  │
  ├──► BM25 Encoder      (sparse vector) ──────┐
  │                                             ▼
  └──► all-MiniLM-L6-v2  (dense vector) ──► Pinecone dotproduct
                                               │
                                         Alpha fusion
                                      (0=BM25, 1=Semantic)
                                               │
                                          Top-K Results
```

| Component | Detail |
|---|---|
| Data Sources | Wikipedia (20 articles) + ArXiv (10 papers) |
| Dense Embeddings | `all-MiniLM-L6-v2` (HuggingFace, 384-dim) |
| Sparse Encoding | BM25 (`pinecone-text`) |
| Vector Database | Pinecone Serverless (AWS us-east-1, dotproduct) |
| Retriever | `PineconeHybridSearchRetriever` (LangChain) |
| UI | Streamlit |
| Python | 3.11.9 |

---

## 📁 Project Structure

```
hybrid-search-pinecone/
│
├── financial_hybrid_search.ipynb  # Main notebook — financial RAG pipeline
├── hybrid_search_intro.ipynb      # Intro notebook — concepts & fundamentals
├── app.py                         # Streamlit web app
├── retriever_config.json          # Auto-generated retriever settings
├── bm25_financial.json            # Auto-generated BM25 model
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template
├── .gitignore
└── README.md
```

---

## 📓 Notebooks

### 1. `financial_hybrid_search.ipynb` — Main Project
The full production pipeline over real financial data:

| Section | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Load environment variables securely |
| 3 | Initialize Pinecone Serverless index (fresh) |
| 4 | Fetch 20 Wikipedia financial articles via API |
| 5 | Fetch 10 ArXiv finance + LLM papers via API |
| 6 | Chunk documents with LangChain text splitter |
| 7 | Load `all-MiniLM-L6-v2` + fit BM25 encoder |
| 8 | Upload 2,195 chunks to Pinecone (batched) |
| 9 | Run real financial queries |
| 10 | Alpha tuning — keyword vs semantic balance |
| 11 | **Hard benchmark** — MRR@5, Hit Rate@5, latency across 3 difficulty categories |
| 12 | Save config for Streamlit app |

### 2. `hybrid_search_intro.ipynb` — Concepts & Fundamentals
A step-by-step introduction to hybrid search before the full project:

| Section | Description |
|---|---|
| 1–3 | Setup, env vars, Pinecone index creation |
| 4–5 | Dense embeddings + BM25 sparse encoding explained |
| 6–8 | Build retriever, index sample documents |
| 9 | Run queries and inspect results |
| 10 | Alpha tuning walkthrough |
| 11 | Benchmark — MRR@3, Hit Rate@3, latency |

> Start here if you are new to hybrid search before running the financial notebook.

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Krishnakant25/GenerativeAI_Projects.git
cd GenerativeAI_Projects
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
Open `financial_hybrid_search.ipynb` and run all cells top to bottom.

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

```
PINECONE_API_KEY=your_pinecone_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Get your keys:
- Pinecone → https://app.pinecone.io
- HuggingFace → https://huggingface.co/settings/tokens

> Never commit `.env` — it is listed in `.gitignore`

---

## 💡 Why Hybrid Search?

| Query Type | Example | Best Mode |
|---|---|---|
| Exact keyword | "Sharpe ratio formula" | BM25 (alpha=0.0) |
| Paraphrased | "measuring daily investment loss" | Semantic (alpha=1.0) |
| Mixed real-world | "LLM earnings prediction NLP" | **Hybrid (alpha=0.5)** ✅ |

Pure keyword search fails on paraphrased or intent-based queries. Pure semantic search can miss exact financial term matches. Hybrid combines both signals for robust retrieval — proven by +14.2% MRR gain over BM25 on hard financial domain queries.

---

## 🛠️ Tech Stack

`Python 3.11` · `LangChain` · `Pinecone Serverless` · `HuggingFace` · `sentence-transformers` · `pinecone-text` · `Wikipedia API` · `ArXiv API` · `Streamlit` · `python-dotenv`

---

## 📜 License

MIT
