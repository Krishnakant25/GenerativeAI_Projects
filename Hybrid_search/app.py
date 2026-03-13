"""
💹 Financial Knowledge Hybrid Search Engine
Streamlit app — connects to existing Pinecone index, no re-indexing needed.
Run: streamlit run app.py
"""

import os, json, time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Hybrid Search",
    page_icon="💹",
    layout="wide",
)

# ── Load config ───────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "retriever_config.json")) as f:
    CONFIG = json.load(f)

# ── Cache retriever (loads once, reused across queries) ───────
@st.cache_resource(show_spinner="Loading retrieval models...")
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["model_name"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    bm25 = BM25Encoder().load(os.path.join(BASE_DIR, CONFIG["bm25_path"]))
    pc   = Pinecone(api_key=PINECONE_API_KEY)
    idx  = pc.Index(CONFIG["index_name"])
    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=idx,
        top_k=CONFIG["top_k"],
        alpha=CONFIG["default_alpha"],
    )

retriever = load_retriever()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Search Settings")

    alpha = st.slider(
        label="Alpha — Retrieval Mode",
        min_value=0.0, max_value=1.0,
        value=CONFIG["default_alpha"], step=0.25,
        help="0 = Pure BM25 keyword | 0.5 = Hybrid | 1 = Pure Semantic",
    )

    alpha_labels = {
        0.0 : "🔤 Pure BM25 (Keyword)",
        0.25: "🔤➕🧠 BM25-leaning Hybrid",
        0.5 : "⚖️ Balanced Hybrid",
        0.75: "🧠➕🔤 Semantic-leaning Hybrid",
        1.0 : "🧠 Pure Semantic",
    }
    st.info(alpha_labels.get(alpha, "Custom"))

    top_k = st.selectbox("Results to return", [3, 5, 7, 10], index=1)

    st.divider()
    st.markdown("**📊 Index Stats**")
    st.metric("Document chunks", f"{CONFIG['total_chunks']:,}")
    st.metric("Data sources", "Wikipedia + ArXiv")
    st.metric("Embedding model", CONFIG["model_name"])
    st.metric("Vector DB", "Pinecone Serverless")

    st.divider()
    st.markdown("**🔬 How it works**")
    st.markdown("""
- **BM25** matches exact financial terms
- **Dense embeddings** capture meaning & context
- **Hybrid** combines both signals via alpha fusion
- **Benchmark:** Hybrid MRR@5 = 0.73, +14.2% vs BM25
    """)

# ── Main UI ───────────────────────────────────────────────────
st.title("💹 Financial Knowledge Hybrid Search")
st.caption(f"Searching over {CONFIG['total_chunks']:,} chunks from 20 Wikipedia articles + 10 ArXiv papers")

# Sample queries
st.markdown("**💡 Try these queries:**")
sample_queries = [
    "What is systematic risk in portfolio management?",
    "How do hedge funds generate alpha?",
    "LLM applications in financial forecasting",
    "How is Value at Risk calculated?",
    "Explain the Capital Asset Pricing Model",
    "What causes financial crises?",
    "NLP sentiment analysis for stock prediction",
]

cols = st.columns(3)
for i, q in enumerate(sample_queries):
    if cols[i % 3].button(q, use_container_width=True, key=f"sample_{i}"):
        st.session_state["query_input"] = q

# Search box
query = st.text_input(
    label="🔎 Enter your financial query",
    placeholder="e.g. What is the Sharpe ratio?",
    key="query_input",
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

# ── Search & Display Results ──────────────────────────────────
if query and search_clicked or (query and st.session_state.get("query_input")):
    retriever.alpha = alpha
    retriever.top_k = top_k

    with st.spinner("Searching..."):
        start   = time.perf_counter()
        results = retriever.invoke(query)
        elapsed = (time.perf_counter() - start) * 1000

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Results returned", len(results))
    m2.metric("Query latency", f"{elapsed:.0f} ms")
    m3.metric("Alpha used", alpha)
    m4.metric("Mode", alpha_labels.get(alpha, "Custom").split(" ", 1)[-1])

    st.divider()
    st.subheader(f"📄 Top {len(results)} Results")

    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("score", None)
        score_str = f" — Score: `{score:.4f}`" if score else ""

        with st.expander(f"**Result {i}**{score_str}", expanded=(i <= 3)):
            st.write(doc.page_content)

    # Compare modes
    st.divider()
    with st.expander("🔬 Compare all retrieval modes for this query"):
        compare_cols = st.columns(3)
        mode_configs = [(0.0, "Pure BM25"), (0.5, "Hybrid"), (1.0, "Pure Semantic")]

        for col, (a, label) in zip(compare_cols, mode_configs):
            retriever.alpha = a
            retriever.top_k = 3
            mode_results = retriever.invoke(query)
            col.markdown(f"**{label} (alpha={a})**")
            for j, r in enumerate(mode_results, 1):
                col.markdown(f"*[{j}]* {r.page_content[:150]}...")
            col.divider()

        retriever.alpha = alpha
        retriever.top_k = top_k
