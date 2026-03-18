"""
main.py
-------
Business Intelligence Chat Assistant
Natural language → SQL → Answer + Auto Chart
"""

import streamlit as st
from pathlib import Path
import sqlite3
import re
from datetime import datetime

import pandas as pd
import plotly.express as px

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from sqlalchemy import create_engine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Intelligence Chat Assistant",
    page_icon="💼",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "successful_queries" not in st.session_state:
    st.session_state.successful_queries = 0
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💼 Business Intelligence Chat Assistant")
    st.caption("Ask questions about your business data in plain English")
with col2:
    if st.session_state.query_count > 0:
        rate = (st.session_state.successful_queries / st.session_state.query_count) * 100
        st.metric("Success Rate", f"{rate:.0f}%",
                  f"{st.session_state.successful_queries}/{st.session_state.query_count} queries")

# ── Constants ─────────────────────────────────────────────────────────────────
LOCALDB = "USE_LOCALDB"
MYSQL   = "USE_MYSQL"

# ── Sidebar: DB selection ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

radio_opt = [
    "Use SQLite 3 Database - retail_business.db",
    "Connect to your MySQL Database",
]
selected_opt = st.sidebar.radio(label="Choose the database", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    with st.sidebar.expander("MySQL Connection Details", expanded=True):
        mysql_host     = st.text_input("Host")
        mysql_user     = st.text_input("User")
        mysql_password = st.text_input("Password", type="password")
        mysql_db       = st.text_input("Database Name")
else:
    db_uri = LOCALDB

# ── Sidebar: API key ──────────────────────────────────────────────────────────
api_key = st.sidebar.text_input(label="🔑 Groq API Key", type="password")

# ── Security: Query validation ────────────────────────────────────────────────
def validate_query(query: str) -> tuple[bool, str]:
    """Block all write/DDL operations. Only allow SELECT."""
    q = query.upper()
    for kw in ["DROP","DELETE","TRUNCATE","ALTER","UPDATE","INSERT",
               "GRANT","REVOKE","CREATE","REPLACE"]:
        if kw in q:
            return False, f"⚠️ Dangerous operation '{kw}' blocked."
    if not q.strip().startswith("SELECT"):
        return False, "⚠️ Only SELECT queries are allowed."
    return True, ""

# ── Chart engine ──────────────────────────────────────────────────────────────
def get_db_connection():
    """Direct sqlite3 connection for chart queries (read-only)."""
    dbfilepath = (Path(__file__).parent / "retail_business.db").absolute()
    return sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)

def try_show_chart(user_query: str) -> bool:
    """
    Detect query intent and render an appropriate Plotly chart.
    Returns True if a chart was shown.
    """
    q = user_query.lower()
    conn = get_db_connection()
    shown = False

    try:
        # ── Top products → Horizontal bar chart ───────────────────────────────
        if ("top" in q or "best" in q) and ("product" in q or "selling" in q or "revenue" in q):
            nums  = re.findall(r'\d+', user_query)
            limit = int(nums[0]) if nums else 5

            df = pd.read_sql(f"""
                SELECT p.product_name AS Product,
                       SUM(s.total_amount) AS Revenue,
                       COUNT(s.sale_id)    AS Sales
                FROM sales s
                JOIN products p ON s.product_id = p.product_id
                GROUP BY p.product_name
                ORDER BY Revenue DESC
                LIMIT {limit}
            """, conn)

            fig = px.bar(
                df, x="Revenue", y="Product", orientation="h",
                title=f"🏆 Top {limit} Products by Revenue",
                color="Revenue", color_continuous_scale="Blues",
                text=df["Revenue"].apply(lambda x: f"₹{x:,.0f}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
            shown = True

        # ── City / revenue by city → Bar chart ───────────────────────────────
        elif "city" in q or "cities" in q:
            df = pd.read_sql("""
                SELECT customer_city AS City,
                       SUM(total_amount) AS Revenue,
                       COUNT(*)          AS Transactions
                FROM sales
                GROUP BY customer_city
                ORDER BY Revenue DESC
            """, conn)

            fig = px.bar(
                df, x="City", y="Revenue",
                title="🏙️ Revenue by City",
                color="Revenue", color_continuous_scale="Teal",
                text=df["Revenue"].apply(lambda x: f"₹{x/1e5:.1f}L"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
            shown = True

        # ── Payment methods → Donut pie chart ────────────────────────────────
        elif any(w in q for w in ["payment", "upi", "cash", "card", "wallet"]):
            df = pd.read_sql("""
                SELECT payment_method AS Method,
                       COUNT(*)          AS Transactions,
                       SUM(total_amount) AS Revenue
                FROM sales
                GROUP BY payment_method
                ORDER BY Transactions DESC
            """, conn)

            fig = px.pie(
                df, names="Method", values="Transactions",
                title="💳 Payment Method Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            shown = True

        # ── Category → Bar chart ──────────────────────────────────────────────
        elif "category" in q or "categories" in q:
            df = pd.read_sql("""
                SELECT p.category AS Category,
                       SUM(s.total_amount) AS Revenue,
                       COUNT(s.sale_id)    AS Sales
                FROM sales s
                JOIN products p ON s.product_id = p.product_id
                GROUP BY p.category
                ORDER BY Revenue DESC
            """, conn)

            fig = px.bar(
                df, x="Category", y="Revenue",
                title="📦 Revenue by Category",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text=df["Revenue"].apply(lambda x: f"₹{x/1e5:.1f}L"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
            shown = True

        # ── Daily / weekly trend → Line chart ────────────────────────────────
        elif any(w in q for w in ["trend", "daily", "week", "over time"]):
            nums = re.findall(r'\d+', user_query)
            days = int(nums[0]) if nums else 14

            df = pd.read_sql(f"""
                SELECT sale_date AS Date,
                       SUM(total_amount) AS Revenue,
                       COUNT(*)           AS Transactions
                FROM sales
                GROUP BY sale_date
                ORDER BY sale_date DESC
                LIMIT {days}
            """, conn)
            df = df.sort_values("Date")

            fig = px.line(
                df, x="Date", y="Revenue",
                title=f"📈 Daily Sales Trend (Last {days} Days)",
                markers=True, line_shape="spline",
            )
            fig.update_traces(line_color="#0068c9", line_width=2.5,
                              marker=dict(size=7))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            shown = True

        # ── Stock / inventory → Horizontal bar chart ──────────────────────────
        elif any(w in q for w in ["stock", "restock", "inventory", "low"]):
            nums      = re.findall(r'\d+', user_query)
            threshold = int(nums[0]) if nums else 50

            df = pd.read_sql(f"""
                SELECT product_name AS Product,
                       stock_quantity AS Stock,
                       category AS Category
                FROM products
                WHERE stock_quantity < {threshold}
                ORDER BY stock_quantity ASC
            """, conn)

            if not df.empty:
                fig = px.bar(
                    df, x="Stock", y="Product", orientation="h",
                    title=f"⚠️ Low Stock Products (Under {threshold} units)",
                    color="Stock", color_continuous_scale="Reds_r",
                    text="Stock",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    height=max(320, len(df) * 42),
                )
                st.plotly_chart(fig, use_container_width=True)
                shown = True

        # ── Supplier → Horizontal bar chart ──────────────────────────────────
        elif "supplier" in q:
            df = pd.read_sql("""
                SELECT supplier AS Supplier,
                       COUNT(*) AS Products
                FROM products
                GROUP BY supplier
                ORDER BY Products DESC
            """, conn)

            fig = px.bar(
                df, x="Products", y="Supplier", orientation="h",
                title="🏭 Products per Supplier",
                color="Products", color_continuous_scale="Purples",
                text="Products",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False, height=520,
            )
            st.plotly_chart(fig, use_container_width=True)
            shown = True

    except Exception:
        pass  # Chart errors must never break the main response
    finally:
        conn.close()

    return shown

# ── Sidebar: Sample questions ─────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Sample Questions")

sample_questions = [
    "What are the top 5 best-selling products?",
    "Show total sales by city",
    "Which products are low in stock (less than 30)?",
    "What's the total revenue this month?",
    "Compare UPI vs Card payment methods",
    "Which category generates most revenue?",
    "Show daily sales trend for last 7 days",
    "Which supplier has most products?",
]

for i, question in enumerate(sample_questions):
    if st.sidebar.button(question, key=f"sample_{i}", use_container_width=True):
        st.session_state.auto_question = question
        st.rerun()

# ── Sidebar: Session analytics ────────────────────────────────────────────────
if st.session_state.query_count > 0:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Analytics")
    st.sidebar.metric("Total Queries", st.session_state.query_count)
    st.sidebar.metric("Successful", st.session_state.successful_queries)

    if st.sidebar.button("📥 Export Query History"):
        history_text = "\n\n".join([
            f"Query {i+1}: {q['query']}\nResponse: {q['response']}"
            for i, q in enumerate(st.session_state.query_history[-10:])
        ])
        st.sidebar.download_button(
            label="Download History",
            data=history_text,
            file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )

# ── Guard: missing credentials ────────────────────────────────────────────────
if not api_key:
    st.info("👈  Please enter your **Groq API Key** in the sidebar to get started.")
    st.stop()

if db_uri == MYSQL and not all([mysql_host, mysql_user, mysql_password, mysql_db]):
    st.info("👈  Please fill in all **MySQL connection details** in the sidebar.")
    st.stop()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="Llama-3.3-70b-Versatile",
    streaming=True,
)

# ── Database (LangChain, cached) ──────────────────────────────────────────────
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None,
                 mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "retail_business.db").absolute()
        if not dbfilepath.exists():
            st.error("❌ `retail_business.db` not found. Run `python create_retail_db.py` first.")
            st.stop()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )
        return SQLDatabase(engine)

db = (configure_db(db_uri) if db_uri == LOCALDB
      else configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db))

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent   = create_sql_agent(
    llm=llm, toolkit=toolkit, verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state or st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [
        {"role": "assistant",
         "content": "Hello! I'm your BI assistant. Ask me anything about your business data. 📊"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Handle input ──────────────────────────────────────────────────────────────
if "auto_question" in st.session_state:
    user_query = st.session_state.auto_question
    del st.session_state.auto_question
else:
    user_query = st.chat_input(placeholder="Ask anything about the business data…")

# ── Process query ─────────────────────────────────────────────────────────────
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    st.session_state.query_count += 1

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            start_time = datetime.now()
            response   = agent.run(user_query, callbacks=[streamlit_callback])
            exec_time  = (datetime.now() - start_time).total_seconds()

            # 1. Text answer
            st.success(response)

            # 2. Auto chart (fires silently if no match)
            try_show_chart(user_query)

            # 3. Execution time
            st.caption(f"⚡ Executed in {exec_time:.2f}s")

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.successful_queries += 1
            st.session_state.query_history.append({
                "query": user_query, "response": response,
                "time": exec_time, "success": True,
            })

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.query_history.append({
                "query": user_query, "response": error_msg, "success": False,
            })
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                - Rephrase your question more clearly
                - Use the sample questions as templates
                - Check if the data exists for the time period you mentioned
                """)
