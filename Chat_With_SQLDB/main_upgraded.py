"""
main.py (UPGRADED VERSION)
---------------------------
Business Intelligence Chat Assistant with Advanced Features
- Result visualization (metrics, charts)
- Query analytics dashboard
- Export functionality
- Enhanced error handling
"""

import streamlit as st
from pathlib import Path
import sqlite3
import re
from datetime import datetime

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

# ── Initialize session state for analytics ────────────────────────────────────
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
        success_rate = (st.session_state.successful_queries / st.session_state.query_count) * 100
        st.metric("Success Rate", f"{success_rate:.0f}%", 
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
selected_opt = st.sidebar.radio(
    label="Choose the database",
    options=radio_opt,
)

if radio_opt.index(selected_opt) == 1:
    db_uri       = MYSQL
    with st.sidebar.expander("MySQL Connection Details", expanded=True):
        mysql_host   = st.text_input("Host")
        mysql_user   = st.text_input("User")
        mysql_password = st.text_input("Password", type="password")
        mysql_db     = st.text_input("Database Name")
else:
    db_uri = LOCALDB

# ── Sidebar: API key ──────────────────────────────────────────────────────────
api_key = st.sidebar.text_input(label="🔑 Groq API Key", type="password")

# ── Security: Query validation ────────────────────────────────────────────────
def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate SQL query for safety.
    Returns (is_safe: bool, error_message: str).
    """
    query_upper = query.upper()
    dangerous = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE",
        "INSERT", "GRANT", "REVOKE", "CREATE", "REPLACE",
    ]
    for keyword in dangerous:
        if keyword in query_upper:
            return False, f"⚠️ Dangerous operation '{keyword}' blocked."
    if not query_upper.strip().startswith("SELECT"):
        return False, "⚠️ Only SELECT queries are allowed."
    return True, ""

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

# ── Sidebar: Analytics Dashboard ─────────────────────────────────────────────
if st.session_state.query_count > 0:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Analytics")
    st.sidebar.metric("Total Queries", st.session_state.query_count)
    st.sidebar.metric("Successful", st.session_state.successful_queries)
    
    if st.sidebar.button("📥 Export Query History"):
        history_text = "\n\n".join([
            f"Query {i+1}: {q['query']}\nResponse: {q['response']}\n"
            for i, q in enumerate(st.session_state.query_history[-10:])
        ])
        st.sidebar.download_button(
            label="Download History",
            data=history_text,
            file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
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

# ── Database configuration (cached) ──────────────────────────────────────────
@st.cache_resource(ttl="2h")
def configure_db(
    db_uri: str,
    mysql_host: str = None,
    mysql_user: str = None,
    mysql_password: str = None,
    mysql_db: str = None,
) -> SQLDatabase:
    """Build and return a LangChain SQLDatabase instance."""
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "retail_business.db").absolute()
        if not dbfilepath.exists():
            st.error("❌  `retail_business.db` not found. Run `python create_retail_db.py` first.")
            st.stop()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}"
            f"@{mysql_host}/{mysql_db}"
        )
        return SQLDatabase(engine)

# Connect to DB
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# ── LangChain SQL Agent ───────────────────────────────────────────────────────
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state or st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your BI assistant. Ask me anything about your business data. 📊"}
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
    
    # Track query
    st.session_state.query_count += 1

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        
        try:
            # Execute query
            start_time = datetime.now()
            response = agent.run(user_query, callbacks=[streamlit_callback])
            execution_time = (datetime.now() - start_time).total_seconds()
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.successful_queries += 1
            
            # Log to history
            st.session_state.query_history.append({
                "query": user_query,
                "response": response,
                "time": execution_time,
                "success": True
            })
            
            # ── Smart Result Visualization ────────────────────────────────
            visualized = False
            
            # Extract numbers from response
            numbers = re.findall(r'₹?\s*[\d,]+\.?\d*', response)
            
            # Revenue/Total queries → Show as metric
            if any(word in user_query.lower() for word in ['revenue', 'total', 'sum', 'amount']):
                if numbers:
                    clean_number = numbers[0].replace('₹', '').replace(',', '').strip()
                    try:
                        value = float(clean_number)
                        st.metric("💰 Result", f"₹{value:,.2f}")
                        visualized = True
                    except:
                        pass
            
            # Count queries → Show as metric
            elif any(word in user_query.lower() for word in ['how many', 'count', 'number of']):
                if numbers:
                    clean_number = numbers[0].replace(',', '').strip()
                    try:
                        value = int(float(clean_number))
                        st.metric("📊 Count", f"{value:,}")
                        visualized = True
                    except:
                        pass
            
            # "Top N" queries → Show as table
            elif 'top' in user_query.lower() and any(char.isdigit() for char in user_query):
                st.info("💡 Tip: Results show the top items. Click 'Full Response' below for details.")
                visualized = True
            
            # Display response
            if visualized:
                with st.expander("📝 Full Response", expanded=True):
                    st.success(response)
            else:
                st.success(response)
            
            # Show execution time
            st.caption(f"⚡ Executed in {execution_time:.2f}s")
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # Log failed query
            st.session_state.query_history.append({
                "query": user_query,
                "response": error_msg,
                "success": False
            })
            
            # Show troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common issues:**
                - Make sure your question is clear and specific
                - Try rephrasing your question
                - Use sample questions as templates
                - Check if the data exists (e.g., "this month" needs recent data)
                """)
