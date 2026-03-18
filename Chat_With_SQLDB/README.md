# 💼 Business Intelligence Chat Assistant

Natural language interface for querying SQL databases. Ask business questions in plain English and get instant answers.

## 🎯 Features

- 🗣️ **Natural Language to SQL** – Ask questions in plain English
- 📊 **Dual Database Support** – Works with SQLite and MySQL
- 💬 **Chat Interface** – Conversational history maintained
- 🔍 **Transparent Execution** – See the SQL queries being generated
- 🔒 **Read-Only Mode** – Safe querying with validation
- 🇮🇳 **Indian Business Context** – Sample data with Indian brands, cities, ₹ pricing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key ([Get one free here](https://console.groq.com))

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd sql-chat-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

4. Create sample database:
```bash
python create_retail_db.py
```

5. Run the app:
```bash
streamlit run main.py
```

## 💡 Example Questions

Try asking:
- "What are the top 5 best-selling products?"
- "Show total sales by city"
- "Which products are low in stock?"
- "What's the total revenue this month?"
- "Compare UPI vs Card payment methods"
- "Which category generates most revenue?"
- "Show daily sales trend for last 7 days"

## 🏗️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | GROQ – Llama 3.3 70B Versatile |
| Agent | LangChain SQL Agent |
| UI | Streamlit |
| Database | SQLite / MySQL (via SQLAlchemy) |
| Language | Python 3.8+ |

## 🗄️ Database Schema

### Products Table
| Column | Type | Description |
|--------|------|-------------|
| product_id | INTEGER PK | Auto-increment primary key |
| product_name | TEXT | Indian brand product name |
| category | TEXT | Electronics / Groceries / Clothing / Home |
| price | REAL | Price in ₹ |
| stock_quantity | INTEGER | Units available |
| supplier | TEXT | Indian supplier name |
| last_restock_date | TEXT | YYYY-MM-DD |

### Sales Table
| Column | Type | Description |
|--------|------|-------------|
| sale_id | INTEGER PK | Auto-increment primary key |
| product_id | INTEGER FK | References products |
| sale_date | TEXT | YYYY-MM-DD |
| quantity | INTEGER | Units sold (1–5) |
| total_amount | REAL | price × quantity in ₹ |
| customer_city | TEXT | Indian city |
| payment_method | TEXT | Cash / UPI / Card / Wallet |

## 🔒 Security

- Read-only SQLite connection (`?mode=ro`)
- Dangerous operations blocked: `DROP`, `DELETE`, `TRUNCATE`, `ALTER`, `UPDATE`, `INSERT`, `GRANT`, `REVOKE`, `CREATE`, `REPLACE`
- Query validation runs before every execution
- No database modifications possible

## 📝 Use Cases

- **Retail Businesses** – Quick sales analysis
- **Small Businesses** – Inventory management
- **Analysts** – Ad-hoc data queries
- **Non-technical Users** – Self-service analytics

## 🚢 Deployment on Streamlit Cloud

1. Push code to GitHub (`.db` file is git-ignored; users run `create_retail_db.py` locally or you can commit it separately)
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GROQ_API_KEY` to **Streamlit Secrets**
4. Deploy!

## 📄 License

MIT License – Feel free to use for personal or commercial projects.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

> **Note:** This project demonstrates GenAI + LangChain capabilities for a portfolio. For production deployments with sensitive data, add authentication, rate limiting, and audit logging.
