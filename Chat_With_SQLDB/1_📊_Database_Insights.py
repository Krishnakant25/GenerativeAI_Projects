"""
pages/1_📊_Database_Insights.py
-------------------------------
Dashboard showing database statistics and insights
"""

import streamlit as st
import sqlite3
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Database Insights",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Database Insights Dashboard")
st.caption("Explore your database structure and key metrics")

# Connect to database
dbfilepath = (Path(__file__).parent.parent / "retail_business.db").absolute()

if not dbfilepath.exists():
    st.error("❌ Database not found. Please run `python create_retail_db.py` first.")
    st.stop()

conn = sqlite3.connect(dbfilepath)

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.header("🎯 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

# Total products
total_products = pd.read_sql("SELECT COUNT(*) as count FROM products", conn).iloc[0]['count']
col1.metric("Total Products", f"{total_products:,}")

# Total sales
total_sales = pd.read_sql("SELECT COUNT(*) as count FROM sales", conn).iloc[0]['count']
col2.metric("Total Sales", f"{total_sales:,}")

# Total revenue
total_revenue = pd.read_sql("SELECT SUM(total_amount) as revenue FROM sales", conn).iloc[0]['revenue']
col3.metric("Total Revenue", f"₹{total_revenue:,.2f}")

# Average order value
avg_order = total_revenue / total_sales if total_sales > 0 else 0
col4.metric("Avg Order Value", f"₹{avg_order:,.2f}")

# ── Database Schema ───────────────────────────────────────────────────────────
st.header("🗄️ Database Schema")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Products Table")
    products_schema = pd.read_sql("PRAGMA table_info(products)", conn)
    st.dataframe(products_schema[['name', 'type']], use_container_width=True)

with col2:
    st.subheader("Sales Table")
    sales_schema = pd.read_sql("PRAGMA table_info(sales)", conn)
    st.dataframe(sales_schema[['name', 'type']], use_container_width=True)

# ── Top Insights ──────────────────────────────────────────────────────────────
st.header("💡 Quick Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 5 Products by Revenue")
    top_products = pd.read_sql("""
        SELECT 
            p.product_name,
            SUM(s.total_amount) as revenue,
            COUNT(s.sale_id) as sales_count
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.product_name
        ORDER BY revenue DESC
        LIMIT 5
    """, conn)
    
    for idx, row in top_products.iterrows():
        st.metric(
            f"{idx+1}. {row['product_name']}", 
            f"₹{row['revenue']:,.0f}",
            f"{row['sales_count']} sales"
        )

with col2:
    st.subheader("Sales by City")
    sales_by_city = pd.read_sql("""
        SELECT 
            customer_city,
            COUNT(*) as sales_count,
            SUM(total_amount) as revenue
        FROM sales
        GROUP BY customer_city
        ORDER BY revenue DESC
        LIMIT 5
    """, conn)
    
    for idx, row in sales_by_city.iterrows():
        st.metric(
            f"{idx+1}. {row['customer_city']}", 
            f"₹{row['revenue']:,.0f}",
            f"{row['sales_count']} sales"
        )

# ── Category Breakdown ────────────────────────────────────────────────────────
st.header("📦 Category Performance")

category_data = pd.read_sql("""
    SELECT 
        p.category,
        COUNT(DISTINCT p.product_id) as product_count,
        COUNT(s.sale_id) as sales_count,
        SUM(s.total_amount) as revenue
    FROM products p
    LEFT JOIN sales s ON p.product_id = s.product_id
    GROUP BY p.category
    ORDER BY revenue DESC
""", conn)

st.dataframe(category_data, use_container_width=True)

# ── Payment Methods ───────────────────────────────────────────────────────────
st.header("💳 Payment Method Distribution")

payment_data = pd.read_sql("""
    SELECT 
        payment_method,
        COUNT(*) as transaction_count,
        SUM(total_amount) as total_revenue,
        ROUND(AVG(total_amount), 2) as avg_transaction
    FROM sales
    GROUP BY payment_method
    ORDER BY transaction_count DESC
""", conn)

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(payment_data, use_container_width=True)

with col2:
    st.subheader("Quick Stats")
    most_popular = payment_data.iloc[0]['payment_method']
    st.info(f"**Most Popular:** {most_popular}")
    
    highest_value = payment_data.sort_values('avg_transaction', ascending=False).iloc[0]['payment_method']
    st.info(f"**Highest Avg Value:** {highest_value}")

# ── Low Stock Alert ───────────────────────────────────────────────────────────
st.header("⚠️ Low Stock Alert")

low_stock = pd.read_sql("""
    SELECT 
        product_name,
        category,
        stock_quantity,
        supplier
    FROM products
    WHERE stock_quantity < 30
    ORDER BY stock_quantity ASC
""", conn)

if len(low_stock) > 0:
    st.warning(f"🚨 {len(low_stock)} products are running low on stock!")
    st.dataframe(low_stock, use_container_width=True)
else:
    st.success("✅ All products are adequately stocked!")

# ── Sample Queries ────────────────────────────────────────────────────────────
st.header("💡 Suggested Queries for Chat")

st.markdown("""
Based on this data, try asking:
- "What are the top 5 best-selling products?"
- "Show me sales by city"
- "Which products need restocking?"
- "Compare UPI vs Card payment methods"
- "What's the total revenue this month?"
- "Which category has the highest average price?"
""")

conn.close()
