"""
pages/1_📊_Database_Insights.py
--------------------------------
Full visual dashboard with Plotly charts for all key business metrics.
"""

import streamlit as st
import sqlite3
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Database Insights",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Database Insights Dashboard")
st.caption("Visual overview of your retail business data")

# ── Database connection ───────────────────────────────────────────────────────
dbfilepath = (Path(__file__).parent.parent / "retail_business.db").absolute()

if not dbfilepath.exists():
    st.error("❌ Database not found. Please run `python create_retail_db.py` first.")
    st.stop()

conn = sqlite3.connect(dbfilepath)

# ── KPI Metrics ───────────────────────────────────────────────────────────────
st.header("🎯 Key Metrics")

total_products = pd.read_sql("SELECT COUNT(*) as c FROM products", conn).iloc[0]['c']
total_sales    = pd.read_sql("SELECT COUNT(*) as c FROM sales", conn).iloc[0]['c']
total_revenue  = pd.read_sql("SELECT SUM(total_amount) as r FROM sales", conn).iloc[0]['r']
avg_order      = total_revenue / total_sales if total_sales > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🛒 Total Products",   f"{total_products:,}")
col2.metric("📦 Total Sales",      f"{total_sales:,}")
col3.metric("💰 Total Revenue",    f"₹{total_revenue/1e5:.1f}L")
col4.metric("🧾 Avg Order Value",  f"₹{avg_order:,.0f}")

st.markdown("---")

# ── Row 1: Top Products + Category Revenue ────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top 8 Products by Revenue")
    top_products = pd.read_sql("""
        SELECT p.product_name AS Product,
               SUM(s.total_amount) AS Revenue,
               COUNT(s.sale_id)    AS Sales
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.product_name
        ORDER BY Revenue DESC
        LIMIT 8
    """, conn)

    fig = px.bar(
        top_products, x="Revenue", y="Product", orientation="h",
        color="Revenue", color_continuous_scale="Blues",
        text=top_products["Revenue"].apply(lambda x: f"₹{x/1e3:.0f}K"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=380, margin=dict(l=0, r=20, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📦 Revenue by Category")
    category_data = pd.read_sql("""
        SELECT p.category AS Category,
               SUM(s.total_amount) AS Revenue,
               COUNT(s.sale_id)    AS Sales
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        GROUP BY p.category
        ORDER BY Revenue DESC
    """, conn)

    fig = px.pie(
        category_data, names="Category", values="Revenue",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label",
                      textfont_size=13)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Row 2: Sales by City + Payment Methods ────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏙️ Revenue by City")
    city_data = pd.read_sql("""
        SELECT customer_city AS City,
               SUM(total_amount) AS Revenue,
               COUNT(*)          AS Transactions
        FROM sales
        GROUP BY customer_city
        ORDER BY Revenue DESC
    """, conn)

    fig = px.bar(
        city_data, x="City", y="Revenue",
        color="Revenue", color_continuous_scale="Teal",
        text=city_data["Revenue"].apply(lambda x: f"₹{x/1e5:.1f}L"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=380, margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💳 Payment Method Distribution")
    payment_data = pd.read_sql("""
        SELECT payment_method AS Method,
               COUNT(*)          AS Transactions,
               SUM(total_amount) AS Revenue
        FROM sales
        GROUP BY payment_method
        ORDER BY Transactions DESC
    """, conn)

    fig = px.pie(
        payment_data, names="Method", values="Transactions",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=13)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Row 3: Daily Sales Trend ──────────────────────────────────────────────────
st.subheader("📈 Daily Sales Trend (Last 30 Days)")

trend_data = pd.read_sql("""
    SELECT sale_date AS Date,
           SUM(total_amount) AS Revenue,
           COUNT(*)           AS Transactions
    FROM sales
    GROUP BY sale_date
    ORDER BY sale_date DESC
    LIMIT 30
""", conn)
trend_data = trend_data.sort_values("Date")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=trend_data["Date"], y=trend_data["Revenue"],
    mode="lines+markers",
    name="Revenue",
    line=dict(color="#0068c9", width=2.5),
    marker=dict(size=6),
    fill="tozeroy",
    fillcolor="rgba(0,104,201,0.1)",
))
fig.update_layout(
    height=350,
    xaxis_title="Date",
    yaxis_title="Revenue (₹)",
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Row 4: Supplier breakdown + Low Stock ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏭 Products per Supplier")
    supplier_data = pd.read_sql("""
        SELECT supplier AS Supplier,
               COUNT(*) AS Products
        FROM products
        GROUP BY supplier
        ORDER BY Products DESC
    """, conn)

    fig = px.bar(
        supplier_data, x="Products", y="Supplier", orientation="h",
        color="Products", color_continuous_scale="Purples",
        text="Products",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=420, margin=dict(l=0, r=20, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚠️ Low Stock Alert (< 50 units)")
    low_stock = pd.read_sql("""
        SELECT product_name AS Product,
               stock_quantity AS Stock,
               category AS Category
        FROM products
        WHERE stock_quantity < 50
        ORDER BY stock_quantity ASC
    """, conn)

    if not low_stock.empty:
        st.warning(f"🚨 {len(low_stock)} products are running low on stock!")
        fig = px.bar(
            low_stock, x="Stock", y="Product", orientation="h",
            color="Stock", color_continuous_scale="Reds_r",
            text="Stock",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=max(300, len(low_stock) * 38),
            margin=dict(l=0, r=20, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ All products are adequately stocked!")

st.markdown("---")

# ── Database Schema (collapsible) ────────────────────────────────────────────
with st.expander("🗄️ View Database Schema"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Products Table**")
        products_schema = pd.read_sql("PRAGMA table_info(products)", conn)
        st.dataframe(products_schema[['name', 'type']], use_container_width=True)
    with c2:
        st.markdown("**Sales Table**")
        sales_schema = pd.read_sql("PRAGMA table_info(sales)", conn)
        st.dataframe(sales_schema[['name', 'type']], use_container_width=True)

# ── Suggested queries ─────────────────────────────────────────────────────────
st.markdown("### 💡 Try These in the Chat")
st.markdown("""
- "What are the top 5 best-selling products?"
- "Show me sales by city"
- "Which products need restocking?"
- "Compare UPI vs Card payment methods"
- "What's the total revenue this month?"
- "Show daily sales trend for last 7 days"
""")

conn.close()
