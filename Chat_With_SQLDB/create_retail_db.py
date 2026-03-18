"""
create_retail_db.py
-------------------
Generates a realistic Indian retail business database (retail_business.db)
with 30+ products and 1000+ sales records.
"""

import sqlite3
import random
from datetime import datetime, timedelta

# ── Database connection ──────────────────────────────────────────────────────
conn = sqlite3.connect("retail_business.db")
cursor = conn.cursor()

# Drop tables if they exist (clean slate on re-run)
cursor.execute("DROP TABLE IF EXISTS sales")
cursor.execute("DROP TABLE IF EXISTS products")

# ── Create PRODUCTS table ────────────────────────────────────────────────────
cursor.execute("""
CREATE TABLE products (
    product_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name     TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    price            REAL    NOT NULL,
    stock_quantity   INTEGER NOT NULL,
    supplier         TEXT    NOT NULL,
    last_restock_date TEXT   NOT NULL
)
""")

# ── Seed data: 32 Indian retail products ────────────────────────────────────
# Format: (name, category, price, supplier)
product_seed = [
    # Electronics (price ₹1,000 – ₹50,000)
    ("Samsung Galaxy M34 5G",      "Electronics", 18999, "Samsung India"),
    ("OnePlus Nord CE3 Lite",       "Electronics", 19999, "OnePlus"),
    ("Redmi Note 13 Pro",           "Electronics", 21999, "Xiaomi India"),
    ("boAt Airdopes 131",           "Electronics",  1299, "boAt"),
    ("boAt Rockerz 450 Headphones", "Electronics",  1499, "boAt"),
    ("HP 15s Laptop (i3, 8GB)",     "Electronics", 38999, "HP India"),
    ("Lenovo IdeaPad Slim 3",       "Electronics", 42999, "Lenovo India"),
    ("Sony WH-1000XM5",             "Electronics", 29990, "Sony India"),
    ("Mi Smart TV 43\" 4K",         "Electronics", 27999, "Xiaomi India"),
    ("Realme Buds Air 5",           "Electronics",  3999, "Realme"),

    # Groceries (price ₹100 – ₹500)
    ("Tata Tea Gold 1kg",           "Groceries",    380,  "Tata Consumer"),
    ("Fortune Sunlite Oil 1L",      "Groceries",    145,  "Adani Wilmar"),
    ("Amul Butter 500g",            "Groceries",    280,  "Amul"),
    ("Aashirvaad Atta 10kg",        "Groceries",    380,  "ITC"),
    ("Maggi Noodles 12-pack",       "Groceries",    180,  "Nestle India"),
    ("Surf Excel Matic 3kg",        "Groceries",    480,  "HUL"),
    ("Nescafe Classic 200g",        "Groceries",    390,  "Nestle India"),
    ("Haldiram Bhujia 1kg",         "Groceries",    320,  "Haldiram"),
    ("Saffola Gold Oil 1L",         "Groceries",    170,  "Marico"),
    ("Dabur Honey 500g",            "Groceries",    210,  "Dabur"),

    # Clothing (price ₹500 – ₹5,000)
    ("Allen Solly Formal Shirt",    "Clothing",    1799,  "Madura Fashion"),
    ("Levi's 511 Slim Jeans",       "Clothing",    3499,  "Levi Strauss India"),
    ("Puma Running Shoes",          "Clothing",    3999,  "Puma India"),
    ("Fabindia Kurta",              "Clothing",    1490,  "Fabindia"),
    ("Peter England Chinos",        "Clothing",    2199,  "Madura Fashion"),
    ("Nike Air Max Shoes",          "Clothing",    7999,  "Nike India"),
    ("Van Heusen T-Shirt",          "Clothing",     999,  "Madura Fashion"),

    # Home (price ₹200 – ₹3,000)
    ("Prestige Aluminium Cooker 5L","Home",        1850,  "TTK Prestige"),
    ("Milton Thermosteel 1L",       "Home",         699,  "Milton"),
    ("Pigeon Non-stick Kadai",      "Home",         799,  "Pigeon"),
    ("Cello Polypropylene Box Set", "Home",         450,  "Cello"),
    ("Bajaj Mixer Grinder 750W",    "Home",        2799,  "Bajaj Electricals"),
]

# Randomise stock and restock dates (Jan – Feb 2024)
random.seed(42)
restock_base = datetime(2024, 1, 1)

products_rows = []
for name, category, price, supplier in product_seed:
    stock = random.randint(20, 200)
    days_offset = random.randint(0, 55)          # Jan 1 – Feb 25
    restock_date = (restock_base + timedelta(days=days_offset)).strftime("%Y-%m-%d")
    products_rows.append((name, category, price, stock, supplier, restock_date))

cursor.executemany("""
    INSERT INTO products (product_name, category, price, stock_quantity, supplier, last_restock_date)
    VALUES (?, ?, ?, ?, ?, ?)
""", products_rows)

print(f"✅  Inserted {len(products_rows)} products.")

# ── Create SALES table ───────────────────────────────────────────────────────
cursor.execute("""
CREATE TABLE sales (
    sale_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id      INTEGER NOT NULL,
    sale_date       TEXT    NOT NULL,
    quantity        INTEGER NOT NULL,
    total_amount    REAL    NOT NULL,
    customer_city   TEXT    NOT NULL,
    payment_method  TEXT    NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
""")

# ── Reference data for sales generation ─────────────────────────────────────
cities = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]

# UPI ~40 %, Cash ~25 %, Card ~25 %, Wallet ~10 %
payment_methods = (
    ["UPI"] * 40 +
    ["Cash"] * 25 +
    ["Card"] * 25 +
    ["Wallet"] * 10
)

# Fetch product ids and prices for FK integrity
cursor.execute("SELECT product_id, price FROM products")
product_price_map = {pid: price for pid, price in cursor.fetchall()}
product_ids = list(product_price_map.keys())

# Sales spread over 75 days starting Jan 1 2024
sales_start = datetime(2024, 1, 1)

sales_rows = []
for _ in range(1100):                            # ~1,100 records
    pid = random.choice(product_ids)
    days_offset = random.randint(0, 74)
    sale_date = (sales_start + timedelta(days=days_offset)).strftime("%Y-%m-%d")
    qty = random.randint(1, 5)
    total = round(product_price_map[pid] * qty, 2)
    city = random.choice(cities)
    payment = random.choice(payment_methods)
    sales_rows.append((pid, sale_date, qty, total, city, payment))

cursor.executemany("""
    INSERT INTO sales (product_id, sale_date, quantity, total_amount, customer_city, payment_method)
    VALUES (?, ?, ?, ?, ?, ?)
""", sales_rows)

print(f"✅  Inserted {len(sales_rows)} sales records.")

# ── Commit & close ───────────────────────────────────────────────────────────
conn.commit()
conn.close()

print("\n🎉  retail_business.db created successfully!")
print("    Tables: products (32 rows)  |  sales (1,100 rows)")
print("    Run: streamlit run main.py")
