"""Enterprise-scale database seeder (5,000+ relational records).

Populates both Turso Cloud (via libsql_client.batch) and local SQLite
with a high-cardinality enterprise supply chain dataset grounded in Olist:
- 500 Vendors across Brazilian commercial hubs
- 1,200 Products across 8 enterprise categories
- 3,500 Historical Purchase Orders with dates, quantities, and statuses
- 16 Departmental Budget Ceilings across fiscal quarters
- Relational B-Tree Indexes on foreign keys and filter fields
"""

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import libsql_client
from src.utils.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_enterprise_scale")

CATEGORIES = [
    "Hardware",
    "Software",
    "Services",
    "Logistics",
    "Electronics",
    "Office_Supplies",
    "Industrial",
    "Security",
]

BRAZILIAN_CITIES = [
    ("Sao Paulo", "SP"),
    ("Rio de Janeiro", "RJ"),
    ("Curitiba", "PR"),
    ("Belo Horizonte", "MG"),
    ("Porto Alegre", "RS"),
    ("Campinas", "SP"),
    ("Florianopolis", "SC"),
    ("Salvador", "BA"),
    ("Brasilia", "DF"),
    ("Recife", "PE"),
]

STATUSES = ["Approved", "Fulfilled", "Draft", "Cancelled"]


def generate_enterprise_data(
    seed: int = 42,
    num_vendors: int = 500,
    num_items: int = 1200,
    num_orders: int = 3500,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate high-cardinality enterprise dataset with strict referential integrity."""
    rng = random.Random(seed)

    # 1. Vendors (500)
    vendors = []
    for i in range(num_vendors):
        city, state = rng.choice(BRAZILIAN_CITIES)
        cat = CATEGORIES[i % len(CATEGORIES)]
        vendors.append({
            "vendor_id": f"VND-BR-{1000 + i}",
            "name": f"Enterprise {cat} Supply {city} #{i+1} Ltda",
            "category": cat,
            "city": city,
            "state": state,
            "rating": round(rng.uniform(3.2, 5.0), 2),
            "contract_status": "Active" if rng.random() > 0.1 else "UnderReview",
        })

    # 2. Inventory Items (1200)
    items = []
    for j in range(num_items):
        cat = CATEGORIES[j % len(CATEGORIES)]
        unit_cost = round(rng.uniform(15.0, 1250.0), 2)
        stock = rng.randint(10, 800)
        items.append({
            "item_sku": f"SKU-BR-{2000 + j}",
            "item_name": f"Enterprise {cat} Component Model-{j+1}",
            "category": cat,
            "weight_g": round(rng.uniform(100.0, 5000.0), 1),
            "unit_cost": unit_cost,
            "stock_level": stock,
            "reorder_threshold": max(15, int(stock * 0.15)),
        })

    # 3. Purchase Orders (3500)
    orders = []
    for k in range(num_orders):
        v = rng.choice(vendors)
        # Find item matching category or random
        matching = [it for it in items if it["category"] == v["category"]]
        it = rng.choice(matching) if matching else rng.choice(items)
        qty = rng.randint(1, 50)
        price = round(it["unit_cost"] * rng.uniform(0.92, 1.15), 2)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        orders.append({
            "po_id": f"PO-BR-{10000 + k}",
            "vendor_id": v["vendor_id"],
            "item_sku": it["item_sku"],
            "quantity": qty,
            "unit_price": price,
            "freight_value": round(rng.uniform(10.0, 120.0), 2),
            "order_status": rng.choice(STATUSES),
            "order_date": f"2026-{month:02d}-{day:02d}",
        })

    # 4. Budget Caps (16)
    caps = []
    for cat in CATEGORIES:
        for q in ["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"]:
            caps.append({
                "category": cat,
                "fiscal_quarter": q,
                "allocated_budget": round(rng.uniform(150000.0, 750000.0), 2),
            })

    return vendors, items, orders, caps


def seed_local_sqlite(
    db_path: Path,
    vendors: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    caps: List[Dict[str, Any]],
) -> None:
    """Seed local SQLite file with enterprise data and indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # DDL
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS vendors (
        vendor_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        city TEXT DEFAULT 'Sao Paulo',
        state TEXT DEFAULT 'SP',
        rating REAL DEFAULT 4.5,
        contract_status TEXT DEFAULT 'Active'
    );
    CREATE TABLE IF NOT EXISTS inventory_items (
        item_sku TEXT PRIMARY KEY,
        item_name TEXT NOT NULL,
        category TEXT NOT NULL,
        weight_g REAL DEFAULT 500.0,
        unit_cost REAL NOT NULL,
        stock_level INTEGER DEFAULT 100,
        reorder_threshold INTEGER DEFAULT 20
    );
    CREATE TABLE IF NOT EXISTS purchase_orders (
        po_id TEXT PRIMARY KEY,
        vendor_id TEXT NOT NULL,
        item_sku TEXT NOT NULL,
        quantity INTEGER DEFAULT 1,
        unit_price REAL NOT NULL,
        freight_value REAL DEFAULT 15.0,
        order_status TEXT DEFAULT 'Approved',
        order_date TEXT NOT NULL,
        FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id),
        FOREIGN KEY (item_sku) REFERENCES inventory_items (item_sku)
    );
    CREATE TABLE IF NOT EXISTS budget_caps (
        category TEXT NOT NULL,
        fiscal_quarter TEXT NOT NULL,
        allocated_budget REAL NOT NULL,
        PRIMARY KEY (category, fiscal_quarter)
    );
    CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category);
    CREATE INDEX IF NOT EXISTS idx_items_cat ON inventory_items(category);
    CREATE INDEX IF NOT EXISTS idx_po_vendor ON purchase_orders(vendor_id);
    CREATE INDEX IF NOT EXISTS idx_po_item ON purchase_orders(item_sku);
    CREATE INDEX IF NOT EXISTS idx_po_status ON purchase_orders(order_status);
    """)

    cur.executemany(
        "INSERT OR REPLACE INTO vendors VALUES (:vendor_id, :name, :category, :city, :state, :rating, :contract_status)",
        vendors,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO inventory_items VALUES (:item_sku, :item_name, :category, :weight_g, :unit_cost, :stock_level, :reorder_threshold)",
        items,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO purchase_orders VALUES (:po_id, :vendor_id, :item_sku, :quantity, :unit_price, :freight_value, :order_status, :order_date)",
        orders,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO budget_caps VALUES (:category, :fiscal_quarter, :allocated_budget)",
        caps,
    )

    conn.commit()
    conn.close()
    logger.info("Local SQLite (%s) seeded successfully.", db_path)


def seed_turso_cloud(
    url: str,
    token: str,
    vendors: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    orders: List[Dict[str, Any]],
    caps: List[Dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """Seed Turso Cloud cluster with batched statements."""
    client = libsql_client.create_client_sync(url=url, auth_token=token)

    # 1. DDL
    ddl = [
        """CREATE TABLE IF NOT EXISTS vendors (
            vendor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            city TEXT DEFAULT 'Sao Paulo',
            state TEXT DEFAULT 'SP',
            rating REAL DEFAULT 4.5,
            contract_status TEXT DEFAULT 'Active'
        );""",
        """CREATE TABLE IF NOT EXISTS inventory_items (
            item_sku TEXT PRIMARY KEY,
            item_name TEXT NOT NULL,
            category TEXT NOT NULL,
            weight_g REAL DEFAULT 500.0,
            unit_cost REAL NOT NULL,
            stock_level INTEGER DEFAULT 100,
            reorder_threshold INTEGER DEFAULT 20
        );""",
        """CREATE TABLE IF NOT EXISTS purchase_orders (
            po_id TEXT PRIMARY KEY,
            vendor_id TEXT NOT NULL,
            item_sku TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            unit_price REAL NOT NULL,
            freight_value REAL DEFAULT 15.0,
            order_status TEXT DEFAULT 'Approved',
            order_date TEXT NOT NULL,
            FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id),
            FOREIGN KEY (item_sku) REFERENCES inventory_items (item_sku)
        );""",
        """CREATE TABLE IF NOT EXISTS budget_caps (
            category TEXT NOT NULL,
            fiscal_quarter TEXT NOT NULL,
            allocated_budget REAL NOT NULL,
            PRIMARY KEY (category, fiscal_quarter)
        );""",
        "CREATE INDEX IF NOT EXISTS idx_vendors_cat ON vendors(category);",
        "CREATE INDEX IF NOT EXISTS idx_items_cat ON inventory_items(category);",
        "CREATE INDEX IF NOT EXISTS idx_po_vendor ON purchase_orders(vendor_id);",
        "CREATE INDEX IF NOT EXISTS idx_po_item ON purchase_orders(item_sku);",
        "CREATE INDEX IF NOT EXISTS idx_po_status ON purchase_orders(order_status);",
    ]
    for stmt in ddl:
        client.execute(stmt)

    logger.info("Turso Cloud DDL & Indexes verified.")

    # 2. Batched Vendors
    logger.info("Ingesting %d vendors to Turso...", len(vendors))
    v_sql = "INSERT OR REPLACE INTO vendors VALUES (?, ?, ?, ?, ?, ?, ?)"
    for i in range(0, len(vendors), batch_size):
        chunk = vendors[i : i + batch_size]
        stmts = [
            libsql_client.Statement(v_sql, [v["vendor_id"], v["name"], v["category"], v["city"], v["state"], v["rating"], v["contract_status"]])
            for v in chunk
        ]
        client.batch(stmts)

    # 3. Batched Items
    logger.info("Ingesting %d inventory items to Turso...", len(items))
    it_sql = "INSERT OR REPLACE INTO inventory_items VALUES (?, ?, ?, ?, ?, ?, ?)"
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        stmts = [
            libsql_client.Statement(it_sql, [it["item_sku"], it["item_name"], it["category"], it["weight_g"], it["unit_cost"], it["stock_level"], it["reorder_threshold"]])
            for it in chunk
        ]
        client.batch(stmts)

    # 4. Batched Purchase Orders
    logger.info("Ingesting %d purchase orders to Turso...", len(orders))
    po_sql = "INSERT OR REPLACE INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    for i in range(0, len(orders), batch_size):
        chunk = orders[i : i + batch_size]
        stmts = [
            libsql_client.Statement(po_sql, [po["po_id"], po["vendor_id"], po["item_sku"], po["quantity"], po["unit_price"], po["freight_value"], po["order_status"], po["order_date"]])
            for po in chunk
        ]
        client.batch(stmts)

    # 5. Budget Caps
    c_sql = "INSERT OR REPLACE INTO budget_caps VALUES (?, ?, ?)"
    stmts = [
        libsql_client.Statement(c_sql, [c["category"], c["fiscal_quarter"], c["allocated_budget"]])
        for c in caps
    ]
    client.batch(stmts)

    logger.info("Turso Cloud ingestion complete.")


def run_migration() -> Dict[str, int]:
    """Execute end-to-end enterprise scale migration."""
    settings = get_settings()
    logger.info("Generating 5,216 enterprise records...")
    vendors, items, orders, caps = generate_enterprise_data()

    # 1. Local SQLite sync
    local_path = Path(__file__).resolve().parent.parent.parent / "data" / "lexsi_production.db"
    seed_local_sqlite(local_path, vendors, items, orders, caps)

    # 2. Turso Cloud sync
    if settings.database_url and settings.turso_auth_token:
        logger.info("Connecting to Turso Cloud (%s)...", settings.database_url)
        seed_turso_cloud(settings.database_url, settings.turso_auth_token, vendors, items, orders, caps)

    return {
        "vendors": len(vendors),
        "inventory_items": len(items),
        "purchase_orders": len(orders),
        "budget_caps": len(caps),
        "total": len(vendors) + len(items) + len(orders) + len(caps),
    }


if __name__ == "__main__":
    result = run_migration()
    print("Migration finished:", result)

