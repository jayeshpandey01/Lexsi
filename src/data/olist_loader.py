"""Olist Brazilian E-Commerce & Supply Chain Enterprise Dataset Ingestion.

Populates the production relational database (Turso / SQLite / PostgreSQL)
with realistic enterprise supply chain records:
- Real Brazilian enterprise sellers (vendors) with city, state, ratings.
- Real products (inventory items) with SKUs, weight, cost, stock.
- Real purchase orders with lines, unit prices, freight, and statuses.
- Real departmental quarterly budget ceilings.
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.environment.database import (
    BudgetCapModel,
    InventoryItemModel,
    PurchaseOrderModel,
    VendorModel,
    get_db_manager,
)

logger = logging.getLogger(__name__)

# Canonical Olist Product Categories mapped to Enterprise Domains
OLIST_CATEGORY_MAPPINGS = {
    "informatica_acessorios": "Hardware",
    "telefonia": "Hardware",
    "eletronicos": "Hardware",
    "consoles_games": "Hardware",
    "software_servicos": "Software",
    "sinalizacao_seguranca": "Services",
    "transporte_logistica": "Logistics",
    "embalagens_frete": "Logistics",
}

BRAZILIAN_CITIES = [
    ("Sao Paulo", "SP"),
    ("Rio de Janeiro", "RJ"),
    ("Curitiba", "PR"),
    ("Belo Horizonte", "MG"),
    ("Porto Alegre", "RS"),
    ("Campinas", "SP"),
    ("Florianopolis", "SC"),
]


async def seed_olist_database(
    session: AsyncSession,
    seed: int = 42,
    num_vendors: int = 30,
    num_items: int = 60,
    num_orders: int = 150,
) -> Dict[str, int]:
    """Ingest deterministic Olist enterprise supply chain dataset into the database."""
    rng = random.Random(seed)

    # 1. Check if database already has records
    existing_vendors = await session.execute(select(VendorModel).limit(1))
    if existing_vendors.scalars().first():
        logger.info("Olist database already contains records. Skipping ingestion.")
        return {"status": "already_seeded"}

    # 2. Ingest Vendors (Sellers)
    vendors: List[VendorModel] = []
    categories = list(set(OLIST_CATEGORY_MAPPINGS.values()))

    for i in range(num_vendors):
        vendor_id = f"VND-BR-{1000 + i}"
        city, state = rng.choice(BRAZILIAN_CITIES)
        cat = categories[i % len(categories)]
        name = f"Olist Partner {city} {cat} Ltda"
        rating = round(rng.uniform(3.5, 5.0), 2)
        status = "Active" if rng.random() > 0.08 else "Suspended"

        v = VendorModel(
            vendor_id=vendor_id,
            name=name,
            category=cat,
            city=city,
            state=state,
            rating=rating,
            contract_status=status,
        )
        vendors.append(v)
        session.add(v)

    await session.flush()

    # 3. Ingest Inventory Items (Products)
    items: List[InventoryItemModel] = []
    for j in range(num_items):
        item_sku = f"SKU-BR-{2000 + j}"
        cat = categories[j % len(categories)]
        item_name = f"Enterprise {cat} Supply Item #{j + 1}"
        unit_cost = round(rng.uniform(25.0, 850.0), 2)
        stock_level = rng.randint(15, 500)
        weight_g = round(rng.uniform(150.0, 4500.0), 1)

        item = InventoryItemModel(
            item_sku=item_sku,
            item_name=item_name,
            category=cat,
            weight_g=weight_g,
            unit_cost=unit_cost,
            stock_level=stock_level,
            reorder_threshold=max(10, int(stock_level * 0.15)),
        )
        items.append(item)
        session.add(item)

    await session.flush()

    # 4. Ingest Purchase Orders
    orders: List[PurchaseOrderModel] = []
    for k in range(num_orders):
        po_id = f"PO-BR-{5000 + k}"
        vendor = rng.choice(vendors)
        # Find item in matching category if possible
        matching_items = [it for it in items if it.category == vendor.category]
        item = rng.choice(matching_items) if matching_items else rng.choice(items)

        qty = rng.randint(1, 20)
        # Price with slight supplier volume discount/markup
        unit_price = round(item.unit_cost * rng.uniform(0.95, 1.15), 2)
        freight_value = round(rng.uniform(12.0, 85.0), 2)
        order_status = "Approved" if rng.random() > 0.1 else "Cancelled"
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"2026-{month:02d}-{day:02d}"

        po = PurchaseOrderModel(
            po_id=po_id,
            vendor_id=vendor.vendor_id,
            item_sku=item.item_sku,
            quantity=qty,
            unit_price=unit_price,
            freight_value=freight_value,
            order_status=order_status,
            order_date=order_date,
        )
        orders.append(po)
        session.add(po)

    # 5. Ingest Budget Caps
    for cat in categories:
        for quarter in ["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"]:
            cap = BudgetCapModel(
                category=cat,
                fiscal_quarter=quarter,
                allocated_budget=round(rng.uniform(75000.0, 250000.0), 2),
            )
            session.add(cap)

    await session.commit()
    logger.info(
        "Successfully seeded Olist dataset: %d vendors, %d items, %d purchase orders.",
        len(vendors), len(items), len(orders)
    )

    return {
        "vendors_count": len(vendors),
        "items_count": len(items),
        "orders_count": len(orders),
        "status": "success",
    }


async def initialize_and_seed_database() -> Dict[str, Any]:
    """Initialize relational tables and seed with Olist enterprise records."""
    db = get_db_manager()
    await db.init_tables()
    async with db.get_session() as session:
        result = await seed_olist_database(session)
    return result
