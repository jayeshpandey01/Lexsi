"""Deterministic in-memory relational database fixtures for Procurement & Inventory domain."""

from __future__ import annotations

import random
from typing import Any, Dict, List


def get_mock_database(seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a reproducible, closed-world relational database from seed."""
    rng = random.Random(seed)

    categories = ["Hardware", "Software", "Services", "Logistics"]
    countries = ["US", "DE", "JP", "GB", "SG"]

    # 1. Vendors table
    vendors: List[Dict[str, Any]] = [
        {
            "vendor_id": f"VND-{100 + i}",
            "name": f"Enterprise {categories[i % len(categories)]} Solutions {i}",
            "category": categories[i % len(categories)],
            "rating": round(3.5 + (rng.random() * 1.5), 2),
            "contract_status": "Active" if i % 4 != 0 else "UnderReview",
            "country": countries[i % len(countries)],
        }
        for i in range(12)
    ]

    # 2. Inventory items table
    inventory_items: List[Dict[str, Any]] = [
        {
            "item_sku": f"SKU-{8000 + i}",
            "item_name": f"Enterprise Component {i}",
            "category": categories[i % len(categories)],
            "stock_level": rng.randint(10, 500),
            "reorder_threshold": rng.randint(20, 100),
            "unit_cost": round(15.0 + (rng.random() * 150.0), 2),
        }
        for i in range(20)
    ]

    # 3. Purchase orders table
    purchase_orders: List[Dict[str, Any]] = []
    statuses = ["Draft", "Approved", "Fulfilled", "Cancelled"]
    for i in range(30):
        vendor = vendors[i % len(vendors)]
        item = inventory_items[i % len(inventory_items)]
        qty = rng.randint(5, 100)
        price = round(item["unit_cost"] * (1.0 + (rng.random() * 0.1)), 2)
        purchase_orders.append(
            {
                "po_id": f"PO-{9000 + i}",
                "vendor_id": vendor["vendor_id"],
                "item_sku": item["item_sku"],
                "quantity": qty,
                "unit_price": price,
                "order_status": statuses[i % len(statuses)],
                "order_date": f"2026-0{(i % 6) + 1}-15",
            }
        )

    # 4. Budget caps table
    budget_caps: List[Dict[str, Any]] = [
        {
            "category": cat,
            "fiscal_quarter": "2026-Q1",
            "allocated_budget": 50000.0 + (idx * 25000.0),
        }
        for idx, cat in enumerate(categories)
    ]

    return {
        "vendors": vendors,
        "inventory_items": inventory_items,
        "purchase_orders": purchase_orders,
        "budget_caps": budget_caps,
    }
