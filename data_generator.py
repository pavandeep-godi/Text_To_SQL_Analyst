"""Create deterministic, realistic sample sales and procurement data for the demo."""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AS_OF = date(2026, 10, 5)

CUSTOMERS = [
    ("C001", "Northstar Retail", "Retail", "Consumer Goods", "North America"),
    ("C002", "Blue Ridge Systems", "Enterprise", "Technology", "North America"),
    ("C003", "Harbor Health Group", "Enterprise", "Healthcare", "North America"),
    ("C004", "Cedar & Stone", "SMB", "Hospitality", "Europe"),
    ("C005", "Vertex Manufacturing", "Enterprise", "Manufacturing", "Europe"),
    ("C006", "Sunrise Market", "SMB", "Retail", "Asia Pacific"),
    ("C007", "Atlas Public Services", "Public Sector", "Government", "North America"),
    ("C008", "Evergreen Labs", "SMB", "Healthcare", "Europe"),
    ("C009", "Summit Logistics", "Enterprise", "Transportation", "Asia Pacific"),
    ("C010", "Golden Fields Co.", "Mid-Market", "Agriculture", "Latin America"),
]
PRODUCTS = [
    ("P100", "Analytics Platform", "Software", "Analytics", 950),
    ("P101", "Cloud Data Suite", "Software", "Cloud", 720),
    ("P102", "Edge Gateway", "Hardware", "Networking", 410),
    ("P103", "Secure Access Kit", "Hardware", "Security", 285),
    ("P104", "Insight Pro License", "Software", "Licenses", 180),
    ("P105", "Storage Array", "Hardware", "Storage", 1250),
    ("P106", "Integration Services", "Services", "Consulting", 1500),
    ("P107", "Support Renewal", "Services", "Support", 360),
    ("P108", "Mobile Scanner", "Hardware", "Devices", 195),
    ("P109", "Forecasting Add-on", "Software", "Analytics", 330),
]
SUPPLIERS = [
    ("S001", "Apex Components", "Hardware", "North America", "Strategic"),
    ("S002", "Nimbus Cloud Supply", "Technology", "Europe", "Preferred"),
    ("S003", "Pacific Industrial Ltd", "Manufacturing", "Asia Pacific", "Preferred"),
    ("S004", "Greenline Packaging", "Packaging", "North America", "Approved"),
    ("S005", "Meridian Logistics", "Logistics", "Europe", "Strategic"),
    ("S006", "Vertex Office Goods", "Office Supplies", "Latin America", "Approved"),
    ("S007", "Sterling Security Group", "Security", "North America", "Preferred"),
    ("S008", "Cobalt Professional Services", "Services", "Asia Pacific", "Approved"),
]


def random_date(rng: random.Random, year: int) -> date:
    start = date(year, 1, 1)
    end = AS_OF if year == AS_OF.year else date(year, 12, 31)
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate_sales(rng: random.Random, count: int = 2000) -> list[dict[str, object]]:
    rows = []
    for index in range(1, count + 1):
        year = rng.choice([2023, 2024, 2025, 2026])
        order_date = random_date(rng, year)
        customer = rng.choice(CUSTOMERS)
        product = rng.choice(PRODUCTS)
        quantity = rng.randint(1, 35)
        discount = round(rng.choices([0, 0.05, 0.10, 0.15, 0.20], [35, 20, 25, 15, 5])[0], 2)
        unit_price = round(product[4] * rng.uniform(0.88, 1.12), 2)
        gross = round(quantity * unit_price, 2)
        net = round(gross * (1 - discount), 2)
        cost = round(net * rng.uniform(0.48, 0.76), 2)
        shipping_days = rng.randint(1, 16)
        ship_date = order_date + timedelta(days=shipping_days)
        status = rng.choices(["Delivered", "Processing", "Shipped", "Cancelled", "Returned"], [72, 8, 12, 4, 4])[0]
        rows.append({
            "sale_id": f"SAL-{index:05d}", "order_id": f"ORD-{index // 2 + 1:05d}",
            "order_line": index % 2 + 1, "order_date": order_date.isoformat(), "fiscal_year": year,
            "order_month": order_date.strftime("%Y-%m"), "customer_id": customer[0], "customer_name": customer[1],
            "customer_segment": customer[2], "industry": customer[3], "customer_region": customer[4],
            "country": rng.choice(["United States", "Canada", "United Kingdom", "Germany", "Singapore", "Brazil"]),
            "product_id": product[0], "product_name": product[1], "product_category": product[2],
            "product_subcategory": product[3], "sku": f"SKU-{product[0]}-{rng.randint(1, 4)}", "quantity": quantity,
            "unit_price": unit_price, "discount_pct": discount, "gross_revenue": gross, "net_revenue": net,
            "currency": "USD", "sales_channel": rng.choice(["Direct", "Partner", "Online", "Marketplace"]),
            "sales_rep": rng.choice(["A. Patel", "J. Chen", "M. Garcia", "S. Johnson", "R. Kim"]),
            "unit_cost": round(cost / quantity, 2), "gross_margin": round(net - cost, 2),
            "payment_method": rng.choice(["Credit Card", "Wire Transfer", "ACH", "Net Terms"]),
            "payment_status": rng.choices(["Paid", "Pending", "Overdue", "Refunded"], [70, 15, 10, 5])[0],
            "ship_date": ship_date.isoformat(), "delivery_date": (ship_date + timedelta(days=rng.randint(1, 8))).isoformat() if status == "Delivered" else "",
            "fulfillment_status": status, "warehouse": rng.choice(["WH-East", "WH-West", "WH-Central", "WH-EU"]),
            "return_quantity": rng.randint(1, quantity) if status == "Returned" else 0,
            "contract_type": rng.choice(["Standard", "Annual", "Multi-year"]),
        })
    return rows


def generate_procurement(rng: random.Random, count: int = 2000) -> list[dict[str, object]]:
    rows = []
    for index in range(1, count + 1):
        year = rng.choice([2023, 2024, 2025, 2026])
        po_date = random_date(rng, year)
        supplier = rng.choice(SUPPLIERS)
        product = rng.choice(PRODUCTS)
        # Keep purchase quantities in a plausible range relative to sales lines so
        # generated annual procurement spend is comparable to, not wildly above, sales.
        quantity = rng.randint(5, 70)
        unit_cost = round(product[4] * rng.uniform(0.38, 0.72), 2)
        total = round(quantity * unit_cost, 2)
        lead_days = max(1, int(rng.gauss(24, 9)))
        expected = po_date + timedelta(days=lead_days)
        late_days = rng.choices([-5, 0, 3, 8, 15], [20, 42, 20, 12, 6])[0]
        actual = expected + timedelta(days=late_days)
        status = rng.choices(["Received", "In Transit", "Open", "Cancelled", "Closed"], [62, 13, 15, 3, 7])[0]
        received = quantity if status in ("Received", "Closed") else (rng.randint(0, quantity) if status == "In Transit" else 0)
        rows.append({
            "po_id": f"PO-{index:05d}", "po_line": 1, "requisition_id": f"REQ-{index:05d}",
            "po_date": po_date.isoformat(), "fiscal_year": year, "expected_delivery_date": expected.isoformat(),
            "actual_delivery_date": actual.isoformat() if status in ("Received", "Closed") else "",
            "supplier_id": supplier[0], "supplier_name": supplier[1], "supplier_category": supplier[2],
            "supplier_region": supplier[3], "supplier_tier": supplier[4], "item_id": product[0],
            "item_name": product[1], "item_category": product[2], "item_subcategory": product[3],
            "quantity_ordered": quantity, "quantity_received": received, "unit_cost": unit_cost,
            "line_total": total, "currency": "USD", "payment_terms": rng.choice(["Net 30", "Net 45", "Net 60", "2/10 Net 30"]),
            "buyer_name": rng.choice(["L. Morgan", "D. Singh", "K. Wilson", "P. Nguyen", "T. Brown"]),
            "purchase_status": status, "invoice_status": rng.choices(["Matched", "Pending", "Disputed", "Not Received"], [55, 25, 5, 15])[0],
            "quality_rating": round(rng.uniform(2.8, 5.0), 1), "lead_time_days": lead_days,
            "late_days": max(0, late_days) if status in ("Received", "Closed") else 0,
            "on_time": status in ("Received", "Closed") and late_days <= 0,
            "cost_center": rng.choice(["Operations", "Technology", "Sales", "Facilities", "Research"]),
            "ship_to_region": rng.choice(["North America", "Europe", "Asia Pacific"]),
            "contract_flag": rng.choice([True, True, False]), "tax_amount": round(total * rng.choice([0, 0.05, 0.08, 0.1]), 2),
            "freight_amount": round(rng.uniform(0, 450), 2), "payment_status": rng.choices(["Paid", "Scheduled", "Overdue"], [62, 28, 10])[0],
        })
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261005)
    sales = generate_sales(rng)
    procurement = generate_procurement(rng)
    write_csv(DATA_DIR / "sales.csv", list(sales[0]), sales)
    write_csv(DATA_DIR / "procurement.csv", list(procurement[0]), procurement)
    print(f"Generated {len(sales):,} sales rows and {len(procurement):,} procurement rows in {DATA_DIR}.")


if __name__ == "__main__":
    main()
