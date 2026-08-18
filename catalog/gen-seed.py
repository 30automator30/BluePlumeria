#!/usr/bin/env python3
"""
Generate supabase/seed_products.sql from catalog/products.json.

Idempotent: re-running the SQL upserts by SKU, so it's safe to apply
any number of times as the catalog grows. products.json stays the
single source of truth; this just projects it into the database.

Run:  python catalog/gen-seed.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "catalog" / "products.json").read_text(encoding="utf-8"))
products = data["products"]

def q(v):
    """SQL-quote a text value (or NULL)."""
    if v is None:
        return "null"
    return "'" + str(v).replace("'", "''") + "'"

def jsonb(v):
    return "'" + json.dumps(v, ensure_ascii=False).replace("'", "''") + "'::jsonb"

rows = []
for p in products:
    rows.append(
        "  (" + ", ".join([
            q(p["sku"]), q(p["name"]), f"{p['price']:.2f}", q(p.get("collection")),
            q(p.get("tier") or "signature"), q(p.get("label")),
            str(p.get("maxQuantity", 1)), q(p.get("itemUrl") or "/shop.html"),
            q(p.get("cartImage")), jsonb(p.get("images", [])),
            "true" if p.get("available", True) else "false",
        ]) + ")"
    )

sql = (
    "-- Generated from catalog/products.json — do not edit by hand.\n"
    "-- Regenerate with: python catalog/gen-seed.py\n\n"
    "insert into public.products\n"
    "  (sku, name, price, collection, tier, label, max_quantity, item_url, cart_image, images, available)\n"
    "values\n" + ",\n".join(rows) + "\n"
    "on conflict (sku) do update set\n"
    "  name = excluded.name, price = excluded.price, collection = excluded.collection,\n"
    "  tier = excluded.tier, label = excluded.label, max_quantity = excluded.max_quantity,\n"
    "  item_url = excluded.item_url, cart_image = excluded.cart_image,\n"
    "  images = excluded.images, available = excluded.available;\n"
)

out = ROOT / "supabase" / "seed_products.sql"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(sql, encoding="utf-8")
print(f"Wrote {out.relative_to(ROOT)} ({len(products)} products)")
