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
    # STRUCTURE-ONLY upsert. price / available / name / label are owner-editable
    # in the admin and the DB is authoritative for them — re-applying an older
    # seed must NOT revert an admin price change or resurrect a sold-out piece.
    # New pieces still insert their full row (including the initial price/name).
    "on conflict (sku) do update set\n"
    "  collection = excluded.collection, tier = excluded.tier,\n"
    "  max_quantity = excluded.max_quantity, item_url = excluded.item_url,\n"
    "  cart_image = excluded.cart_image, images = excluded.images;\n"
)

out = ROOT / "supabase" / "seed_products.sql"
out.parent.mkdir(parents=True, exist_ok=True)
# LF explicitly so Windows and the Linux CI runner emit identical bytes.
out.write_bytes(sql.encode("utf-8"))
print(f"Wrote {out.relative_to(ROOT)} ({len(products)} products)")
