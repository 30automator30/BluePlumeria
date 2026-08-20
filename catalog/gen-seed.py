#!/usr/bin/env python3
"""
Generate supabase/seed_products.sql from catalog/products.json.

This is a BOOTSTRAP / disaster-recovery snapshot only. The DB is now the full
source of truth for the catalog (owner adds items + edits everything in the
admin), so the seed inserts NEW rows but does NOTHING on conflict — it must
never overwrite live admin-owned data (price, availability, images, order…).
Restores come from Supabase backups, not from re-applying this seed.

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
            "true" if p.get("aiNote") else "false",
            str(p.get("position", "null")) if p.get("position") is not None else "null",
            "true" if p.get("published", True) else "false",
        ]) + ")"
    )

sql = (
    "-- Generated from catalog/products.json — do not edit by hand.\n"
    "-- Regenerate with: python catalog/gen-seed.py\n"
    "-- BOOTSTRAP ONLY: inserts new rows, DOES NOTHING on conflict. The DB is\n"
    "-- the source of truth; this must never overwrite live admin-owned data.\n\n"
    "insert into public.products\n"
    "  (sku, name, price, collection, tier, label, max_quantity, item_url,\n"
    "   cart_image, images, available, ai_note, position, published)\n"
    "values\n" + ",\n".join(rows) + "\n"
    "on conflict (sku) do nothing;\n"
)

out = ROOT / "supabase" / "seed_products.sql"
out.parent.mkdir(parents=True, exist_ok=True)
# LF explicitly so Windows and the Linux CI runner emit identical bytes.
out.write_bytes(sql.encode("utf-8"))
print(f"Wrote {out.relative_to(ROOT)} ({len(products)} products)")
