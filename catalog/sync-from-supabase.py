#!/usr/bin/env python3
"""
Overlay the owner-editable fields from Supabase onto catalog/products.json.

Ownership split (see catalog/README.md):
  * STRUCTURE — which pieces exist, their order, images, collection, tier —
    lives in products.json (git). Adding/removing a piece is a git change.
  * LIVE STATE — price, availability, name, label — is edited by the owner in
    the admin (docs/admin.html) and stored in Supabase.

This script reads the live state from Supabase and writes ONLY those fields
back onto the pieces already in products.json. It never adds, removes, or
reorders pieces, and never touches images — so an incomplete or extra DB row
can't corrupt the storefront, and hand-added pieces are never erased.

Reads with the public anon key (products are world-readable via RLS), so no
secret is required — this runs in CI with nothing but the key below.

Run:  python catalog/sync-from-supabase.py
Then: python catalog/gen-shop.py  &&  python catalog/gen-seed.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "catalog" / "products.json"

SB_URL = "https://ktjxrxchrxtmyvlfsyof.supabase.co"
SB_ANON = "sb_publishable_8NSnU6lGLIt9GplZ-7hHUw_XvtDmX2s"

# Fields the owner edits in the admin; the DB is authoritative for these.
OVERLAY = ("price", "available", "name", "label")


def fetch_products():
    url = (f"{SB_URL}/rest/v1/products"
           f"?select=sku,{','.join(OVERLAY)}&limit=2000")
    req = urllib.request.Request(url, headers={
        "apikey": SB_ANON,
        "Authorization": f"Bearer {SB_ANON}",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"Unexpected Supabase response: {rows!r}")
    return {r["sku"]: r for r in rows}


def main():
    db = fetch_products()
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    products = doc["products"]

    changed, missing = 0, []
    for p in products:
        row = db.get(p["sku"])
        if not row:
            missing.append(p["sku"])
            continue
        for f in OVERLAY:
            new = row.get(f)
            if f == "price":
                new = float(new)
            if p.get(f) != new:
                p[f] = new
                changed += 1

    for sku in missing:
        print(f"WARN: {sku} is in products.json but not in Supabase — "
              f"keeping its committed values.", file=sys.stderr)

    # Write LF explicitly (not the OS default) so a run on Windows and a run
    # on the Linux CI runner produce byte-identical output — no churn commits.
    # Pinned to LF in .gitattributes to match.
    DATA.write_bytes(
        (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(f"Synced {len(products)} pieces from Supabase "
          f"({changed} field value(s) updated, {len(missing)} not in DB).")


if __name__ == "__main__":
    main()
