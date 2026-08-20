#!/usr/bin/env python3
"""
Generate catalog/products.json from the live Supabase catalog.

Since the admin can now add items and manage photos, the DB is the FULL source
of truth for the catalog. This script fully regenerates products.json from it:
only PUBLISHED + COMPLETE pieces, ordered by position. products.json is a
generated snapshot — do not hand-edit it.

Reads with the public anon key (RLS exposes only published rows), so no secret
is required — this runs in CI with nothing but the key below.

SAFETY — floor guard: a bad query, half-applied migration, or accidental mass
unpublish must NOT silently wipe the storefront. If the new published+complete
count is 0, or drops by more than the allowed delta versus the committed
products.json, this exits NON-ZERO (failing the CI run) instead of writing.
Set ALLOW_CATALOG_SHRINK=1 for an intentional bulk removal.

Run:  python catalog/sync-from-supabase.py
Then: python catalog/gen-shop.py  &&  python catalog/gen-seed.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "catalog" / "products.json"

SB_URL = "https://ktjxrxchrxtmyvlfsyof.supabase.co"
SB_ANON = "sb_publishable_8NSnU6lGLIt9GplZ-7hHUw_XvtDmX2s"

# DB (snake) -> products.json (camel). Unlisted fields pass through unchanged.
RENAME = {"max_quantity": "maxQuantity", "item_url": "itemUrl",
          "cart_image": "cartImage", "ai_note": "aiNote"}
KEY_ORDER = ["sku", "name", "price", "collection", "tier", "label", "maxQuantity",
             "itemUrl", "cartImage", "images", "available", "aiNote", "position", "published"]
SELECT = "sku,name,price,collection,tier,label,max_quantity,item_url,cart_image,images,available,ai_note,position,published"


def is_complete(p):
    """Shared definition of 'complete' — MUST match the admin's predicate:
    at least one image, a non-empty name, and a price > 0."""
    return (bool(p.get("images"))
            and bool(str(p.get("name") or "").strip())
            and float(p.get("price") or 0) > 0)


def fetch():
    url = (f"{SB_URL}/rest/v1/products?select={SELECT}"
           f"&published=eq.true&order=position.asc,sku.asc&limit=2000")
    req = urllib.request.Request(url, headers={
        "apikey": SB_ANON, "Authorization": f"Bearer {SB_ANON}",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.loads(r.read().decode("utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"Unexpected Supabase response (aborting, no write): {rows!r}")
    return rows


def to_camel(row):
    p = {RENAME.get(k, k): v for k, v in row.items()}
    p["price"] = float(p["price"])
    p["images"] = [{"url": im.get("url"), "alt": im.get("alt")}
                   for im in (p.get("images") or [])]
    return {k: p[k] for k in KEY_ORDER if k in p}


def main():
    rows = fetch()
    kept = [to_camel(r) for r in rows if is_complete(r)]
    skipped = [r["sku"] for r in rows if not is_complete(r)]
    for sku in skipped:
        print(f"WARN: {sku} is published but incomplete (missing image/name/price) "
              f"— not rendered.", file=sys.stderr)

    # --- Floor guard: refuse to shrink the catalog to zero / drastically ---
    # A drop of >2 pieces in one run is almost always a bad query / mass
    # unpublish, not a real edit. Fixed at 2 (not a percentage) so the guard
    # can't be widened by, then ratchet down through, a growing baseline.
    committed = json.loads(DATA.read_text(encoding="utf-8"))["products"]
    prev, now = len(committed), len(kept)
    allowed_drop = 2
    # Fail-closed: only an explicit affirmative value disables the guard, so a
    # typo'd/unexpected env value ("off", "flase", …) can't silently open it.
    override = os.environ.get("ALLOW_CATALOG_SHRINK", "").strip().lower() in ("1", "true", "yes")
    if not override:
        if now == 0 and prev > 0:
            raise SystemExit(f"ABORT: catalog would drop to 0 (was {prev}). "
                             f"No write. Set ALLOW_CATALOG_SHRINK=1 if intentional.")
        if now < prev - allowed_drop:
            raise SystemExit(f"ABORT: catalog would shrink {prev} -> {now} "
                             f"(allowed drop {allowed_drop}). No write. "
                             f"Set ALLOW_CATALOG_SHRINK=1 if intentional.")

    DATA.write_bytes(
        (json.dumps({"products": kept}, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"Synced {now} published pieces from Supabase "
          f"({len(skipped)} skipped incomplete).")


if __name__ == "__main__":
    main()
