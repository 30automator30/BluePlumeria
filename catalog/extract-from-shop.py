#!/usr/bin/env python3
"""
Blue Plumeria — one-time catalog extractor.

Reads the live, hand-authored docs/shop.html and lifts every product
(.showcase-set) into a structured catalog/products.json — the single
source of truth the shop grid and the Snipcart validation feed will be
generated FROM going forward. Lossless: captures the Snipcart item
fields, collection/tier, visible label, and slider images.

Run:  python catalog/extract-from-shop.py
"""
import json
import re
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOP = ROOT / "docs" / "shop.html"
OUT = ROOT / "catalog" / "products.json"

# Corrections applied on top of the raw live data (documented, not silent):
#   the live shop.html stored a broken character for this piece.
NAME_FIXES = {
    "BP-STUDIO-003": "Blue Ombré Crystal Necklace",
}

html = SHOP.read_text(encoding="utf-8", errors="replace")

# Each product card is <div class="showcase-set ..."> ... </div> closing
# after the .showcase-info block. Slice generously and rely on data-item-id
# to mark a real product.
set_re = re.compile(r'<div class="showcase-set[^"]*"(.*?)</div>\s*</div>\s*</div>', re.S)

def grab(pattern, s, group=1):
    m = re.search(pattern, s, re.S)
    return unescape(m.group(group)).strip() if m else None

products = []
for block in set_re.findall(html):
    sku = grab(r'data-item-id="([^"]*)"', block)
    if not sku:
        continue
    name = NAME_FIXES.get(sku) or grab(r'data-item-name="([^"]*)"', block)
    imgs = [unescape(m) for m in re.findall(r'<img src="(images/[^"]+)"', block)]
    products.append({
        "sku": sku,
        "name": name,
        "price": float(grab(r'data-item-price="([^"]*)"', block) or 0),
        "collection": grab(r'data-collection="([^"]*)"', block),
        "tier": grab(r'data-tier="([^"]*)"', block) or "signature",
        "label": grab(r'<span class="showcase-label">([^<]*)</span>', block),
        "maxQuantity": int(grab(r'data-item-max-quantity="([^"]*)"', block) or 1),
        "itemUrl": grab(r'data-item-url="([^"]*)"', block) or "/shop.html",
        "cartImage": grab(r'data-item-image="([^"]*)"', block),
        "images": imgs,
        "available": True,
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"products": products}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Extracted {len(products)} products -> {OUT.relative_to(ROOT)}")
missing = [p["sku"] for p in products if not (p["name"] and p["price"] and p["label"] and p["images"])]
print("incomplete records:", missing or "none")
