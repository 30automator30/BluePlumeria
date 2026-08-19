#!/usr/bin/env python3
"""
Generate the product grid in docs/shop.html from catalog/products.json.

products.json is the single source of truth. This script projects it into
the storefront: it rewrites ONLY the cards between the BEGIN/END markers
inside <section class="showcase-section">, leaving the head, nav, footer,
Snipcart loader, and the live sold-out script untouched.

Because Snipcart re-crawls the page and validates the cart price against
the HTML it sees, the price baked here IS the price customers pay — so the
storefront must always be regenerated after editing products.json.

Card template is chosen per piece, entirely from existing fields:
  * tier == "everyday"                    -> single-photo card (no slider)
  * image path under images/featured/     -> slider + AI-model note + lazy
  * otherwise (Studio pieces)             -> slider, no note, eager-loaded

Run:  python catalog/gen-shop.py
Then also refresh the DB seed:  python catalog/gen-seed.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOP = ROOT / "docs" / "shop.html"
DATA = ROOT / "catalog" / "products.json"

BEGIN = "<!-- BEGIN generated cards — do not edit by hand. Regenerate: python catalog/gen-shop.py -->"
END = "<!-- END generated cards -->"
AI_NOTE = ('Shown on an AI model to illustrate styling &mdash; '
           "swipe for a photo of the actual piece.")
EVERYDAY_DIVIDER = (
    '      <!-- ═══ EVERYDAY tier — data-tier="everyday" renders these under their own\n'
    '           "Everyday Collection" banner. Same card format as above, single photo. ═══ -->'
)


def esc(s):
    """Escape a text/attribute value for HTML. Unicode (—, é) is left
    literal — it renders identically and keeps the source readable."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def price_display(price):
    """$195 for whole amounts, $42.50 for fractional ones."""
    return f"{int(price)}" if float(price) == int(price) else f"{price:.2f}"


def kind(p):
    if p.get("tier") == "everyday":
        return "everyday"
    imgs = p.get("images") or []
    return "featured" if imgs and imgs[0].startswith("images/featured/") else "studio"


def card(p):
    sku = p["sku"]
    name_html = esc(p["name"])
    alt = esc(p["name"])
    label = esc(p.get("label") or "")
    collection = p.get("collection") or ""
    tier = p.get("tier") or "signature"
    tier_attr = ' data-tier="everyday"' if tier == "everyday" else ""
    maxq = p.get("maxQuantity", 1)
    item_url = p.get("itemUrl") or "/shop.html"
    cart_img = p.get("cartImage") or (p["images"][0] if p.get("images") else "")
    k = kind(p)

    # Cart button — identical across templates; this is the Snipcart contract.
    button = (
        f'          <div class="showcase-cta">\n'
        f'            <button class="snipcart-add-item btn btn-primary" data-item-max-quantity="{maxq}"\n'
        f'              data-item-id="{sku}" data-item-name="{name_html}"\n'
        f'              data-item-price="{float(p["price"]):.2f}" data-item-url="{item_url}"\n'
        f'              data-item-image="{cart_img}">Add to Cart</button>\n'
        f'          </div>'
    )
    info = (
        f'        <div class="showcase-info">\n'
        f'          <span class="showcase-label">{label}</span>\n'
        f'          <h3 class="showcase-name">{name_html}</h3>\n'
        f'          <p class="showcase-price">${price_display(p["price"])}</p>\n'
        f'{button}\n'
        f'        </div>'
    )

    if k == "everyday":
        img = (p["images"][0] if p.get("images") else cart_img)
        media = (
            f'        <div class="showcase-media">\n'
            f'          <div class="showcase-slider">\n'
            f'            <img src="{img}" alt="{alt}" class="active" loading="lazy">\n'
            f'          </div>\n'
            f'        </div>'
        )
    else:
        lazy = ' loading="lazy"' if k == "featured" else ""
        second = "detail" if k == "featured" else "product"
        imgs = p.get("images") or [cart_img]
        # Tolerate a piece with fewer than two images (fall back to the first /
        # cart image) so one incomplete row can never break the whole rebuild.
        worn = imgs[0] if imgs else cart_img
        other = imgs[1] if len(imgs) > 1 else worn
        ai = (f'\n          <p class="ai-note">{AI_NOTE}</p>'
              if k == "featured" else "")
        media = (
            f'        <div class="showcase-media">\n'
            f'          <div class="showcase-slider" data-slider>\n'
            f'            <img src="{worn}" alt="{alt} — worn" class="active"{lazy}>\n'
            f'            <img src="{other}" alt="{alt} — {second}"{lazy}>\n'
            f'            <button class="slider-btn prev" aria-label="Previous">&lsaquo;</button>\n'
            f'            <button class="slider-btn next" aria-label="Next">&rsaquo;</button>\n'
            f'          </div>\n'
            f'          <div class="slider-dots">\n'
            f'            <button class="slider-dot active" aria-label="Slide 1"></button>\n'
            f'            <button class="slider-dot" aria-label="Slide 2"></button>\n'
            f'          </div>{ai}\n'
            f'        </div>'
        )

    # Name is owner-editable free text; neutralize anything that could close
    # the HTML comment early (`-->`, stray `>`, `--`) so it can't corrupt the page.
    comment_name = re.sub(r"-{2,}", "-", p["name"]).replace(">", "")
    return (
        f'      <!-- {comment_name} ({sku}) -->\n'
        f'      <div class="showcase-set reveal" data-collection="{collection}"{tier_attr}>\n'
        f'{media}\n'
        f'{info}\n'
        f'      </div>'
    )


def main():
    products = json.loads(DATA.read_text(encoding="utf-8"))["products"]

    parts, everyday_started = [], False
    for p in products:
        if p.get("tier") == "everyday" and not everyday_started:
            parts.append(EVERYDAY_DIVIDER)
            everyday_started = True
        parts.append(card(p))

    body = (
        f"\n      {BEGIN}\n\n"
        + "\n\n".join(parts)
        + f"\n\n      {END}\n"
    )

    html = SHOP.read_text(encoding="utf-8")

    # Replace only the card region inside the showcase section's container.
    region = re.compile(
        r'(<section class="showcase-section[^"]*">\s*<div class="container">\n)'
        r'(?:.*?)'
        r'(\n[ \t]*</div>\s*</section>)',
        re.S,
    )
    if not region.search(html):
        raise SystemExit("Could not locate the showcase-section card region in shop.html")
    html = region.sub(lambda m: m.group(1) + body + m.group(2), html, count=1)

    # Keep the visible piece count in sync with the catalog.
    html = re.sub(
        r'(<span id="product-count">)\d+(</span>)',
        lambda m: f"{m.group(1)}{len(products)}{m.group(2)}",
        html,
        count=1,
    )

    # LF explicitly so Windows and the Linux CI runner emit identical bytes.
    SHOP.write_bytes(html.encode("utf-8"))
    print(f"Wrote {SHOP.relative_to(ROOT)} ({len(products)} cards)")


if __name__ == "__main__":
    main()
