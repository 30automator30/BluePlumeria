#!/usr/bin/env python3
"""
Generate the product grid in docs/shop.html from catalog/products.json.

products.json is a generated snapshot of the Supabase catalog (see
sync-from-supabase.py). This script projects it into the storefront: it
rewrites ONLY the cards between the BEGIN/END markers inside
<section class="showcase-section">, leaving the head, nav, footer, Snipcart
loader, and the live sold-out script untouched.

Because Snipcart re-crawls the page and validates the cart price against the
HTML it sees, the price baked here IS the price customers pay.

Each product carries everything the card needs — no path heuristics:
  * images: [{ "url", "alt" }, ...]   (>=2 -> slider, <=1 -> single photo)
  * aiNote: bool                       (show the "shown on an AI model" note + lazy-load)
  * tier == "everyday"                 -> single-photo card, grouped under its own banner
Only published + complete pieces reach products.json, so every entry renders.

Run:  python catalog/gen-shop.py
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
    """Escape a text/attribute value for HTML. Unicode (—, é, ') is left
    literal — it renders identically and keeps the source readable."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def price_display(price):
    """$195 for whole amounts, $42.50 for fractional ones."""
    return f"{int(price)}" if float(price) == int(price) else f"{price:.2f}"


def card(p):
    sku = p["sku"]
    name_html = esc(p["name"])
    label = esc(p.get("label") or "")
    collection = p.get("collection") or ""
    tier = p.get("tier") or "signature"
    tier_attr = ' data-tier="everyday"' if tier == "everyday" else ""
    maxq = p.get("maxQuantity", 1)
    item_url = p.get("itemUrl") or "/shop.html"
    images = p.get("images") or []
    cart_img = esc(p.get("cartImage") or (images[0]["url"] if images else ""))
    ai_note = bool(p.get("aiNote"))
    single = tier == "everyday" or len(images) <= 1
    lazy = ' loading="lazy"' if (ai_note or tier == "everyday") else ""

    button = (
        f'          <div class="showcase-cta">\n'
        f'            <button class="snipcart-add-item btn btn-primary" data-item-max-quantity="{maxq}"\n'
        f'              data-item-id="{esc(sku)}" data-item-name="{name_html}"\n'
        f'              data-item-price="{float(p["price"]):.2f}" data-item-url="{esc(item_url)}"\n'
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

    if single:
        im = images[0] if images else {"url": p.get("cartImage") or "", "alt": p["name"]}
        media = (
            f'        <div class="showcase-media">\n'
            f'          <div class="showcase-slider">\n'
            f'            <img src="{esc(im["url"])}" alt="{esc(im.get("alt") or "")}" class="active"{lazy}>\n'
            f'          </div>\n'
            f'        </div>'
        )
    else:
        slides = []
        for i, im in enumerate(images):
            active = ' class="active"' if i == 0 else ''
            slides.append(
                f'            <img src="{esc(im["url"])}" alt="{esc(im.get("alt") or "")}"{active}{lazy}>')
        dots = "".join(
            f'            <button class="slider-dot{" active" if i == 0 else ""}" '
            f'aria-label="Slide {i + 1}"></button>\n'
            for i in range(len(images)))
        ai = (f'\n          <p class="ai-note">{AI_NOTE}</p>' if ai_note else "")
        media = (
            f'        <div class="showcase-media">\n'
            f'          <div class="showcase-slider" data-slider>\n'
            + "\n".join(slides) + "\n"
            f'            <button class="slider-btn prev" aria-label="Previous">&lsaquo;</button>\n'
            f'            <button class="slider-btn next" aria-label="Next">&rsaquo;</button>\n'
            f'          </div>\n'
            f'          <div class="slider-dots">\n'
            f'{dots}'
            f'          </div>{ai}\n'
            f'        </div>'
        )

    # Name and SKU are owner-editable free text; neutralize anything that could
    # close the HTML comment early (`-->`, stray `>`, `--`) so it can't corrupt
    # the page. (SKU is also esc()'d in the attributes above.)
    safe = lambda s: re.sub(r"-{2,}", "-", str(s)).replace(">", "")
    return (
        f'      <!-- {safe(p["name"])} ({safe(sku)}) -->\n'
        f'      <div class="showcase-set reveal" data-collection="{esc(collection)}"{tier_attr}>\n'
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

    region = re.compile(
        r'(<section class="showcase-section[^"]*">\s*<div class="container">\n)'
        r'(?:.*?)'
        r'(\n[ \t]*</div>\s*</section>)',
        re.S,
    )
    if not region.search(html):
        raise SystemExit("Could not locate the showcase-section card region in shop.html")
    html = region.sub(lambda m: m.group(1) + body + m.group(2), html, count=1)

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
