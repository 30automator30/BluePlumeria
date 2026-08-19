# Catalog pipeline

The storefront (`docs/shop.html`) is static — Snipcart charges the price baked
into that HTML and validates it by crawling the page **without running
JavaScript**. So every price must physically be in the static file. This
pipeline keeps that true while letting the owner edit prices from the admin.

## Who owns what

| Field | Edited where | Source of truth |
|-------|--------------|-----------------|
| **price, availability, name, label** | Owner Dashboard (`docs/admin.html`) | Supabase `products` |
| **images, collection, tier, order, which pieces exist** | `catalog/products.json` (git) | git |

## How an admin edit reaches the live store

```
Owner edits price in admin  ─▶  Supabase products table
                                      │
        GitHub Action "Publish store" │ (every ~10 min, or "Run workflow" now)
                                      ▼
  sync-from-supabase.py  ─▶  overlay price/availability/name/label onto products.json
  gen-shop.py            ─▶  regenerate the cards in docs/shop.html
  gen-seed.py            ─▶  refresh supabase/seed_products.sql
                                      │
                          commit (only if changed) ─▶ GitHub Pages redeploys
```

`sync-from-supabase.py` is an **overlay**: it only rewrites the four
owner-editable fields on pieces already in `products.json`. It never adds,
removes, or reorders pieces and never touches images — so an incomplete or
extra database row can't corrupt the storefront, and pieces you add by hand
are never erased.

## Adding or removing a piece (structural — still a git change)

1. Edit `catalog/products.json` (add/remove the piece; set its images,
   collection, tier, and an initial price).
2. `python catalog/gen-shop.py && python catalog/gen-seed.py`
3. Commit and push.
4. For a **new** piece, also insert it into Supabase so the admin can manage it:
   run the `insert … on conflict` in `supabase/seed_products.sql` in the
   Supabase SQL editor. (The seed is structure-only on conflict, so this will
   **not** overwrite prices/availability the owner has already set.)

## Scripts

- `sync-from-supabase.py` — pull live price/availability/name/label from
  Supabase onto `products.json` (public read key; no secret).
- `gen-shop.py` — regenerate the product cards in `docs/shop.html`.
- `gen-seed.py` — regenerate `supabase/seed_products.sql`.
- `extract-from-shop.py` — one-time bootstrap that built `products.json` from
  the original hand-written `shop.html` (kept for reference).

## One-time setup (required before enabling auto-publish)

1. **Apply the seed to production** once so Supabase matches git:
   run `supabase/seed_products.sql` in the Supabase SQL editor.
2. **Lock down writes**: apply `supabase/migrations/20260818120000_pin_owner_email.sql`
   (confirm the email in it is your Supabase login), and disable email
   sign-ups in Supabase → Authentication.

## Notes

- Generated files are pinned to LF (`.gitattributes`) so local (Windows) and CI
  (Linux) regeneration produce identical bytes — no line-ending churn commits.
- GitHub disables scheduled workflows after 60 days with no repo activity. If
  the shop goes quiet that long, re-enable "Publish store" in the Actions tab
  (the manual "Run workflow" button always works).
