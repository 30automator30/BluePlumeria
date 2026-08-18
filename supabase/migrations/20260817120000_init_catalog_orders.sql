-- ============================================================
-- Blue Plumeria — backend schema (catalog + orders + inquiries)
--
-- Design notes (this is the teaching part):
--  * Snipcart stays the payment processor. We NEVER store card data.
--    This DB is the OWNER'S system of record: catalog, real orders
--    (captured from Snipcart webhooks), inventory state, and inquiries.
--  * Every table has Row-Level Security ON. Nothing is readable by
--    default; each policy grants the minimum needed:
--      - products  : world-readable (the storefront needs it), owner-writable
--      - orders/items/inquiries : owner-only (they hold customer PII);
--        the webhook writes via the service_role key, which BYPASSES RLS.
--  * The webhook is idempotent on snipcart_token so a retried delivery
--    can't create a duplicate order.
-- ============================================================

-- updated_at helper -------------------------------------------------
create or replace function public.touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

-- CATALOG -----------------------------------------------------------
create table if not exists public.products (
  sku           text primary key,
  name          text not null,
  price         numeric(10,2) not null check (price >= 0),
  collection    text,
  tier          text not null default 'signature',
  label         text,
  max_quantity  int  not null default 1 check (max_quantity >= 0),
  item_url      text not null default '/shop.html',
  cart_image    text,
  images        jsonb not null default '[]'::jsonb,
  available     boolean not null default true,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create trigger products_touch before update on public.products
  for each row execute function public.touch_updated_at();

-- ORDERS (captured from Snipcart) -----------------------------------
create table if not exists public.orders (
  id              uuid primary key default gen_random_uuid(),
  snipcart_token  text unique not null,          -- idempotency key
  invoice_number  text,
  email           text,
  customer_name   text,
  shipping        jsonb,                          -- address block
  subtotal        numeric(10,2),
  shipping_cost   numeric(10,2),
  taxes           numeric(10,2),
  total           numeric(10,2),
  currency        text default 'USD',
  status          text not null default 'paid',
  placed_at       timestamptz,
  raw             jsonb,                          -- full payload, for audit
  created_at      timestamptz not null default now()
);

create index if not exists orders_placed_at_idx on public.orders (placed_at desc);

create table if not exists public.order_items (
  id          uuid primary key default gen_random_uuid(),
  order_id    uuid not null references public.orders(id) on delete cascade,
  sku         text,
  name        text,
  unit_price  numeric(10,2),
  quantity    int not null default 1,
  line_total  numeric(10,2)
);

create index if not exists order_items_order_idx on public.order_items (order_id);
create index if not exists order_items_sku_idx   on public.order_items (sku);

-- INQUIRIES (contact / commission intake — replaces dead Formspree) --
create table if not exists public.inquiries (
  id          uuid primary key default gen_random_uuid(),
  source      text not null default 'blueplumeria',   -- or 'desmitdesigns'
  name        text,
  email       text,
  message     text,
  status      text not null default 'new',
  created_at  timestamptz not null default now()
);

create index if not exists inquiries_created_idx on public.inquiries (created_at desc);

-- ROW-LEVEL SECURITY -------------------------------------------------
alter table public.products    enable row level security;
alter table public.orders      enable row level security;
alter table public.order_items enable row level security;
alter table public.inquiries   enable row level security;

-- products: anyone may read the catalog; only signed-in owner may write.
drop policy if exists products_public_read on public.products;
create policy products_public_read on public.products
  for select using (true);

drop policy if exists products_owner_write on public.products;
create policy products_owner_write on public.products
  for all to authenticated using (true) with check (true);

-- orders / order_items / inquiries: owner-only reads (hold PII).
-- Writes come from the webhook using the service_role key, which
-- bypasses RLS entirely, so no INSERT policy is needed for it.
drop policy if exists orders_owner_read on public.orders;
create policy orders_owner_read on public.orders
  for select to authenticated using (true);

drop policy if exists order_items_owner_read on public.order_items;
create policy order_items_owner_read on public.order_items
  for select to authenticated using (true);

drop policy if exists inquiries_owner_read on public.inquiries;
create policy inquiries_owner_read on public.inquiries
  for select to authenticated using (true);

-- Public may CREATE an inquiry (the contact form), but never read them.
drop policy if exists inquiries_public_insert on public.inquiries;
create policy inquiries_public_insert on public.inquiries
  for insert to anon with check (true);
