-- ============================================================
-- Catalog admin fields — enables adding items + managing photos from the
-- Owner Dashboard. After this, the DB is the full source of truth for the
-- catalog (products.json becomes a generated snapshot).
--
-- Adds: position (display order), published (draft gate), ai_note (the
-- "shown on an AI model" caption — replaces the old images/featured/ path
-- guess). Constrains tier/collection to the storefront's fixed taxonomy.
-- Restricts the public read to published rows only. Adds an atomic
-- position-swap RPC for the reorder arrows, and a trigger that auto-places
-- new drafts at the end.
--
-- Apply order: this (schema) -> 20260819120001 (data backfill) -> 20260819120002 (storage).
-- ============================================================

alter table public.products
  add column if not exists position   int,
  add column if not exists published  boolean not null default false,
  add column if not exists ai_note    boolean not null default false;

-- Fixed taxonomy — free text here would make an item unfilterable / mis-headed
-- on the storefront (docs/js/main.js hardcodes these slugs).
alter table public.products
  drop constraint if exists products_tier_check,
  add  constraint products_tier_check check (tier in ('signature','everyday'));
alter table public.products
  drop constraint if exists products_collection_check,
  add  constraint products_collection_check
    check (collection is null or collection in
      ('sea-shore','stone-earth','pearl-crystal','hand-woven'));

-- New drafts auto-place at the end (gaps of 10 so reordering never renumbers).
create or replace function public.products_default_position()
returns trigger language plpgsql as $$
begin
  if new.position is null then
    select coalesce(max(position), 0) + 10 into new.position from public.products;
  end if;
  return new;
end $$;

drop trigger if exists products_set_position on public.products;
create trigger products_set_position before insert on public.products
  for each row execute function public.products_default_position();

-- Public sees only PUBLISHED rows (drafts stay private). The owner still sees
-- everything via the owner FOR ALL policy (20260818120000), which is OR'd in.
drop policy if exists products_public_read on public.products;
create policy products_public_read on public.products
  for select using (published = true);

-- Atomic swap for the reorder arrows (SECURITY INVOKER: owner RLS applies, so
-- only the owner can actually move rows). Both updates commit together.
-- Returns the number of rows the caller actually moved (0 when RLS blocks a
-- non-owner) so the client can tell a real swap from a silent no-op; raises if
-- either SKU is missing (so a bad call never nulls a live position).
create or replace function public.swap_positions(sku_a text, sku_b text)
returns int language plpgsql as $$
declare pa int; pb int; n int;
begin
  select position into pa from public.products where sku = sku_a;
  select position into pb from public.products where sku = sku_b;
  if pa is null or pb is null then
    raise exception 'swap_positions: unknown or unreadable sku (% / %)', sku_a, sku_b;
  end if;
  update public.products set position = pb where sku = sku_a;
  get diagnostics n = row_count;
  update public.products set position = pa where sku = sku_b;
  return n;
end $$;

revoke execute on function public.swap_positions(text, text) from public;
grant  execute on function public.swap_positions(text, text) to authenticated;
