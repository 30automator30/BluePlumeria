-- ============================================================
-- 1) Atomic order recording. The webhook previously wrote orders,
--    order_items, and the sell-through update as three separate REST calls;
--    a failure between them could drop line items and leave a sold
--    one-of-a-kind still `available = true` (double-sell), because the
--    idempotency check keyed only on the order row existing. This records
--    all three in ONE transaction, idempotent on snipcart_token. The webhook
--    (service_role) extracts fields and passes clean JSON.
-- ============================================================
create or replace function public.record_order(p jsonb)
returns text
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_order_id uuid;
  v_token text := p->'order'->>'snipcart_token';
begin
  if v_token is null or v_token = '' then
    raise exception 'record_order: missing snipcart_token';
  end if;

  select id into v_order_id from public.orders where snipcart_token = v_token;
  if found then
    return 'duplicate';
  end if;

  insert into public.orders (
    snipcart_token, invoice_number, email, customer_name, shipping,
    subtotal, shipping_cost, taxes, total, currency, status, placed_at, raw
  ) values (
    v_token,
    p->'order'->>'invoice_number',
    p->'order'->>'email',
    p->'order'->>'customer_name',
    p->'order'->'shipping',
    nullif(p->'order'->>'subtotal','')::numeric,
    nullif(p->'order'->>'shipping_cost','')::numeric,
    nullif(p->'order'->>'taxes','')::numeric,
    nullif(p->'order'->>'total','')::numeric,
    coalesce(p->'order'->>'currency','USD'),
    coalesce(p->'order'->>'status','paid'),
    coalesce(nullif(p->'order'->>'placed_at','')::timestamptz, now()),
    p->'order'->'raw'
  )
  returning id into v_order_id;

  insert into public.order_items (order_id, sku, name, unit_price, quantity, line_total)
  select v_order_id,
         it->>'sku',
         it->>'name',
         nullif(it->>'unit_price','')::numeric,
         coalesce(nullif(it->>'quantity','')::int, 1),
         nullif(it->>'line_total','')::numeric
  from jsonb_array_elements(coalesce(p->'items', '[]'::jsonb)) as it;

  update public.products
     set available = false
   where max_quantity = 1
     and sku in (
       select jsonb_array_elements_text(coalesce(p->'sold_skus', '[]'::jsonb))
     );

  return 'recorded';
end;
$$;

revoke all on function public.record_order(jsonb) from public, anon, authenticated;
grant execute on function public.record_order(jsonb) to service_role;

-- ============================================================
-- 2) Security-advisor hygiene.
-- ============================================================
-- Take the SECURITY DEFINER event-trigger helper off the public REST surface
-- (it only acts inside a DDL event, but shouldn't be anon/authenticated-callable).
revoke all on function public.rls_auto_enable() from public, anon, authenticated;

-- Pin search_path on the flagged functions (non-behavioral hardening).
alter function public.products_default_position() set search_path = pg_catalog, public;
alter function public.swap_positions(text, text) set search_path = pg_catalog, public;
alter function public.touch_updated_at() set search_path = pg_catalog, public;
