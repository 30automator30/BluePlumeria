-- ============================================================
-- Lock owner data to a single admin account (defense in depth).
--
-- The initial policies allowed ANY authenticated user to read orders.
-- That's only safe if signups are disabled. This narrows reads/writes
-- to one admin email so even if a stray account exists, it sees nothing.
--
-- >>> REPLACE 'you@example.com' (4 places) with your admin login email,
--     then run this in the Supabase SQL Editor. <<<
-- ============================================================

-- Orders — owner-only read (writes come from the webhook's service_role,
-- which bypasses RLS, so no insert policy needed).
drop policy if exists orders_owner_read on public.orders;
create policy orders_owner_read on public.orders
  for select to authenticated
  using ((auth.jwt() ->> 'email') = 'you@example.com');

drop policy if exists order_items_owner_read on public.order_items;
create policy order_items_owner_read on public.order_items
  for select to authenticated
  using ((auth.jwt() ->> 'email') = 'you@example.com');

drop policy if exists inquiries_owner_read on public.inquiries;
create policy inquiries_owner_read on public.inquiries
  for select to authenticated
  using ((auth.jwt() ->> 'email') = 'you@example.com');

-- Products stay world-readable (storefront needs them); only the owner
-- may write (e.g. the dashboard's mark-sold / mark-available toggle).
drop policy if exists products_owner_write on public.products;
create policy products_owner_write on public.products
  for all to authenticated
  using ((auth.jwt() ->> 'email') = 'you@example.com')
  with check ((auth.jwt() ->> 'email') = 'you@example.com');
