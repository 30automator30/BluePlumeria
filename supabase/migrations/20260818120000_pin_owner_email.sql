-- ============================================================
-- Pin all owner-only policies to the real admin login email.
--
-- WHY THIS EXISTS (security-critical):
--   * The initial schema granted products WRITE to ANY authenticated user
--     (`using (true)`). With the admin price editor + auto-publish pipeline,
--     that would let any signed-up account change the price customers are
--     actually charged. This restricts writes to the single owner account.
--   * The earlier 20260818000000 migration did this correctly but left the
--     email as the placeholder 'you@example.com'. This migration supersedes
--     it with the real address so there is no placeholder to forget.
--
-- >>> BEFORE APPLYING: confirm the email below is the exact address you log
--     into the Owner Dashboard (Supabase Auth) with. A wrong value either
--     locks you out of writes or leaves the door open. Then run this in the
--     Supabase SQL Editor. Also confirm email sign-ups are DISABLED in
--     Supabase → Authentication → Providers. <<<
-- ============================================================

-- The one admin account. Everything below keys off this.
-- (Change here if your Supabase login email differs from your contact email.)
do $$
declare owner_email text := 'desmitdesignz@gmail.com';
begin
  -- Products: world-readable (storefront needs it), owner-only write.
  execute 'drop policy if exists products_owner_write on public.products';
  execute format(
    'create policy products_owner_write on public.products '
    'for all to authenticated using (%L = (auth.jwt() ->> ''email'')) '
    'with check (%L = (auth.jwt() ->> ''email''))',
    owner_email, owner_email);

  -- Orders / items / inquiries: owner-only reads (they hold customer PII).
  execute 'drop policy if exists orders_owner_read on public.orders';
  execute format(
    'create policy orders_owner_read on public.orders '
    'for select to authenticated using (%L = (auth.jwt() ->> ''email''))',
    owner_email);

  execute 'drop policy if exists order_items_owner_read on public.order_items';
  execute format(
    'create policy order_items_owner_read on public.order_items '
    'for select to authenticated using (%L = (auth.jwt() ->> ''email''))',
    owner_email);

  execute 'drop policy if exists inquiries_owner_read on public.inquiries';
  execute format(
    'create policy inquiries_owner_read on public.inquiries '
    'for select to authenticated using (%L = (auth.jwt() ->> ''email''))',
    owner_email);
end $$;
