-- ============================================================
-- More than one store owner: an allow-list instead of one pinned email.
--
-- Until now every owner-only policy (product edits, orders, order items,
-- inquiries, product-image uploads) compared the login email to the single
-- literal 'desmitdesignz@gmail.com'. This moves that to a table, bp_owners,
-- and one helper, bp_is_owner(), so adding Amy (or removing someone) is one
-- row — no policy edits.
--
-- Same security model as before:
--   * Only logins whose email is in bp_owners can write products/images or read
--     orders / inquiries (customer PII). Everyone else sees only the public
--     storefront. Email sign-ups must stay DISABLED (Authentication ▸ Providers);
--     owner logins are created by you in Authentication ▸ Users.
--   * bp_owners itself is unreadable/unwritable from the website: manage it
--     here in the SQL Editor only.
--
-- Run in the Blue Plumeria project (ktjxrxchrxtmyvlfsyof) ▸ SQL Editor.
-- Safe to run again.
-- ============================================================

create table if not exists public.bp_owners (
  email    text primary key check (email = lower(email)),
  added_at timestamptz not null default now()
);
alter table public.bp_owners enable row level security;   -- no policies: SQL Editor only
revoke all on public.bp_owners from anon, authenticated;

insert into public.bp_owners (email) values
  ('desmitdesignz@gmail.com'),
  ('amydesmit@gmail.com')
on conflict (email) do nothing;

create or replace function public.bp_is_owner()
returns boolean language sql stable security definer set search_path = '' as $$
  select exists (
    select 1 from public.bp_owners o
    where o.email = lower(coalesce(auth.jwt() ->> 'email', ''))
  );
$$;
revoke all on function public.bp_is_owner() from public;
grant execute on function public.bp_is_owner() to anon, authenticated;

-- Products: world-readable (unchanged), owners write.
drop policy if exists products_owner_write on public.products;
create policy products_owner_write on public.products
  for all to authenticated
  using (public.bp_is_owner()) with check (public.bp_is_owner());

-- Orders / items / inquiries: owners read (customer PII).
drop policy if exists orders_owner_read on public.orders;
create policy orders_owner_read on public.orders
  for select to authenticated using (public.bp_is_owner());

drop policy if exists order_items_owner_read on public.order_items;
create policy order_items_owner_read on public.order_items
  for select to authenticated using (public.bp_is_owner());

drop policy if exists inquiries_owner_read on public.inquiries;
create policy inquiries_owner_read on public.inquiries
  for select to authenticated using (public.bp_is_owner());

-- Product images: public read (unchanged), owners upload / replace / delete.
drop policy if exists product_images_owner_insert on storage.objects;
create policy product_images_owner_insert on storage.objects
  for insert to authenticated
  with check (bucket_id = 'product-images' and public.bp_is_owner());

drop policy if exists product_images_owner_update on storage.objects;
create policy product_images_owner_update on storage.objects
  for update to authenticated
  using (bucket_id = 'product-images' and public.bp_is_owner());

drop policy if exists product_images_owner_delete on storage.objects;
create policy product_images_owner_delete on storage.objects
  for delete to authenticated
  using (bucket_id = 'product-images' and public.bp_is_owner());

-- Check: both owners listed, and no policy still pins a single email.
select email from public.bp_owners order by email;
select schemaname, tablename, policyname from pg_policies
 where qual ilike '%desmitdesignz%' or with_check ilike '%desmitdesignz%';
