-- ============================================================
-- Storage for admin-uploaded product photos.
--
-- Public bucket (storefront + Snipcart load images by public URL); only the
-- owner may write/replace/delete. Objects are written under <sku>/<name> and
-- are write-once (the admin uploads a fresh timestamped key and deletes the
-- old one) so Supabase's ~1h CDN cache never serves a stale image.
-- ============================================================

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('product-images', 'product-images', true, 10485760,
        array['image/jpeg','image/png','image/webp'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- Public read (bucket is public anyway; explicit policy for the objects table).
drop policy if exists product_images_public_read on storage.objects;
create policy product_images_public_read on storage.objects
  for select using (bucket_id = 'product-images');

-- Owner-only write / replace / delete, pinned to the admin email.
drop policy if exists product_images_owner_insert on storage.objects;
create policy product_images_owner_insert on storage.objects
  for insert to authenticated
  with check (bucket_id = 'product-images'
              and (auth.jwt() ->> 'email') = 'desmitdesignz@gmail.com');

drop policy if exists product_images_owner_update on storage.objects;
create policy product_images_owner_update on storage.objects
  for update to authenticated
  using (bucket_id = 'product-images'
         and (auth.jwt() ->> 'email') = 'desmitdesignz@gmail.com');

drop policy if exists product_images_owner_delete on storage.objects;
create policy product_images_owner_delete on storage.objects
  for delete to authenticated
  using (bucket_id = 'product-images'
         and (auth.jwt() ->> 'email') = 'desmitdesignz@gmail.com');
