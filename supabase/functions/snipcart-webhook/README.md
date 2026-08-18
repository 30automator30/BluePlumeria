# Snipcart order webhook

Records completed Snipcart orders into the Blue Plumeria database
(`orders`, `order_items`) and marks one-of-a-kind pieces sold. Snipcart
and Stripe handle payment; this only records the paid result.

## One-time setup (you run these — secrets stay on your side)

1. **Find your Snipcart secret key**
   Snipcart dashboard → Account → API Keys → **Secret API Key**.

2. **Store it as a function secret** (from `I:\BluePlumeria`):
   ```
   supabase secrets set SNIPCART_SECRET_API_KEY=sk_xxx_your_secret_key
   ```
   `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are injected automatically.

3. **Deploy the function** (JWT off — Snipcart has no Supabase token):
   ```
   supabase functions deploy snipcart-webhook --no-verify-jwt
   ```
   Endpoint becomes:
   `https://ktjxrxchrxtmyvlfsyof.supabase.co/functions/v1/snipcart-webhook`

4. **Point Snipcart at it**
   Snipcart dashboard → Account → **Webhooks** → set the order-events
   endpoint URL to the function URL above.

## Test it safely

- Put Snipcart in **Test** mode and place a test order (use Stripe test
  card `4242 4242 4242 4242`).
- The webhook records it with `status = 'test'` so test orders are easy
  to tell apart and delete later.
- Verify in the Supabase SQL Editor:
  ```sql
  select placed_at, customer_name, total, status from public.orders
  order by placed_at desc limit 5;
  ```

## How verification works

Snipcart sends `X-Snipcart-RequestToken`. The function calls
`GET https://app.snipcart.com/api/requestvalidation/{token}` using your
secret key. Only a genuine Snipcart request validates — spoofed calls
get `401`. The function is idempotent on the order token, so Snipcart's
automatic retries never create duplicate orders.
