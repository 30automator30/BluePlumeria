// ============================================================
// Blue Plumeria — Snipcart order webhook (Supabase Edge Function)
//
// Dependency-free: no npm/esm imports (so it deploys cleanly from the
// dashboard editor). Talks to the DB over Supabase's REST API with the
// service-role key, which bypasses RLS.
//
// On a completed order we:
//   1) VERIFY the request is genuinely from Snipcart (validate its
//      request-token against Snipcart's API with our SECRET key).
//   2) Are IDEMPOTENT on the order token (retries can't duplicate).
//   3) Record the order + line items.
//   4) Mark one-of-a-kind pieces (max_quantity = 1) as sold.
// We never see card data — Snipcart/Stripe handle payment.
//
// Secrets: SNIPCART_SECRET_API_KEY (set by owner).
// SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are injected automatically.
// Deploy with Verify JWT OFF (Snipcart sends no Supabase token).
// ============================================================
const SNIPCART_SECRET = Deno.env.get("SNIPCART_SECRET_API_KEY") ?? "";
const SB_URL = Deno.env.get("SUPABASE_URL")!;
const SB_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const REST = `${SB_URL}/rest/v1`;
const sbHeaders = {
  apikey: SB_KEY,
  Authorization: `Bearer ${SB_KEY}`,
  "Content-Type": "application/json",
};

const json = (obj: unknown, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });

Deno.serve(async (req) => {
  if (req.method !== "POST") return json({ ok: true });

  // 1) Verify authenticity with Snipcart.
  const token = req.headers.get("x-snipcart-requesttoken");
  if (!token) return json({ error: "missing request token" }, 401);
  if (!SNIPCART_SECRET) return json({ error: "server not configured" }, 500);

  const check = await fetch(
    `https://app.snipcart.com/api/requestvalidation/${token}`,
    {
      headers: {
        Accept: "application/json",
        Authorization: "Basic " + btoa(`${SNIPCART_SECRET}:`),
      },
    },
  );
  if (!check.ok) return json({ error: "request validation failed" }, 401);

  const body = await req.json().catch(() => null);
  if (!body) return json({ error: "bad payload" }, 400);
  if (body.eventName !== "order.completed") {
    return json({ ignored: body.eventName ?? "unknown" });
  }

  const o = body.content ?? {};
  const orderToken: string | undefined = o.token;
  if (!orderToken) return json({ error: "no order token" }, 400);

  // 2) Idempotency — skip if already recorded.
  const dup = await fetch(
    `${REST}/orders?snipcart_token=eq.${encodeURIComponent(orderToken)}&select=id&limit=1`,
    { headers: sbHeaders },
  );
  const dupRows = await dup.json();
  if (Array.isArray(dupRows) && dupRows.length) {
    return json({ status: "already processed", id: dupRows[0].id });
  }

  // 3) Insert the order (return=representation gives us the new id).
  const orderRes = await fetch(`${REST}/orders`, {
    method: "POST",
    headers: { ...sbHeaders, Prefer: "return=representation" },
    body: JSON.stringify({
      snipcart_token: orderToken,
      invoice_number: o.invoiceNumber ?? null,
      email: o.email ?? null,
      customer_name: o.billingAddressName ?? o.shippingAddressName ?? null,
      shipping: o.shippingAddress ?? o.billingAddress ?? null,
      subtotal: o.itemsTotal ?? null,
      shipping_cost: o.shippingFees ?? null,
      taxes: o.taxesTotal ?? null,
      total: o.finalGrandTotal ?? o.total ?? null,
      currency: String(o.currency ?? "usd").toUpperCase(),
      status: body.mode === "Test" ? "test" : "paid",
      placed_at: o.completionDate ?? o.creationDate ?? new Date().toISOString(),
      raw: body,
    }),
  });
  if (!orderRes.ok) {
    return json({ error: "order insert failed", detail: await orderRes.text() }, 500);
  }
  const orderId = (await orderRes.json())[0]?.id;

  // 4) Insert line items.
  const items = (o.items ?? []).map((it: Record<string, unknown>) => ({
    order_id: orderId,
    sku: (it.id as string) ?? null,
    name: (it.name as string) ?? null,
    unit_price: (it.price as number) ?? null,
    quantity: (it.quantity as number) ?? 1,
    line_total: (it.totalPrice as number) ??
      (it.price != null ? (it.price as number) * ((it.quantity as number) ?? 1) : null),
  }));
  if (items.length) {
    const itemsRes = await fetch(`${REST}/order_items`, {
      method: "POST",
      headers: sbHeaders,
      body: JSON.stringify(items),
    });
    if (!itemsRes.ok) {
      return json({ error: "items insert failed", detail: await itemsRes.text() }, 500);
    }
  }

  // 5) Sell-through: one-of-a-kind pieces become unavailable.
  const soldSkus = items.map((i) => i.sku).filter(Boolean) as string[];
  if (soldSkus.length) {
    await fetch(
      `${REST}/products?max_quantity=eq.1&sku=in.(${soldSkus.map(encodeURIComponent).join(",")})`,
      { method: "PATCH", headers: sbHeaders, body: JSON.stringify({ available: false }) },
    );
  }

  return json({ status: "recorded", order: orderId, items: items.length });
});
