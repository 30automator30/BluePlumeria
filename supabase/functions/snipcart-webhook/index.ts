// ============================================================
// Blue Plumeria — Snipcart order webhook (Supabase Edge Function)
//
// Dependency-free (no npm/esm imports). Talks to the DB over Supabase's
// REST API with the service-role key (bypasses RLS).
//
// On a completed order we:
//   1) VERIFY the request is genuinely from Snipcart (validate its
//      request-token against Snipcart's API with our SECRET key).
//   2) Record the order + line items + sell-through in ONE atomic
//      transaction via the public.record_order() RPC — idempotent on the
//      order token, so retries can't duplicate and a partial failure can't
//      drop items or leave a sold one-of-a-kind still available.
//   3) Fail closed: any error returns non-2xx so Snipcart safely retries.
// We never see card data — Snipcart/Stripe handle payment.
//
// Secrets: SNIPCART_SECRET_API_KEY (set by owner).
// SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are injected automatically.
// Deploy with Verify JWT OFF (Snipcart sends no Supabase token).
// ============================================================
const SNIPCART_SECRET = Deno.env.get("SNIPCART_SECRET_API_KEY") ?? "";
const SB_URL = Deno.env.get("SUPABASE_URL")!;
const SB_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const FETCH_TIMEOUT_MS = Number(Deno.env.get("FETCH_TIMEOUT_MS") ?? "8000");

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

// Fetch with a hard wall-clock timeout so a slow upstream can't hang the
// invocation until the platform kills it.
async function fetchT(
  url: string,
  init: RequestInit = {},
  timeoutMs = FETCH_TIMEOUT_MS,
): Promise<Response> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ac.signal });
  } finally {
    clearTimeout(timer);
  }
}

Deno.serve(async (req) => {
  if (req.method !== "POST") return json({ ok: true });

  try {
    // 1) Verify authenticity with Snipcart.
    const token = req.headers.get("x-snipcart-requesttoken");
    if (!token) return json({ error: "missing request token" }, 401);
    if (!SNIPCART_SECRET) return json({ error: "server not configured" }, 500);

    const check = await fetchT(
      `https://app.snipcart.com/api/requestvalidation/${encodeURIComponent(token)}`,
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

    // 2) Build clean, pre-extracted payload for the atomic RPC.
    const order = {
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
    };
    const items = (o.items ?? []).map((it: Record<string, unknown>) => ({
      sku: (it.id as string) ?? null,
      name: (it.name as string) ?? null,
      unit_price: (it.price as number) ?? null,
      quantity: (it.quantity as number) ?? 1,
      line_total: (it.totalPrice as number) ??
        (it.price != null ? (it.price as number) * ((it.quantity as number) ?? 1) : null),
    }));
    const sold_skus = items.map((i: { sku: string | null }) => i.sku).filter(Boolean);

    // 3) One transaction: order + items + sell-through, idempotent on token.
    const rpc = await fetchT(`${REST}/rpc/record_order`, {
      method: "POST",
      headers: sbHeaders,
      body: JSON.stringify({ p: { order, items, sold_skus } }),
    });
    if (!rpc.ok) {
      // Fail closed — non-2xx makes Snipcart retry (the RPC is idempotent).
      return json({ error: "record failed", detail: await rpc.text() }, 500);
    }
    const result = await rpc.json(); // "recorded" | "duplicate"
    return json({ status: result, order: orderToken, items: items.length });
  } catch (e) {
    console.error("snipcart-webhook error", e);
    return json({ error: "webhook error" }, 500); // retryable
  }
});
