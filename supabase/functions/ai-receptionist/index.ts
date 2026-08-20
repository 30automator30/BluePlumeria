// ============================================================
// Blue Plumeria — AI receptionist (Supabase Edge Function)
//
// A concierge for the storefront. It:
//   1) Answers store questions (materials, custom orders, how buying works)
//      grounded in real brand facts — it does NOT invent shipping/return
//      policies it wasn't given.
//   2) Helps visitors find pieces, using the LIVE catalog pulled from the
//      products table each request (so sold-out 1-of-1s aren't recommended).
//   3) Captures leads + custom-order requests via a save_inquiry tool that
//      writes to public.inquiries — the same table the Owner Dashboard reads.
//
// Dependency-free (raw fetch, no npm/esm imports) to match snipcart-webhook
// and deploy cleanly from the dashboard editor. Talks to the DB over the
// REST API with the service-role key (bypasses RLS). Calls the Anthropic
// Messages API directly.
//
// Secrets (set by owner):  ANTHROPIC_API_KEY
// Injected automatically:  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
// Deploy with Verify JWT OFF (the widget sends no Supabase user token).
// ============================================================

const ANTHROPIC_KEY = Deno.env.get("ANTHROPIC_API_KEY") ?? "";
const SB_URL = Deno.env.get("SUPABASE_URL")!;
const SB_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

// Fast + inexpensive, which is the right trade for a customer-facing
// receptionist. Bump to "claude-sonnet-5" or "claude-opus-5" for more depth.
const MODEL = "claude-haiku-4-5";
const MAX_TOKENS = 800;

// Only browsers on the real storefront may call this paid endpoint. Override
// with an ALLOWED_ORIGINS secret (comma-separated) if the domain changes.
const ALLOWED_ORIGINS = (Deno.env.get("ALLOWED_ORIGINS") ??
  "https://blue-plumeria.com,https://www.blue-plumeria.com")
  .split(",").map((s) => s.trim()).filter(Boolean);

// Guardrails on the incoming conversation (cost + abuse).
const MAX_MESSAGES = 24;
const MAX_CHARS_PER_MSG = 2000;
const MAX_TOOL_HOPS = 3;

const REST = `${SB_URL}/rest/v1`;
const sbHeaders = {
  apikey: SB_KEY,
  Authorization: `Bearer ${SB_KEY}`,
  "Content-Type": "application/json",
};

function corsHeaders(origin: string | null) {
  const allow = origin && (
    ALLOWED_ORIGINS.includes(origin) || /^http:\/\/localhost(:\d+)?$/.test(origin)
  );
  return {
    "Access-Control-Allow-Origin": allow ? origin! : "null",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "authorization, apikey, content-type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

const json = (obj: unknown, status: number, cors: Record<string, string>) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...cors },
  });

// ── Live catalog → a compact block the model can reason over ──────────
async function loadCatalog(): Promise<string> {
  try {
    const res = await fetch(
      `${REST}/products?published=eq.true` +
        `&select=name,price,collection,tier,label,available,max_quantity` +
        `&order=available.desc,price.asc`,
      { headers: sbHeaders },
    );
    if (!res.ok) return "(catalog temporarily unavailable)";
    const rows = await res.json() as Array<Record<string, unknown>>;
    if (!Array.isArray(rows) || !rows.length) return "(no published pieces)";
    return rows.map((p) => {
      const price = p.price != null ? `$${Number(p.price).toFixed(0)}` : "price on request";
      const oneOfAKind = Number(p.max_quantity) === 1;
      const stock = p.available === false
        ? "SOLD OUT"
        : oneOfAKind ? "available (one-of-a-kind)" : "available";
      const coll = p.collection ? ` · ${p.collection}` : "";
      return `- ${p.name} — ${price}${coll} — ${stock}`;
    }).join("\n");
  } catch {
    return "(catalog temporarily unavailable)";
  }
}

// ── Brand + policy grounding. Deliberately states only what's TRUE and
//    tells the model to route unknowns to a human instead of inventing. ──
function systemPrompt(catalog: string): string {
  return `You are the receptionist for Blue Plumeria (blue-plumeria.com), a small working
atelier that makes limited studio editions of handcrafted jewelry — earrings,
necklaces, and bracelets. You greet visitors warmly, answer their questions,
help them find a piece, and take a message when the studio should follow up.

VOICE: warm, unhurried, and personal — like a maker who knows every piece.
Keep replies short (1–3 sentences unless asked for more). Never pushy.

WHAT'S TRUE (you may state these):
- Every piece is handcrafted in the USA in small batches. Many are one-of-a-kind:
  once a one-of-a-kind piece sells, it's gone.
- Materials: natural stones (agate, sodalite, turquoise, jade, lava stone),
  glass and gemstone crystals, freshwater pearls, hand-shaped copper and wire,
  seed and artisan glass beads.
- Checkout is secure (card handled by Snipcart/Stripe — the studio never sees
  card details). The cart is on every page.
- Custom orders and commissions are welcome.
- Direct contact: hello@blue-plumeria.com. The Shop page is /shop.html, the
  studio story is /about.html.

RULES:
- Do NOT invent specifics you weren't told — exact shipping times/costs, return
  windows, sizing, or whether a specific unlisted item exists. If asked, say the
  studio will confirm, and offer to take their name + email (use save_inquiry).
- Recommend only pieces from the LIVE CATALOG below. Never suggest a SOLD OUT
  piece as available; if their favorite is gone, say so and offer a similar one
  or a custom order. Prices below are current.
- For a custom order, repair, wholesale, or any request needing a human, collect
  the visitor's name, email, and what they want, then call save_inquiry. Confirm
  once saved. Also use save_inquiry when a visitor asks to be contacted or leaves
  a question you can't fully answer.
- To point someone at the shop or a page, use a plain relative link like
  /shop.html or the mailto: hello@blue-plumeria.com.

LIVE CATALOG (published pieces, cheapest first):
${catalog}`;
}

// The single tool: hand a lead to the studio.
const TOOLS = [{
  name: "save_inquiry",
  description:
    "Save a visitor's message so the Blue Plumeria studio can follow up by " +
    "email. Use for custom orders, commissions, repairs, wholesale, 'please " +
    "contact me' requests, or any question you cannot fully answer. Only call " +
    "this once you have at least the visitor's email and their message.",
  input_schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      name: { type: "string", description: "Visitor's name, if given." },
      email: { type: "string", description: "Visitor's email address." },
      message: {
        type: "string",
        description: "What the visitor wants, in their words or summarized.",
      },
      kind: {
        type: "string",
        enum: ["question", "custom-order", "lead"],
        description:
          "custom-order for a bespoke/commission/repair request; question " +
          "for an unanswered question; lead for a general 'contact me'.",
      },
    },
    required: ["email", "message", "kind"],
  },
}];

async function saveInquiry(
  input: Record<string, unknown>,
  transcript: string,
): Promise<string> {
  const email = String(input.email ?? "").trim();
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return "That email doesn't look valid — could you double-check it?";
  }
  const res = await fetch(`${REST}/inquiries`, {
    method: "POST",
    headers: sbHeaders,
    body: JSON.stringify({
      source: "blueplumeria",
      name: String(input.name ?? "").trim() || null,
      email,
      message: String(input.message ?? "").trim() || null,
      kind: ["question", "custom-order", "lead"].includes(String(input.kind))
        ? input.kind
        : "lead",
      status: "new",
      meta: { via: "ai-receptionist", transcript: transcript.slice(0, 4000) },
    }),
  });
  if (!res.ok) return "Sorry — I couldn't save that just now.";
  return "Saved — the studio will be in touch by email soon.";
}

// ── Anthropic Messages API (raw HTTP) with a small tool-use loop ──────
async function callClaude(
  messages: Array<Record<string, unknown>>,
  system: string,
): Promise<Response> {
  return await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": ANTHROPIC_KEY,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      system,
      tools: TOOLS,
      messages,
    }),
  });
}

Deno.serve(async (req) => {
  const origin = req.headers.get("origin");
  const cors = corsHeaders(origin);

  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  if (req.method !== "POST") return json({ error: "method not allowed" }, 405, cors);
  if (cors["Access-Control-Allow-Origin"] === "null") {
    return json({ error: "origin not allowed" }, 403, cors);
  }
  if (!ANTHROPIC_KEY) return json({ error: "server not configured" }, 500, cors);

  const body = await req.json().catch(() => null);
  const incoming = body?.messages;
  if (!Array.isArray(incoming) || !incoming.length) {
    return json({ error: "no messages" }, 400, cors);
  }

  // Sanitize: keep only user/assistant text turns, capped in size and count.
  const messages: Array<Record<string, unknown>> = incoming
    .slice(-MAX_MESSAGES)
    .filter((m: unknown): m is { role: string; content: unknown } =>
      !!m && typeof m === "object" &&
      ((m as { role?: unknown }).role === "user" ||
        (m as { role?: unknown }).role === "assistant") &&
      typeof (m as { content?: unknown }).content === "string")
    .map((m) => ({
      role: m.role,
      content: String(m.content).slice(0, MAX_CHARS_PER_MSG),
    }));
  if (!messages.length || messages[0].role !== "user") {
    return json({ error: "conversation must start with a visitor message" }, 400, cors);
  }

  const transcript = messages
    .map((m) => `${m.role === "user" ? "Visitor" : "Receptionist"}: ${m.content}`)
    .join("\n");

  const system = systemPrompt(await loadCatalog());

  let leadSaved = false;
  try {
    for (let hop = 0; hop <= MAX_TOOL_HOPS; hop++) {
      const res = await callClaude(messages, system);
      if (!res.ok) {
        const detail = await res.text();
        console.error("anthropic error", res.status, detail);
        return json({ error: "assistant unavailable" }, 502, cors);
      }
      const data = await res.json();

      if (data.stop_reason === "tool_use") {
        // Echo the assistant turn back, then answer each tool call.
        messages.push({ role: "assistant", content: data.content });
        const toolResults: Array<Record<string, unknown>> = [];
        for (const block of data.content ?? []) {
          if (block.type !== "tool_use") continue;
          let result = "Unknown tool.";
          if (block.name === "save_inquiry") {
            result = await saveInquiry(block.input ?? {}, transcript);
            if (result.startsWith("Saved")) leadSaved = true;
          }
          toolResults.push({
            type: "tool_result",
            tool_use_id: block.id,
            content: result,
          });
        }
        messages.push({ role: "user", content: toolResults });
        continue; // let the model phrase the confirmation
      }

      // Normal completion — extract the text.
      const reply = (data.content ?? [])
        .filter((b: Record<string, unknown>) => b.type === "text")
        .map((b: Record<string, unknown>) => b.text)
        .join("\n")
        .trim() ||
        "I'm here to help — could you say a little more about what you're after?";
      return json({ reply, leadSaved }, 200, cors);
    }
    // Ran out of tool hops without a final answer.
    return json({
      reply: "Thanks — I've noted that for the studio. Anything else I can help with?",
      leadSaved,
    }, 200, cors);
  } catch (e) {
    console.error("receptionist error", e);
    return json({ error: "assistant unavailable" }, 502, cors);
  }
});
