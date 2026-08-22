// ============================================================
// Blue Plumeria — contact form handler (Supabase Edge Function)
//
// Receives the storefront contact form, saves it to public.inquiries
// (the same table the Owner Dashboard + AI receptionist use), then:
//   1) emails the studio owner a notification (reply-to = the visitor),
//   2) emails the visitor a warm auto-acknowledgement.
// Both emails go through Resend. If RESEND_API_KEY isn't set yet, the
// inquiry is STILL saved — email is skipped gracefully.
//
// Abuse controls (this endpoint sends branded email to an arbitrary address,
// so it's throttled and never echoes visitor content back in the auto-reply):
//   - global + per-IP hourly rate limit (shared public.receptionist_hits),
//   - honeypot field, origin allowlist, hard fetch timeouts.
//
// Dependency-free (raw fetch). Talks to the DB over the REST API with the
// service-role key (bypasses RLS).
//
// Secrets:  RESEND_API_KEY (required for email), optional OWNER_EMAIL,
//           FROM_EMAIL, ALLOWED_ORIGINS, CONTACT_RATE_LIMIT_PER_HOUR,
//           GLOBAL_LIMIT_PER_HOUR, FETCH_TIMEOUT_MS.
// Injected: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
// Deploy with Verify JWT OFF (the public form sends no Supabase token).
// ============================================================

const SB_URL = Deno.env.get("SUPABASE_URL")!;
const SB_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const RESEND_KEY = Deno.env.get("RESEND_API_KEY") ?? "";
const FROM_EMAIL = Deno.env.get("FROM_EMAIL") ??
  "Blue Plumeria <hello@blue-plumeria.com>";
const OWNER_EMAIL = Deno.env.get("OWNER_EMAIL") ?? "desmitdesignz@gmail.com";

const ALLOWED_ORIGINS = (Deno.env.get("ALLOWED_ORIGINS") ??
  "https://blue-plumeria.com,https://www.blue-plumeria.com")
  .split(",").map((s) => s.trim()).filter(Boolean);

// Reject malformed env (a NaN cap would silently disable the limiter).
function posInt(v: string | undefined, dflt: number): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : dflt;
}
const RATE_LIMIT_PER_HOUR = posInt(Deno.env.get("CONTACT_RATE_LIMIT_PER_HOUR"), 20);
const GLOBAL_LIMIT_PER_HOUR = posInt(Deno.env.get("GLOBAL_LIMIT_PER_HOUR"), 600);
const FETCH_TIMEOUT_MS = posInt(Deno.env.get("FETCH_TIMEOUT_MS"), 20000);

const REST = `${SB_URL}/rest/v1`;
const sbHeaders = {
  apikey: SB_KEY,
  Authorization: `Bearer ${SB_KEY}`,
  "Content-Type": "application/json",
};

function corsHeaders(origin: string | null) {
  const allow = origin && (
    ALLOWED_ORIGINS.includes(origin) ||
    /^http:\/\/localhost(:\d+)?$/.test(origin)
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

const esc = (s: string) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

// Fetch with a hard wall-clock timeout (fail fast instead of hanging).
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

function clientIp(req: Request): string {
  // Prefer platform-set headers; XFF is client-appendable, so use its LAST hop.
  return req.headers.get("cf-connecting-ip") ??
    req.headers.get("x-real-ip") ??
    (req.headers.get("x-forwarded-for")?.split(",").pop()?.trim() ?? "");
}

async function countHits(filter: string): Promise<number | null> {
  const res = await fetchT(
    `${REST}/receptionist_hits?${filter}&select=id`,
    { headers: { ...sbHeaders, Prefer: "count=exact", Range: "0-0" } },
    5000,
  ).catch(() => null);
  if (!res || !res.ok) {
    console.error("rate: count query failed", res?.status ?? "network");
    return null;
  }
  const cr = res.headers.get("content-range"); // "0-0/<total>" or "*/0"
  const total = cr && cr.includes("/") ? Number(cr.split("/")[1]) : 0;
  return Number.isFinite(total) ? total : 0;
}

// Global + per-IP hourly caps (shared public.receptionist_hits). GLOBAL is the
// real spend/abuse backstop; per-IP is best-effort. Fails OPEN; logs failures.
async function rateLimited(ip: string, ns: string): Promise<boolean> {
  const sinceEnc = encodeURIComponent(new Date(Date.now() - 3600_000).toISOString());
  // Namespace the per-IP key per endpoint (see receptionist). Global count is
  // created_at-only, so it stays a shared spend backstop across both functions.
  const key = `${ns}${ip || "unknown"}`;
  try {
    const globalN = await countHits(`created_at=gte.${sinceEnc}`);
    if (globalN === null) return false;
    if (globalN >= GLOBAL_LIMIT_PER_HOUR) {
      console.error("rate: global cap reached", globalN);
      return true;
    }
    if (ip) {
      const ipN = await countHits(
        `ip=eq.${encodeURIComponent(key)}&created_at=gte.${sinceEnc}`,
      );
      if (ipN !== null && ipN >= RATE_LIMIT_PER_HOUR) return true;
    }
    const ins = await fetchT(`${REST}/receptionist_hits`, {
      method: "POST",
      headers: { ...sbHeaders, Prefer: "return=minimal" },
      body: JSON.stringify({ ip: key }),
    }, 5000).catch(() => null);
    if (!ins || !ins.ok) console.error("rate: hit insert failed", ins?.status ?? "network");
    // Both functions prune, so the shared table stays bounded even if one goes quiet.
    if (Math.random() < 0.02) {
      const old = encodeURIComponent(new Date(Date.now() - 7200_000).toISOString());
      await fetchT(`${REST}/receptionist_hits?created_at=lt.${old}`, {
        method: "DELETE",
        headers: { ...sbHeaders, Prefer: "return=minimal" },
      }, 5000).catch(() => {});
    }
    return false;
  } catch {
    return false;
  }
}

// Resend send. Never throws — returns false on any problem (incl. key unset).
async function sendEmail(opts: {
  to: string | string[];
  subject: string;
  html: string;
  replyTo?: string;
}): Promise<boolean> {
  if (!RESEND_KEY) return false;
  try {
    const res = await fetchT("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${RESEND_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: FROM_EMAIL,
        to: Array.isArray(opts.to) ? opts.to : [opts.to],
        subject: opts.subject,
        html: opts.html,
        ...(opts.replyTo ? { reply_to: opts.replyTo } : {}),
      }),
    });
    if (!res.ok) {
      console.error("resend error", res.status);
      return false;
    }
    return true;
  } catch (e) {
    console.error("resend exception", e);
    return false;
  }
}

Deno.serve(async (req) => {
  const origin = req.headers.get("origin");
  const cors = corsHeaders(origin);

  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
  if (req.method !== "POST") return json({ error: "method not allowed" }, 405, cors);
  if (cors["Access-Control-Allow-Origin"] === "null") {
    return json({ error: "origin not allowed" }, 403, cors);
  }

  const body = await req.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return json({ error: "bad payload" }, 400, cors);
  }

  // Honeypot: real users leave this empty. Bots fill it. Pretend success.
  if (String((body as Record<string, unknown>).company ?? "").trim()) {
    return json({ ok: true }, 200, cors);
  }

  const name = String((body as Record<string, unknown>).name ?? "").trim().slice(0, 200);
  const email = String((body as Record<string, unknown>).email ?? "").trim().slice(0, 200);
  const subject = String((body as Record<string, unknown>).subject ?? "").replace(/[\r\n]+/g, " ").trim().slice(0, 200);
  const message = String((body as Record<string, unknown>).message ?? "").trim().slice(0, 5000);

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return json({ error: "Please enter a valid email address." }, 400, cors);
  }
  if (!message) {
    return json({ error: "Please include a message." }, 400, cors);
  }

  // Throttle before doing any work — this endpoint sends branded email.
  if (await rateLimited(clientIp(req), "contact:")) {
    return json({
      error: "You've sent several messages recently. Please try again a bit later, " +
        "or email hello@blue-plumeria.com directly.",
    }, 429, cors);
  }

  // 1) Save the inquiry (source of truth — must succeed).
  const ins = await fetchT(`${REST}/inquiries`, {
    method: "POST",
    headers: sbHeaders,
    body: JSON.stringify({
      source: "blueplumeria",
      name: name || null,
      email,
      message: subject ? `[${subject}] ${message}` : message,
      kind: "general",
      status: "new",
      meta: { via: "contact-form", subject: subject || null },
    }),
  }, 5000).catch(() => null);
  if (!ins || !ins.ok) {
    console.error("inquiry insert failed", ins?.status ?? "network");
    return json({ error: "Sorry — something went wrong. Please try again." }, 502, cors);
  }

  // 2) Notify the studio (reply-to the visitor so a reply reaches them). The
  //    visitor's message only goes to the OWNER address, never echoed outward.
  const who = name ? `${esc(name)} (${esc(email)})` : esc(email);
  // Greet by first name only if it looks like a name (visitor-controlled).
  const firstName = /^[\p{L}][\p{L}'-]{0,39}$/u.test(name.split(" ")[0] || "")
    ? name.split(" ")[0]
    : "";
  await sendEmail({
    to: OWNER_EMAIL,
    replyTo: email,
    subject: `New contact form message${subject ? ` — ${subject}` : ""} · Blue Plumeria`,
    html: `
      <div style="font-family:Georgia,serif;color:#2b2b2b;max-width:560px">
        <h2 style="font-weight:normal;margin:0 0 12px">New message from the website</h2>
        <p style="margin:0 0 6px"><strong>From:</strong> ${who}</p>
        ${subject ? `<p style="margin:0 0 6px"><strong>Subject:</strong> ${esc(subject)}</p>` : ""}
        <p style="margin:14px 0 6px"><strong>Message:</strong></p>
        <p style="white-space:pre-wrap;margin:0;padding:12px 14px;background:#f6f4ef;border-radius:6px">${esc(message)}</p>
        <p style="margin:16px 0 0;color:#888;font-size:13px">Reply to this email to respond directly to ${esc(email)}.</p>
      </div>`,
  });

  // 3) Auto-acknowledge the visitor. Deliberately does NOT echo their message —
  //    the address is attacker-controllable, so this must never carry content.
  await sendEmail({
    to: email,
    subject: "Thanks for reaching out — Blue Plumeria",
    html: `
      <div style="font-family:Georgia,serif;color:#2b2b2b;max-width:560px">
        <p>Hi${firstName ? " " + esc(firstName) : ""},</p>
        <p>Thank you for reaching out to Blue Plumeria — your message came through
        and we'll get back to you as soon as we can.</p>
        <p>With gratitude,<br>Blue Plumeria</p>
        <p style="color:#888;font-size:13px">Handcrafted in the USA · blue-plumeria.com</p>
      </div>`,
  });

  return json({ ok: true }, 200, cors);
});
