// ============================================================
// Blue Plumeria — contact form handler (Supabase Edge Function)
//
// Receives the storefront contact form, saves it to public.inquiries
// (the same table the Owner Dashboard + AI receptionist use), then:
//   1) emails the studio owner a notification (reply-to = the visitor),
//   2) emails the visitor a warm auto-acknowledgement.
// Both emails go through Resend. If RESEND_API_KEY isn't set yet, the
// inquiry is STILL saved — email is skipped gracefully — so wiring up
// Resend later is non-breaking.
//
// Dependency-free (raw fetch, no npm/esm imports) to match the other
// functions and deploy cleanly from the dashboard editor. Talks to the DB
// over the REST API with the service-role key (bypasses RLS).
//
// Secrets:  RESEND_API_KEY (required for email), optional OWNER_EMAIL,
//           FROM_EMAIL, ALLOWED_ORIGINS.
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

// ── Resend: send one email. Returns false (never throws) so a mail
//    failure can't break the form submission itself. ───────────────────
async function sendEmail(opts: {
  to: string | string[];
  subject: string;
  html: string;
  replyTo?: string;
}): Promise<boolean> {
  if (!RESEND_KEY) return false; // email not configured yet — skip
  try {
    const res = await fetch("https://api.resend.com/emails", {
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
      console.error("resend error", res.status, await res.text());
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

  // Honeypot: real users leave this hidden field empty. Bots fill it.
  // Pretend success so the bot doesn't retry, but save/send nothing.
  if (String((body as Record<string, unknown>).company ?? "").trim()) {
    return json({ ok: true }, 200, cors);
  }

  const name = String((body as Record<string, unknown>).name ?? "").trim().slice(0, 200);
  const email = String((body as Record<string, unknown>).email ?? "").trim().slice(0, 200);
  const subject = String((body as Record<string, unknown>).subject ?? "").trim().slice(0, 200);
  const message = String((body as Record<string, unknown>).message ?? "").trim().slice(0, 5000);

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return json({ error: "Please enter a valid email address." }, 400, cors);
  }
  if (!message) {
    return json({ error: "Please include a message." }, 400, cors);
  }

  // 1) Save the inquiry (source of truth — must succeed).
  const ins = await fetch(`${REST}/inquiries`, {
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
  });
  if (!ins.ok) {
    console.error("inquiry insert failed", ins.status, await ins.text());
    return json({ error: "Sorry — something went wrong. Please try again." }, 502, cors);
  }

  // 2) Notify the studio (reply-to the visitor so a reply reaches them).
  const who = name ? `${esc(name)} (${esc(email)})` : esc(email);
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

  // 3) Auto-acknowledge the visitor (best-effort).
  await sendEmail({
    to: email,
    subject: "Thanks for reaching out — Blue Plumeria",
    html: `
      <div style="font-family:Georgia,serif;color:#2b2b2b;max-width:560px">
        <p>Hi${name ? " " + esc(name.split(" ")[0]) : ""},</p>
        <p>Thank you for reaching out to Blue Plumeria — your message came through
        and we'll get back to you as soon as we can.</p>
        <p style="white-space:pre-wrap;margin:14px 0;padding:12px 14px;background:#f6f4ef;border-radius:6px;color:#555">${esc(message)}</p>
        <p>With gratitude,<br>Blue Plumeria</p>
        <p style="color:#888;font-size:13px">Handcrafted in the USA · blue-plumeria.com</p>
      </div>`,
  });

  return json({ ok: true }, 200, cors);
});
