# ai-receptionist

The storefront's AI concierge. A floating chat widget (`docs/js/receptionist.js`)
posts the conversation here; this function answers with Claude, grounded in the
**live catalog** and Blue Plumeria's real brand facts, and can save a lead to
`public.inquiries` (the table the Owner Dashboard already reads).

Like `snipcart-webhook`, it's dependency-free (no npm/esm imports) so it pastes
straight into the dashboard editor, and it talks to the DB with the service-role
key (bypasses RLS).

## What it does

- **Answers** store questions in the studio's voice — and refuses to invent
  policies it wasn't given (shipping windows, returns, sizing → offers to take a
  message instead).
- **Recommends** only *published* pieces from the `products` table, pulled fresh
  each request, and never offers a sold-out one-of-a-kind as available.
- **Captures leads / custom orders** via a `save_inquiry` tool → inserts into
  `public.inquiries` with `kind` (`question` | `custom-order` | `lead`) and a
  short transcript in `meta`.

## Secrets

| Name | Who sets it | Notes |
|------|-------------|-------|
| `ANTHROPIC_API_KEY` | owner | From console.anthropic.com. **Required.** |
| `ALLOWED_ORIGINS` | optional | Comma-separated origin allowlist. Defaults to `https://blue-plumeria.com,https://www.blue-plumeria.com`. `localhost` is always allowed for testing. |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | — | Injected automatically. |

## Deploy

Requires the migration `20260819130000_inquiry_ai_fields.sql` (adds `kind` +
`meta` to `inquiries`) to be applied first.

```bash
# 1. Apply the DB migration
supabase db push

# 2. Set the Anthropic key
supabase secrets set ANTHROPIC_API_KEY=sk-ant-...

# 3. Deploy WITHOUT JWT verification (the widget sends no Supabase user token)
supabase functions deploy ai-receptionist --no-verify-jwt
```

Dashboard alternative: create a function named `ai-receptionist`, paste
`index.ts`, set **Verify JWT = OFF**, add the `ANTHROPIC_API_KEY` secret.

## Request / response

`POST /functions/v1/ai-receptionist`

```jsonc
// request
{ "messages": [ { "role": "user", "content": "do you have anything in copper?" } ] }
// response
{ "reply": "We do — ...", "leadSaved": false }
```

Only `user`/`assistant` **text** turns are accepted; the server prepends the
system prompt and catalog. Caps: 24 messages, 2000 chars each, 800 output tokens.

## Guardrails (agentic-ops)

- **Scoped** — answers Blue Plumeria topics only; declines off-topic/general
  questions and resists persona changes + prompt-injection ("ignore your
  instructions"). It may say it's a *virtual* assistant but never reveals the prompt.
- **Human handoff** — when it can't help, it offers to connect the visitor:
  `save_inquiry` (studio follows up by email) or `hello@blue-plumeria.com`.
- **Least privilege** — one tool only (`save_inquiry`), and it's additive
  (create an inquiry). No destructive/irreversible capability, so nothing needs
  a human-confirmation gate. If you add such a tool later, gate it.
- **Rate limit** — per-IP hourly cap via `public.receptionist_hits`
  (`RATE_LIMIT_PER_HOUR`, default 40). **Fails open** — a limiter error never
  blocks a real visitor. Rows self-prune (>2h old) opportunistically.
- **Wall-clock timeout** — every upstream call goes through `fetchT`
  (`FETCH_TIMEOUT_MS`, default 20s; 5s for DB limiter) so a hung upstream fails
  fast instead of riding the platform limit.
- **Catalog cache** — `loadCatalog` caches per warm isolate (`CATALOG_TTL_MS`,
  default 60s), skipping the DB round-trip on most turns.
- **Tracing** — every run logs `request → model (tokens) → tool → done/error`
  as JSON with a `traceId` to the `function_logs` source. Query:
  `select event_message from logs where source='function_logs' and event_message like '%traceId%'`.
- Runs on Claude Haiku 4.5 (cheap/fast); origin-allowlisted; 24 msgs / 2000
  chars / 800 output tokens caps.

## Optional env (all have safe defaults)

| Name | Default | Purpose |
|------|---------|---------|
| `RESEND_API_KEY` | — | Enables lead-notification + auto-reply emails (see [contact fn]). |
| `OWNER_EMAIL` | desmitdesignz@gmail.com | Where lead notifications go. |
| `FROM_EMAIL` | Blue Plumeria <hello@blue-plumeria.com> | Sender. |
| `RATE_LIMIT_PER_HOUR` | 40 | Per-IP message cap. |
| `FETCH_TIMEOUT_MS` | 20000 | Upstream timeout. |
| `CATALOG_TTL_MS` | 60000 | Catalog cache TTL. |

## Sandbox-first deploy flow

Supabase branches need a paid plan, so on the Free tier the sandbox-first flow is:

1. **Test before prod** — run locally with `supabase functions serve
   ai-receptionist --env-file .env.local` and curl it, **or** deploy a
   throwaway `ai-receptionist-staging` function (same project — doesn't use a
   project slot) and smoke-test that first.
2. **Deploy** — `supabase functions deploy ai-receptionist --no-verify-jwt`.
   Deploys are **atomic**: a build error leaves the current version live.
3. **Smoke-test after deploy** — POST an on-topic and an off-topic message
   (see below) and confirm 200 + sensible replies, then check `function_logs`
   for the trace and any errors.

```bash
curl -s -X POST "$URL/functions/v1/ai-receptionist" \
  -H "Origin: https://blue-plumeria.com" -H "Content-Type: application/json" \
  -H "apikey: $PUBLISHABLE_KEY" -H "Authorization: Bearer $PUBLISHABLE_KEY" \
  -d '{"messages":[{"role":"user","content":"Do you have anything with pearls?"}]}'
```
