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

## Cost / abuse notes

- Origin-allowlisted + size-capped, and runs on Claude Haiku 4.5 (cheap/fast).
  Change the `MODEL` constant in `index.ts` for a deeper model.
- The endpoint is public and unauthenticated by design. For higher traffic, add
  per-IP rate limiting (e.g. a small `chat_rate` table keyed on IP + minute, or
  an edge KV) before the Anthropic call.
