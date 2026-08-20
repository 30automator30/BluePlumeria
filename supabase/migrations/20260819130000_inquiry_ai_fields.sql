-- ============================================================
-- Inquiry AI fields — lets the AI receptionist tag and enrich the leads
-- it captures without changing how the Owner Dashboard reads inquiries.
--
-- The contact form and the receptionist both write to public.inquiries.
-- Two additive, nullable columns keep every existing row valid:
--   * kind : what the visitor wanted — 'general' (contact form),
--            'question', 'custom-order', or 'lead'. Lets the owner triage.
--   * meta : free-form JSON the receptionist fills with a short transcript
--            and channel, so a captured lead carries its context.
--
-- Writes still arrive via the service_role key (edge function), which
-- BYPASSES RLS — so no policy changes are needed here.
-- ============================================================

alter table public.inquiries
  add column if not exists kind text not null default 'general',
  add column if not exists meta jsonb;

-- Constrain kind to the known set (keeps triage filters honest). Existing
-- rows default to 'general', so this passes without a backfill.
alter table public.inquiries
  drop constraint if exists inquiries_kind_check,
  add  constraint inquiries_kind_check
    check (kind in ('general','question','custom-order','lead'));

create index if not exists inquiries_kind_idx on public.inquiries (kind);
