-- Per-IP request log powering the ai-receptionist rate limiter.
-- The edge function counts recent rows per IP (last hour) before calling the
-- model, and opportunistically prunes rows older than ~2h. Only the
-- service-role function touches this table, so RLS is on with no policies.
create table if not exists public.receptionist_hits (
  id bigint generated always as identity primary key,
  ip text not null,
  created_at timestamptz not null default now()
);

create index if not exists receptionist_hits_ip_time
  on public.receptionist_hits (ip, created_at);

alter table public.receptionist_hits enable row level security;

comment on table public.receptionist_hits is
  'Per-IP request log for the ai-receptionist rate limiter. Rows expire opportunistically (>2h old).';
