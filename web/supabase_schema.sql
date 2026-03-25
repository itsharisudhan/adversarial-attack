create extension if not exists pgcrypto;

create table if not exists public.analysis_history (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default timezone('utc', now()),
    filename text not null,
    verdict text not null,
    verdict_short text,
    ensemble_score numeric(5, 4) not null,
    input_preview_url text,
    fft_spectrum_url text,
    ela_heatmap_url text,
    image_info jsonb not null default '{}'::jsonb,
    detector_scores jsonb not null default '{}'::jsonb
);

create index if not exists analysis_history_created_at_idx
    on public.analysis_history (created_at desc);
