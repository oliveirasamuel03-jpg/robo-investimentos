-- PREPARATION ONLY. Never run against the old database.
-- A new history is not recovery or migration of the old history.
-- Requires an explicitly approved, dedicated, empty database; no app/worker running.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
SET LOCAL search_path = public;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
               WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','S','f')) THEN
        RAISE EXCEPTION 'Refusing initialization: public schema is not empty';
    END IF;
END $$;
CREATE TABLE public.app_state (
    namespace TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE public.app_rows (
    seq BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    row_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_app_rows_namespace_seq ON public.app_rows(namespace, seq);
COMMIT;
