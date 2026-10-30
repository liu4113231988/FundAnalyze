-- Additive migration. Nullable state supports existing scheduler versions.
-- Rollback: stop new workers, drop state/requested_at columns; prices unaffected.
ALTER TABLE sync_assets ADD COLUMN IF NOT EXISTS state VARCHAR;
ALTER TABLE sync_assets ADD COLUMN IF NOT EXISTS requested_at VARCHAR;
