-- Additive, idempotent migration. Rollback: stop scheduler; drop sync_assets.
-- Existing datasets and prices are unaffected.
CREATE TABLE IF NOT EXISTS sync_assets (
  asset_key VARCHAR PRIMARY KEY,
  kind VARCHAR NOT NULL,
  code VARCHAR NOT NULL,
  basis VARCHAR NOT NULL,
  last_success VARCHAR,
  last_attempt VARCHAR,
  error VARCHAR,
  mode VARCHAR,
  rows BIGINT DEFAULT 0
);
