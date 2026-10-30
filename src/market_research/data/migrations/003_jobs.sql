-- Additive task-control table; no changes to stored market prices.
-- Rollback: stop workers, export desired results, DROP TABLE analysis_jobs.
CREATE TABLE IF NOT EXISTS analysis_jobs (
  id VARCHAR PRIMARY KEY, code VARCHAR, models VARCHAR,
  operation VARCHAR, arguments VARCHAR, task_key VARCHAR,
  status VARCHAR, stage VARCHAR, progress DOUBLE, detail VARCHAR,
  created DOUBLE, updated DOUBLE, started DOUBLE, lease_until DOUBLE,
  token VARCHAR, attempts INTEGER DEFAULT 0, error VARCHAR, result VARCHAR
);
