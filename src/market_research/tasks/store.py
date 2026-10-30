"""Cross-process durable job queue using the existing DuckDB/file-lock boundary."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
from importlib.resources import files
import time
import uuid

import market_research.data.storage as storage
from market_research.errors import JobsBusy
from market_research.settings import Settings

_TOKEN = ContextVar("job_attempt_token", default=None)
LEASE_SECONDS = 45


@contextmanager
def bind_attempt(token):
    marker = _TOKEN.set(token)
    try:
        yield
    finally:
        _TOKEN.reset(marker)


def _schema(db):
    db.execute((files("market_research.data") / "migrations/003_jobs.sql").read_text(encoding="utf-8"))


def _row(cursor):
    values = cursor.fetchone()
    if values is None:
        return None
    result = dict(zip([c[0] for c in cursor.description], values))
    for field in ("models", "detail", "arguments", "result"):
        if field in result and result[field] is not None:
            result[field] = json.loads(result[field])
    return result


class DurableJobStore:
    def migrate(self):
        with storage.connection() as db:
            _schema(db)

    def enqueue(self, code, models, operation, arguments, key=None):
        settings, current = Settings.load(), time.time()
        with storage.connection() as db:
            _schema(db)
            db.execute("DELETE FROM analysis_jobs WHERE status IN ('done','error') AND updated < ?", [current - 86400])
            if key:
                found = db.execute("SELECT id FROM analysis_jobs WHERE task_key=? AND status IN ('queued','running') ORDER BY created LIMIT 1", [key]).fetchone()
                if found:
                    return found[0], True
            count = db.execute("SELECT count(*) FROM analysis_jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= settings.queue_limit:
                raise JobsBusy(f"待处理任务已达上限 {settings.queue_limit}，请稍后重试")
            jid = uuid.uuid4().hex[:16]
            db.execute("INSERT INTO analysis_jobs(id,code,models,operation,arguments,task_key,status,stage,progress,detail,created,updated) VALUES (?,?,?,?,?,?,'queued','等待计算进程',0,'[]',?,?)",
                       [jid, code, json.dumps(models), operation, json.dumps(arguments, ensure_ascii=False, allow_nan=False), key, current, current])
            return jid, False

    def submit(self, code, models, operation, arguments, key=None):
        return self.enqueue(code, models, operation, arguments, key)

    def recover(self, db, current):
        maximum = Settings.load().max_attempts
        db.execute("UPDATE analysis_jobs SET status=CASE WHEN attempts < ? THEN 'queued' ELSE 'error' END, stage=CASE WHEN attempts < ? THEN '恢复中断任务' ELSE '恢复失败' END, error=CASE WHEN attempts < ? THEN NULL ELSE '计算进程多次中断' END, token=NULL, updated=? WHERE status='running' AND lease_until < ?",
                   [maximum, maximum, maximum, current, current])

    def claim(self):
        current = time.time()
        with storage.connection() as db:
            _schema(db)
            self.recover(db, current)
            if db.execute("SELECT count(*) FROM analysis_jobs WHERE status='running'").fetchone()[0] >= Settings.load().max_jobs:
                return None
            job = _row(db.execute("SELECT * FROM analysis_jobs WHERE status='queued' ORDER BY created,id LIMIT 1"))
            if job is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE analysis_jobs SET status='running',stage='启动计算进程',started=?,updated=?,lease_until=?,token=?,attempts=attempts+1 WHERE id=?",
                       [current, current, current + LEASE_SECONDS, token, job["id"]])
            job.update(token=token, started=current, status="running", attempts=job["attempts"] + 1)
            return job

    def heartbeat(self, jid, token):
        with storage.connection() as db:
            row = db.execute("UPDATE analysis_jobs SET lease_until=?,updated=? WHERE id=? AND token=? AND status='running' RETURNING id",
                             [time.time() + LEASE_SECONDS, time.time(), jid, token]).fetchone()
            return row is not None

    def update(self, jid, **values):
        allowed = {"stage", "progress", "detail"}
        values = {k:v for k,v in values.items() if k in allowed}
        if not values:
            return
        if "detail" in values:
            values["detail"] = json.dumps(values["detail"], ensure_ascii=False)
        assignments = ",".join(f"{k}=?" for k in values)
        with storage.connection() as db:
            _schema(db)
            db.execute(f"UPDATE analysis_jobs SET {assignments},updated=? WHERE id=? AND status='running' AND token=?",
                       [*values.values(), time.time(), jid, _TOKEN.get()])

    def _terminal(self, jid, status, result=None, error=None, token=None):
        from market_research.serialization import _jsafe
        serialized = json.dumps(_jsafe(result), ensure_ascii=False, allow_nan=False) if result is not None else None
        with storage.connection() as db:
            _schema(db)
            db.execute("UPDATE analysis_jobs SET status=?,stage=?,progress=?,result=?,error=?,updated=? WHERE id=? AND status='running' AND token=?",
                       [status, "完成" if status == "done" else "失败", 1.0 if status == "done" else 0.0, serialized, error, time.time(), jid, token or _TOKEN.get()])

    def finish(self, jid, result):
        self._terminal(jid, "done", result=result)

    def fail(self, jid, error, token=None):
        self._terminal(jid, "error", error=error, token=token)

    def retry_crashed(self, jid, token, error):
        with storage.connection() as db:
            maximum = Settings.load().max_attempts
            db.execute("UPDATE analysis_jobs SET status=CASE WHEN attempts < ? THEN 'queued' ELSE 'error' END, stage='计算进程中断',error=?,updated=?,token=NULL WHERE id=? AND token=? AND status='running'",
                       [maximum, error[:500], time.time(), jid, token])

    def get(self, jid):
        with storage.connection() as db:
            _schema(db)
            return _row(db.execute("SELECT * FROM analysis_jobs WHERE id=?", [jid]))

    def public(self, jid):
        with storage.connection() as db:
            _schema(db)
            job = _row(db.execute("SELECT id,code,models,status,stage,progress,detail,created,updated,attempts,error FROM analysis_jobs WHERE id=?", [jid]))
        return job

    def cancel(self, jid):
        with storage.connection() as db:
            row = db.execute("UPDATE analysis_jobs SET status='error',stage='已取消',error='任务已取消',updated=? WHERE id=? AND status IN ('queued','running') RETURNING id", [time.time(), jid]).fetchone()
            return row is not None

    def is_cancelled(self, jid):
        job = self.public(jid)
        return not job or job["status"] != "running"
