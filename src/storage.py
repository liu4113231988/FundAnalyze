"""DuckDB 行情缓存；短连接和文件锁允许 Web 与 CLI 交替访问。"""
from contextlib import contextmanager
import hashlib
import json
import os
import threading

import duckdb
import pandas as pd
from . import config as C

_LOCK = threading.RLock()


@contextmanager
def connection():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK, (C.DATA_DIR / "market.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            with duckdb.connect(str(C.DATA_DIR / "market.duckdb")) as db:
                db.execute("CREATE TABLE IF NOT EXISTS datasets (name VARCHAR PRIMARY KEY, metadata VARCHAR, schema_json VARCHAR, revision VARCHAR)")
                db.execute("CREATE TABLE IF NOT EXISTS prices (dataset VARCHAR, date TIMESTAMP_NS, asset VARCHAR, value DOUBLE, PRIMARY KEY(dataset, date, asset))")
                yield db
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def metadata(name):
    with connection() as db:
        row = db.execute("SELECT metadata FROM datasets WHERE name=?", [name]).fetchone()
    return json.loads(row[0]) if row else None


def read(name):
    with connection() as db:
        row = db.execute("SELECT schema_json FROM datasets WHERE name=?", [name]).fetchone()
        if row is None:
            return None
        frame = db.execute("SELECT date, asset, value FROM prices WHERE dataset=? ORDER BY date", [name]).df()
    schema = json.loads(row[0])
    result = frame.pivot(index="date", columns="asset", values="value").reindex(columns=schema["columns"])
    result.columns.name = None
    result.index.name = schema["index_name"]
    result.index = result.index.as_unit(schema["index_unit"])
    return result.astype(schema["dtypes"])


def write(name, frame, meta):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or frame.index.hasnans or frame.index.tz is not None:
        raise ValueError("行情必须使用唯一的日期索引")
    if frame.empty or frame.columns.has_duplicates or not all(isinstance(c, str) for c in frame.columns):
        raise ValueError("行情不能为空，列名必须为唯一字符串")
    schema = json.dumps({"columns": list(frame.columns), "index_name": frame.index.name, "index_unit": frame.index.unit,
                         "dtypes": {c: str(t) for c, t in frame.dtypes.items()}})
    values = frame.rename_axis("date").reset_index().melt(id_vars="date", var_name="asset", value_name="value")
    revision = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes() + schema.encode() + json.dumps({k:v for k,v in meta.items() if k != "info"}, sort_keys=True).encode()).hexdigest()
    with connection() as db:
        db.register("incoming", values)
        db.execute("BEGIN")
        try:
            db.execute("DELETE FROM prices WHERE dataset=?", [name])
            db.execute("INSERT INTO prices SELECT ?, date, asset, value FROM incoming", [name])
            db.execute("INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?)", [name, json.dumps(meta), schema, revision])
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise


def write_metadata(name, meta):
    with connection() as db:
        db.execute("UPDATE datasets SET metadata=? WHERE name=?", [json.dumps(meta), name])


def stamp():
    with connection() as db:
        rows = db.execute("SELECT name, revision FROM datasets ORDER BY name").fetchall()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def summary():
    """运行记录使用的数据库统计，与旧 CSV 备份数量无关。"""
    with connection() as db:
        total, funds, indices = db.execute(
            "SELECT count(*), count(*) FILTER (WHERE starts_with(name, 'fund_nav_')), "
            "count(*) FILTER (WHERE starts_with(name, 'index_universe_')) FROM datasets"
        ).fetchone()
        rows = db.execute("SELECT count(*) FROM prices").fetchone()[0]
    return {"storage": "duckdb", "n_datasets": total, "n_fund_nav_cached": funds,
            "n_index_cache": indices, "n_price_rows": rows}
