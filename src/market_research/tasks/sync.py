"""Only synchronize registered assets; durable state, daily Shanghai schedule.

Run with the web service, or independently: python -m market_research.tasks.sync.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from importlib.resources import files
import threading

import numpy as np
import pandas as pd

import market_research.config as C
import market_research.data.storage as storage

LOG = logging.getLogger(__name__)
SHANGHAI = timezone(timedelta(hours=8))
HISTORY_START = "1990-01-01"
_START_LOCK = threading.Lock()
_THREAD = None


def now():
    return datetime.now(SHANGHAI)


def schedule_time():
    value = os.environ.get("FIL_SYNC_TIME", "18:30")
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise ValueError("FIL_SYNC_TIME 必须为 HH:MM，例如 18:30") from error


def migrate():
    with storage.connection() as db:
        for path in sorted((files("market_research.data") / "migrations").iterdir()):
            db.execute(path.read_text(encoding="utf-8"))


def register(kind, code, basis=""):
    if kind not in ("stock", "index", "fund"):
        raise ValueError("未知同步类型")
    if kind == "fund":
        if not (isinstance(code, str) and len(code) == 6 and code.isdigit()):
            raise ValueError("基金代码必须为六位数字")
        basis = basis or "total"
        if basis not in ("total", "cum"):
            raise ValueError("未知净值口径")
    else:
        from market_research.market.analysis import normalize_code
        # SW industry indices are already used by fund analysis.
        if not (kind == "index" and code in C.SW_INDUSTRY):
            code = normalize_code(kind, code)
        if kind == "index":
            basis = ""
        elif basis not in ("", "qfq", "hfq"):
            raise ValueError("未知复权口径")
    key = f"{kind}:{code}:{basis}"
    migrate()
    with storage.connection() as db:
        db.execute("INSERT INTO sync_assets(asset_key,kind,code,basis) VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                   [key, kind, code, basis])
    return key


def datasets():
    with storage.connection() as db:
        rows = db.execute("SELECT name,metadata,schema_json FROM datasets").fetchall()
    return [(name, json.loads(meta), json.loads(schema)["columns"]) for name, meta, schema in rows]


def discover():
    """Import only assets actually stored, never the global catalogue."""
    known = {a["asset_key"] for a in assets()}

    def add(kind, code, basis=""):
        key = f"{kind}:{code}:{basis}"
        if key not in known:
            known.add(register(kind, code, basis))

    for name, meta, columns in datasets():
        try:
            if name.startswith("market_v1_"):
                add(meta["kind"], meta["code"], meta.get("adjust", ""))
            elif name.startswith("fund_nav_"):
                add("fund", meta["code"], meta.get("basis", "cum"))
            elif name.startswith("index_universe_"):
                for code in columns:
                    add("index", code)
        except (KeyError, ValueError):
            LOG.warning("不能识别历史数据集 %s", name)


def assets():
    migrate()
    with storage.connection() as db:
        cursor = db.execute("SELECT * FROM sync_assets ORDER BY asset_key")
        names = [x[0] for x in cursor.description]
        result = [dict(zip(names, row)) for row in cursor.fetchall()]
    for asset in result:
        if not asset.get("state"):
            asset["state"] = ("error" if asset["error"] else "running" if asset["last_attempt"] and
                              (not asset["last_success"] or asset["last_attempt"] > asset["last_success"])
                              else "success" if asset["last_success"] else "pending")
    return result


def request_sync(key):
    """Durable request; active work is never duplicated or reset."""
    migrate()
    with storage.connection() as db:
        row = db.execute("SELECT state,last_attempt,last_success,error FROM sync_assets WHERE asset_key=?", [key]).fetchone()
        if not row:
            raise ValueError("资产尚未添加")
        state, attempt, success, error = row
        if state == "running" and attempt and (now() - datetime.fromisoformat(attempt)).total_seconds() < 1800:
            return False
        db.execute("UPDATE sync_assets SET state='pending', requested_at=?,last_attempt=NULL,error=NULL WHERE asset_key=?", [now().isoformat(), key])
    return True


def history_name(asset):
    return "sync_history_" + asset["asset_key"].replace(":", "_")


def cached(kind, code, basis, start, end):
    """Canonical histories serve analysis windows after successful sync."""
    key = f"{kind}:{code}:{basis}"
    name = history_name({"asset_key": key})
    meta = storage.metadata(name)
    if not meta or meta.get("through", "") < (end or now().date().isoformat()):
        return None
    if (start or C.START_DATE) < HISTORY_START:
        return None
    frame = storage.read(name)
    if frame is None:
        return None
    frame = frame.loc[start or C.START_DATE:end].copy()
    frame.attrs.update(meta.get("info", {}))
    return frame if len(frame) else None


def due(asset, current):
    attempt = asset.get("last_attempt")
    if attempt and (current - datetime.fromisoformat(attempt)).total_seconds() < 1800:
        return False
    if asset.get("state") == "pending":
        return True
    success = asset.get("last_success")
    if not success:
        return True
    scheduled = datetime.combine(current.date(), schedule_time(), SHANGHAI)
    if current < scheduled:
        scheduled -= timedelta(days=1)
    return datetime.fromisoformat(success) < scheduled


def fetch(asset, start, end):
    import market_research.data.marketdata as D
    kind, code, basis = asset["kind"], asset["code"], asset["basis"]
    if kind == "fund":
        series = D.fetch_fund_total_return(code, start) if basis == "total" else D.fetch_fund_nav(code, start)
        if series.attrs.get("degraded"):
            raise ValueError("总回报构造降级，保留上一份数据")
        return series.to_frame()
    if kind == "index" and code in C.SW_INDUSTRY:
        return D.fetch_index_sw(code, start).to_frame(name="close")
    from market_research.market.analysis import load_bars
    return load_bars(kind, code, start, end, basis, refresh=True, _fetch_only=True)


def synchronize(asset, current):
    import market_research.data.marketdata as D
    end = current.date().isoformat()
    name = history_name(asset)
    old = storage.read(name)
    # Full adjusted history is intentional: distributions and provider corrections
    # can revise old prices without a reliable corporate-action feed.
    full = old is None or asset["kind"] == "fund" or bool(asset["basis"])
    start = HISTORY_START if full else max(HISTORY_START, (old.index[-1] - pd.Timedelta(days=10)).strftime("%Y-%m-%d"))
    incoming = fetch(asset, start, end)
    if incoming is None or incoming.empty:
        raise ValueError("数据源返回空历史，保留旧数据")
    incoming = incoming.loc[:end].sort_index()
    if incoming.empty or not np.isfinite(incoming.to_numpy(dtype=float)[~pd.isna(incoming.to_numpy(dtype=float))]).all():
        raise ValueError("历史数据包含非法数值")
    if old is not None and incoming.index[-1] < old.index[-1]:
        raise ValueError("数据源历史倒退，保留旧数据")
    if full:
        if old is not None and incoming.index[0] > old.index[0]:
            raise ValueError("数据源历史被截断，拒绝部分复权覆盖")
        if old is not None and len(old.index.difference(incoming.index)):
            raise ValueError("数据源缺失已有历史交易日，拒绝部分覆盖")
        frame = incoming
    else:
        frame = pd.concat([old.loc[old.index < incoming.index[0]], incoming])
    meta = {"kind": asset["kind"], "code": asset["code"], "basis": asset["basis"], "through": end,
            "info": {**incoming.attrs, "fetched_at": current.isoformat(), "data_end": frame.index[-1].date().isoformat()}}
    storage.write(name, frame, meta)
    # Refresh old window caches too, including fund candidate pools. Only existing
    # columns are updated, so this never expands the synchronization universe.
    for target, saved, columns in datasets():
        if asset["kind"] == "fund" and target.startswith("fund_nav_") and saved.get("code") == asset["code"] and saved.get("basis", "cum") == asset["basis"]:
            sliced = frame.loc[saved.get("start"):].copy()
            sliced.columns = columns
        elif target.startswith("market_v1_") and saved.get("kind") == asset["kind"] and saved.get("code") == asset["code"] and saved.get("adjust", "") == asset["basis"]:
            sliced = frame.loc[saved.get("start"):saved.get("end")]
        elif asset["kind"] == "index" and target.startswith("index_universe_") and asset["code"] in columns:
            sliced = storage.read(target)
            values = frame.close.loc[saved.get("start"):saved.get("end")]
            sliced = sliced.reindex(sliced.index.union(values.index))
            sliced[asset["code"]] = values.reindex(sliced.index)
        else:
            continue
        if len(sliced):
            storage.write(target, sliced, D._with_info({k:v for k,v in saved.items() if k != "info"}, sliced))
    return "full" if full else "incremental", len(frame)


@contextmanager
def leader():
    """Nonblocking process lock; released automatically after process death."""
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (C.DATA_DIR / "sync.lock").open("a+b") as file:
        file.seek(0, 2)
        if not file.tell():
            file.write(b"0"); file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            if os.name == "nt":
                file.seek(0); msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def tick(current=None):
    current = current or now()
    with leader() as acquired:
        if not acquired:
            return
        discover()
        for asset in assets():
            if not due(asset, current):
                continue
            with storage.connection() as db:
                db.execute("UPDATE sync_assets SET last_attempt=?,error=NULL,state='running' WHERE asset_key=?", [current.isoformat(), asset["asset_key"]])
            try:
                mode, rows = synchronize(asset, current)
                with storage.connection() as db:
                    db.execute("UPDATE sync_assets SET last_success=?,mode=?,rows=?,error=NULL,state='success' WHERE asset_key=?",
                               [current.isoformat(), mode, rows, asset["asset_key"]])
            except Exception as error:
                LOG.exception("同步失败 %s", asset["asset_key"])
                with storage.connection() as db:
                    db.execute("UPDATE sync_assets SET error=?,state='error' WHERE asset_key=?", [str(error)[:500], asset["asset_key"]])


def serve():
    schedule_time()
    while True:
        try:
            tick()
        except Exception:
            LOG.exception("同步调度失败")
        threading.Event().wait(30)


def start():
    global _THREAD
    if os.environ.get("FIL_SYNC_ENABLED", "1") == "0":
        return
    schedule_time()
    from logging.handlers import RotatingFileHandler
    if not LOG.handlers:
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(C.DATA_DIR / "sync.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
    with _START_LOCK:
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=serve, name="market-sync", daemon=True)
            _THREAD.start()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="执行本次到期任务后退出")
    parser.add_argument("--time", help="覆盖北京时间执行时间 HH:MM")
    args = parser.parse_args()
    if args.time:
        os.environ["FIL_SYNC_TIME"] = args.time
    schedule_time()
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=str(C.DATA_DIR / "sync.log"), encoding="utf-8")
    if args.once:
        tick()
    else:
        serve()


if __name__ == "__main__":
    main()
