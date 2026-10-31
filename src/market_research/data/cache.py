from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional
import pandas as pd
import market_research.config as C
from market_research.data import storage

def data_stamp(*, exclude_stocks=False, fund_code=None) -> str:
    """内容指纹；可隔离股票/其他基金，保守保留共享与未知依赖。"""
    return storage.stamp(exclude_stocks=exclude_stocks, fund_code=fund_code)


def _cache_meta_path(name: str) -> Path:
    return C.DATA_DIR / f"{name}.meta.json"


def _fingerprint(meta: dict) -> dict:
    """
    只取**参与有效性判等**的字段。

    ★ 必须把 "info" 排除掉（N-07）：info 里放的是抓取时间、数据末日、行数、
    缺失资产这类**描述性**信息，它们每次抓取都会变。若让它们参与相等比较，
    缓存会每次都判为失效 → 每次全量重建，等于没有缓存。
    """
    return {k: v for k, v in (meta or {}).items() if k != "info"}


def _cache_meta_valid(name: str, meta: dict) -> bool:
    """优先校验数据库口径；仅为旧 CSV 迁移读取 sidecar。"""
    saved = storage.metadata(name)
    if saved is None:
        fp = _cache_meta_path(name)
        if not fp.exists():
            return True
        try:
            saved = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
    return _fingerprint(saved) == _fingerprint(meta)


def _cache_meta_write(name: str, meta: dict):
    if storage.metadata(name) is None:
        _cache_meta_path(name).write_text(json.dumps(meta), encoding="utf-8")
    else:
        storage.write_metadata(name, meta)


def cache_info(name: str) -> dict:
    return (storage.metadata(name) or {}).get("info", {})


def _cache_info_write(name: str, info: dict):
    storage.update_info(name, info)


def _cached(name: str, builder, meta: Optional[dict] = None, refresh=False):
    """DuckDB 为唯一写入目标；兼容导入旧 CSV，刷新失败保留上一份数据。"""
    if not refresh and (meta is None or _cache_meta_valid(name, meta)):
        frame = storage.read(name)
        if frame is not None:
            return frame
        legacy = C.DATA_DIR / f"{name}.csv"
        if legacy.exists():
            frame = pd.read_csv(legacy, index_col=0, parse_dates=True)
            if len(frame):
                saved = _cache_meta_path(name)
                old_meta = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
                storage.write(name, frame, _with_info(meta or old_meta, frame))
                return frame
    frame = builder()
    storage.write(name, frame, _with_info(meta or {}, frame))
    return frame


def _with_info(meta: dict, df) -> dict:
    """
    给 meta 补上描述性 info（N-07）：抓取时间、数据末日、行数。

    这些字段**不参与** _cache_meta_valid 的判等（见 _fingerprint），
    否则每次抓取时间不同都会让缓存失效。
    """
    meta = dict(meta or {})
    info = dict(meta.get("info") or {})
    info.setdefault("fetched_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(df, pd.DataFrame) and len(df.index):
        try:
            info.setdefault("data_end", str(pd.Timestamp(df.index[-1]).date()))
            info.setdefault("data_start", str(pd.Timestamp(df.index[0]).date()))
            info.setdefault("n_rows", int(len(df)))
            info.setdefault("n_cols", int(df.shape[1]))
        except (TypeError, ValueError, IndexError):        # noqa: BLE001
            pass
    meta["info"] = info
    return meta
