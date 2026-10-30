"""Shared finite JSON conversion for research payloads."""
from __future__ import annotations
import math
from typing import List
import numpy as np
import pandas as pd
def _jnum(v, nd: int = 6):
    """数值 → JSON 安全值。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _jsafe(obj):
    """递归把 numpy / pandas 转成 JSON 可序列化结构。"""
    if isinstance(obj, dict):
        return {str(k): _jsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsafe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _jnum(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return _jnum(obj)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pd.Series):
        return [_jsafe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return _jsafe(obj.to_dict("list"))
    return str(obj)


def _dates(idx) -> List[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(idx)]
