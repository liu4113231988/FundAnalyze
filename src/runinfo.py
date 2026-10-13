# -*- coding: utf-8 -*-
"""
运行指纹（P1-09 / N-09）。

**问题**：旧的研究产物只写结论数字，不写"这些数字是在哪套代码、哪套依赖、
哪份数据、哪些参数下跑出来的"。于是同一份代码在不同机器上给出不同数字时，
无法判断是代码变了、依赖变了，还是数据变了 —— 结论不可复现也不可追责。

本模块把指纹随产物一起落盘。三个来源：
  1. 环境：Python / 平台 / 关键依赖版本 / git commit
  2. 数据：候选池规模、缓存基金数、数据指纹（与结果缓存键同一个 data_stamp）
  3. 参数：影响结果的全部配置快照（窗口、refit、成本、门控阈值、rebal_mode…）
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _git_head(root: Optional[Path] = None) -> Optional[str]:
    try:
        root = root or Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()[:40]
    except Exception:                                         # noqa: BLE001
        pass
    return None


def _lib_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:                                     # noqa: BLE001
            out[mod] = None
    return out


def _config_snapshot() -> Dict[str, Any]:
    """只取会影响结果的键，避免把整个 config  dump 成噪音。"""
    try:
        from . import config as C
        snap: Dict[str, Any] = {}
        for name, blk in (("RBSA", getattr(C, "RBSA", {})),
                          ("BACKTEST", getattr(C, "BACKTEST", {})),
                          ("SIGNAL", getattr(C, "SIGNAL", {}))):
            snap[name] = {k: _jsonable(v) for k, v in dict(blk).items()}
        snap["CACHE_VERSION"] = getattr(C, "CACHE_VERSION", None)
        snap["DATA_CACHE_VERSION"] = getattr(C, "DATA_CACHE_VERSION", None)
        snap["n_candidates"] = len(getattr(C, "NAME_OF", {}))
        snap["n_extended_funds"] = len(getattr(C, "EXTENDED_FUNDS", {}))
        return snap
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _jsonable(v):
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:                                         # noqa: BLE001
        pass
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def data_summary() -> Dict[str, Any]:
    """数据侧指纹：与结果缓存键共用一个 data_stamp，便于交叉比对。"""
    try:
        from . import config as C, data as D
        files = sorted(Path(C.DATA_DIR).glob("*.csv"))
        return {
            "data_stamp": D.data_stamp(),
            "n_csv": len(files),
            "n_fund_nav_cached": len([f for f in files
                                      if f.name.startswith("fund_nav_")]),
            "n_index_cache": len([f for f in files
                                  if f.name.startswith("index_universe_")]),
            "start_date": getattr(C, "START_DATE", None),
            # ★ 分析窗口（2026-10-12）：START_DATE 是**滚动**的，只记录它无法
            #   判断"这次跑的是几年窗口"。复现需要窗口年数与抓取起点一起记。
            "window_years": getattr(C, "DEFAULT_WINDOW_YEARS", None),
            "min_window_years": getattr(C, "MIN_WINDOW_YEARS", None),
            "fetch_start": getattr(C, "FETCH_START_DATE", None),
        }
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def run_fingerprint(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """完整运行指纹。extra 里可放本次运行的参数（基金列表、模型集、max_rebal…）。"""
    info = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_head(),
        "libs": _lib_versions(),
        "data": data_summary(),
        "config": _config_snapshot(),
        # ★ 随机性说明：本流水线的 sklearn 组件都固定了 random_state=0，
        #   进程池/spawn 不引入随机性；但 bootstrap 的 seed 在
        #   src/robustness.block_bootstrap 里显式给出，也一并记录。
        "seeds": {"bootstrap": 20261011, "sklearn_random_state": 0},
    }
    if extra:
        info["run"] = _jsonable(extra)
    return info


def write_fingerprint(path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """落盘为 JSON，同时返回内容（便于打印）。"""
    import json
    info = run_fingerprint(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(p)                        # 原子替换，避免半截 JSON
    return info


def fingerprint_lines(info: Dict[str, Any], prefix: str = "  ") -> list:
    """人类可读的一页摘要，用于控制台输出。"""
    L = [f"{prefix}生成时间   {info.get('generated_at')}",
         f"{prefix}git commit  {info.get('git_commit') or '（非 git 环境）'}",
         f"{prefix}Python      {info.get('python')}",
         f"{prefix}依赖        " + ", ".join(
             f"{k} {v}" for k, v in (info.get("libs") or {}).items() if v),
         f"{prefix}数据指纹    {(info.get('data') or {}).get('data_stamp')}",
         f"{prefix}候选指数    {(info.get('config') or {}).get('n_candidates')} 个"]
    run = info.get("run") or {}
    if run:
        L.append(f"{prefix}本次运行    " + ", ".join(
            f"{k}={v}" for k, v in run.items()))
    return L
