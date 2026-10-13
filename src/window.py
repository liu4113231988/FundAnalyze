# -*- coding: utf-8 -*-
"""
分析窗口解析（2026-10-13）。

背景：分析窗口默认近 5 年（`C.DEFAULT_WINDOW_YEARS`）、下限 2 年
（`C.MIN_WINDOW_YEARS`）。但各研究脚本此前**没有任何入口**能改窗口，
恒用默认值 —— 想跑一次近 2 年的对比只能手动改代码或设环境变量。

本模块提供三件事：

1. `add_window_args(ap)`   给 argparse 统一加 `--window-years / --window-start
                           / --window-end`（不复用 `--start/--end`：`run.py`
                           的 `--start/--end` 是**回测**窗口，语义不同，
                           同名会让两者混为一谈）。
2. `resolve_window(...)`   纯函数，只计算不碰全局状态，便于测试。
3. `apply_window(args)`    解析并**就地覆盖** `C.START_DATE` / `C.END_DATE`。

★ 为什么要"就地覆盖"而不是把窗口逐个传进几十个调用点：
   这些脚本里有大量 `load_panel(code)` / `run_fund(code)` 不显式传窗口的调用，
   逐个改既啰嗦又容易漏。`data.load_panel` 的 `start/end` 默认值已在函数体内
   惰性解析（`start or C.START_DATE`，见 src/data.py 的说明），所以覆盖属性
   就能让它们全部跟随。

   注意这个"惰性解析"是**前提**：Python 的默认参数在函数定义时求值，
   若写成 `def load_panel(..., start=C.START_DATE)`，运行时改属性毫无作用。
"""
from __future__ import annotations

import argparse
import datetime as _dt
from typing import Optional, Tuple

import pandas as pd

from . import config as C


def _iso(x, label: str) -> str:
    """
    严格校验 YYYY-MM-DD 并原样返回字符串。

    ★ 不能用 `pd.Timestamp(x)` 判断：pandas 会接受 "2021/10/12"、"2021.10.12"、
      甚至 "Oct 12 2021" 等一堆格式。而下游 `data.fetch_start_for()` 做的是
      **字符串字典序**比较，"2021/10/12" 与 "2016-01-01" 的字典序关系和日期
      先后无关，于是窗口会被静默判成"晚于固定起点"从而走错分支。
      实测（R13）：`resolve_window(start="2021/10/12")` 第一版**没有报错**。
    """
    s = str(x).strip()
    try:
        d = _dt.date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{label} {x!r} 不是 YYYY-MM-DD 格式")
    # ★ 还要挡"基本格式"：Python 3.11+ 的 fromisoformat 接受 "20211012"
    #   （无分隔符）。它解析出来是同一个日期，但下游 fetch_start_for 做的是
    #   **字符串**字典序比较 —— "20211012" 与 "2016-01-01" 的字典序关系和
    #   日期先后毫无关系，窗口会被静默判错。所以要求往返一致。
    if d.isoformat() != s:
        raise ValueError(f"{label} {x!r} 不是 YYYY-MM-DD 格式（应为四位年-两位月-两位日）")
    return s


def add_window_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """给 argparse 加统一的窗口参数（返回同一个 parser 以便链式调用）。"""
    g = ap.add_argument_group(
        "分析窗口",
        f"默认近 {C.DEFAULT_WINDOW_YEARS} 年，下限 {C.MIN_WINDOW_YEARS} 年。"
        "十年跨度会混入风格漂移与经理变更，拟合出的画像是历史平均，"
        "对当下没有指导意义。")
    g.add_argument("--window-years", type=float, default=None, metavar="N",
                   help=f"分析窗口年数（默认 {C.DEFAULT_WINDOW_YEARS}，"
                        f"最小 {C.MIN_WINDOW_YEARS}）")
    g.add_argument("--window-start", default=None, metavar="YYYY-MM-DD",
                   help="分析窗口起始日；与 --window-years 二选一，"
                        "给了则本项优先")
    g.add_argument("--window-end", default=None, metavar="YYYY-MM-DD",
                   help="分析窗口结束日，默认取到最新")
    return ap


def resolve_window(years: Optional[float] = None,
                   start: Optional[str] = None,
                   end: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    把窗口参数解析成 `(start, end)`，并校验 2 年下限。

    优先级：`start` > `years` > 当前配置（`C.START_DATE`）。
    `end` 为 None 表示取到最新。
    校验不通过抛 `ValueError`（由调用方决定是退出还是转 4xx）。
    """
    if start:
        s = _iso(start, "窗口起始日")
    elif years is not None:
        s = C._rolling_start(years)
    else:
        s = C.START_DATE

    e = _iso(end, "窗口结束日") if end else None
    try:
        s_ts = pd.Timestamp(s)
    except (ValueError, TypeError):
        raise ValueError(f"窗口起始日 {s!r} 不是有效日期（应为 YYYY-MM-DD）")
    e_eff = pd.Timestamp(e) if e else pd.Timestamp.today()
    if e_eff <= s_ts:
        raise ValueError(f"窗口起始日 {s_ts.date()} 需早于结束日 {e_eff.date()}")
    days = int((e_eff - s_ts).days)
    if days < C.MIN_WINDOW_DAYS:
        raise ValueError(
            f"分析窗口只有 {days} 天（约 {days / 365.25:.1f} 年），"
            f"最短需 {C.MIN_WINDOW_DAYS} 天（{C.MIN_WINDOW_YEARS} 年）。"
            f"样本太短时拟合窗口与预热会吃掉大部分数据。")
    return s, e


def apply_window(args, verbose: bool = True) -> Tuple[str, Optional[str]]:
    """
    解析 args 上的窗口参数，就地覆盖 `C.START_DATE` / `C.END_DATE` 并返回之。

    失败时抛 `ValueError` —— 由调用方打印并退出（研究脚本）或转 4xx（Web）。
    """
    s, e = resolve_window(getattr(args, "window_years", None),
                          getattr(args, "window_start", None),
                          getattr(args, "window_end", None))
    C.START_DATE = s
    C.END_DATE = e
    if verbose:
        print(f"[窗口] 分析窗口 {s} ~ {e or '最新'}"
              + (f"（{getattr(args, 'window_years', None)} 年）"
                 if getattr(args, "window_years", None) else ""))
    return s, e
