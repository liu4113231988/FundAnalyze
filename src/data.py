# -*- coding: utf-8 -*-
"""
数据层：基金复权净值 + 候选指数行情，统一对齐后落盘缓存。

数据源（均为免费公开接口）
    - 基金净值 : 天天基金 pingzhongdata（akshare: fund_open_fund_info_em，累计净值走势）
    - 宽基/风格: 新浪指数日线（akshare: stock_zh_index_daily）
    - 申万行业 : 申万宏源一级行业指数（akshare: index_hist_sw）

注意
    累计净值已含分红再投效果，用于收益分析是标准的；停牌/缺失按交易日历前向填充，
    但不允许跨超过 5 个交易日填充（防止长期停牌被伪装成 0 收益）。
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from . import config as C


# ------------------------------------------------------------------ 通用重试
def _retry(fn, tries: int = 4, wait: float = 1.2, **kw):
    last = None
    for i in range(tries):
        try:
            return fn(**kw)
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(wait * (i + 1))
    raise last


def _clean_series(s: pd.Series, start: Optional[str] = None) -> pd.Series:
    """通用清洗：剔除非正值、按日期去重、截断起点。"""
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")]
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s


# ------------------------------------------------------------------ 基金净值
def fetch_fund_nav(code: str, start: Optional[str] = None) -> pd.Series:
    """返回基金累计净值序列（index=交易日）。"""
    import akshare as ak

    df = _retry(ak.fund_open_fund_info_em, symbol=code, indicator="累计净值走势")
    df.columns = ["date", "nav"]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["nav"].astype(float).sort_index()
    # 注意：必须按"索引（日期）"去重，不能按数值去重——
    # 累计净值存在大量数值相同的相邻记录，按数值去重会误删四成以上样本。
    s.name = f"fund_{code}"
    return _clean_series(s, start)


# ------------------------------------------------------------------ 指数行情
def fetch_index_sina(code: str, start: Optional[str] = None) -> pd.Series:
    """新浪指数日线收盘价。code 形如 sh000300 / sz399006。"""
    import akshare as ak

    df = _retry(ak.stock_zh_index_daily, symbol=code)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["close"].astype(float).sort_index()
    s.name = code
    return _clean_series(s, start)


def fetch_index_sw(code: str, start: Optional[str] = None) -> pd.Series:
    """申万一级行业指数收盘价。code 形如 801010。"""
    import akshare as ak

    df = _retry(ak.index_hist_sw, symbol=code, period="day")
    dcol = "日期" if "日期" in df.columns else df.columns[1]
    ccol = "收盘" if "收盘" in df.columns else "close"
    df[dcol] = pd.to_datetime(df[dcol])
    s = df.set_index(dcol)[ccol].astype(float).sort_index()
    s.name = code
    return _clean_series(s, start)


# ------------------------------------------------------------------ 批量抓取
def _fetch_one(code: str, start: str, end: Optional[str]):
    if code in C.SW_INDUSTRY:
        s = fetch_index_sw(code, start)
    else:
        s = fetch_index_sina(code, start)
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    return code, s


def fetch_universe(codes: Iterable[str], start: str, end: Optional[str] = None,
                   verbose: bool = True, workers: int = 6) -> pd.DataFrame:
    """
    并发抓取整个候选池，返回 宽表 DataFrame(index=日期, columns=指数代码)。

    串行抓 44 个指数约需 45 秒，6 线程并发可压到 8 秒左右。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    codes = list(codes)
    series: Dict[str, pd.Series] = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, c, start, end): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                code, s = fut.result()
                if len(s) > 200:
                    series[code] = s
                    ok += 1
                else:
                    fail += 1
                    if verbose:
                        print(f"  [data] {code} {C.NAME_OF.get(code,''):8s} "
                              f"跳过（仅 {len(s)} 行）")
            except Exception as e:      # noqa: BLE001
                fail += 1
                if verbose:
                    print(f"  [data] {code} {C.NAME_OF.get(code,''):8s} "
                          f"失败: {type(e).__name__} {str(e)[:50]}")

    if verbose:
        print(f"  [data] 候选池抓取完成：成功 {ok} / 失败 {fail}")
    px = pd.DataFrame(series).sort_index()
    return px


# ------------------------------------------------------------------ 清洗对齐
def align_prices(px: pd.DataFrame, nav: pd.Series, max_ffill: int = 5,
                 min_coverage: float = 0.97) -> pd.DataFrame:
    """
    把基金净值与指数价格对齐到共同交易日。

    规则：
      1. 以基金净值的交易日为基准（基金开放日才是可交易日）；
      2. 指数缺失值前向填充，但连续填充不超过 max_ffill 天；
      3. 先剔除区间内覆盖率不足 min_coverage 的指数（如已停编的老指数），
         再掐头去尾保留全部序列均有有效值的区间。
    """
    df = px.reindex(nav.index).ffill(limit=max_ffill)
    df["__FUND__"] = nav.reindex(df.index)

    # 覆盖率过滤：优先剔除"中途停编"的指数，避免整段历史被截断
    cov = df.notna().mean()
    dropped = cov[(cov < min_coverage) & (df.columns != "__FUND__")].index.tolist()
    if dropped:
        print(f"[data] 覆盖率不足已剔除 {len(dropped)} 个：" +
              ", ".join(f"{c}({C.NAME_OF.get(c, c)})" for c in dropped))
    df = df.drop(columns=dropped)

    # 掐头去尾并剔除中间缺失日（连续缺失超过 max_ffill 的交易日直接放弃）
    return df.dropna()


def to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """价格 → 对数收益。"""
    r = np.log(df).diff().dropna()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


# ------------------------------------------------------------------ 指数估值
def fetch_pe_panel(symbols: Iterable[str], lookback_years: int = 8,
                   verbose: bool = True) -> pd.DataFrame:
    """
    拉取宽基指数的滚动市盈率历史（月度），并转为 **滚动历史分位**（0~1）。

    用途：判断风格基准的"绝对估值水位"。分位=1 表示处于历史最贵档。
    返回列名为指数中文名。
    """
    import akshare as ak

    out = {}
    for sym in symbols:
        try:
            df = _retry(ak.stock_index_pe_lg, tries=2, wait=1.0, symbol=sym)
            df["日期"] = pd.to_datetime(df["日期"])
            s = df.set_index("日期")["滚动市盈率"].astype(float).sort_index()
            s = s[s > 0]
            win = 12 * lookback_years
            pct = s.rolling(win, min_periods=win // 3).apply(
                lambda a: float((a <= a[-1]).mean()), raw=True)
            out[sym] = pct
            if verbose:
                print(f"  [pe] {sym:8s} {len(s):4d} 期  {s.index[0].date()} ~ "
                      f"{s.index[-1].date()}  当前分位={pct.iloc[-1]:.0%}")
        except Exception as e:      # noqa: BLE001
            if verbose:
                print(f"  [pe] {sym:8s} 失败 {type(e).__name__} {str(e)[:40]}")
    return pd.DataFrame(out).sort_index()


def load_pe_panel(verbose: bool = True) -> pd.DataFrame:
    from .signal import PE_SYMBOL, PE_FALLBACK

    syms = sorted(set(PE_SYMBOL.values()) | {PE_FALLBACK})
    df = _cached("pe_panel", lambda: fetch_pe_panel(syms, verbose=verbose))
    # 缓存可能因当初部分标的抓取失败而缺列：增量补取后合并并回写
    missing = [s for s in syms if s not in df.columns]
    if missing:
        add = fetch_pe_panel(missing, verbose=verbose)
        if add.shape[1]:
            df = pd.concat([df, add], axis=1).sort_index()
            df.to_csv(C.DATA_DIR / "pe_panel.csv", encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ 缓存层
def _cached(name: str, builder):
    fp = C.DATA_DIR / f"{name}.csv"
    if fp.exists():
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df):
                return df
        except Exception as e:      # noqa: BLE001
            print(f"[data] 缓存 {name} 损坏（{type(e).__name__}），重新构建")
    df = builder()
    df.to_csv(fp, encoding="utf-8-sig")
    return df


def load_panel(fund_code: str, start: str = C.START_DATE,
               end: Optional[str] = C.END_DATE, refresh: bool = False,
               verbose: bool = True) -> pd.DataFrame:
    """
    返回对齐后的价格面板：列 = 候选指数 + __FUND__。
    """
    cache_name = f"panel_{fund_code}_{start}"
    if refresh and (C.DATA_DIR / f"{cache_name}.csv").exists():
        (C.DATA_DIR / f"{cache_name}.csv").unlink()

    def build():
        if verbose:
            print(f"[data] 抓取基金净值 {fund_code} ...")
        nav = fetch_fund_nav(fund_code, start)
        if verbose:
            print(f"       净值 {len(nav)} 行  {nav.index[0].date()} ~ {nav.index[-1].date()}")
            print(f"[data] 抓取候选指数池 {len(C.NAME_OF)} 个 ...")
        px = fetch_universe(C.NAME_OF.keys(), start, end, verbose=verbose)
        px = align_prices(px, nav)
        if verbose:
            print(f"[data] 对齐后：{px.shape[0]} 交易日 × {px.shape[1]} 条序列 "
                  f"({px.index[0].date()} ~ {px.index[-1].date()})")
        return px

    return _cached(cache_name, build)
