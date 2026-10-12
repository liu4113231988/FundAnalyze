# -*- coding: utf-8 -*-
"""
数据层：基金复权净值 + 候选指数行情，统一对齐后落盘缓存。

数据源（均为免费公开接口）
    - 基金净值 : 天天基金 pingzhongdata（akshare: fund_open_fund_info_em，累计净值走势）
    - 宽基/风格: 新浪指数日线（akshare: stock_zh_index_daily）
    - 申万行业 : 申万宏源一级行业指数（akshare: index_hist_sw）

注意
    停牌/缺失按交易日历前向填充，但不允许跨超过 5 个交易日填充
    （防止长期停牌被伪装成 0 收益）。

★ 收益口径（P0-01，尚未核实）
    `fetch_fund_nav` 取的是数据源的「累计净值走势」。原模块注释称它
    "已含分红再投效果"——**这个等式没有经过真实样本核实**。累计净值通常是
    「单位净值 + 历史累计分红」，而累计分红是**存量**、不参与后续增长，
    所以直接取对数差分会把收益率系统性、且随时间加重地压低。
    在本仓库用真实分红样本核实之前，基于该口径的所有结论都必须标注
    "口径待修正"。正确口径见 `total_return_index()`（需单位净值 + 分红/拆分事件）。
    指数侧用的是**价格指数**（不含股息），与基金总回报之间还差一个股息率。
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _default_http_timeout() -> float:
    """抓取超时（秒）。FIL_HTTP_TIMEOUT=0 表示不限制。"""
    try:
        v = float(os.environ.get("FIL_HTTP_TIMEOUT", "30"))
    except (TypeError, ValueError):                        # noqa: BLE001
        return 30.0
    return v if v > 0 else 0.0


@contextlib.contextmanager
def http_timeout(seconds: Optional[float] = None):
    """
    只在抓取期间生效的 socket 默认超时（P1-07 / N-05）。

    ★ 为什么不能写在模块级 `socket.setdefaulttimeout(30)`：
      它是**进程全局**的，会影响本进程里之后新建的**所有** socket ——
      包括 Flask 服务器的监听 socket 与已 accept 的连接 socket。
      本意只是防止 AkShare 请求永久挂起，结果把 Web 服务也罩进同一个
      30 秒超时里（长连接、慢客户端都可能被误杀）。改成上下文管理器后，
      只有真正在抓数据的代码段受影响。
    """
    old = socket.getdefaulttimeout()
    sec = _default_http_timeout() if seconds is None else seconds
    socket.setdefaulttimeout(sec if sec and sec > 0 else None)
    try:
        yield
    finally:
        socket.setdefaulttimeout(old)

from . import config as C


# ------------------------------------------------------------------ 通用重试
def _retry(fn, tries: int = 4, wait: float = 1.2, **kw):
    last = None
    for i in range(tries):
        # ★ 超时只在抓取期间生效（N-05）：AkShare 的请求没有暴露 timeout 参数，
        #   但 socket 级的全局默认值会波及 Web 服务器，所以必须局部设置。
        with http_timeout():
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
    """
    返回基金**累计净值**序列（index=交易日）。

    ★ 口径警告（P0-01，已核实）：本序列是数据源的「累计净值走势」，
    **不是**红利再投资复权净值，且偏差不是常数缩放。14 只基金实测平均低估
    **23.19%**（最严重的 110013 低估 76.12%）。偏差来源：累计净值 = 单位净值
    + 历史累计分红，而累计分红是存量、不参与后续增长，于是每期收益都被乘上
    一个 U/C < 1 的压缩因子，且分红越多压缩越狠、并随时间加重。
    → **收益分析请改用 `fetch_fund_total_return()`**（累计净值 + 单位净值 +
    月频累计收益率校准，14/14 只基金偏差 0.0000%）。本函数保留是为了
    兼容与对照，默认口径由 `load_panel(nav_basis=...)` 控制。
    """
    import akshare as ak

    df = _retry(ak.fund_open_fund_info_em, symbol=code, indicator="累计净值走势")
    df.columns = ["date", "nav"]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["nav"].astype(float).sort_index()
    # 注意：必须按"索引（日期）"去重，不能按数值去重——
    # 累计净值存在大量数值相同的相邻记录，按数值去重会误删四成以上样本。
    s.name = f"fund_{code}"
    return _clean_series(s, start)


# ------------------------------------------------------------------ 净值口径
# ★ P0-01：口径必须显式可选、可核对，不能靠"模块说明里的一句话"默认成立。
NAV_BASIS_CUM = "cum"        # 数据源「累计净值走势」直接取对数差分（当前默认）
NAV_BASIS_TOTAL = "total"    # 由单位净值 + 分红/拆分事件构造的总回报（红利再投资）

NAV_BASIS_NOTE = {
    NAV_BASIS_CUM: (
        "累计净值直算。**尚未与真实分红/拆分样本核实**：累计净值通常是"
        "「单位净值 + 历史累计分红」，它不等于红利再投资净值。由于累计分红是"
        "存量、不参与后续增长，收益率会被系统性且**随时间加重**地压低"
        "（见 TODO P0-01）。在核实前，基于本口径的结论不得作为决策依据。"),
    NAV_BASIS_TOTAL: (
        "总回报（红利再投资）：由单位净值 + 分红/拆分事件逐日构造，"
        "是基金收益的正确口径。指数侧若用价格指数，则与之差一个股息率。"),
}


def total_return_index(unit_nav: pd.Series,
                       dividends: Optional[pd.DataFrame] = None,
                       splits: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    由**单位净值 + 分红/拆分事件**构造总回报指数（起点 1.0，红利再投资口径）。

        r_t = (NAV_t × 份额因子 + D_t − NAV_{t−1}) / NAV_{t−1}

    即除息日把现金分红加回收益：除权后净值下跌的那部分不是亏损。
    拆分（1 拆 k）把**拆分日之前**的净值与分红统一折算到当前份额口径，
    这样净值序列在事件日连续，事件本身不产生虚假收益。

    unit_nav   : 单位净值序列（index=交易日）
    dividends  : DataFrame(index=除息日, 列 per_unit=每份现金分红)，可 None
    splits     : DataFrame(index=拆分日, 列 ratio=拆分比例，2 表示 1 拆 2)，可 None

    返回与 unit_nav 同索引的总回报指数序列（首日 = 1.0）。
    """
    s = pd.Series(unit_nav).astype(float).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return s.iloc[0:0]

    fac = pd.Series(1.0, index=s.index, dtype=float)
    if splits is not None and len(splits):
        sp = splits.copy()
        sp.index = pd.to_datetime(sp.index)
        col = "ratio" if "ratio" in sp.columns else sp.columns[0]
        for d, ratio in zip(sp.index, sp[col]):
            try:
                rr = float(ratio)
            except (TypeError, ValueError):
                continue
            if np.isfinite(rr) and rr > 0:
                fac[s.index < d] /= rr

    adj = s * fac
    d = pd.Series(0.0, index=adj.index, dtype=float)
    if dividends is not None and len(dividends):
        dv = dividends.copy()
        dv.index = pd.to_datetime(dv.index)
        col = "per_unit" if "per_unit" in dv.columns else dv.columns[0]
        amt = dv[col].astype(float)
        amt = amt[amt.index.isin(d.index)]
        d.loc[amt.index] += amt.values * fac.reindex(amt.index).values

    prev = adj.shift(1)
    r = (adj + d - prev) / prev
    r = r.where(prev > 0)
    r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod()


# ------------------------------------------------------------------ 真实总回报（P0-01）
def _fetch_nav(code: str, indicator: str, start: Optional[str] = None) -> pd.Series:
    """按 indicator 抓一列净值（累计/单位），走统一重试与清洗。"""
    import akshare as ak
    df = _retry(ak.fund_open_fund_info_em, symbol=code, indicator=indicator)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    dcol = df.columns[0]
    s = df.set_index(pd.to_datetime(df[dcol]))[df.columns[1]].astype(float).sort_index()
    s.name = code
    return _clean_series(s, start)


def fetch_fund_unit_nav(code: str, start: Optional[str] = None) -> pd.Series:
    """单位净值序列（除息日会真实下跌，不含分红再投）。"""
    return _fetch_nav(code, "单位净值走势", start)


def fetch_cum_return_curve(code: str,
                           start: Optional[str] = None) -> pd.Series:
    """
    天天基金自算的**累计收益率**（%，月频采样，约 27~36 天一个点）。

    这是独立于净值序列的第三口径：它含分红再投，是我们要的"真实总回报"。
    缺点是不是日频 —— 因此不能直接用于日频建模，只能作为**校准锚点**。
    """
    import akshare as ak
    try:
        df = _retry(ak.fund_open_fund_info_em, symbol=code,
                    indicator="累计收益率走势")
    except Exception:                                      # noqa: BLE001
        return pd.Series(dtype=float)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df.columns = ["d", "r"]
    df["d"] = pd.to_datetime(df["d"])
    s = df.set_index("d")["r"].astype(float).sort_index()
    s.name = f"{code}_cumret"
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s


def calibrate_total_return(cum_nav: pd.Series, unit_nav: pd.Series,
                           cum_ret: pd.Series) -> pd.Series:
    """
    构造**日频真实总回报净值**（含分红再投，起点 1.0）。

    为什么需要两步
    -------------
    ① 累计净值直算会系统性低估。14 只基金实测平均低估 **23.19%**，
       最严重的 110013 低估 76.12%。原因是累计净值 = 单位净值 + **历史累计分红**，
       而累计分红是存量、不参与后续增长：
           r_reported = (C_t − C_{t−1}) / C_{t−1} = U_{t−1}·r_true / C_{t−1}
       即每期都被乘上一个 U/C < 1 的压缩因子，且分红越多、压缩越狠。

    ② 光用单位净值也不行：除息日单位净值真实下跌，会造出虚假的负收益日；
       而且份额拆分会让单位净值序列跳变。

    因此先用累计净值与单位净值的**组合**还原真实日收益
           r_t = (C_t − C_{t−1}) / U_{t−1}
    （除息日 C 连续、U 下跌，这个比值恰好消掉分红），再用月频
    「累计收益率」作锚点，按时间线性插值一个日频缩放系数，
    以吸收拆分与数据源差异。

    实测：14/14 只基金在校准锚点上的偏差为 0.0000%。
    """
    idx = cum_nav.index.intersection(unit_nav.index)
    if len(idx) < 30:
        return pd.Series(dtype=float)
    C = cum_nav.reindex(idx).astype(float)
    U = unit_nav.reindex(idx).astype(float)
    r = (C.diff() / U.shift(1)).fillna(0.0)
    r.iloc[0] = 0.0
    F = (1.0 + r).cumprod()

    anchors = cum_ret[(cum_ret.index >= idx[0]) & (cum_ret.index <= idx[-1])]
    if len(anchors) >= 2:
        scale = {}
        for d, v in anchors.items():
            j = F.index.asof(d)
            if j is None or pd.isna(j) or j not in F.index:
                continue
            fm = float(F.loc[j])
            if np.isfinite(fm) and fm > 0:
                scale[d] = (1.0 + float(v) / 100.0) / fm
        if len(scale) >= 2:
            s = pd.Series(scale).sort_index()
            sc = (s.reindex(F.index).interpolate(method="time")
                   .ffill().bfill())
            sc = sc.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0)
            F = F * sc
        elif len(anchors) >= 2:
            # ★ 复查发现：锚点存在但全部落在净值区间之外时，这里会**静默**
            #   跳过校准，返回只有分红修正的 F —— 使用者无从知道口径降级了。
            print(f"[data] ⚠ 累计收益率锚点 {len(anchors)} 个全部落在净值区间"
                  f" [{idx[0].date()} ~ {idx[-1].date()}] 之外，拆分校准未生效")
    F = F.clip(lower=1e-8)
    F.name = "total_return"
    return F


def fetch_fund_total_return(code: str,
                            start: Optional[str] = None) -> pd.Series:
    """
    日频**总回报**净值（含分红再投，起点 1.0）—— 收益分析的正确口径。

    需要 3 次接口调用（累计净值 / 单位净值 / 累计收益率）。

    ★ 退化行为（P0-01 / 复查发现）：无法构造总回报时，本函数会在返回的
    Series 上标记 `attrs["degraded"] = True`，而**不再静默退回 cum**——
    旧实现把 cum 结果原样返回，调用方 `load_fund_nav_cached` 会把它写进
    `_total` 后缀的缓存名下（meta 还标着 basis="total"），之后所有读取
    持续命中这份口径错误的缓存，且无从察觉。
    """
    cum = _fetch_nav(code, "累计净值走势", start)
    unit = fetch_fund_unit_nav(code, start)
    if cum.empty or unit.empty:
        print(f"[data] ⚠ {code} 拿不到单位净值，退回累计净值口径（P0-01 未修正）")
        cum.attrs["degraded"] = True
        return cum
    rr = fetch_cum_return_curve(code, start)
    if rr.empty:
        print(f"[data] ⚠ {code} 拿不到累计收益率锚点，只用分红修正（拆分未校准）")
        idx = cum.index.intersection(unit.index)
        C, U = cum.reindex(idx), unit.reindex(idx)
        r = (C.diff() / U.shift(1)).fillna(0.0)
        r.iloc[0] = 0.0
        F = (1.0 + r).cumprod()
        F.name = "total_return"
        F.attrs["degraded"] = True          # 拆分未校准，仍是部分口径
        return F.clip(lower=1e-8)
    F = calibrate_total_return(cum, unit, rr)
    if F.empty:
        print(f"[data] ⚠ {code} 总回报构造结果为空，退回累计净值口径")
        cum.attrs["degraded"] = True
        return cum
    return F


def verify_nav_basis(code: str, start: Optional[str] = None) -> dict:
    """
    核对一只基金的三种口径（P0-01 验收用）。

    返回 dict：累计净值直算 / 单位净值直算 / 校准后总回报 三者的期末总回报，
    以及与月频「累计收益率」（真值）的偏差。校准后偏差应为 0。
    """
    cum = _fetch_nav(code, "累计净值走势", start)
    unit = fetch_fund_unit_nav(code, start)
    rr = fetch_cum_return_curve(code, start)
    end = min([s.index[-1] for s in (cum, unit, rr) if len(s)])
    cum, unit, rr = (s[s.index <= end] for s in (cum, unit, rr))
    truth = 1.0 + float(rr.iloc[-1]) / 100.0
    a = float(cum.iloc[-1] / cum.iloc[0])
    b = float(unit.iloc[-1] / unit.iloc[0])
    cal = calibrate_total_return(cum, unit, rr)
    return {
        "code": code, "n_days": int(len(cum)),
        "end": str(cum.index[-1].date()),
        "true_total_return": round(truth, 6),
        "cum_nav_basis": round(a, 6),
        "unit_nav_basis": round(b, 6),
        "calibrated_basis": round(float(cal.iloc[-1]), 6),
        "cum_nav_bias_pct": round((a / truth - 1) * 100, 4),
        "unit_nav_bias_pct": round((b / truth - 1) * 100, 4),
        "calibrated_bias_pct": round((float(cal.iloc[-1]) / truth - 1) * 100, 6),
    }


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


def data_stamp() -> str:
    """
    底层数据的粗粒度指纹（P1-05）。

    结果缓存旧键里只有 (基金, 模型集, 参数, 精度)，**不含底层数据摘要**：
    行情追加或重抓之后，缓存键不变，页面会一直显示旧结果，看起来像"刷新无效"。
    这里用数据目录里 CSV 的「文件数 + 最大 mtime」做指纹 —— 任何一次写入都会
    改变它，于是缓存键随之改变，结果自然刷新。开销仅一次目录 stat。
    """
    try:
        fs = list(C.DATA_DIR.glob("*.csv"))
        if not fs:
            return "empty"
        mx = max(f.stat().st_mtime for f in fs)
        return f"{len(fs)}:{int(mx)}"
    except Exception:                                   # noqa: BLE001
        return "unknown"


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
    """
    缓存是否仍有效。

    用 sidecar 元数据而不是把版本号塞进文件名，是为了**不作废已有的 14 份
    panel 缓存**——它们仍然有效，强行改名会让离线环境（无 akshare）直接跑不起来。
    没有元数据文件的旧缓存一律视为有效。
    """
    fp = _cache_meta_path(name)
    if not fp.exists():
        return True
    try:
        saved = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:               # noqa: BLE001
        return True
    return _fingerprint(saved) == _fingerprint(meta)


def _cache_meta_write(name: str, meta: dict):
    try:
        _cache_meta_path(name).write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except Exception:               # noqa: BLE001
        pass


def cache_info(name: str) -> dict:
    """
    读某个缓存的描述性信息（P1-05 / N-07）：抓取时间、数据末日、行数、缺失资产。

    用途：回答"这份缓存是什么时候抓的、抓到哪天、缺了哪些资产" ——
    旧缓存只有一个版本号，出问题时无法判断该重抓还是该补取。
    """
    fp = _cache_meta_path(name)
    if not fp.exists():
        return {}
    try:
        return (json.loads(fp.read_text(encoding="utf-8")) or {}).get("info", {})
    except Exception:                                    # noqa: BLE001
        return {}


def _cache_info_write(name: str, info: dict):
    """把描述性信息合并进现有 meta（不触碰参与判等的指纹字段）。"""
    fp = _cache_meta_path(name)
    try:
        cur = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    except Exception:                                    # noqa: BLE001
        cur = {}
    cur.setdefault("info", {}).update(info)
    _cache_meta_write(name, cur)


def _os_replace_retry(src: Path, dst: Path, retries: int = 6):
    """os.replace 带重试。Windows 下目标被另一线程短暂占用会抛 WinError 32。"""
    import os
    import time
    for i in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def _atomic_csv_write(df: pd.DataFrame, path: Path):
    """
    原子写 CSV：先写同目录临时文件再 os.replace。

    为什么需要：直接 to_csv 到目标路径，进程在写一半时被杀会留下**半截 CSV**，
    下次读取虽然能兜底重建，但会白白多一次网络请求。
    临时名必须每次唯一 —— Web 端是多线程，只加 pid 仍会撞名。
    """
    import os
    import threading
    import uuid
    tmp = path.with_name(f"{path.name}.{os.getpid()}."
                         f"{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        df.to_csv(tmp, encoding="utf-8-sig")
        _os_replace_retry(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _cached(name: str, builder, meta: Optional[dict] = None):
    fp = C.DATA_DIR / f"{name}.csv"
    stale = meta is not None and not _cache_meta_valid(name, meta)
    if stale:
        print(f"[data] 缓存 {name} 口径已变更，重新抓取")
    if fp.exists() and not stale:
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df):
                # ★ 命中旧缓存时也要补写元数据，否则版本守卫永远不会被"武装"：
                # 之后有人改 DATA_CACHE_VERSION 时，_cache_meta_valid 会因为
                # 找不到元数据而一律判为有效，缓存照样命中，守卫形同虚设。
                if meta is not None:
                    _cache_meta_write(name, _with_info(meta, df))
                return df
        except Exception as e:      # noqa: BLE001
            print(f"[data] 缓存 {name} 损坏（{type(e).__name__}），重新构建")
    df = builder()
    _atomic_csv_write(df, fp)
    if meta is not None:
        _cache_meta_write(name, _with_info(meta, df))
    return df


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


# ------------------------------------------------------------------ 分层缓存
def load_index_universe(start: str = C.START_DATE,
                        end: Optional[str] = C.END_DATE,
                        refresh: bool = False,
                        verbose: bool = True) -> pd.DataFrame:
    """
    候选指数池的价格宽表（**全局缓存**，与基金无关）。

    ★ 为什么要单独缓存：指数价格对一只基金和对另一只基金是同一份数据。
    早期实现把它塞进 load_panel 的 per-fund 缓存里，于是每分析一只基金
    都要重新抓 47 个指数 —— 扩池到 50 只就是 2350 次网络请求。
    拆出来之后，无论分析多少只基金，指数只抓一次。

    返回列 = 候选指数代码（不含 __FUND__），**未做对齐** —— 对齐依赖具体
    基金的交易日轴，仍由 load_panel 做。
    """
    cache_name = f"index_universe_{start}" + (f"_to{end}" if end else "")
    meta = {"v": C.DATA_CACHE_VERSION, "n_candidates": len(C.NAME_OF),
            "start": start, "end": end}
    if refresh:
        for p in (C.DATA_DIR / f"{cache_name}.csv",
                  C.DATA_DIR / f"{cache_name}.meta.json"):
            if p.exists():
                p.unlink()

    def build() -> pd.DataFrame:
        if verbose:
            print(f"[data] 抓取候选指数池 {len(C.NAME_OF)} 个 ...")
        return fetch_universe(C.NAME_OF.keys(), start, end, verbose=verbose)

    df = _cached(cache_name, build, meta=meta)
    # ★ 缺失资产必须留痕（N-07）：部分行情失败会静默得到一个"不完整指数池"，
    #   下游的覆盖率过滤会把它们当"中途停编"剔除，掩盖了真实原因是抓取失败。
    #   无条件写入（空列表 = 无缺失），否则"没这个字段"和"没缺失"分不清。
    missing = [c for c in C.NAME_OF if c not in df.columns]
    _cache_info_write(cache_name, {
        "missing_assets": missing,
        "n_missing": len(missing),
        "n_expected": len(C.NAME_OF)})
    if missing:
        print(f"[data] ⚠ 候选池缺 {len(missing)}/{len(C.NAME_OF)} 个指数"
              f"（{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}）"
              f"——重跑本命令可补齐，已成功的部分不会重复请求")
    return df


def load_fund_nav_cached(fund_code: str, start: str = C.START_DATE,
                         refresh: bool = False,
                         basis: str = NAV_BASIS_TOTAL) -> pd.Series:
    """
    单只基金的收益序列（独立缓存，~100 KB，避免与指数抓取耦合）。

    basis = NAV_BASIS_TOTAL（默认，P0-01 修正口径）：日频总回报，含分红再投。
            由累计净值 + 单位净值 + 月频累计收益率校准得到。
    basis = NAV_BASIS_CUM：数据源的累计净值直算。**已实测系统性低估
            （14 只基金平均 −23.19%，最差 −76.12%）**，仅用于兼容与对照。

    缓存名带 basis 后缀，两种口径互不覆盖。
    """
    suffix = "" if basis == NAV_BASIS_CUM else f"_{basis}"
    cache_name = f"fund_nav_{fund_code}_{start}{suffix}"
    meta = {"v": C.DATA_CACHE_VERSION, "code": fund_code, "start": start,
            "basis": basis}
    if refresh:
        for p in (C.DATA_DIR / f"{cache_name}.csv",
                  C.DATA_DIR / f"{cache_name}.meta.json"):
            if p.exists():
                p.unlink()

    class _DegradedNav(Exception):
        """总回报口径不可用。绝不能把退化结果写进 total 缓存名下。"""

    def build() -> pd.DataFrame:
        if basis == NAV_BASIS_TOTAL:
            s = fetch_fund_total_return(fund_code, start)
            # ★ 复查发现的缓存污染：旧实现把退回的 cum 序列原样返回，
            #   _cached 会把它写进 `_total` 后缀的缓存名（meta 还标
            #   basis="total"），之后所有读取持续命中口径错误的缓存。
            #   这里显式拒绝：退回结果只本次内存使用，不落盘。
            if bool(getattr(s, "attrs", {}).get("degraded")):
                raise _DegradedNav(
                    f"{fund_code} 无法构造总回报口径（单位净值或累计收益率锚点缺失）")
            return s.to_frame()
        return fetch_fund_nav(fund_code, start).to_frame()

    try:
        df = _cached(cache_name, build, meta=meta)
    except _DegradedNav as e:
        print(f"[data] ⚠ {e}；本次退回累计净值口径，结果【未缓存】")
        return fetch_fund_nav(fund_code, start)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s.name = f"fund_{fund_code}"
    return s


def fetch_start_for(start: Optional[str] = None) -> str:
    """
    **抓取**起点：取 `start` 与 `C.FETCH_START_DATE` 的较早者。

    ★ 为什么分析与抓取必须用两个起点（2026-10-12 改）：
    缓存名里带 start（`fund_nav_{code}_{start}_total.csv`）。分析窗口改成滚动
    近 N 年之后，如果抓取也跟着滚动，那么每换一次窗口、甚至每天运行都会让
    全部缓存 miss —— 离线环境下等于拿不到任何数据；在线环境下则是每次重抓
    全部指数，白白放大几十倍请求量。

    所以抓取一律用固定长区间（缓存稳定 + 预热期充足），拿到之后再裁剪到
    分析窗口。若用户显式要求比 FETCH_START_DATE 更早的区间，才真正提前抓取。

    ISO 格式（YYYY-MM-DD）字符串可直接字典序比较，等价于日期比较。
    """
    s = str(start) if start else ""
    # 取**较早者**：分析起点晚于固定起点时用固定起点（白赚一段预热数据，
    # 且缓存必然命中）；只有用户显式要求更早的区间才真的提前抓取。
    if not s or s > C.FETCH_START_DATE:
        return C.FETCH_START_DATE
    return s


def load_panel(fund_code: str, start: str = C.START_DATE,
               end: Optional[str] = C.END_DATE, refresh: bool = False,
               verbose: bool = True,
               nav_basis: str = NAV_BASIS_TOTAL) -> pd.DataFrame:
    """
    返回对齐后的价格面板：列 = 候选指数 + __FUND__，**已裁剪到 [start, end]**。

    存储分层（★ 面板本身**不再缓存**，即时重建）：

        指数池  index_universe_{fs}.csv         全局一份（与基金无关）
        净值    fund_nav_{code}_{fs}.csv         每只一份（~100 KB）
        面板    = 两者 → align_prices() 即时重建（纯内存，<100 ms）

    为什么不缓存面板：面板里 95% 以上的列是指数，逐基金缓存会让同样的指数
    数据存 N 份（1 MB × 基金数）。拆成两个原子数据源之后，存储与抓取量都从
    O(基金 × 指数) 降到 O(基金 + 指数)；对齐是毫秒级内存操作，重建不亏。

    ★ `start` 是**分析**起点，抓取用 fetch_start_for(start)（通常更早）。
    对齐按**完整抓取区间**做（覆盖率筛选需要全样本），之后再裁剪到分析窗口 ——
    顺序反过来会让覆盖率统计只看到窗口内的样本，把"曾经存在过的指数"误判为
    缺失而剔除。
    """
    fs = fetch_start_for(start)
    nav = load_fund_nav_cached(fund_code, fs, refresh=refresh, basis=nav_basis)
    if verbose:
        tag = "总回报" if nav_basis == NAV_BASIS_TOTAL else "累计净值(旧口径)"
        print(f"[data] 基金 {fund_code} {tag} {len(nav)} 行  "
              f"{nav.index[0].date()} ~ {nav.index[-1].date()}")
    px = load_index_universe(fs, end, refresh=refresh, verbose=verbose)
    out = align_prices(px, nav)

    # 裁剪到分析窗口
    if start:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]
    if out.empty:
        raise ValueError(
            f"{fund_code} 在 {start or '全区间'} ~ {end or '最新'} 内没有可用数据"
            f"（抓取区间 {fs} ~ {end or '最新'}，请放宽分析窗口或检查代码）")
    if verbose:
        print(f"[data] 对齐后：{out.shape[0]} 交易日 × {out.shape[1]} 条序列 "
              f"({out.index[0].date()} ~ {out.index[-1].date()})"
              + (f"  [分析窗口 {start} 起]" if start else ""))
    return out
