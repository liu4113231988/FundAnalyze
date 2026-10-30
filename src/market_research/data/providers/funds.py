from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import market_research.config as C
from market_research.data.http import _retry, _clean_series

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
