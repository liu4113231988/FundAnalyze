# -*- coding: utf-8 -*-
"""
回测框架：严格滚动样本外 + 信号有效性检验 + 对照实验。

无前瞻偏差的三条硬约束
----------------------
  1. t 日收盘后，只用 ≤ t 的数据估计风格权重与信号（RBSA 本身已滚动）；
  2. t 日的仓位建议，只能在 t+1 日生效；
  3. 交易成本按「仓位变动绝对值 × 单边费率」计提，并计入净值。

对照实验（关键设计）
--------------------
只把策略和「买入持有」比是不够的：择时策略平均仓位往往远低于 100%，
收益天然更低、回撤天然更小。必须加入
        ★ 恒定仓位对照组（仓位 = 策略的时间平均仓位）
才能分离出**真正的择时贡献**：
    - 策略 ≈ 恒定仓位  → 策略的全部差异来自"仓位水平"，择时零贡献
    - 策略 > 恒定仓位  → 择时真的有价值

★★ 关键口径：**每一个被测方案都必须配一份自己的恒定仓位对照**。
对照组只在"平均仓位一致"时才成立，因此主策略、每个 variant、每个消融分支
各自拿自己的时间平均仓位做对照（见返回值的 controls / timing_contrib_all）。
若拿主策略（平均 43%）的对照去减一个平均 100% 的方案，得到的"超额"
只是仓位差，不是择时能力——这是本项目最需要避免的自欺方式。

信号有效性指标
--------------
  · RankIC / ICIR : 综合分与未来 h 日收益的秩相关及其稳定性
  · 分组单调性     : 按综合分五等分，看未来收益是否单调
  · 择时胜率       : 建议仓位 > 50% 时未来 h 日上涨的比例
  · 年化换手       : 双边换手倍数
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ 绩效指标
def perf_metrics(ret: pd.Series, freq: int = 252, rf: float = None,
                 name: str = "") -> dict:
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    rf = C.BACKTEST["cash_rate"] if rf is None else rf
    n = len(ret)
    eq = (1 + ret).cumprod()
    years = n / freq
    ann = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = ret.std(ddof=1) * np.sqrt(freq)
    sharpe = (ann - rf) / vol if vol > 1e-9 else 0.0
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    calmar = ann / abs(mdd) if mdd < -1e-9 else 0.0
    monthly = (1 + ret).resample("ME").prod() - 1
    return {
        "方案": name,
        "年化收益": ann,
        "年化波动": vol,
        "夏普": sharpe,
        "最大回撤": mdd,
        "Calmar": calmar,
        "月胜率": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "累计收益": float(eq.iloc[-1] - 1),
        "样本天数": n,
    }


def _fwd_ret(ret: pd.Series, h: int) -> pd.Series:
    """未来 h 日累计收益（simple），对齐到 t 日。"""
    return (1 + ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1


def _rolling_spearman(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    """滚动 Spearman 秩相关（秩的 Pearson 相关），滑窗批量向量化。"""
    from scipy.stats import rankdata

    av, bv = a.to_numpy(float), b.to_numpy(float)
    n = len(av)
    out = np.full(n, np.nan)
    if n < win:
        return pd.Series(out, index=a.index)

    sw_view = np.lib.stride_tricks.sliding_window_view
    wa = sw_view(av, win)
    wb = sw_view(bv, win)
    valid = ~(np.isnan(wa).any(axis=1) | np.isnan(wb).any(axis=1))
    if not valid.any():
        return pd.Series(out, index=a.index)

    # NaN 窗口替换为 0，避免 rankdata 对 NaN 的行为未定义
    wa = np.where(np.isnan(wa), 0.0, wa)
    wb = np.where(np.isnan(wb), 0.0, wb)
    ra = rankdata(wa, axis=1)
    rb = rankdata(wb, axis=1)
    rac = ra - ra.mean(axis=1, keepdims=True)
    rbc = rb - rb.mean(axis=1, keepdims=True)
    den = np.sqrt((rac ** 2).sum(axis=1) * (rbc ** 2).sum(axis=1))

    corr = np.full(len(wa), np.nan)
    ok = valid & (den > 1e-12)
    corr[ok] = (rac[ok] * rbc[ok]).sum(axis=1) / den[ok]
    out[win - 1:] = corr
    return pd.Series(out, index=a.index)


def score_to_position(score: pd.Series, smooth: int = 5) -> pd.Series:
    """综合分 → 目标仓位（0~1），EMA 平滑。"""
    p = pd.Series(np.clip(0.5 + 0.5 * score.fillna(0).values, 0, 1),
                  index=score.index)
    return p.ewm(span=smooth, adjust=False).mean()


# ------------------------------------------------------------------ 主回测
def run_backtest(nav: pd.Series, sig: pd.DataFrame, cfg: Optional[dict] = None,
                 fund_name: str = "", variants: Optional[Dict[str, pd.Series]] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 main_name: str = "择时策略(融合)",
                 decision_trades: Optional[list] = None
                 ) -> dict:
    cfg = {**C.BACKTEST, **(cfg or {})}
    cost, cash_ann, warm, h = (cfg["cost_rate"], cfg["cash_rate"],
                               cfg["warmup"], cfg["fwd_horizon"])
    cash_d = (1 + cash_ann) ** (1 / 252) - 1

    nav = nav.dropna()
    r_f = nav.pct_change()
    idx = nav.index

    pos_main = sig["position_final"].reindex(idx).ffill().fillna(0.5)
    variant_pos = {main_name: pos_main, **dict(variants or {})}

    # ---- 回测窗口：预热起点与用户指定日期的交集，吸附到真实交易日
    # start_i 必须夹在 [0, n-30] 内：旧写法 min(warm, len(idx)-30) 在样本
    # 少于 30 天时会得到负索引，idx[-10] 会静默取到序列末尾的元素。
    n = len(idx)
    start_i = max(0, min(int(warm), n - 30)) if n > 30 else 0
    start = idx[start_i]
    if start_date:
        start = max(start, min(pd.Timestamp(start_date), idx[-1]))
    end = idx[-1]
    if end_date:
        end = min(end, max(pd.Timestamp(end_date), idx[0]))
    win_idx = idx[(idx >= start) & (idx <= end)]
    if len(win_idx) < 60:
        raise ValueError(
            f"回测区间过短：{start.date()} ~ {end.date()} 只有 {len(win_idx)} 个交易日，"
            f"至少需要 60 个。请检查 start_date / end_date。")
    start, end = win_idx[0], win_idx[-1]

    # ---- 恒定仓位对照：★ 每个方案各配一份，仓位 = 该方案在窗口内的时间平均仓位
    # 对照组只有与被比较方案"平均仓位一致"时，差值才代表择时贡献。
    # 旧实现只按主策略算一份对照、再拿它去减所有 variant：
    # 平均仓位 100% 的"买入持有 / 健康决策"减去平均仓位 43% 的对照，
    # 会得到 +12% 的表观"超额年化"—— 那纯粹是仓位差，不是择时能力。
    variant_pos = {k: v.reindex(idx).ffill().fillna(0.5)
                   for k, v in variant_pos.items()}
    ctrl_series: Dict[str, pd.Series] = {}
    controls: Dict[str, str] = {}
    avg_position_all: Dict[str, float] = {}
    for vname, pos_v in variant_pos.items():
        fm = float(pos_v.loc[start:end].mean())
        if not np.isfinite(fm):
            fm = 0.5
        avg_position_all[vname] = fm
        cname = f"恒定仓位({fm:.0%})"
        if cname not in ctrl_series:      # 平均仓位相同的方案共用同一份对照
            ctrl_series[cname] = pd.Series(fm, index=idx)
        controls[vname] = cname

    all_pos = {**variant_pos, **ctrl_series}
    all_pos["买入持有(100%)"] = pd.Series(1.0, index=idx)

    # ---- 逐方案回测
    series, rows, eqs = {}, [], {}
    dpos_oos_all: Dict[str, pd.Series] = {}
    for vname, pos_v in all_pos.items():
        pos_eff = pos_v.shift(1)
        dpos = pos_v.diff().abs().fillna(0)
        # 仓位 t 日决定、t+1 日执行，成本与换手均按执行日（dpos 滞后一期）计入
        dpos_exec = dpos.shift(1).fillna(0)
        r = pos_eff.fillna(0.5) * r_f.fillna(0) + (1 - pos_eff.fillna(0.5)) * cash_d
        r = (r - dpos_exec * cost)[lambda s: (s.index >= start) & (s.index <= end)]
        series[vname] = r
        eqs[vname] = (1 + r).cumprod()
        # 样本过短时 perf_metrics 返回 {}，补回"方案"列，
        # 否则 set_index("方案") 会得到 NaN 索引，下游 .loc[方案名] 全部 KeyError
        rows.append(perf_metrics(r, name=vname) or {"方案": vname})
        dpos_oos_all[vname] = dpos_exec.loc[start:end]

    metrics = pd.DataFrame(rows).set_index("方案")

    # ------------------------------ 信号有效性（窗口内）
    fwd = _fwd_ret(r_f, h)
    valid = pd.concat([sig["score"].rename("score"), fwd.rename("fwd")],
                      axis=1).dropna()
    valid = valid.loc[start:end]
    ic = float(valid["score"].corr(valid["fwd"], method="spearman")) if len(valid) > 30 else np.nan
    roll_ic = _rolling_spearman(valid["score"], valid["fwd"], 120)
    icir = float(roll_ic.mean() / roll_ic.std()) if roll_ic.std() > 1e-9 else np.nan

    groups = pd.DataFrame()
    try:
        q = pd.qcut(valid["score"], 5, labels=["Q1(最空)", "Q2", "Q3", "Q4", "Q5(最多)"])
        g = valid.groupby(q, observed=False)["fwd"]
        groups = pd.DataFrame({"平均前瞻收益": g.mean(), "波动": g.std(),
                               "样本数": g.size()})
    except Exception:      # noqa: BLE001
        pass

    score_oos = sig["score"].reindex(idx).ffill().loc[start:end]
    fwd_oos = fwd.reindex(score_oos.index)
    lm, sm = (score_oos > 0), (score_oos < 0)
    win_long = float((fwd_oos[lm] > 0).mean()) if lm.any() else np.nan
    win_short = float((fwd_oos[sm] < 0).mean()) if sm.any() else np.nan

    # ------------------------------ 择时净贡献（每个方案 − 它自己的恒定仓位对照）
    def _contrib(vname: str) -> dict:
        cn = controls.get(vname)
        if cn is None or vname not in metrics.index or cn not in metrics.index:
            return {}
        out = {}
        for m in ("年化收益", "夏普", "最大回撤", "Calmar"):
            try:
                out[m] = float(metrics.loc[vname, m] - metrics.loc[cn, m])
            except (KeyError, TypeError, ValueError):
                out[m] = float("nan")
        return out

    timing_contrib_all = {v: _contrib(v) for v in variant_pos}
    # 向后兼容：主策略的净贡献仍然是平铺字典（run.py / report.py / webapi 都读它）
    timing_contrib = timing_contrib_all.get(main_name, {})

    # ------------------------------ 交易记录：决策日 → 次一交易日执行
    trades_out = []
    if decision_trades:
        for t in decision_trades:
            later = idx[idx > t["date"]]
            if not len(later):
                continue
            exec_day = later[0]
            if start <= exec_day <= end:
                trades_out.append({**t, "exec_date": exec_day})

    years = max(len(series[main_name]) / 252, 1e-9)
    turnover_all = {v: float(d.sum() / years) for v, d in dpos_oos_all.items()}
    return {
        "equity": pd.DataFrame(eqs),
        "series": series,
        "position": pos_main.loc[start:end],
        "metrics": metrics,
        "ic": ic, "icir": icir, "roll_ic": roll_ic,
        "groups": groups,
        "win_long": win_long, "win_short": win_short,
        "turnover": turnover_all.get(main_name, float("nan")),
        "turnover_all": turnover_all,
        "avg_position": avg_position_all.get(main_name, float("nan")),
        "avg_position_all": avg_position_all,
        "timing_contrib": timing_contrib,
        "timing_contrib_all": timing_contrib_all,
        "controls": controls,
        "control_name": controls.get(main_name),
        "main_name": main_name,
        "window_start": start, "window_end": end,
        "trades": trades_out,
    }
