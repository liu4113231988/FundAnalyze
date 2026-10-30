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

★ 可交易性边界（P1-01，尚未建模）
--------------------------------
本框架**没有**建模真实交易的时点与成本结构，因此它的输出是"研究口径的诊断"，
不是可执行的交易方案。具体缺失：

  1. 收盘后发布的基金净值何时可知、申赎以哪个净值成交、资金何时到账，
     均未建模。当前假设"t 日信号 → t+1 日收盘到收盘收益"，对场外基金偏乐观。
  2. 固定目标仓位隐含**持续再平衡**。`pos.diff()` 只统计目标仓位的变化，
     未必等于维持该比例所需的实际成交量。
  3. 成本只有单一的 `cost_rate`（按仓位变动计费）：不含申购费/持有期相关的
     赎回费、滑点、初始建仓与终止清算。E10 用三档费率做了敏感性，
     但那是"费率参数扫描"，不是真实的费用结构。
  4. 「Σw 约束成立」只说明合成基准在统计上自洽，**不能**证明它可交易。

日志收益加权用于拟合、自融资组合收益用于绩效，这两者也需要区分。
在这些建模完成之前，回测数字只能用于横向比较与排序，不能当作预期收益。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import market_research.config as C


# ------------------------------------------------------------------ 绩效指标
def perf_metrics(ret: pd.Series, freq: int = 252, rf: float = None,
                 name: str = "") -> dict:
    """
    绩效指标。三个口径必须分清，不能混用：

      · **年化收益** = 几何年化（CAGR）：(∏(1+r))^(252/n) − 1。
        它是"净值实际增长了多少"，**不可加**——两个方案的 CAGR 之差没有意义。
      · **夏普**   = (几何年化 − rf) / 年化波动。沿用几何年化口径，
        因此也**不可加**（见 run_backtest 里 timing_contrib 的说明）。
      · **算术夏普** = (mean(r) − rf/252) × 252 / (std(r) × √252)。
        这是教科书 Sharpe 的常见写法，**可加**，用于需要跨方案做差或
        做统计检验的场合。
      · **IR（信息比率）** 在 run_backtest._contrib 里单独定义：
        分子是**超额**日收益序列的均值×252，分母是该超额序列的年化波动。
        它衡量的是"相对某个基准"，不是"相对无风险利率"，分子不再扣 rf。

    最大回撤从**初始资本 1.0** 起算（净值曲线前置一个 1.0 点），
    否则首日亏损不会被计入回撤。
    """
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
    # 算术夏普：可加口径，用于跨方案求差与显著性检验
    rf_d = (1 + rf) ** (1 / freq) - 1
    ex_arith = ret.mean() - rf_d
    sharpe_arith = (ex_arith * freq / vol) if vol > 1e-9 else 0.0

    # 回撤：净值曲线前置初始资本 1.0，避免漏掉首日亏损
    vals = np.concatenate([[1.0], (1.0 + ret.to_numpy()).cumprod()])
    peak = np.maximum.accumulate(vals)
    dd = pd.Series(vals[1:] / peak[1:] - 1.0, index=ret.index)
    mdd = float(dd.min())
    calmar = ann / abs(mdd) if mdd < -1e-9 else 0.0
    monthly = (1 + ret).resample("ME").prod() - 1
    return {
        "方案": name,
        "年化收益": ann,
        "年化波动": vol,
        "夏普": sharpe,
        "算术夏普": sharpe_arith,
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
    control_labels: Dict[str, str] = {}
    avg_position_all: Dict[str, float] = {}
    _ctrl_by_fm: Dict[float, str] = {}    # 精确平均暴露 → 对照唯一键
    _used: set = set()
    for vname, pos_v in variant_pos.items():
        # ★ 平均暴露按**实际生效的持仓** pos_eff = pos_v.shift(1) 计算：
        #   窗口第一天执行的是 start 前一日决定的仓位（回测口径同理），
        #   直接用 pos_v 会把首日算成"决策日仓位"，与真实暴露差一档。
        fm = float(pos_v.shift(1).fillna(0.5).loc[start:end].mean())
        if not np.isfinite(fm):
            fm = 0.5
        avg_position_all[vname] = fm
        if fm in _ctrl_by_fm:             # 平均暴露完全相同 → 合法共用同一份对照
            controls[vname] = _ctrl_by_fm[fm]
            continue
        # ★ 对照键必须携带足够精度：旧实现用 .0% 四舍五入，导致 50.1% 与
        #   50.4% 两个方案共用"恒定仓位(50%)"，后者凭空多出虚假超额。
        #   这里用 .1%，并在展示名仍撞车时追加序号，保证**身份唯一**。
        base = f"恒定仓位({fm:.1%})"
        cname, k = base, 2
        while cname in _used:
            cname = f"{base}#{k}"
            k += 1
        _used.add(cname)
        _ctrl_by_fm[fm] = cname
        ctrl_series[cname] = pd.Series(fm, index=idx)
        control_labels[cname] = base
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
    # ★ 分母只计**有效前瞻标签**：窗口尾部 fwd_horizon 天没有完整未来收益，
    #   fwd 为 NaN，而 `NaN > 0` 会被判为 False —— 旧实现把它们算进分母，
    #   于是"持续上涨样本"的多头胜率也不是 100%（窗口越短，拉低越多）。
    fl, fs = fwd_oos[lm].dropna(), fwd_oos[sm].dropna()
    win_long = float((fl > 0).mean()) if len(fl) else np.nan
    win_short = float((fs < 0).mean()) if len(fs) else np.nan
    n_valid_label = int(len(fl) + len(fs))

    # ------------------------------ 择时净贡献（每个方案 − 它自己的恒定仓位对照）
    # ★ 夏普与年化收益都**不可加**，所以不能拿 metrics 表里的两行相减：
    #   perf_metrics 的年化是几何年化 (∏(1+r))^(252/n)−1，而几何年化之差
    #   ≠ 差值的年化。实测同一份数据，两种口径给出的"为正基金数"
    #   从 14/14 变成 9/14 —— 足以翻转结论。
    # 正确口径是直接在**超额日收益序列**上算（信息比率 IR）：
    #   超额年化 = mean(r_v − r_c) × 252
    #   超额夏普 = 超额年化 / std(r_v − r_c) × √252
    # ★ 分子**不扣无风险利率**：超额已经是"相对某个基准"的收益，
    #   再扣一次 rf 会把"零超额"算成负值，系统性拉负全部结果
    #   （实测 14 只基金的平均超额夏普从 −0.19 掉到 −0.88，纯属口径错误）。
    def _contrib(vname: str) -> dict:
        cn = controls.get(vname)
        if cn is None or vname not in metrics.index or cn not in metrics.index:
            return {}
        out = {}
        try:
            ex = (series[vname] - series[cn]).dropna()
            ann_ex = float(ex.mean() * 252)
            vol_ex = float(ex.std(ddof=1) * np.sqrt(252))
            out["年化收益"] = ann_ex
            if vol_ex > 1e-9:
                out["夏普"] = ann_ex / vol_ex
            else:
                # 超额恒为 0（方案 ≡ 对照）时 IR 无定义，记 0 而不是 NaN，
                # 否则"满仓方案 vs 恒定仓位(100%)"这种天然为 0 的情形
                # 会在表格里显示成空值，看起来像计算失败。
                out["夏普"] = 0.0 if abs(ann_ex) < 1e-9 else float("nan")
        except (KeyError, TypeError, ValueError, IndexError):
            out["年化收益"] = out["夏普"] = float("nan")
        # 回撤不可加（无"超额回撤"的自然定义），故只给描述性的差值
        for m in ("最大回撤", "Calmar"):
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
        "n_valid_label": n_valid_label,
        "control_labels": control_labels,
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
