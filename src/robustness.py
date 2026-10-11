# -*- coding: utf-8 -*-
"""
稳健性检验：回答"这个结论的**有效样本量**到底有多大"。

背景
----
E4 报告"250 日窗口 + 120 日再平衡，14 只基金超额夏普**全部**为正"。
乍看是极强的证据，但这 14 只基金的样本外区间**完全重叠**（同为 2116 天），
且都是 A 股主动权益基金 —— 它们共享同一个市场因子，横截面并不独立。
如果超额主要来自共同的市场暴露，那"14/14 为正"只是**同一个信号被重复计了 14 次**。

本模块提供三个互补的检验：

1. ``effective_sample_size``  横截面相关矩阵 → 有效独立样本数
   n_eff = (Σλ)² / Σλ²   （λ 为相关系数矩阵的特征值）
   n_eff = 14 表示完全独立；n_eff ≈ 3 表示"14 只基金 ≈ 3 个独立观测"。

2. ``common_factor_analysis``  对共同因子做回归
   超额_t = α_i + β_i·m_t + ε_t，m 为沪深300。
   若 α_i 显著为正 → 超额不是市场暴露带来的，而是真正的择时。
   同时报告第一主成分解释的方差比例。

3. ``block_bootstrap``  移动块 bootstrap（保留时序自相关）
   **对所有基金使用同一组重采样日期**，从而保留横截面相关性 ——
   这是关键：如果各自独立重采样，就等于假设"14 个观测互不相关"，
   恰恰抹掉了我们要检验的东西。
   输出"净贡献为正的基金比例"的抽样分布与 p 值。

三条全部**只用回测产出的样本外日收益**，不引入任何新的前瞻。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C


# ====================================================================== 1. 有效样本量
def effective_sample_size(excess: pd.DataFrame) -> dict:
    """
    由横截面相关矩阵估计有效独立样本数。

    excess : index=日期, columns=基金, 值=该基金相对自身恒定仓位对照的**日**超额收益
    """
    cols = [c for c in excess.columns if excess[c].notna().sum() > 30]
    d = excess[cols].dropna()
    if len(cols) < 2 or len(d) < 60:
        return {"n_funds": len(cols), "n_eff": float("nan")}
    corr = d.corr().values
    corr = np.nan_to_num(corr)
    lam = np.linalg.eigvalsh(corr)
    lam = np.clip(lam.real, 0.0, None)
    n_eff = float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12))
    # 特征值谱：前几个特征值占多少
    order = np.sort(lam)[::-1]
    top1 = float(order[0] / max(order.sum(), 1e-12))
    return {
        "n_funds": int(len(cols)),
        "n_days": int(len(d)),
        "n_eff": n_eff,
        "n_eff_ratio": n_eff / len(cols),
        "pc1_share": top1,
        "avg_corr": float(np.nanmean(corr[np.triu_indices(len(cols), 1)])),
    }


# ====================================================================== 2. 共同因子
def common_factor_analysis(excess: pd.DataFrame,
                           market: pd.Series) -> Dict[str, pd.DataFrame]:
    """
    对每只基金的日超额做共同因子分析。

    超额 = 策略 − 对照，两者仓位相近，所以**方向暴露天然抵消**，
    市场 beta 本来就应接近 0（实测 −0.05 ~ +0.04，市场 R² < 2%）——
    这本身是个好消息：超额不是"低仓位"或"市场择时"的另一种写法。

    显著性用 **Newey-West 均值检验**而非回归系数的 t 值：
        t = x̄ / sqrt(LRV(x̄) / n)
    直接对 X=[1, m] 做两参数 NW 回归在数值上极不稳定（常数列与 0.01 量级
    收益列尺度悬殊，标准误会爆炸到 t≈1e-16），而我们要检验的本来
    就只是"超额均值是否为 0"。

    ★ P1-03 口径澄清：这里的 t_nw 检验的是**原始超额均值是否为 0**，
    它**不是**"扣掉市场因子后的 alpha 是否显著"。两者不是一回事 ——
    不能拿本函数的显著性去宣称"存在因子调整后的 alpha"。
    默认 lags=5 是固定值，未做带宽敏感性；需要时用 `hac_sensitivity()`。
    """
    m = np.log(market).diff().reindex(excess.index)
    rows = []
    for c in excess.columns:
        y = excess[c].dropna()
        mm = m.reindex(y.index)
        ok = y.notna() & mm.notna()
        y, mm = y[ok], mm[ok]
        if len(y) < 120:
            continue
        X = np.column_stack([np.ones(len(mm)), mm.values])
        beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        resid = y.values - X @ beta
        b = float(beta[1])
        t_nw = _nw_mean_t(y.values, lags=5)
        yv = float(y.values.var())
        r2_mkt = (float("nan") if yv <= 1e-18
                  else float(np.clip(1 - resid.var() / yv, -1.0, 1.0)))
        rows.append({
            "基金": c,
            "年化超额": float(y.mean()) * 252,
            "NW_t值": float(t_nw),
            "显著(5%)": bool(abs(t_nw) > 1.96),
            "市场beta": b,
            "市场R2": r2_mkt,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("NW_t值", ascending=False).reset_index(drop=True)
    return {"per_fund": df,
            "n_sig": int(df["显著(5%)"].sum()) if len(df) else 0,
            "n_total": int(len(df)),
            "n_pos_alpha": int((df["年化超额"] > 0).sum()) if len(df) else 0}


def _nw_mean_t(x: np.ndarray, lags: int = 5) -> float:
    """
    序列均值的 Newey-West t 统计量：t = x̄ / sqrt(LRV / n)

    LRV = γ₀ + 2 Σ_{l=1..L} w_l·γ_l，γ_l = Σ_{t=l}^{n−1}(x_t−x̄)(x_{t−l}−x̄)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 40:
        return float("nan")
    d = x - x.mean()
    lrv = float(d @ d) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        lrv += 2.0 * w * float(d[l:] @ d[:-l]) / n
    return float(x.mean() / np.sqrt(max(lrv, 1e-18) / n))


# ====================================================================== 3. 块 bootstrap
def block_bootstrap(excess: pd.DataFrame, sharpe_rf: float = 0.0,
                    block: int = 60, n_boot: int = 2000,
                    seed: int = 20261011) -> dict:
    """
    移动块 bootstrap：所有基金**共用同一组重采样日期**。

    为什么必须共用日期：如果每只基金各自独立重采样，就等于假设
    "14 个观测互不相关"，恰好抹掉了我们要检验的横截面相关性。
    共用日期 = 保留"这 14 只基金今天一起涨、一起跌"的事实。

    输出：
      * share_pos  观测到的"净贡献为正"基金比例
      * boot_share 该比例在零假设下的抽样分布
      * p_value   P(bootstrap 比例 ≥ 观测值)

    ★ 与 run_backtest 的净贡献口径保持一致：超额夏普 = IR = 年化超额 / 年化超额波动，
      **不扣无风险利率**（超额已经是相对基准的收益，再扣 rf 会把零超额算成负值）。
    """
    d = excess.dropna()
    n, m = d.shape
    if n < block * 3 or m < 2:
        return {"share_pos": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    vals = d.values
    n_blocks = int(np.ceil(n / block))
    max_start = n - block

    def sharpe_of(mat: np.ndarray) -> np.ndarray:
        mu = mat.mean(axis=0) * 252
        sd = mat.std(axis=0, ddof=1) * np.sqrt(252)
        sd = np.where(sd > 1e-9, sd, np.nan)
        return (mu - sharpe_rf) / sd

    # 零假设：把超额全部平移成"均值为 0"，再用同一批重采样日期
    centered = vals - vals.mean(axis=0, keepdims=True)
    share = np.empty(n_boot)
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    for b in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in starts[b]])
        idx = idx[idx < n][:n]
        if len(idx) < n:
            idx = np.resize(idx, n)
        share[b] = float(np.nanmean(sharpe_of(centered[idx]) > 0))

    return {
        "share_pos": float(np.nanmean(sharpe_of(vals) > 0)),
        "boot_mean": float(share.mean()),
        "boot_p05": float(np.quantile(share, 0.05)),
        "boot_p95": float(np.quantile(share, 0.95)),
        "p_value": float(np.mean(share >= float(np.nanmean(sharpe_of(vals) > 0)))),
        "block": block, "n_boot": n_boot,
    }


# ====================================================================== 3b. 敏感性 / 多重比较
def bootstrap_sensitivity(excess: pd.DataFrame, sharpe_rf: float = 0.0,
                          blocks=(20, 60, 120, 250), n_boot: int = 1000,
                          seed: int = 20261011) -> pd.DataFrame:
    """
    块长敏感性（P1-03）。

    默认块长 60 是拍出来的。若结论只在某一个块长下成立，那它依赖的是
    调参而不是数据 —— 必须把 p 值随块长的变化一起报出来。
    """
    rows = []
    for b in blocks:
        r = block_bootstrap(excess, sharpe_rf=sharpe_rf, block=int(b),
                            n_boot=int(n_boot), seed=seed)
        rows.append({"块长": int(b),
                     "为正比例": r["share_pos"],
                     "p值": r["p_value"],
                     "零假设5%分位": r["boot_p05"],
                     "零假设95%分位": r["boot_p95"]})
    return pd.DataFrame(rows)


def hac_sensitivity(excess: pd.DataFrame, lags=(0, 5, 10, 21)) -> pd.DataFrame:
    """NW 带宽敏感性（P1-03）：逐基金超额均值 t 值随滞后的变化。"""
    rows = []
    for c in excess.columns:
        y = excess[c].dropna().values
        if len(y) < 120:
            continue
        row = {"基金": c}
        for L in lags:
            row[f"t(lag={int(L)})"] = float(_nw_mean_t(y, lags=int(L)))
        rows.append(row)
    return pd.DataFrame(rows)


def multiplicity_adjust(pvals, names=None, alpha: float = 0.05,
                        method: str = "fdr"):
    """
    多重比较校正（P1-03）。

    从 108 组参数网格或十几个模型里挑出最大值，再拿单次检验的 α=0.05
    去判定"显著"，等于把假阳性率放大了几十倍 —— E5 里"37% 的格子天然为正"
    就是这个效应的直接证据。

    method:
      "bonferroni"  阈值 α/m，最保守，适合候选数少、结论要硬的场合
      "fdr"         Benjamini-Hochberg，控制错误发现率，候选多时更实用

    返回 (DataFrame, 说明文字)。
    """
    p = np.asarray([float(x) for x in pvals], dtype=float)
    m = int(np.isfinite(p).sum())
    names = list(names) if names is not None else [f"#{i}" for i in range(len(p))]
    if m == 0:
        return pd.DataFrame({"检验": names, "p值": p, "拒绝原假设": False}), "无有效 p 值"

    ok = np.isfinite(p)
    pv = p[ok]
    rej = np.zeros(m, dtype=bool)
    if method == "bonferroni":
        thr = alpha / m
        rej = pv <= thr
        note = f"Bonferroni（m={m}）：阈值 {alpha}/{m} = {thr:.5f}，拒绝 {int(rej.sum())} 个"
    else:
        order = np.argsort(pv)
        ranked = pv[order]
        line = alpha * (np.arange(1, m + 1) / float(m))
        passed = np.where(ranked <= line)[0]
        cut = int(passed[-1]) + 1 if len(passed) else 0
        rej[order[:cut]] = True
        note = (f"BH-FDR（m={m}, α={alpha}）：{cut}/{m} 个通过"
                f"（最大通过 p = {ranked[cut-1]:.4f}）" if cut else
                f"BH-FDR（m={m}, α={alpha}）：无一项通过")

    reject_full = np.zeros(len(p), dtype=bool)
    reject_full[ok] = rej
    return pd.DataFrame({"检验": names, "p值": p,
                         "拒绝原假设": reject_full}), note


# ====================================================================== 编排
def assess_positions(nav_by_fund: Dict[str, pd.Series],
                     pos_by_fund: Dict[str, pd.Series],
                     panel: pd.DataFrame,
                     label: str = "策略") -> dict:
    """
    对任意"仓位序列"做完整的稳健性检验（通用入口）。

    E4 那条长周期线索、以及波动率目标，都先用它评估 ——
    避免重复"表面上 9/14 为正就当有效果"的错误。
    """
    from .backtest import run_backtest

    market = panel["sh000300"].astype(float) if "sh000300" in panel.columns else None
    excess: Dict[str, pd.Series] = {}
    diag: Dict[str, dict] = {}
    for code, nav in nav_by_fund.items():
        pos = pos_by_fund.get(code)
        if pos is None:
            continue
        try:
            p = pos.reindex(nav.index).ffill().fillna(1.0)
            sig = pd.DataFrame({"position_final": p,
                                "score": pd.Series(0.5, index=p.index)})
            bt = run_backtest(nav, sig, main_name=label)
        except Exception as e:                            # noqa: BLE001
            diag[code] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
            continue
        ctrl = bt["control_name"]
        d = (bt["series"][label] - bt["series"][ctrl])
        d = d[d.index >= bt["window_start"]]
        excess[code] = d
        diag[code] = {
            "超额夏普": bt["timing_contrib"].get("夏普"),
            "超额年化": bt["timing_contrib"].get("年化收益"),
            "平均仓位": bt["avg_position_all"].get(label),
            "换手": bt["turnover_all"].get(label),
            "对照": ctrl,
        }
    if not excess:
        return {"ok": False, "diag": diag, "label": label}
    df = pd.concat(excess, axis=1)
    df.columns = [C.EXTENDED_FUNDS.get(c, c) for c in df.columns]
    df = df.dropna()
    out = {"ok": True, "label": label, "per_fund": pd.DataFrame(diag).T,
           "excess": df,
           "effective": effective_sample_size(df),
           "bootstrap": block_bootstrap(df)}
    if market is not None:
        out["factor"] = common_factor_analysis(df, market)
    return out


def assess_signal_edge(nav_by_fund: Dict[str, pd.Series],
                       syn_by_fund: Dict[str, pd.Series],
                       panel: pd.DataFrame,
                       weights_by_fund: Dict[str, pd.DataFrame],
                       market_col: str = "sh000300") -> dict:
    """
    对一格信号配置（这里是 250 日窗口 + 120 日再平衡）做完整的稳健性检验。

    返回：逐基金的样本外超额序列 + 三项检验的结论。
    """
    from .backtest import run_backtest
    from .signal import SignalEngine

    try:
        from .data import load_pe_panel
        pe = load_pe_panel(verbose=False)
    except Exception:                                       # noqa: BLE001
        pe = None
    market = panel[market_col].astype(float) if market_col in panel.columns else None

    excess: Dict[str, pd.Series] = {}
    diag: Dict[str, dict] = {}
    for code, nav in nav_by_fund.items():
        syn = syn_by_fund[code]
        w = weights_by_fund[code]
        cfg = dict(trend_ma=250, trend_mom=250, trend_slope=250,
                   w_trend=1.0, w_dev=0.0, w_val_pe=0.0)
        try:
            sig = SignalEngine(cfg).compute(nav, syn, w, pe)
        except Exception as e:                            # noqa: BLE001
            diag[code] = {"error": f"signal: {type(e).__name__}: {e}"}
            continue
        raw = sig["position_raw"].copy()
        mask = np.zeros(len(raw), dtype=bool)
        mask[::120] = True                                  # 季度再平衡
        sampled = raw.where(pd.Series(mask, index=raw.index)).ffill().bfill()
        try:
            bt = run_backtest(nav, sig, variants={"低频再平衡": sampled},
                              main_name="低频再平衡")
        except Exception as e:                            # noqa: BLE001
            diag[code] = {"error": f"backtest: {type(e).__name__}: {e}"}
            continue
        key, ctrl = "低频再平衡", bt["control_name"]
        d = (bt["series"][key] - bt["series"][ctrl])
        d = d[d.index >= bt["window_start"]]
        excess[code] = d
        # ★ P1-10：旧实现把两个**几何**夏普相减当作"超额夏普"，而夏普不可加
        #   （实测同一份数据上"为正基金数"会从 14/14 变成 9/14，足以翻转结论）。
        #   正确口径是 run_backtest 在超额日序列上算的 IR。
        #   旧值保留在 sharpe_diff_geometric 里，仅供对照，不得用于判定。
        _tc = bt["timing_contrib_all"].get(key, {}) or {}
        diag[code] = {
            "sharpe": float(_tc.get("夏普", float("nan"))),
            "ann": float(_tc.get("年化收益", float("nan"))),
            "sharpe_diff_geometric": float(bt["metrics"].loc[key, "夏普"]
                                           - bt["metrics"].loc[ctrl, "夏普"]),
            "ann_diff_geometric": float(bt["metrics"].loc[key, "年化收益"]
                                        - bt["metrics"].loc[ctrl, "年化收益"]),
                      "avg_pos": float(bt["avg_position_all"].get(key, np.nan)),
                      "control": ctrl}

    if not excess:
        return {"ok": False, "diag": diag}
    df = pd.concat(excess, axis=1)
    df.columns = [C.EXTENDED_FUNDS.get(c, c) for c in df.columns]
    df = df.dropna()

    out = {
        "ok": True,
        "per_fund": pd.DataFrame(diag).T,
        "excess": df,
        "effective": effective_sample_size(df),
        "bootstrap": block_bootstrap(df),
    }
    if market is not None:
        out["factor"] = common_factor_analysis(df, market)
    return out