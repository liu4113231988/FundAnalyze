# -*- coding: utf-8 -*-
"""
诊断实验模块：回答"信号到底有没有信息、在哪个时间尺度上有信息"。

三个实验
--------
  E1 IC 期限结构   : RankIC 在不同前瞻期（5/20/60/120/250 日）上的表现。
                     动量型信号若与短期收益负相关、与中长期正相关，
                     说明问题不在"没信息"，而在"持有期错配"。
  E2 通道消融      : 趋势 / 相对偏离 / 赛道估值 三个通道单独与组合，
                     全部与「恒定仓位对照」比较，分离真实择时贡献。
  E3 参数敏感性    : 窗口长度、通道权重、估值压缩系数做网格，
                     检验结论是否稳健（还是某个参数下的偶然结果）。
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config as C
from .backtest import perf_metrics, score_to_position, _fwd_ret
from .signal import SignalEngine

HORIZONS = (5, 20, 60, 120, 250)


# ====================================================================== E1
def ic_term_structure(sig: pd.DataFrame, nav: pd.Series,
                      horizons=HORIZONS) -> pd.DataFrame:
    """各通道信号 → 不同前瞻期的 RankIC / ICIR。"""
    r = nav.pct_change()
    cols = {
        "综合分": "score", "趋势通道": "trend_score",
        "相对偏离通道": "dev_score", "赛道估值通道": "val_score",
    }
    rows = []
    for label, col in cols.items():
        if col not in sig.columns:
            continue
        for h in horizons:
            v = pd.concat([sig[col].rename("s"), _fwd_ret(r, h).rename("f")],
                          axis=1).dropna()
            if len(v) < 100:
                rows.append({"信号": label, "前瞻期": f"{h}日", "样本": len(v),
                             "RankIC": np.nan, "ICIR": np.nan, "t值": np.nan})
                continue
            ic = float(v["s"].corr(v["f"], method="spearman"))
            # 分块 IC（每块 60 日）用于算 ICIR
            blocks = [v.iloc[i:i + 60] for i in range(0, len(v) - 59, 60)]
            ics = [float(b["s"].corr(b["f"], method="spearman")) for b in blocks
                   if len(b) > 20]
            icir = float(np.mean(ics) / np.std(ics)) if np.std(ics) > 1e-9 else np.nan
            t = ic * np.sqrt(len(v) / max(h, 1))     # 重叠样本的粗略 t 值
            rows.append({"信号": label, "前瞻期": f"{h}日", "样本": len(v),
                         "RankIC": ic, "ICIR": icir, "t值": t})
    return pd.DataFrame(rows)


# ====================================================================== E2
def channel_ablation(sig: pd.DataFrame, nav: pd.Series,
                     cfg: Optional[dict] = None,
                     syn_nav: Optional[pd.Series] = None) -> pd.DataFrame:
    """各通道单独回测（含恒定仓位对照）。"""
    from .backtest import run_backtest

    variants = {
        "仅趋势通道": score_to_position(sig["trend_score"]),
        "仅相对偏离通道": score_to_position(sig["dev_score"]),
        "仅赛道估值通道": score_to_position(sig["val_score"]),
        "趋势+估值(去偏离)": score_to_position(
            # 剔除偏离通道后，按原 0.50:0.15 的相对比例重新归一
            0.77 * sig["trend_score"].fillna(0) + 0.23 * sig["val_score"].fillna(0)),
    }
    # 波动率目标：风险归一化，不需要信号，适合放在消融表里对照
    if syn_nav is not None:
        vt = SignalEngine().vol_target_channel(syn_nav)
        variants["波动率目标"] = vt["vol_target_pos"].reindex(nav.index).ffill().fillna(1.0)
    bt = run_backtest(nav, sig, cfg=cfg, variants=variants)
    m = bt["metrics"].copy()
    # ★ 每个方案减自己的恒定仓位对照（平均仓位一致才可比）。
    # 旧实现让所有方案都减主策略那一份对照，于是平均仓位 100% 的方案
    # 会被记上一笔纯粹来自仓位差的"超额"。对照行与买入持有没有对照，记 NaN。
    ca = bt["timing_contrib_all"]
    out = m.copy()
    for col, key in (("超额夏普_vs对照", "夏普"),
                     ("超额年化_vs对照", "年化收益"),
                     ("超额回撤_vs对照", "最大回撤")):
        out[col] = [ca.get(v, {}).get(key, np.nan) for v in out.index]
    out["对照组"] = [bt["controls"].get(v) for v in out.index]
    return out


# ====================================================================== E3
def param_sensitivity(nav: pd.Series, syn_nav: pd.Series, weights: pd.DataFrame,
                      pe: Optional[pd.DataFrame], base_cfg: Optional[dict] = None
                      ) -> pd.DataFrame:
    """信号参数网格：只重算信号层，风格权重复用（省时且口径一致）。"""
    from .backtest import run_backtest

    grid = {
        "trend_slope": [60, 120, 250],
        "w_trend": [0.0, 0.5, 1.0],
        "kappa": [0.5, 1.0, 2.0],
    }
    rows = []
    combos = list(product(grid["trend_slope"], grid["w_trend"], grid["kappa"]))
    for i, (slope_w, wt, kappa) in enumerate(combos, 1):
        cfg = {"trend_slope": slope_w, "trend_ma": max(20, slope_w // 2),
               "trend_mom": slope_w, "w_trend": wt,
               "w_dev": (1 - wt) * 0.7, "w_val_pe": (1 - wt) * 0.3,
               "kappa": kappa}
        try:
            eng = SignalEngine(cfg)
            sig = eng.compute(nav, syn_nav, weights, pe)
            bt = run_backtest(nav, sig, cfg=base_cfg)
            m = bt["metrics"]
            ctrl = bt["control_name"]
            # 同上：超额夏普取自超额序列的 IR，不能用两个夏普相减
            ca = bt["timing_contrib_all"].get("择时策略(融合)", {})
            rows.append({
                "趋势窗口": slope_w, "趋势权重": wt, "kappa": kappa,
                "策略夏普": m.loc["择时策略(融合)", "夏普"],
                "对照夏普": m.loc[ctrl, "夏普"],
                "超额夏普": ca.get("夏普", np.nan),
                "超额年化": ca.get("年化收益", np.nan),
                "策略年化": m.loc["择时策略(融合)", "年化收益"],
                "对照年化": m.loc[ctrl, "年化收益"],
                "RankIC": bt["ic"],
            })
        except Exception as e:      # noqa: BLE001
            print(f"  [sensitivity] 组合{i} 失败 {type(e).__name__}: {e}")
    return pd.DataFrame(rows)


# ====================================================================== E4
def lowfreq_grid(nav: pd.Series, syn_nav: pd.Series, weights: pd.DataFrame,
                 pe: Optional[pd.DataFrame], refit_days=(5, 20, 60, 120),
                 windows=(60, 120, 250)) -> pd.DataFrame:
    """
    E4 · 周期匹配实验：趋势信号的时间尺度必须与调仓周期匹配。

    E1 发现趋势通道在 120/250 日前瞻下 RankIC 转正（+0.10 ~ +0.11），
    但日频调仓为负。本实验把「信号窗口」与「再平衡周期」同时扫一遍，
    检验低频化能否把这部分信息兑现出来。
    """
    from .backtest import run_backtest

    rows = []
    for win in windows:
        cfg = dict(trend_ma=win, trend_mom=win, trend_slope=win,
                   w_trend=1.0, w_dev=0.0, w_val_pe=0.0)
        try:
            sig = SignalEngine(cfg).compute(nav, syn_nav, weights, pe)
        except Exception as e:      # noqa: BLE001
            print(f"  [lowfreq] win={win} 信号失败 {e}")
            continue

        for R in refit_days:
            # 只在每 R 天更新仓位，其余时间持有不动
            raw = sig["position_raw"].copy()
            mask = np.zeros(len(raw), dtype=bool)
            mask[::R] = True
            sampled = raw.where(pd.Series(mask, index=raw.index)).ffill().bfill()
            # ★ main_name 必须是被评估的那个方案，否则恒定仓位对照会按
            # sig["position_final"]（日频 EMA+死区）的平均仓位构造，
            # 与 sampled（每 R 天采样的 position_raw）的平均仓位不一致，
            # 两者相差的几个百分点会被全额记进"超额夏普"。
            # 旧代码正是这么写的，E4 的结论因此带有仓位偏差。
            bt = run_backtest(nav, sig, variants={"低频再平衡": sampled},
                              main_name="低频再平衡")
            m = bt["metrics"]
            ctrl = bt["control_name"]
            # ★ 超额夏普必须由 run_backtest 在**超额日序列**上算（信息比率口径），
            #   不能用 metrics 表里两个几何夏普相减 —— 夏普不可加。
            #   旧写法在同一份数据上给出 100% 为正，正确口径下只有 9/14。
            ca = bt["timing_contrib_all"].get("低频再平衡", {})
            avg_pos = float(sampled.reindex(nav.index).ffill()
                            .loc[bt["window_start"]:bt["window_end"]].mean())
            rows.append({
                "信号窗口": win, "再平衡周期": R,
                "策略年化": m.loc["低频再平衡", "年化收益"],
                "对照年化": m.loc[ctrl, "年化收益"],
                "策略夏普": m.loc["低频再平衡", "夏普"],
                "对照夏普": m.loc[ctrl, "夏普"],
                "超额夏普": ca.get("夏普", np.nan),
                "超额年化": ca.get("年化收益", np.nan),
                "策略回撤": m.loc["低频再平衡", "最大回撤"],
                "对照回撤": m.loc[ctrl, "最大回撤"],
                "平均仓位": avg_pos,
                "对照组": ctrl,
            })
    return pd.DataFrame(rows)


# ====================================================================== 汇总
def summarize_funds(results: Dict[str, dict]) -> pd.DataFrame:
    """跨基金汇总：择时净贡献与 IC 的一致性检验。"""
    rows = []
    for code, res in results.items():
        bt = res["backtest"]
        m = bt["metrics"]
        ctrl = bt["control_name"]
        rows.append({
            "代码": code, "名称": res["name"],
            "样本天数": int(m.loc["择时策略(融合)", "样本天数"]),
            "策略夏普": m.loc["择时策略(融合)", "夏普"],
            "对照夏普": m.loc[ctrl, "夏普"],
            "净贡献_夏普": bt["timing_contrib"]["夏普"],
            "净贡献_年化": bt["timing_contrib"]["年化收益"],
            "净贡献_回撤": bt["timing_contrib"]["最大回撤"],
            "RankIC": bt["ic"], "ICIR": bt["icir"],
            "多头胜率": bt["win_long"], "平均仓位": bt["avg_position"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df.loc["均值"] = ["—", "平均"] + list(df.iloc[:, 2:].mean().round(4))
    return df
