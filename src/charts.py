# -*- coding: utf-8 -*-
"""图表模块：把所有可视化渲染成 base64 PNG，嵌进 HTML 报告（不依赖外网）。"""
from __future__ import annotations

import base64
import io
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import config as C

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "text.color": "#1e293b",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "font.size": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfdfe",
    "legend.frameon": False,
})

# 中国习惯：红涨绿跌
UP, DOWN = "#d92b37", "#0f9d58"
C_FUND, C_SYN, C_STRAT = "#2563eb", "#94a3b8", "#d92b37"


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _pct(ax, axis="y"):
    f = FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def _rebase(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s / s.iloc[0]


def _col(eq: pd.DataFrame, prefix: str) -> Optional[str]:
    for c in eq.columns:
        if c.startswith(prefix):
            return c
    return None


# ------------------------------------------------------------------ 1 净值对比
def chart_equity(res: dict) -> str:
    eq = res["backtest"]["equity"]
    nav = res["nav"].reindex(eq.index)
    syn = res["syn_nav"].reindex(eq.index)
    c_main = _col(eq, "择时策略") or eq.columns[0]
    c_ctrl = _col(eq, "恒定仓位")
    c_bh = _col(eq, "买入持有") or eq.columns[-1]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(eq.index, _rebase(nav), color=C_FUND, lw=1.5, label="基金净值（买入持有）")
    ax.plot(eq.index, _rebase(syn), color=C_SYN, lw=1.3, ls="--",
            label="合成基准（风格 beta）")
    if c_ctrl:
        ax.plot(eq.index, _rebase(eq[c_ctrl]), color="#f59e0b", lw=1.4, ls="-.",
                label=f"对照：{c_ctrl}")
    ax.plot(eq.index, _rebase(eq[c_main]), color=C_STRAT, lw=2.0,
            label=f"{c_main}")
    ax.set_title(f"{res['name']}（{res['code']}）· 净值曲线对比")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylabel("累计净值（起点=1）")
    return _b64(fig)


# ------------------------------------------------------------------ 2 风格暴露演变
def chart_exposure(res: dict, top_n: int = 6) -> str:
    w = res["weights_hist"]
    tot = w.sum(0).sort_values(ascending=False)
    keep = [c for c in tot.index[:top_n] if tot[c] > 0]
    sub = w[keep].copy()
    other = w.drop(columns=keep).sum(1)
    if other.abs().sum() > 1e-6:
        sub["其他"] = other
    labels = [C.NAME_OF.get(c, c) for c in sub.columns]

    palette = ["#2563eb", "#0ea5e9", "#f59e0b", "#10b981", "#8b5cf6",
               "#ef4444", "#64748b", "#14b8a6"]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.stackplot(sub.index, [sub[c].values for c in sub.columns],
                 labels=labels, colors=palette[:len(sub.columns)], alpha=0.9)
    ax.set_title("风格暴露演变（滚动 RBSA 权重）")
    ax.set_ylabel("权重")
    ax.set_ylim(0, 1.0)
    _pct(ax)
    ax.legend(loc="upper center", ncol=min(len(sub.columns), 6),
              bbox_to_anchor=(0.5, -0.13))
    ax.margins(x=0)
    return _b64(fig)


# ------------------------------------------------------------------ 3 偏离与估值
def chart_deviation(res: dict) -> str:
    sig = res["signal"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})

    ax = axes[0]
    spread = sig["spread"].dropna()
    ax.plot(spread.index, spread, color="#334155", lw=1.3,
            label="价差 = log(基金) − log(合成基准)")
    long_ma = spread.rolling(C.SIGNAL["z_long"], min_periods=60).mean()
    sd = spread.rolling(C.SIGNAL["z_long"], min_periods=60).std()
    ax.plot(long_ma.index, long_ma, color="#f59e0b", lw=1.2, ls="--",
            label=f"{C.SIGNAL['z_long']}日均值（中枢）")
    ax.fill_between(spread.index, long_ma - 2 * sd, long_ma + 2 * sd,
                    color="#f59e0b", alpha=0.10, label="±2σ 区间")
    ax.set_title("相对偏离：基金相对自身风格基准的累计超额（贵 / 便宜）")
    ax.legend(loc="upper left", ncol=2)

    ax2 = axes[1]
    z = sig["z_spread"]
    ax2.plot(z.index, z, color="#7c3aed", lw=1.4, label="偏离度 z-score")
    ax2.axhline(1.5, color=UP, lw=1, ls=":")
    ax2.axhline(-1.5, color=DOWN, lw=1, ls=":")
    ax2.axhline(0, color="#94a3b8", lw=0.8)
    ax2.fill_between(z.index, 0, z.where(z > 0), color=UP, alpha=0.18)
    ax2.fill_between(z.index, 0, z.where(z < 0), color=DOWN, alpha=0.18)
    ax2.text(0.01, 0.9, "↑ 偏贵（相对透支）", transform=ax2.transAxes,
             color=UP, fontsize=8.5)
    ax2.text(0.01, 0.05, "↓ 偏便宜（相对落后）", transform=ax2.transAxes,
             color=DOWN, fontsize=8.5)
    ax2.set_title("偏离度 z-score（>1.5 相对高估，<−1.5 相对低估）")
    ax2.legend(loc="upper right")
    return _b64(fig)


# ------------------------------------------------------------------ 4 仓位与信号
def chart_position(res: dict) -> str:
    sig = res["signal"]
    score = sig["score"]
    pos = sig["position_final"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True)

    ax = axes[0]
    ax.fill_between(score.index, 0, score.clip(lower=0), color=UP, alpha=0.35)
    ax.fill_between(score.index, 0, score.clip(upper=0), color=DOWN, alpha=0.35)
    ax.plot(score.index, score, color="#1e293b", lw=1.0)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_title("综合信号分（趋势 0.5 + 相对偏离 0.35 + 赛道估值 0.15）")
    ax.set_ylim(-1.05, 1.05)

    ax2 = axes[1]
    ax2.fill_between(pos.index, 0, pos, color="#2563eb", alpha=0.25)
    ax2.plot(pos.index, pos, color="#2563eb", lw=1.6, label="建议仓位")
    ax2.axhline(0.5, color="#94a3b8", lw=0.8, ls=":")
    ax2.set_ylim(0, 1.02)
    _pct(ax2)
    ax2.set_title("建议仓位（0 = 空仓，100% = 满仓）")
    ax2.legend(loc="upper left")
    return _b64(fig)


# ------------------------------------------------------------------ 5 回撤
def chart_drawdown(res: dict) -> str:
    eq = res["backtest"]["equity"]
    c_main = _col(eq, "择时策略") or eq.columns[0]
    c_ctrl = _col(eq, "恒定仓位")
    c_bh = _col(eq, "买入持有") or eq.columns[-1]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for col, color, alpha in ((c_bh, C_FUND, 0.18), (c_main, C_STRAT, 0.35)):
        dd = eq[col] / eq[col].cummax() - 1
        ax.fill_between(dd.index, 0, dd, color=color, alpha=alpha,
                        label=f"{col}（最大回撤 {dd.min():.1%}）")
        ax.plot(dd.index, dd, color=color, lw=1.1)
    if c_ctrl:
        dd = eq[c_ctrl] / eq[c_ctrl].cummax() - 1
        ax.plot(dd.index, dd, color="#f59e0b", lw=1.2, ls="-.",
                label=f"{c_ctrl}（最大回撤 {dd.min():.1%}）")
    ax.set_title("回撤对比")
    _pct(ax)
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    return _b64(fig)


# ------------------------------------------------------------------ 6 IC 分组
def chart_ic(res: dict) -> Optional[str]:
    g = res["backtest"]["groups"]
    if g is None or g.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2),
                            gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    vals = g["平均前瞻收益"].values
    colors = [DOWN if v >= 0 else UP for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(g.index, fontsize=8.5)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_title(f"按信号五分组的前瞻{C.BACKTEST['fwd_horizon']}日收益")
    _pct(ax)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2%}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.margins(y=0.22)

    ax2 = axes[1]
    ric = res["backtest"]["roll_ic"].dropna()
    ax2.plot(ric.index, ric, color="#7c3aed", lw=1.0)
    ax2.axhline(0, color="#94a3b8", lw=0.8)
    ax2.fill_between(ric.index, 0, ric, color="#7c3aed", alpha=0.2)
    ax2.set_title(f"滚动 RankIC（均值 {res['backtest']['ic']:.3f}）")
    return _b64(fig)


# ------------------------------------------------------------------ 7 当前状态仪表
def chart_gauge(res: dict) -> str:
    """四象限定位图：横轴趋势分，纵轴偏离分。"""
    sig = res["signal"].dropna(subset=["score"])
    t = sig["trend_score"]
    d = sig["dev_score"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.axvline(0, color="#94a3b8", lw=1)
    ax.add_patch(plt.Rectangle((0, 0), 1.05, 1.05, color=UP, alpha=0.07))
    ax.add_patch(plt.Rectangle((-1.05, 0), 1.05, 1.05, color="#f59e0b", alpha=0.08))
    ax.add_patch(plt.Rectangle((-1.05, -1.05), 1.05, 1.05, color="#22c55e", alpha=0.07))
    ax.add_patch(plt.Rectangle((0, -1.05), 1.05, 1.05, color="#94a3b8", alpha=0.10))
    ax.scatter(t.iloc[:-1], d.iloc[:-1], s=5, color="#94a3b8", alpha=0.35)
    ax.plot(t.iloc[-120:], d.iloc[-120:], color="#2563eb", lw=1.3, alpha=0.8,
            label="近 120 日轨迹")
    ax.scatter([t.iloc[-1]], [d.iloc[-1]], s=190, color="#d92b37",
               edgecolor="white", zorder=5, label="当前位置")
    for tx, ty, s, c in [(0.55, 0.88, "趋势↑ 相对低估\n（最佳买点）", UP),
                         (-0.98, 0.88, "趋势↓ 相对低估\n（小仓试探）", "#15803d"),
                         (0.55, -0.98, "趋势↑ 相对高估\n（持有警惕）", "#b45309"),
                         (-0.98, -0.98, "趋势↓ 相对高估\n（减仓离场）", "#64748b")]:
        ax.text(tx, ty, s, fontsize=8, color=c, ha="left", va="top" if ty > 0 else "bottom")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("趋势分（合成基准）")
    ax.set_ylabel("相对偏离分（估值）")
    ax.set_title("当前状态定位")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.15)
    return _b64(fig)


# ------------------------------------------------------------------ 8 对照实验
def chart_variants(res: dict) -> str:
    """各方案横向对比 + 择时净贡献——研究的核心证据图。"""
    m = res["backtest"]["metrics"].copy()
    order = ["择时策略(融合)", "仅趋势通道", "仅相对偏离通道", "仅赛道估值通道"]
    ctrl = res["backtest"]["control_name"]
    order = [o for o in order if o in m.index] + [ctrl, "买入持有(100%)"]
    m = m.loc[[o for o in order if o in m.index]]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    specs = [("年化收益", "年化收益", True), ("夏普", "夏普比率", False),
             ("最大回撤", "最大回撤（越浅越好）", True)]
    for ax, (col, title, is_pct) in zip(axes, specs):
        vals = m[col].values
        labels = [i.replace("(", "\n(").replace("通道", "") for i in m.index]
        colors = []
        for i in m.index:
            if i == ctrl:
                colors.append("#f59e0b")
            elif i == "买入持有(100%)":
                colors.append(C_FUND)
            elif i.startswith("择时策略"):
                colors.append(C_STRAT)
            else:
                colors.append("#cbd5e1")
        ax.barh(range(len(vals)), vals, color=colors, alpha=0.9, height=0.62)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#94a3b8", lw=0.8)
        ax.set_title(title)
        if is_pct:
            _pct(ax, "x")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1%}" if is_pct else f" {v:.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.margins(x=0.28)
    fig.suptitle(f"{res['name']} · 对照实验：择时是否真的创造了价值？",
                 fontsize=12, fontweight="bold", y=1.04)
    return _b64(fig)


ALL_CHARTS = {
    "equity": chart_equity,
    "exposure": chart_exposure,
    "deviation": chart_deviation,
    "position": chart_position,
    "drawdown": chart_drawdown,
    "ic": chart_ic,
    "gauge": chart_gauge,
    "variants": chart_variants,
}


def build_all(res: dict) -> Dict[str, str]:
    out = {}
    for k, fn in ALL_CHARTS.items():
        try:
            v = fn(res)
            if v:
                out[k] = v
        except Exception as e:      # noqa: BLE001
            print(f"[charts] {k} 失败: {type(e).__name__}: {e}")
    return out
