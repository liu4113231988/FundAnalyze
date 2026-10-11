# -*- coding: utf-8 -*-
"""
信号引擎：把「合成基准」翻译成可执行的趋势 / 估值判断，再融合成仓位建议。

三个通道（互相独立、各自可解释）
--------------------------------
  ① 趋势通道 trend_score   —— 作用于**合成基准**（基金的风格 beta）
        · 均线结构   log(S) − log(MA(S, n))
        · 区间动量   log(S) − log(S_{t−n})
        · 趋势斜率   log(S) 对时间滚动回归的年化斜率
        三者各自滚动 z-score 后等权平均，再 tanh 压到 [−1, 1]

  ② 相对偏离 dev_score     —— 作用于**基金 vs 合成基准的价差**
        spread = log(NAV) − log(S)      （基金相对其风格基准的累计超额）
        z = 滚动 z-score(spread, L)
        z > 0 → 基金相对自身风格"透支/偏贵" → dev_score < 0

  ③ 赛道估值 val_score     —— 作用于**风格基准的绝对估值水位**
        用基准成分加权得到的 PE 历史分位（月度数据前向填充）
        分位越高 → 赛道越贵 → val_score < 0

融合
----
       score    = w_trend·trend + w_dev·dev + w_val·val        ∈ [−1, 1]
       目标仓位 = clip(0.5 + 0.5·score, 0, 1)
       再经 EMA 平滑 + 调仓死区，得到最终建议仓位。

象限语义（报告里给用户看）
----
   趋势↑ + 估值低 → 最佳买点        趋势↑ + 估值高 → 持有但警惕
   趋势↓ + 估值低 → 小仓试探        趋势↓ + 估值高 → 减仓/离场
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config as C

# 候选指数 → 乐咕乐股 PE 接口可用名称（月度，2005 年起）
PE_SYMBOL = {
    "sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500",
    "sh000852": "中证1000", "sh000903": "中证100", "sh000009": "上证380",
}
PE_FALLBACK = "沪深300"


# ------------------------------------------------------------------ 工具函数
def _rz(s: pd.Series, win: int) -> pd.Series:
    """滚动 z-score（至少一半窗口有效才计算）。"""
    m = s.rolling(win, min_periods=max(20, win // 2)).mean()
    v = s.rolling(win, min_periods=max(20, win // 2)).std()
    return (s - m) / v.replace(0, np.nan)


def _roll_slope(s: pd.Series, win: int) -> pd.Series:
    """对数价格对时间的滚动 OLS 斜率（已年化，单位：1/年）。"""
    y = np.log(s)
    x = np.arange(win, dtype=float)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    w = pd.Series(xc, index=range(win))
    slope = y.rolling(win).apply(
        lambda a: float(np.dot(a - a.mean(), xc) / denom), raw=True)
    return slope * 252.0


def _tanh(x, k=1.0):
    return np.tanh(k * np.asarray(x, dtype=float))


# ------------------------------------------------------------------ 主引擎
class SignalEngine:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = dict(C.SIGNAL)
        if cfg:
            self.cfg.update(cfg)

    # -------------------------------------------------- 三通道
    def trend_channel(self, syn_nav: pd.Series) -> pd.DataFrame:
        c = self.cfg
        ls = np.log(syn_nav)
        ma = syn_nav.rolling(c["trend_ma"], min_periods=c["trend_ma"] // 2).mean()
        gap = ls - np.log(ma)
        mom = ls - ls.shift(c["trend_mom"])
        slope = _roll_slope(syn_nav, c["trend_slope"])

        win = c["trend_slope"]
        z_gap, z_mom, z_slope = _rz(gap, win), _rz(mom, win), _rz(slope, win)
        raw = pd.concat([z_gap, z_mom, z_slope], axis=1).mean(axis=1)
        score = pd.Series(_tanh(raw.values, c["kappa"]), index=syn_nav.index)
        return pd.DataFrame({"ma_gap": gap, "momentum": mom, "slope": slope,
                             "z_ma_gap": z_gap, "z_mom": z_mom, "z_slope": z_slope,
                             "trend_raw": raw, "trend_score": score})

    def deviation_channel(self, nav: pd.Series, syn_nav: pd.Series) -> pd.DataFrame:
        c = self.cfg
        spread = np.log(nav) - np.log(syn_nav)
        z = _rz(spread, c["z_long"])
        score = pd.Series(_tanh(-z.values, c["kappa_z"]), index=spread.index)
        # 短期动能：最近 z_short 日的超额，用于判断偏离是否还在扩大
        short_excess = spread - spread.shift(c["z_short"])
        return pd.DataFrame({"spread": spread, "z_spread": z,
                             "short_excess": short_excess, "dev_score": score})

    def valuation_channel(self, weights_hist: pd.DataFrame,
                          pe_pct: Optional[pd.DataFrame],
                          idx: pd.DatetimeIndex) -> pd.Series:
        """风格基准的绝对估值分位 → val_score ∈ [−1, 1]。"""
        c = self.cfg
        if pe_pct is None or pe_pct.empty:
            return pd.DataFrame({"pe_pct": pd.Series(index=idx, dtype=float),
                                 "val_score": pd.Series(0.0, index=idx)})

        # 资产列 → PE 标的（无映射或对应列缺失时用宽基兜底）
        sym_of = {}
        for a in weights_hist.columns:
            s = PE_SYMBOL.get(a)
            sym_of[a] = s if s in pe_pct.columns else (
                PE_FALLBACK if PE_FALLBACK in pe_pct.columns else None)

        # 各资产权重按 PE 标的聚合，避免逐行逐资产 reindex
        syms = sorted({s for s in sym_of.values() if s})
        w_sym = pd.DataFrame(0.0, index=weights_hist.index, columns=syms)
        for a, s in sym_of.items():
            if s is not None:
                w_sym[s] += weights_hist[a].fillna(0.0)

        # 月度 PE asof 对齐到每个重估日。
        # reindex(method="ffill") 在**索引单调递增**的前提下等价于 asof 连接，
        # 会取"不晚于该日的最近一期"，正是我们要的语义（早期注释说不能用它，是错的）。
        # ★ 但要叠加发布滞后：乐咕乐股的月末 PE（如 2020-01-31）要到次月初才发布，
        # 直接 ffill 会让 1 月 31 日当天就"知道"1 月末的估值，含 1~10 天前瞻。
        lag = int(c.get("pe_lag_days", 10))
        pe_src = pe_pct[syms]
        if lag > 0:
            pe_src = pe_src.copy()
            pe_src.index = pe_src.index + pd.Timedelta(days=lag)
        pe_v = pe_src.reindex(w_sym.index, method="ffill")
        w_eff = w_sym.where(pe_v.notna(), 0.0)
        denom = w_eff.sum(axis=1).replace(0.0, np.nan)
        pe_agg = (w_sym * pe_v.fillna(0.0)).sum(axis=1) / denom

        pct = pe_agg.reindex(idx, method="ffill")
        score = pd.Series(_tanh(-(pct.values - 0.5) * 2.0, c["kappa_z"]), index=idx)
        return pd.DataFrame({"pe_pct": pct, "val_score": score})

    # -------------------------------------------------- 波动率目标（风险归一化）
    def vol_target_channel(self, syn_nav: pd.Series,
                           idx: Optional[pd.Index] = None) -> pd.DataFrame:
        """
        波动率目标：把仓位按"已实现波动 / 目标波动"缩放，而不是预测涨跌方向。

            w_t = clip(σ_target / σ̂_t(L), w_min, w_max)

        为什么它值得单独做（而前面三通道不值得）：

        1. **不预测方向。** 三通道日频 IC 在所有前瞻期均为负（E1），
           说明方向判断没有信息；而"波动率聚集"是 A 股被反复验证的统计事实，
           不是待验假设。
        2. **参数极少。** 只有 σ_target 与观测窗 L 两个参数，
           几乎不存在网格过拟合的空间 —— 而 E3/E5 显示，
           本项目所有"调参调出来的正贡献"都扛不住样本外检验。
        3. **换手天然低。** 波动率变化慢，季度调仓即可，
           正好避开场外基金 1.5% 申购 / 0.5% 赎回的费用陷阱。
        4. **它改善的是风险调整后收益**，不是绝对收益 ——
           所以必须与「同平均仓位的恒定仓位对照」比，不能与买入持有比。

        σ̂ 用**合成基准**的已实现波动（而非基金自身）：合成基准更平滑、
        更少受个股噪声影响，但也因此会低估基金自身的特质波动。

        ★ mode="inverse" 反转方向：波动率高时**加仓**。这不是拍脑袋，
        本项目实测（14 只基金，各自的恒定仓位对照）：

        | 模式          | 超额年化 | 超额夏普 | 为正基金     |
        |-------------|--------|---------|-----------|
        | 顺波动（降波）    | **−1.32%** | −0.398 | 2 / 14  |
        | 逆波动（升波）    | **+0.36%** | +0.177 | 9 / 14  |

        顺波动失败**不是实现问题**：实现波动 14.8%，精准命中 15% 的目标，
        换手仅 0.45x。失败的原因是**风险平价假设在这批样本上不成立** ——
        波动率与收益同向，机械降波动等于系统性放弃收益。
        """
        c = self.cfg
        win = int(c.get("vol_target_win", 60))
        tgt = float(c.get("vol_target", 0.15))
        lo = float(c.get("vol_target_min", 0.30))
        hi = float(c.get("vol_target_max", 1.00))
        mode = str(c.get("vol_target_mode", "target"))

        r = np.log(syn_nav).diff()
        rv = r.rolling(win, min_periods=max(10, win // 2)).std() * np.sqrt(252)
        rv = rv.replace(0.0, np.nan)
        ratio = (rv / tgt) if mode == "inverse" else (tgt / rv)
        raw = ratio.clip(lo, hi)
        # 限频：波动率目标本身不需要频繁调仓
        reb = int(c.get("vol_target_rebalance", 60))
        if reb > 1:
            mask = np.zeros(len(raw), dtype=bool)
            mask[::reb] = True
            pos = raw.where(pd.Series(mask, index=raw.index)).ffill().bfill()
        else:
            pos = raw
        pos = pos.fillna(1.0)
        if idx is not None:
            pos = pos.reindex(idx, method="ffill").fillna(1.0)
        return pd.DataFrame({"rv": rv, "vol_target_pos": pos})

    # -------------------------------------------------- 融合
    def compute(self, nav: pd.Series, syn_nav: pd.Series,
                weights_hist: pd.DataFrame,
                pe_pct: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        c = self.cfg
        t = self.trend_channel(syn_nav)
        d = self.deviation_channel(nav, syn_nav)
        v = self.valuation_channel(weights_hist, pe_pct, syn_nav.index)

        df = pd.concat([t, d, v], axis=1)
        wt, wv, wl = c["w_trend"], c["w_dev"], c["w_val_pe"]
        # 权重全为 0 时分母为 0 → score 变 inf/NaN，且不会报错，
        # 只会让下游仓位全部塌到 0.5。这是配置错误，明确抛出。
        denom = float(wt) + float(wv) + float(wl)
        if denom <= 0:
            raise ValueError(
                f"三通道权重之和必须为 0 以外的值，当前 "
                f"w_trend={wt}, w_dev={wv}, w_val_pe={wl}")
        df["score"] = (wt * df["trend_score"].fillna(0)
                       + wv * df["dev_score"].fillna(0)
                       + wl * df["val_score"].fillna(0)) / denom

        raw_pos = np.clip(0.5 + 0.5 * df["score"], 0.0, 1.0)
        df["position_raw"] = raw_pos
        df["position"] = raw_pos.ewm(span=c["smooth"], adjust=False).mean()

        # 调仓死区：偏离不足阈值就不动，控制换手
        pos, cur = [], 0.5
        for p in df["position"].values:
            if abs(p - cur) >= c["rebalance_band"]:
                cur = p
            pos.append(cur)
        df["position_final"] = pos

        df["quadrant"] = [_quadrant(a, b) for a, b in
                          zip(df["trend_score"], df["dev_score"])]
        return df

    # -------------------------------------------------- 拟合健康度通道
    def fit_health_channel(self, nav: pd.Series, syn_ret: pd.Series,
                           fit_diag: pd.DataFrame,
                           weights_hist: pd.DataFrame) -> pd.DataFrame:
        """
        把拟合诊断翻译成「健康分」∈ [−1,1]。四个独立的失效模式：

          c_r2    拟合优度的绝对水位（<0.6 → 风格模型基本失效）
          c_te    跟踪误差相对两年中枢的扩张（TE 翻倍 → 风险结构突变）
          c_alpha 基金相对自身风格篮子的实现 alpha 的 t 值（显著为负 → 跑不赢复制组合）
          c_drift 相邻重估点的权重变动（漂移剧烈 → 风格不稳定）

        聚合方式：**取四者的最小值（短板决定 / 或门）**。
        这四个分量描述的是互相独立的失效模式，任何一个单独发生就足以触发风控，
        取平均会让三个正常分量把一个严重恶化的分量稀释掉。

        ★ 这里踩过一次致命的坑：早期实现取四分量的**均值**，而 c_r2 在正常基金上
        恒为 +1（R² 0.85~0.97 → (0.85−0.6)/0.3 ≈ 0.83~1.0），于是 health 的
        理论下界只有 (1 + (−1)×3)/4 = −0.5，离场线 −0.5 永远够不到。
        实测后果：14 只基金全部零交易，「拟合健康决策」退化成买入持有。

        映射阈值均事先固定且有经济含义，不依据回测结果调整。
        """
        c = self.cfg
        idx = syn_ret.index
        nav = nav.reindex(idx)

        if fit_diag is not None and not fit_diag.empty:
            diag = fit_diag[["r2", "te"]].reindex(idx, method="ffill")
        else:
            diag = pd.DataFrame(index=idx, columns=["r2", "te"], dtype=float)
        r2, te = diag["r2"], diag["te"]

        # ★ 四个分量一律用「相对自身两年中枢」的口径，绝对水位一概不用。
        # 原因：不同基金的 R² 与权重漂移的**绝对水平**天生差很多
        # （消费基金 R² 0.95、轮动型基金 R² 0.6；20 日重估的 drift 天然偏大）。
        # 用绝对阈值判"不健康"，等价于永久判那些低 R² / 高换手的基金出局
        # —— 实测会让 14 只基金里 13 只触发、其中 4 只空仓 95% 以上。
        win_l = int(c.get("health_win_long", 504))
        r2_base = r2.rolling(win_l, min_periods=120).median()
        te_base = te.rolling(win_l, min_periods=120).median()
        te_ratio = te / te_base

        # 实现 alpha：基金日对数收益 − 合成基准日收益
        active = np.log(nav).diff() - syn_ret
        win_s = int(c.get("health_win_short", 252))
        roll = active.rolling(win_s, min_periods=120)
        alpha_ann = roll.mean() * 252.0
        # ★ 必须除以标准误，不能拿固定刻度（原来是 5%）去量 alpha。
        # 252 日窗口下 alpha 的年化标准误约等于年化 TE（10% 量级），
        # 用 5% 做分母会让这个分量在**纯噪声**下也几乎总是饱和到 ±1，
        # 即它输出的不是"alpha 变差了"，而是"这次抽样的符号"。
        alpha_se = roll.std() / np.sqrt(roll.count())
        alpha_t = roll.mean() / alpha_se.replace(0.0, np.nan)
        t_scale = float(c.get("health_alpha_t", 2.0))

        if weights_hist is not None and not weights_hist.empty:
            dw = weights_hist.diff().abs().sum(axis=1)
            drift = dw.rolling(3, min_periods=1).mean().reindex(idx, method="ffill")
        else:
            drift = pd.Series(np.nan, index=idx)
        drift_base = drift.rolling(win_l, min_periods=120).median()
        drift_ratio = drift / drift_base

        # 拟合可信度：R² 衡量的是"我的模型能不能解释这只基金"，
        # 与"这只基金该不该卖"是两件不同的事。R² 低 → 证据不可信，
        # 正确响应是停用门控（见 build_decision），而不是卖出基金。
        r2_min = float(c.get("health_r2_min", 0.5))
        fit_reliable = ((r2 - (r2_min - 0.15)) / 0.15).clip(0.0, 1.0)

        comps = pd.DataFrame({
            # 只看相对自身中枢的下降。刻度 20pp 而不是 10pp：
            # 实测 R² 的**正常**波动幅度就有 ±7pp（110022 十年间 0.83~0.97），
            # 10pp 的刻度落在噪声带内，健康基金也会被判失效。
            "c_r2": ((r2 - r2_base) / 0.20).clip(-1, 1),
            # TE 相对两年中枢扩张 60% 归零、扩张 120% 触底。
            # 原式 (1.6 − ratio) 未归一化：ratio=1 时就已经是 +0.6，
            # 要跌到负值需要 TE 扩张 60% 以上，跌到 −0.5 需要扩张 110%，
            # 配合"取均值"的聚合方式后这一项事实上从不参与决策。
            "c_te": ((1.6 - te_ratio) / 0.6).clip(-1, 1),
            "c_alpha": (alpha_t / t_scale).clip(-1, 1),
            # 漂移同理取相对，但刻度更宽：20 日重估下 Σ|Δw| 的逐期波动本身就很大
            # （实测各基金常态 0.04~0.61），触发线设在"漂移速度达到常态 2.5 倍"。
            "c_drift": (2.0 - drift_ratio).clip(-1, 1),
        })
        # 聚合：最差项与整体水平各占一半。
        #   · 纯 min（或门）会被最吵的单个分量主导 —— 实测 14 只基金全部被打到 −1；
        #   · 纯 mean 会被三个正常分量稀释到永不触发（−0.5 数学上够不到）。
        # 折中后的语义是清楚的：**至少一项严重恶化，且整体也在恶化，才动手**；
        # 单独一项抖动（其余三项为 +1）只能得到 0.6×(−1)+0.4×0.5 = −0.4，不触发。
        worst = comps.min(axis=1)
        avg = comps.mean(axis=1)
        health_raw = (0.6 * worst + 0.4 * avg).fillna(0.0)
        health = health_raw.ewm(span=int(c.get("health_smooth", 5)),
                                adjust=False).mean()

        return pd.DataFrame({
            "fit_r2": r2, "r2_base": r2_base, "fit_te": te, "te_ratio": te_ratio,
            "active_alpha": alpha_ann, "alpha_t": alpha_t,
            "style_drift": drift, "drift_ratio": drift_ratio,
            "fit_reliable": fit_reliable,
            "c_r2": comps["c_r2"], "c_te": comps["c_te"],
            "c_alpha": comps["c_alpha"], "c_drift": comps["c_drift"],
            "health_raw": health_raw, "health_score": health,
        })

    # -------------------------------------------------- 当前决策快照
    @staticmethod
    def snapshot(sig: pd.DataFrame, name: str = "") -> dict:
        row = sig.dropna(subset=["score"]).iloc[-1]
        pos = float(row["position_final"])
        if pos >= 0.75:
            action = "加仓 / 持有"
        elif pos >= 0.55:
            action = "持有"
        elif pos >= 0.35:
            action = "中性观望"
        elif pos >= 0.15:
            action = "减仓"
        else:
            action = "清仓 / 离场"
        return {
            "基金": name,
            "日期": pd.Timestamp(row.name).strftime("%Y-%m-%d"),
            "趋势分": round(float(row["trend_score"]), 3),
            "偏离z": round(float(row["z_spread"]), 3) if pd.notna(row["z_spread"]) else None,
            "估值分位": round(float(row["pe_pct"]), 3) if pd.notna(row.get("pe_pct")) else None,
            "综合分": round(float(row["score"]), 3),
            "建议仓位": f"{pos:.0%}",
            "形态": row["quadrant"],
            "结论": action,
        }


def _quadrant(trend: float, dev: float) -> str:
    if pd.isna(trend) or pd.isna(dev):
        return "数据不足"
    if trend >= 0 and dev >= 0:
        return "趋势向上 · 相对低估"
    if trend >= 0 and dev < 0:
        return "趋势向上 · 相对高估"
    if trend < 0 and dev >= 0:
        return "趋势向下 · 相对低估"
    return "趋势向下 · 相对高估"


# ------------------------------------------------------------------ 买卖决策
_TRIGGER_NAME = {"c_r2": "拟合R²", "c_te": "跟踪误差",
                 "c_alpha": "实现alpha", "c_drift": "风格漂移"}


def health_stats(health: pd.DataFrame, cfg: Optional[dict] = None) -> dict:
    """
    健康分的分布 vs 离场阈值 —— 用于**诊断门控本身是否可能被触发**。

    为什么必须有这个函数：健康门控一旦阈值不可达，回测结果会静默退化成
    "买入持有"，而 metrics 表里那个方案看起来还"跑赢对照"（其实只是仓位差）。
    只有把「健康分历史最低值距离离场线还有多远」显式打出来，
    才能让"零交易是信号本来就没报警"和"零交易是阈值写错了"区分开。
    """
    cfg = cfg or dict(C.SIGNAL)
    exit_s = float(cfg.get("health_exit", -0.5))
    enter_s = float(cfg.get("health_enter", -0.2))
    h = health["health_score"].dropna() if "health_score" in health else pd.Series(dtype=float)
    if h.empty:
        return {"n": 0, "min": None, "p05": None, "median": None, "max": None,
                "exit": exit_s, "enter": enter_s, "n_below_exit": 0,
                "gap_to_exit": None, "reachable": False}
    n_below = int((h <= exit_s).sum())
    # 拟合不可信的天数占比：若长期为 0，说明合成基准对这只基金根本不成立
    # （典型成因是固定 0.9 的仓位假设），门控会一直处于停用状态。
    if "fit_reliable" in health.columns:
        rel = health["fit_reliable"].reindex(h.index)
        unreliable = float((rel.fillna(0.0) < 0.5).mean())
        r2_med = float(health["fit_r2"].dropna().median()) if "fit_r2" in health else None
    else:
        unreliable, r2_med = 0.0, None
    return {
        "n": int(len(h)),
        "min": float(h.min()),
        "p05": float(h.quantile(0.05)),
        "median": float(h.median()),
        "max": float(h.max()),
        "exit": exit_s, "enter": enter_s,
        "n_below_exit": n_below,
        # >0 表示从未触及离场线，差多少；≤0 表示触发过
        "gap_to_exit": float(h.min() - exit_s),
        "reachable": bool(n_below > 0),
        "unreliable_ratio": unreliable,
        "r2_median": r2_med,
    }


def _trigger_reason(health: pd.DataFrame, i: int) -> str:
    """
    卖出触发时，列出**最差的两个分量**及其取值。

    早期版本只列出 < −0.5 的分量，但重新标定后各分量的 −0.5 都对应
    "该诊断真正告警"的级别，可能一个都不越线（例如 alpha 分量为 −0.45 触发了
    EMA 后的综合分），于是原因串会退化成一句没有信息量的兜底文案。
    改成永远报最差两项，原因串才真正可读。
    """
    vals = []
    for k, nm in _TRIGGER_NAME.items():
        if k not in health.columns:
            continue
        v = health[k].iloc[i]
        if pd.notna(v):
            vals.append((float(v), nm))
    if not vals:
        return "健康分跌破离场线"
    vals.sort()
    return "、".join(f"{nm} {v:.2f}" for v, nm in vals[:2])


def build_decision(health: pd.DataFrame, cfg: Optional[dict] = None
                   ) -> tuple:
    """
    健康分 → 二值买卖决策（1 持有基金 / 0 空仓持币）及交易记录。

    迟滞状态机：跌破 health_exit 卖出，回升到 health_enter 才买入，
    避免在阈值附近频繁翻转。初始状态为持有（主动权益基金的自然基准）。
    决策在 t 日收盘做出、t+1 日生效，执行滞后由回测层的 shift 处理。

    ⚠ 调用方必须检查 trades 是否为空。零交易意味着该方案 ≡ 买入持有，
    此时它与"恒定仓位对照"的差额只是仓位差，不构成任何择时能力的证据。
    用 health_stats() 可以判断是"信号本就没报警"还是"阈值不可达"。
    """
    cfg = cfg or dict(C.SIGNAL)
    exit_s = float(cfg.get("health_exit", -0.5))
    enter_s = float(cfg.get("health_enter", -0.2))
    # 持续性确认：单日尖峰（如某一天 TE 异常跳升）不足以换手。
    # 迟滞带内（exit < v < enter）两边计数都清零，维持现状。
    confirm = max(1, int(cfg.get("health_confirm", 10)))

    h = health["health_score"]
    # 拟合不可信时**只停用卖出**：证据不足就不该动手，而不是反向操作。
    # 买入不受限制（初始状态本就是持有）。
    rel = (health["fit_reliable"] if "fit_reliable" in health.columns
           else pd.Series(1.0, index=h.index)).reindex(h.index)
    state = np.ones(len(h), dtype=int)
    cur, trades = 1, []
    run_bad = run_good = 0
    for i, (dt, v) in enumerate(h.items()):
        if pd.isna(v):
            state[i] = cur
            continue
        run_bad = run_bad + 1 if v <= exit_s else 0
        run_good = run_good + 1 if v >= enter_s else 0
        ok_rel = pd.notna(rel.iloc[i]) and float(rel.iloc[i]) >= 0.5
        if cur == 1 and run_bad >= confirm and ok_rel:
            cur = 0
            trades.append({"date": dt, "action": "卖出",
                           "health": float(v),
                           "reason": _trigger_reason(health, i)})
        elif cur == 0 and run_good >= confirm:
            cur = 1
            trades.append({"date": dt, "action": "买入",
                           "health": float(v),
                           "reason": f"健康分回升至 {v:.2f}"})
        state[i] = cur
    decision = pd.Series(state, index=h.index, name="decision")
    return decision, trades
