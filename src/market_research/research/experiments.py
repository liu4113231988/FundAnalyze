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

import market_research.config as C
from market_research.modeling.backtest import perf_metrics, score_to_position, _fwd_ret
from market_research.modeling.signals import SignalEngine

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
    from market_research.modeling.backtest import run_backtest

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
    from market_research.modeling.backtest import run_backtest

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
    from market_research.modeling.backtest import run_backtest

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
            # ★ 禁止 bfill（P0-04）：它会把后面才算出的仓位回填到序列开头，
            #   历史输出会随"给多少数据"而变。预热段用中性值 0.5 补齐。
            sampled = raw.where(pd.Series(mask, index=raw.index)).ffill()
            sampled = sampled.fillna(0.5)
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
            # 平均暴露必须用 backtest 的 pos_eff 口径（窗口首日执行的是前一日
            # 决定的仓位），自己按 sampled 直接取均值会与对照组构造口径不一致
            # （P0-02），差出来的千分之几会被全额记进"超额"。
            avg_pos = float(bt["avg_position_all"].get("低频再平衡", np.nan))
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


# ====================================================================== E10 成本
# 成本档位：(名称, 单边费率, 说明)
COST_TIERS = (
    ("场内ETF", 0.0015, "ETF/LOF 买卖佣金 + 冲击，本项目默认"),
    ("场外C类", 0.0050, "C 类销售服务费摊薄 / 打折申购 + 赎回费"),
    ("场外A类", 0.0150, "A 类原价申购 1.5%（未打折的最坏情形）"),
)


def cost_sensitivity(nav: pd.Series, sig: pd.DataFrame,
                     cfg: Optional[dict] = None,
                     variants: Optional[Dict[str, pd.Series]] = None,
                     costs=COST_TIERS) -> pd.DataFrame:
    """
    E10 · 成本敏感性：同一组方案在多档交易成本下重跑。

    ★ 为什么必须有这个实验：本项目默认单边 0.15%，那对应的是**场内 ETF**。
    而场外主动基金的真实申赎是申购 1.5% / 赎回 0.5%（持有 < 7 天还有 1.5%
    惩罚性赎回费），相差 10 倍以上。`docs/研究方案.md` §7.3 把这一条列为
    "最容易被忽略的风险"，但此前只跑过最乐观的那一档。

    换手越高，成本档位的影响越大，所以结果里必须同时给出年化换手 ——
    否则看到"某策略在 1.5% 成本下崩掉"无法判断是策略的问题还是换手的问题。

    注意：抬高成本对**恒定仓位对照**几乎没有影响（它只在建仓时换手一次），
    所以成本上升会直接压低"净贡献"—— 这正是要量化的东西。
    """
    from market_research.modeling.backtest import run_backtest

    cfg = {**C.BACKTEST, **(cfg or {})}
    rows = []
    for label, rate, _note in costs:
        bt = run_backtest(nav, sig, cfg={**cfg, "cost_rate": rate},
                          variants=variants)
        for v, tc in bt["timing_contrib_all"].items():
            if not tc:
                continue
            rows.append({
                "成本档位": label,
                "单边费率": rate,
                "方案": v,
                "对照组": bt["controls"].get(v),
                "年化换手": round(bt["turnover_all"].get(v, np.nan), 2),
                "平均仓位": round(bt["avg_position_all"].get(v, np.nan), 4),
                "年化收益": round(float(bt["metrics"].loc[v, "年化收益"]), 4),
                "夏普": round(float(bt["metrics"].loc[v, "夏普"]), 4),
                # 净贡献用超额序列的 IR 口径（夏普不可加）
                "超额夏普": round(tc.get("夏普", np.nan), 4),
                "超额年化": round(tc.get("年化收益", np.nan), 4),
            })
    return pd.DataFrame(rows)


def cost_durability(cost_df: pd.DataFrame) -> pd.DataFrame:
    """
    把 E10 的长表压成"每个方案在哪些成本档位下仍有正净贡献"的耐久性视图。

    判据用**超额年化 > 0**（金额意义上的净贡献）。
    不用超额夏普做跨档比较：成本上升时它的分子分母同时变，单看比率会误判 ——
    一个方案可能"夏普排名上升但绝对超额转负"。
    """
    if cost_df.empty:
        return pd.DataFrame()
    piv = cost_df.pivot_table(index="方案", columns="成本档位",
                              values="超额年化", aggfunc="mean")
    order = [c[0] for c in COST_TIERS]
    piv = piv.reindex(columns=[c for c in order if c in piv.columns])
    rates = {c[0]: c[1] for c in COST_TIERS}
    survive = []
    for _, r in piv.iterrows():
        # 标签用"无正贡献"而不是"全部为负"：净贡献恰为 0（方案 ≡ 自己的对照，
        # 如零换手方案）既不是正贡献也不是负贡献，说成"为负"会误导。
        last = "无正贡献"
        for c in piv.columns:
            if pd.notna(r[c]) and r[c] > 0:
                last = c
        survive.append(last)
    out = piv.reset_index()
    out["最高可承受档位"] = survive
    out["单边费率上限"] = [rates.get(s) for s in survive]
    return out


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
            "策略_算术夏普": m.loc["择时策略(融合)", "算术夏普"],
            "对照_算术夏普": m.loc[ctrl, "算术夏普"],
            "净贡献_夏普": bt["timing_contrib"]["夏普"],
            "净贡献_年化": bt["timing_contrib"]["年化收益"],
            "净贡献_回撤": bt["timing_contrib"]["最大回撤"],
            "RankIC": bt["ic"], "ICIR": bt["icir"],
            "多头胜率": bt["win_long"], "平均仓位": bt["avg_position"],
            "有效标签数": bt.get("n_valid_label"),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df.loc["均值"] = ["—", "平均"] + list(df.iloc[:, 2:].mean().round(4))
    return df


# ====================================================================== 朴素基准（P2-02）
def naive_benchmarks(panel: pd.DataFrame, fund_col: str = "__FUND__",
                     equity_ratio: float = 0.9, train_frac: float = 0.7,
                     max_single: int = 5) -> pd.DataFrame:
    """
    风格画像的**朴素基准对照**（P2-02）。

    为什么要它：一个共线指数池上的非负稀疏回归，几乎总能"拟合出"一个
    看起来合理的权重组合；窗口内 R² = 0.9 并不能说明模型有解释力 ——
    它同样可能只是"指数足够多"。所以必须回答：

        这套权重，比"只用沪深300""所有指数等权"好多少？

    基准（全部只用训练段定权，测试段不再调整）：
      · 单一宽基  ：每个候选指数单独做无约束回归（β），取表现最好的几个
      · 等权混合  ：全部指数等权，缩放到目标仓位
      · 相关加权  ：按训练段与基金的相关度排序取前 8 个等权
      · 受约束回归：非负 + Σw=equity_ratio（即 RBSA 的口径，作为参照系）

    R² 与模型层一致，**保留负值**（P0-05）：负的样本外 R² 是"不如常数"
    的直接证据，截到 0 就看不出来了。
    """
    from market_research.modeling.models import _constrained_fit

    px = panel.astype(float)
    R = np.log(px).diff().dropna()
    assets = [c for c in px.columns if c != fund_col]
    X, y = R[assets].values, R[fund_col].values
    n = len(y)
    sp = int(n * float(np.clip(train_frac, 0.3, 0.9)))
    if sp < 60 or n - sp < 60:
        return pd.DataFrame()
    Xtr, ytr, Xte, yte = X[:sp], y[:sp], X[sp:], y[sp:]

    def _r2(pred, yy):
        sse = float(((np.asarray(pred) - yy) ** 2).sum())
        sst = float(((yy - yy.mean()) ** 2).sum())
        return float("nan") if sst <= 1e-18 else 1.0 - sse / sst

    rows = []

    def _add(name, w):
        w = np.asarray(w, dtype=float)
        rows.append({
            "基准": name,
            "有效指数数": int((np.abs(w) > 1e-9).sum()),
            "Σw": float(w.sum()),
            "样本内R²": _r2(Xtr @ w, ytr),
            "样本外R²": _r2(Xte @ w, yte),
            "样本外年化TE": float(np.std(Xte @ w - yte, ddof=1) * np.sqrt(252)),
        })

    # 1) 单一宽基：各指数单独回归，取样本内最优的若干个
    singles = []
    for j, a in enumerate(assets):
        var = float(np.var(Xtr[:, j]))
        if var <= 1e-18:
            continue
        b = float(np.cov(ytr, Xtr[:, j])[0, 1] / var)
        singles.append((_r2(b * Xtr[:, j], ytr), j, b, a))
    singles.sort(reverse=True)
    for r2in, j, b, a in singles[:max(1, max_single)]:
        w = np.zeros(len(assets))
        w[j] = b
        _add(f"单一：{C.NAME_OF.get(a, a)}", w)

    # 2) 等权混合
    _add(f"等权混合（{len(assets)} 指数）",
         np.full(len(assets), equity_ratio / max(len(assets), 1)))

    # 3) 按相关度取前 8 等权（缩放）
    corr = np.array([abs(np.corrcoef(Xtr[:, j], ytr)[0, 1])
                     for j in range(len(assets))])
    corr = np.nan_to_num(corr)
    top = np.argsort(-corr)[:min(8, len(assets))]
    w = np.zeros(len(assets))
    w[top] = equity_ratio / len(top)
    _add(f"相关Top{len(top)}等权", w)

    # 4) 受约束非负回归（RBSA 口径）
    w = _constrained_fit(Xtr, ytr, equity_ratio)
    _add(f"受约束回归（Σw={equity_ratio:.0%}）", w)

    return pd.DataFrame(rows)


# ====================================================================== 风格漂移（2026-10-12）
def _robust_z(x: np.ndarray) -> np.ndarray:
    """
    稳健 z 值：用中位数 / MAD 而不是均值 / 标准差。

    ★ 为什么不能用普通 z：风格突变本身就是我们要找的**离群点**，它会同时
    抬高均值和标准差 —— 用均值/标准差标准化，恰好把最该被标出来的那次变化
    压回"正常范围"内（掩蔽效应，masking）。MAD 不受单点主导。
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 1e-12:
        sig = float(np.std(x))
    if not np.isfinite(sig) or sig <= 1e-12:
        return np.zeros_like(x)
    return (x - med) / sig


def _top_moves(a: pd.Series, b: pd.Series, nm=None, k: int = 2,
               eps: float = 0.01) -> str:
    """把一次权重变化写成人话："沪深300 +12.3%、中证500 −8.1%"。"""
    delta = (pd.Series(b, dtype=float) - pd.Series(a, dtype=float))
    delta = delta.fillna(0.0)
    order = delta.abs().sort_values(ascending=False).index
    parts = []
    for c in order[:k]:
        dv = float(delta[c])
        if abs(dv) < eps:
            continue
        label = (nm or {}).get(c, c)
        parts.append(f"{label}{dv * 100:+.1f}%")
    return "、".join(parts) if parts else "—"


def style_drift(weights: pd.DataFrame,
                nav: Optional[pd.Series] = None,
                syn_nav: Optional[pd.Series] = None,
                name_of: Optional[dict] = None,
                top_k: int = 5, z_thresh: float = 3.0,
                unstable_dist: float = 0.60) -> dict:
    """
    风格漂移检测（2026-10-12 新增）。

    ★ 为什么要有它
    --------------
    分析窗口原本固定 2016 起（约 10 年）。十年里基金经理可能换过几任、
    投资风格可能整个调过向，把这段样本混在一起拟合出的"风格画像"是十年平均，
    对当下没有指导意义 —— 这正是把默认窗口改成近 5 年的原因。

    但**经理变更数据取不到**：akshare 的 `fund_manager_em()` 只给"现任经理"
    （36454 行、无任职起止日期），`fund_individual_detail_info_xq()` 只有费率。
    免费数据源没有历史任职记录，所以无法直接检测"什么时候换了人"。

    于是改为检测**风格本身**是否变了 —— 这其实比经理变更更贴近问题：
    · 换了经理，风格不一定变；
    · 没换经理，风格也可能漂。
    权重序列是我们本来就有的产物，不需要任何额外数据源。

    三个视角
    --------
    1. 结构突变：相邻两次重估之间权重向量的 L1 变化，用稳健 z 找离群点。
    2. 前后半对比：区间前半与后半的平均权重 L1 距离（0~2，越大越漂移）。
    3. 分年度画像：每年的前三大持仓，以及（若给了 nav/syn_nav）
       基金与合成基准的年度相关性 —— 相关性逐年下滑是"基准跟不住了"的直接证据。

    参数
    ----
    weights   : 重估日 × 资产的权重表（ModelZoo 的产物）
    nav       : 基金净值（可选，用于年度相关性）
    syn_nav   : 合成基准净值（可选，用于年度相关性）
    z_thresh  : 判定"突变"的稳健 z 阈值，默认 3
    unstable_dist : 前后半 L1 距离超过此值即判为不稳定，默认 0.60
                    （权重和约 0.9，L1 上限约 1.8，0.6 ≈ 三分之一的仓位被重构）

    返回 dict，`ok=False` 表示样本不足以判断（不抛异常）。
    """
    nm = name_of if name_of is not None else getattr(C, "NAME_OF", {})

    w = pd.DataFrame(weights)
    if isinstance(w.index, pd.DatetimeIndex):
        w = w.sort_index()
    w = w.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # 全零行 = 预热期/空仓，不构成风格信息
    w = w.loc[w.abs().sum(axis=1) > 1e-12]
    n = len(w)
    if n < 6:
        return {"ok": False,
                "reason": f"重估点只有 {n} 个，不足以判断漂移（至少需要 6 个）"}

    # ---- 1. 结构突变
    step = w.diff().abs().sum(axis=1).iloc[1:]      # 第 i 项 = w[i] → w[i+1]
    sv = step.to_numpy(dtype=float)
    zv = _robust_z(sv) if len(sv) else np.array([])

    hits = [i for i in range(len(sv)) if zv[i] >= z_thresh]
    hits = sorted(hits, key=lambda i: -sv[i])[:max(1, int(top_k))]
    change_points = [{
        "日期": str(w.index[i + 1].date()),
        "变化幅度": round(float(sv[i]), 4),
        "z值": round(float(zv[i]), 2),
        "主要变动": _top_moves(w.iloc[i], w.iloc[i + 1], nm),
    } for i in sorted(hits)]

    # ---- 2. 前后半对比
    h = n // 2
    wa, wb = w.iloc[:h].mean(), w.iloc[h:].mean()
    half_dist = float((wa - wb).abs().sum())

    def _top(s: pd.Series, k: int = 3):
        o = s[s > 1e-6].sort_values(ascending=False).head(k)
        return [{"名称": nm.get(c, c), "权重": round(float(v), 4)}
                for c, v in o.items()]

    half_top = {
        "前半": {"区间": f"{w.index[0].date()} ~ {w.index[h - 1].date()}",
                 "主要持仓": _top(wa)},
        "后半": {"区间": f"{w.index[h].date()} ~ {w.index[-1].date()}",
                 "主要持仓": _top(wb)},
    }

    # ---- 3. 分年度画像
    yearly_top, yearly_corr = {}, {}
    for y, g in w.groupby(w.index.year):
        yearly_top[int(y)] = _top(g.mean())

    if nav is not None and syn_nav is not None:
        try:
            rf = np.log(pd.Series(nav).astype(float)).diff()
            rs = np.log(pd.Series(syn_nav).astype(float)).diff()
            pair = pd.concat([rf.rename("f"), rs.rename("s")], axis=1).dropna()
            for y, g in pair.groupby(pair.index.year):
                if len(g) >= 60 and g["f"].std() > 0 and g["s"].std() > 0:
                    yearly_corr[int(y)] = round(float(g["f"].corr(g["s"])), 4)
        except (TypeError, ValueError):            # noqa: BLE001
            yearly_corr = {}

    # ---- 结论
    stable = bool(half_dist < unstable_dist and not change_points)
    span = f"{w.index[0].date()} ~ {w.index[-1].date()}"
    if stable:
        verdict = (f"风格稳定：前后半权重距离 {half_dist:.2f}（阈值 {unstable_dist}），"
                   f"无显著结构突变。这段区间的风格画像可以代表当下。")
    else:
        bits = []
        if half_dist >= unstable_dist:
            bits.append(f"前后半权重距离 {half_dist:.2f} ≥ {unstable_dist}")
        if change_points:
            bits.append(f"检出 {len(change_points)} 次结构突变"
                        f"（最显著 {change_points[0]['日期']}，"
                        f"{change_points[0]['主要变动']}）")
        verdict = ("风格发生漂移：" + "；".join(bits)
                   + "。当前画像混入了不同性质的样本，"
                     "建议缩短窗口或分段查看（见 yearly_top）。")

    corr_note = ""
    if len(yearly_corr) >= 2:
        vals = list(yearly_corr.values())
        if max(vals) - min(vals) >= 0.15:
            corr_note = (f"注意：年度相关性从 {max(vals):.2f} 到 {min(vals):.2f}，"
                         f"波动 {max(vals) - min(vals):.2f} —— 合成基准在不同年份的"
                         f"贴合度差异较大。")

    return {
        "ok": True,
        "span": span,
        "n_points": int(n),
        "典型相邻变化": round(float(np.median(sv)), 4) if len(sv) else 0.0,
        "最大相邻变化": round(float(np.max(sv)), 4) if len(sv) else 0.0,
        "前后半距离": round(half_dist, 4),
        "不稳定阈值": float(unstable_dist),
        "突变点数": len(change_points),
        "change_points": change_points,
        "half_top": half_top,
        "yearly_top": yearly_top,
        "yearly_corr": yearly_corr,
        "corr_note": corr_note,
        "stable": stable,
        "verdict": verdict,
    }
