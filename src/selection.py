# -*- coding: utf-8 -*-
"""
选择过程感知的显著性检验（P1-03 后半）。

要回答的问题
------------
E9b 给出的"参数选择过拟合代价 0.769"是 **oracle − 迁移** 的差，
它混合了两种完全不同的东西：

  ① **真实的选参过拟合** —— 训练期挑出来的参数在未来确实不成立；
  ② **抽签运气** —— 从 12 个候选里挑最大值，即使所有配置都毫无技能，
     挑出来的那个在测试期也会偏高（E5 已显示 37%~51% 的格子天然为正）。

不把两者分开，就无法判断"该收手"还是"再多给点样本"。

三个互补指标
----------
1. **选中配置的测试期排名百分位**
   训练期选出的配置，其测试期 IR 在全部候选的测试期 IR 中排第几。
   ≈ 50% → 选参没有技能；≈ 100% → 有强技能。这个指标**不需要**任何分布假设。

2. **选择感知 bootstrap 的 p 值**
   对测试期超额序列做移动块 bootstrap（保持基金间横截面相关），
   **在每次重采样里重新从候选中选最优**，记录这个"重选出来的最优"的 IR。
   该分布就是"纯选择运气"能达到的水平。真实的迁移 IR 若落在其 95% 分位
   以下，说明观测到的表现还不如随机挑选 —— 即没有证据支持该线索。

3. **候选被选中频率**
   bootstrap 里每个候选被选为最优的频率。集中在少数几个 → 选参稳定；
   均匀散布 → 选参本身就是噪声。

与旧实现的区别
--------------
`robustness.block_bootstrap` 是对**已经选好的那一个策略**的超额序列重采样，
它回答的是"这个策略的均值是否显著不为 0"，**没有**把"从 12 个里挑 1 个"
这个动作算进去。本模块把选择过程放进零假设里。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _ir_of(mat: np.ndarray) -> float:
    """DataFrame/ndarray（日 × 基金）→ 跨基金平均的日超额 IR（年化）。"""
    mu = mat.mean(axis=0) * 252
    sd = mat.std(axis=0, ddof=1) * np.sqrt(252)
    sd = np.where(sd > 1e-9, sd, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ir = mu / sd
    return float(np.nanmean(ir)) if np.isfinite(ir).any() else float("nan")


def _moving_block_idx(n: int, block: int, rng) -> np.ndarray:
    """移动块 bootstrap 索引：保留序列内部的时序相关。"""
    n_blocks = int(np.ceil(n / block))
    max_start = max(1, n - block)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    if len(idx) < n:
        idx = np.resize(idx, n)
    return idx[:n]


def rank_percentile(cand: pd.DataFrame, fold: Optional[int] = None
                    ) -> pd.DataFrame:
    """
    指标 1：被训练期选中的配置，其测试期 IR 在候选中的百分位排名。

    cand 需含列：折 / 配置 / 训练集IR / 测试集IR / 被训练期选中。

    ★ 并列口径：用 `others < picked`（严格小于），即并列的对手**不计入**
    高于自己的部分。这是保守口径 —— 并列最高时百分位 < 100%。
    若想按"不超过自己"计，用 `others <= picked`。
    """
    d = cand if fold is None else cand[cand["折"] == fold]
    rows = []
    for fi, g in d.groupby("折"):
        g = g.dropna(subset=["测试集IR"])
        if g.empty or not g["被训练期选中"].any():
            continue
        picked = g[g["被训练期选中"]].iloc[0]
        others = g[~g["被训练期选中"]]["测试集IR"]
        pct = float((others < picked["测试集IR"]).mean() * 100) if len(others) else np.nan
        rows.append({
            "折": fi,
            "训练期选中配置": picked["配置"],
            "该配置测试期IR": float(picked["测试集IR"]),
            "候选数": int(len(g)),
            "排名百分位%": pct,
            "该折测试期IR中位数": float(g["测试集IR"].median()),
            "该折测试期IR最大(oracle)": float(g["测试集IR"].max()),
        })
    return pd.DataFrame(rows)


def selection_aware_bootstrap(by_fund: pd.DataFrame, block: int = 60,
                              n_boot: int = 2000, seed: int = 20261012,
                              fold: Optional[int] = None) -> dict:
    """
    指标 2：把"从候选里选最优"这个动作放进零假设。

    by_fund 是**逐基金 IR 长表**（列：折 / 配置 / 基金 / 测试期IR / 被训练期选中）。

    做法：对**基金维度**做有放回重采样（14 只 → 抽 14 个），每次重采样里
    重新计算每个候选配置的跨基金平均 IR 并取最大值。这个"重选出来的最优"
    就是"纯选择运气"能达到的水平。

    ★ 为什么必须在基金维度重采样，而不能对 12 个 IR 标量重采样：
    后者等价于"从 12 个固定值里有放回抽 12 次再取 max"—— 抽样几乎必然覆盖
    全部 12 个值，于是 max 恒等于 oracle，bootstrap 分布退化成 oracle 本身，
    p 值毫无意义（第一版实测：中位数 == 95% 分位 == oracle）。
    在基金维度重采样，不确定性来自"恰好哪些基金给出高 IR"，
    这才是选择过程真实的随机来源。
    """
    d = by_fund if fold is None else by_fund[by_fund["折"] == fold]
    if d.empty:
        return {"ok": False, "reason": "无逐基金 IR 数据"}
    rng = np.random.default_rng(seed)
    out_rows, skipped = [], []
    for fi, g in d.groupby("折"):
        wide = g.pivot_table(index="基金", columns="配置",
                             values="测试期IR", aggfunc="mean").dropna(how="any")
        if wide.shape[0] < 3 or wide.shape[1] < 2:
            continue
        cfgs = list(wide.columns)
        M = wide.to_numpy(dtype=float)              # (基金 × 配置)
        # ★ 选中配置必须**过滤到 cfgs 内**再用 index。
        #   逐基金表的配置过滤比汇总表更严（收益日数 < 30 的配置会被逐基金表
        #   丢弃，而汇总只要部分基金有值就保留），于是"训练期选中的配置"
        #   可能在逐基金表里根本不存在 —— 直接 index() 会 ValueError。
        #   这里取交集；交集为空说明该折无法评估，跳过而不是崩。
        picked = [p for p in g[g["被训练期选中"]]["配置"].unique().tolist()
                  if p in cfgs]
        if not picked:
            skipped.append(
                f"折{fi}：训练期选中的配置在逐基金数据中缺失（样本过短被过滤）")
            continue
        real = float(M[:, cfgs.index(picked[0])].mean())
        n_f, n_c = M.shape
        boot_best = np.empty(n_boot)
        chosen = np.zeros(n_c, dtype=int)
        for b in range(n_boot):
            idx = rng.integers(0, n_f, size=n_f)    # 基金维度有放回重采样
            means = np.nanmean(M[idx], axis=0)
            j = int(np.nanargmax(means))
            boot_best[b] = means[j]
            chosen[j] += 1
        out_rows.append({
            "折": fi,
            "基金数": n_f, "候选数": n_c,
            "真实迁移IR": real,
            "纯选择运气IR_中位数": float(np.median(boot_best)),
            "纯选择运气IR_5%分位": float(np.quantile(boot_best, 0.05)),
            "纯选择运气IR_95%分位": float(np.quantile(boot_best, 0.95)),
            "p值(上尾)": float(np.mean(boot_best >= real)),
            "p值(双侧)": float(np.mean(np.abs(boot_best) >= abs(real))),
            "最常被选中配置": cfgs[int(np.argmax(chosen))],
            "最高频次占比": float(chosen.max() / n_boot),
        })
    df = pd.DataFrame(out_rows)
    if df.empty:
        return {"ok": False,
                "reason": "所有折的基金/候选数不足"
                          + ("；" + "；".join(skipped) if skipped else "")}
    return {"ok": True, "per_fold": df,
            "skipped": skipped,        # 被跳过的折必须可见，不能静默
            "summary": {
                "折数": len(df),
                "真实迁移IR均值": round(float(df["真实迁移IR"].mean()), 4),
                "纯选择运气IR_中位数": round(
                    float(df["纯选择运气IR_中位数"].mean()), 4),
                "纯选择运气IR_95%分位": round(
                    float(df["纯选择运气IR_95%分位"].mean()), 4),
                "p值(上尾)均值": round(float(df["p值(上尾)"].mean()), 4),
                "最高频次占比均值": round(float(df["最高频次占比"].mean()), 4),
            }}


def decompose_overfit_cost(cand: pd.DataFrame) -> pd.DataFrame:
    """
    把 oracle − 迁移 的"过拟合代价"拆成两块：

        总代价 = oracle − 迁移
        运气部分 = oracle − E[测试期IR]      （挑最大值本身就会偏高）
        过拟合部分 = E[测试期IR] − 迁移       （训练期选择相对平均水平的损失）

    运气部分**不是**过拟合：即使所有配置都没技能，oracle 也会偏高。
    过拟合部分才是"训练期挑的参数在未来不如平均水平"的那一块。
    """
    rows = []
    for fi, g in cand.groupby("折"):
        g = g.dropna(subset=["测试集IR"])
        if len(g) < 2:
            continue
        picked = g[g["被训练期选中"]]
        if picked.empty:
            continue
        mig = float(picked["测试集IR"].iloc[0])
        mean_ir = float(g["测试集IR"].mean())
        oracle = float(g["测试集IR"].max())
        rows.append({
            "折": fi,
            "候选数": len(g),
            "迁移IR": mig,
            "候选IR均值": mean_ir,
            "oracle IR": oracle,
            "总代价(oracle−迁移)": oracle - mig,
            "运气部分(oracle−均值)": oracle - mean_ir,
            "过拟合部分(均值−迁移)": mean_ir - mig,
        })
    return pd.DataFrame(rows)


def selection_frequency(cand: pd.DataFrame) -> pd.DataFrame:
    """指标 3（静态版）：训练期与测试期各自选中了哪些配置、是否一致。"""
    rows = []
    for fi, g in cand.groupby("折"):
        tr = g[g["被训练期选中"]]["配置"].tolist()
        te = g[g["被测试期选中"]]["配置"].tolist()
        rows.append({"折": fi,
                     "训练期选中": ",".join(tr) or "—",
                     "测试期(oracle)选中": ",".join(te) or "—",
                     "是否同一配置": bool(set(tr) & set(te))})
    return pd.DataFrame(rows)
