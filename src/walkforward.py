# -*- coding: utf-8 -*-
"""
按管理人分折的 walk-forward 验证：回答"参数选择本身是否过拟合"。

为什么需要它
------------
E4 报告"250 日窗口 + 120 日再平衡，14 只基金 100% 为正"，但那是**在全部样本上
挑出来的最优格子**。即使扣掉 E7 那条"样本不独立"的问题，仍然剩下一个
更基础的问题：

    这个"最优参数"如果是在 A 组基金上选出来的，它用到没见过的 B 组上还灵吗？

E3 已经给出警示信号：108 组参数网格里 **37% 天然为正**。
在这种情况下，任何"从网格里挑最大值"的结论都必须做样本外验证。

为什么按管理人分折
------------------
14 只基金里有 4 只是易方达、2 只兴全、2 只汇添富。同一管理人的多只基金共享
投研平台、股票池与调仓习惯，横截面相关里有一部分是**"同管理人"的机械相关**。
如果训练集和测试集里都有易方达，"参数迁移能力"就被高估了。

按管理人分折保证**训练集与测试集不含同一管理人**，这是最保守也最诚实的设定。

产出
----
    * ``train_best``  每折在训练集上选出的最优配置 → 在该折测试集上的表现
    * ``oracle``      直接用测试集本身选最优（不可实现的上界）
    * ``degradation`` oracle − train_best，即"参数过拟合的代价"
    * ``fixed_best``  固定用全体样本上的最优配置（与 E4 的说法一致）
      在每折测试集上的平均表现
"""
from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ 候选配置
def default_grid() -> List[dict]:
    """
    E4 的 3×4 扫描（信号窗口 × 再平衡周期），纯趋势通道。

    刻意保持与 E4 完全一致 —— 否则"训练集选出的最优"和"E4 报告的最优"
    根本不是同一个候选集，比较就没有意义。
    """
    out = []
    for win, reb in product((60, 120, 250), (5, 20, 60, 120)):
        out.append({"win": int(win), "reb": int(reb),
                    "cfg": dict(trend_ma=win, trend_mom=win, trend_slope=win,
                                w_trend=1.0, w_dev=0.0, w_val_pe=0.0)})
    return out


# ------------------------------------------------------------------ 单配置评估
def eval_config(nav: pd.Series, syn_nav: pd.Series,
                weights: pd.DataFrame, pe: Optional[pd.DataFrame],
                cfg: dict, reb: int) -> Optional[dict]:
    """
    在一只基金上跑一个配置，返回超额序列与其年化/IR。

    超额口径与 run_backtest 完全一致：在超额日序列上算信息比率（不扣 rf）。
    """
    from .backtest import run_backtest
    from .signal import SignalEngine

    try:
        eng = SignalEngine(cfg)
        sig = eng.compute(nav, syn_nav, weights, pe)
        raw = sig["position_raw"]
        mask = np.zeros(len(raw), dtype=bool)
        mask[::reb] = True
        pos = raw.where(pd.Series(mask, index=raw.index)).ffill().bfill()
        pos = pos.reindex(nav.index).ffill().fillna(1.0)
        sig2 = pd.DataFrame({"position_final": pos,
                             "score": pd.Series(0.5, index=pos.index)})
        bt = run_backtest(nav, sig2, cfg=dict(cfg), main_name="候选")
    except Exception:                                      # noqa: BLE001
        return None
    cn = bt["control_name"]
    ex = (bt["series"]["候选"] - bt["series"][cn])
    ex = ex[ex.index >= bt["window_start"]]
    if len(ex) < 100:
        return None
    return {
        "excess": ex,
        "年化超额": float(ex.mean()) * 252,
        "超额夏普": float(bt["timing_contrib"]["夏普"]),
        "平均仓位": float(bt["avg_position_all"].get("候选", np.nan)),
    }


# ------------------------------------------------------------------ 分折
def manager_folds(codes: List[str]) -> List[tuple]:
    """
    按管理人做 GroupKFold（这里手工实现以便确定性）。

    折的划分目标是让每折的基金数尽量均衡，且**不拆分管理人**。
    大管理人优先单独成折（否则它会污染每一折）。
    """
    by_mgr: Dict[str, List[str]] = {}
    for c in codes:
        by_mgr.setdefault(C.MANAGER_OF.get(c, "未知"), []).append(c)
    # 管理人规模降序；同规模按名称排序保证确定性
    mgrs = sorted(by_mgr.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n_folds = min(5, len(mgrs))
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    # 先把大管理人依次放满各折，再顺次分配
    for i in range(n_folds):
        folds[i].append(mgrs[i][0])
    rest = mgrs[n_folds:]
    k = 0
    for mgr, fs in rest:
        # 放到当前最小的那一折
        j = int(np.argmin([len(f) for f in folds]))
        folds[j].append(mgr)
        k += 1
    out = []
    for i, f in enumerate(folds):
        test = [c for mgr, fs in by_mgr.items() if mgr in f for c in fs]
        train = [c for c in codes if c not in test]
        if train and test:
            out.append((train, test, f))
    return out


# ------------------------------------------------------------------ 主流程
def walk_forward(per_fund: Dict[str, dict], grid: Optional[List[dict]] = None
                 ) -> dict:
    """
    per_fund[code] 必须含 nav / syn_nav / weights / pe 四个键。
    """
    grid = grid or default_grid()
    codes = [c for c in C.EXTENDED_FUNDS if c in per_fund]
    if len(codes) < 4:
        raise ValueError(f"样本不足：只有 {len(codes)} 只基金")

    # ---- 一次算完 (基金 × 配置) 的全表，避免重复拟合
    table: Dict[str, Dict[tuple, dict]] = {}
    for c in codes:
        d = per_fund[c]
        table[c] = {}
        for g in grid:
            key = (g["win"], g["reb"])
            r = eval_config(d["nav"], d["syn_nav"], d["weights"], d["pe"],
                            g["cfg"], g["reb"])
            if r is not None:
                table[c][key] = r

    keys = sorted({k for c in codes for k in table[c]})
    if not keys:
        raise ValueError("所有 (基金 × 配置) 组合都失败了")

    def mean_ir(codes_: List[str], key) -> float:
        vals = [table[c][key]["超额夏普"] for c in codes_ if key in table[c]]
        return float(np.mean(vals)) if vals else float("nan")

    rows, picks = [], []
    for fi, (train, test, mgrs) in enumerate(manager_folds(codes), 1):
        # 训练集上选最优
        best_key = max(keys, key=lambda k: (mean_ir(train, k)
                                            if np.isfinite(mean_ir(train, k))
                                            else -9e9))
        # 测试集上的三个数
        ir_train_pick = mean_ir(test, best_key)
        oracle_key = max(keys, key=lambda k: (mean_ir(test, k)
                                              if np.isfinite(mean_ir(test, k))
                                              else -9e9))
        ir_oracle = mean_ir(test, oracle_key)
        rows.append({
            "折": fi,
            "测试管理人": "、".join(mgrs),
            "训练基金数": len(train), "测试基金数": len(test),
            "训练最优配置": f"{best_key[0]}/{best_key[1]}",
            "训练集IR": round(mean_ir(train, best_key), 3),
            "测试集IR(迁移)": round(ir_train_pick, 3),
            "测试集IR(oracle)": round(ir_oracle, 3),
            "oracle配置": f"{oracle_key[0]}/{oracle_key[1]}",
            "过拟合代价": round(ir_oracle - ir_train_pick, 3),
        })
        picks.append(best_key)

    df = pd.DataFrame(rows)

    # ---- 固定配置在每折测试集上的表现（E4 的说法就是这个）
    global_best = max(keys, key=lambda k: (np.nanmean(
        [table[c][k]["超额夏普"] for c in codes if k in table[c]])
        if any(k in table[c] for c in codes) else -9e9))
    fixed_ir = float(np.mean([mean_ir(test, global_best)
                              for _, test, _ in manager_folds(codes)]))
    migrated_ir = float(np.nanmean(df["测试集IR(迁移)"]))
    oracle_ir = float(np.nanmean(df["测试集IR(oracle)"]))

    return {
        "folds": df,
        "global_best": f"{global_best[0]}/{global_best[1]}",
        "picks": picks,
        "summary": {
            "全体最优配置": f"{global_best[0]}/{global_best[1]}",
            "固定配置在测试集均值IR": round(fixed_ir, 3),
            "迁移配置在测试集均值IR": round(migrated_ir, 3),
            "oracle在测试集均值IR": round(oracle_ir, 3),
            "参数选择过拟合代价": round(oracle_ir - migrated_ir, 3),
            "选择一致性": f"{len(set(picks))}/{len(picks)} 折选出同一配置",
        },
        "table": table,
    }