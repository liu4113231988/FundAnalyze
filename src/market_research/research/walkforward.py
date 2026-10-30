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

import market_research.config as C


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
                cfg: dict, reb: int,
                bt_cfg: Optional[dict] = None) -> Optional[dict]:
    """
    在一只基金上跑一个配置，返回超额序列与其年化/IR。

    超额口径与 run_backtest 完全一致：在超额日序列上算信息比率（不扣 rf）。

    bt_cfg 可覆盖回测层配置（如 `warmup`）。做**时间切片**时必须传：
    默认 warmup=252 天，而一个测试折只有两三百天，预热就把它吃光了，
    结果是该折所有配置都返回 None（时间向前验证会静默地"没有结论"）。
    """
    from market_research.modeling.backtest import run_backtest
    from market_research.modeling.signals import SignalEngine

    try:
        eng = SignalEngine(cfg)
        sig = eng.compute(nav, syn_nav, weights, pe)
        raw = sig["position_raw"]
        mask = np.zeros(len(raw), dtype=bool)
        mask[::reb] = True
        # ★ 禁止 bfill（P0-04）：序列开头的 NaN 若从后面回填，那么"只给我训练段"
        #   与"给我整段再截取"会得到不同的历史仓位 —— 时间向前验证会被自己污染。
        #   预热段一律用中性仓位 0.5 补齐。
        pos = raw.where(pd.Series(mask, index=raw.index)).ffill().fillna(0.5)
        pos = pos.reindex(nav.index).ffill().fillna(0.5)
        sig2 = pd.DataFrame({"position_final": pos,
                             "score": pd.Series(0.5, index=pos.index)})
        bt = run_backtest(nav, sig2, cfg={**dict(cfg), **(bt_cfg or {})},
                          main_name="候选")
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
            # ★ P0-03：本函数**没有**时间切分——它先在每只基金的**全时间段**上
            # 算完绩效，再按管理人划分选参。所以它检验的是"参数能否迁移到
            # 未见过的管理人"，**不是**"参数能否迁移到未来的时间段"。
            # 结论里不得出现"时间向前验证通过""可实现"这类措辞。
            "检验类型": "按管理人分组的横截面迁移检验（非时间向前验证）",
            "全体最优配置": f"{global_best[0]}/{global_best[1]}",
            "固定配置在测试集均值IR": round(fixed_ir, 3),
            "迁移配置在测试集均值IR": round(migrated_ir, 3),
            "oracle在测试集均值IR": round(oracle_ir, 3),
            "参数选择过拟合代价": round(oracle_ir - migrated_ir, 3),
            "选择一致性": f"{len(set(picks))}/{len(picks)} 折选出同一配置",
        },
        "table": table,
    }


# ====================================================================== 时间向前验证
def _cut(obj, start, end):
    """把 Series/DataFrame 裁到 [start, end]；None 原样返回。"""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj.loc[(obj.index >= start) & (obj.index <= end)]
    return obj.loc[(obj.index >= start) & (obj.index <= end)]


def time_folds(all_days: pd.DatetimeIndex, n_folds: int = 3,
               purge_days: int = 20, min_train_frac: float = 0.40
               ) -> List[tuple]:
    """
    外层**按时间扩展**切分（P0-03）。

    返回 [(train_start, train_end, test_start, test_end), ...]。
    训练期恒早于测试期，且测试期起点与训练期末尾之间留出 `purge_days`
    个交易日的间隔 —— 因为配置里含 `trend_mom/win` 这类前瞻标签，
    跨越边界的标签会把训练段的"未来"带进测试段。

    注意：这些日期是**全体基金交易日的并集**上的切分，对每只基金再各自裁剪，
    因此不同基金的可用区间可能略有差异（停牌、成立较晚等）。
    """
    n = len(all_days)
    if n < 200 or n_folds < 1:
        return []
    lo = int(n * float(np.clip(min_train_frac, 0.1, 0.8)))
    edges = np.linspace(lo, n - 1, n_folds + 1).round().astype(int)
    out = []
    for i in range(n_folds):
        tr_end_i = int(edges[i])
        te_end_i = int(edges[i + 1])
        te_start_i = min(tr_end_i + int(purge_days), te_end_i - 1)
        if te_start_i <= tr_end_i or te_start_i >= te_end_i:
            continue
        out.append((all_days[0], all_days[tr_end_i],
                    all_days[te_start_i], all_days[te_end_i]))
    return out


def walk_forward_time(per_fund: Dict[str, dict], grid: Optional[List[dict]] = None,
                      n_folds: int = 3, purge_days: int = 20,
                      min_train_frac: float = 0.40) -> dict:
    """
    **真正的时间向前验证**（P0-03）。

    与 `walk_forward`（按管理人分折）的区别：
      · `walk_forward` 先用**全时间段**算出每只基金的绩效，再按管理人划分
        训练集/测试集 → 检验的是"参数能否迁移到未见过的**管理人**"，
        任何一折的训练期都覆盖了测试期的时间范围。
      · 本函数先按**时间**切分，内层只用训练期的数据选模型/参数/阈值，
        再在**训练期之后的测试期**上评估 → 检验的是"参数能否迁移到**未来**"。

    两者正交，应同时报告。配置选择、oracle 与"过拟合代价"的定义与
    `walk_forward` 保持一致，便于对照。
    """
    codes = [c for c in C.EXTENDED_FUNDS if c in per_fund]
    if len(codes) < 2:
        return {"ok": False, "reason": f"样本不足：只有 {len(codes)} 只基金"}
    grid = grid or default_grid()

    days = None
    for c in codes:
        d = per_fund[c]
        idx = d["nav"].index
        days = idx if days is None else days.union(idx)
    days = pd.DatetimeIndex(sorted(days))
    folds = time_folds(days, n_folds, purge_days, min_train_frac)
    if not folds:
        return {"ok": False, "reason": "可用交易日不足以做时间切分"}

    def _table(slice_range) -> Dict[str, Dict[tuple, dict]]:
        s0, s1 = slice_range
        tb: Dict[str, Dict[tuple, dict]] = {}
        for c in codes:
            d = per_fund[c]
            sub = {k: (_cut(v, s0, s1) if k in ("nav", "syn_nav", "weights", "pe")
                       else v) for k, v in d.items()}
            nav_len = len(sub.get("nav", []))
            if nav_len < 160:                # 少于此时连一个评估窗口都撑不起来
                continue
            # ★ 预热必须按切片长度缩放：默认的 252 天会把两三百天的测试折吃光，
            #   导致该折所有配置返回 None（表现为"没有结论"而非"结论为负"）。
            bt_cfg = {"warmup": int(min(252, max(20, nav_len * 0.25)))}
            tb[c] = {}
            for g in grid:
                r = eval_config(sub["nav"], sub["syn_nav"], sub["weights"],
                                sub["pe"], g["cfg"], g["reb"], bt_cfg=bt_cfg)
                if r is not None:
                    tb[c][(g["win"], g["reb"])] = r
        return tb

    def _mean_ir(tb: Dict[str, Dict[tuple, dict]], key) -> float:
        vals = [tb[c][key]["超额夏普"] for c in tb if key in tb[c]]
        return float(np.mean(vals)) if vals else float("nan")

    rows, oos_parts, cand_rows, fund_rows = [], [], [], []
    for fi, (s0, s1, t0, t1) in enumerate(folds, 1):
        tr = _table((s0, s1))
        te = _table((t0, t1))
        keys = sorted({k for c in tr for k in tr[c]})
        if not keys or not te:
            # ★ 必须说清"为什么没结论"。旧文案只有"样本过短"，看不出是
            #   **测试期**太短、还是全部基金都被过滤掉了 —— 窗口改成近 2 年时
            #   每折测试段只有 ~77 个交易日，低于 _table 的 160 下限，
            #   于是每只基金都被跳过，看起来像"跑完了"其实是"没跑成"。
            rows.append({"折": fi, "训练区间": f"{s0.date()}~{s1.date()}",
                         "测试区间": f"{t0.date()}~{t1.date()}",
                         "备注": (f"该折无可用配置：测试期仅 "
                                  f"{len(te)}/{len(codes)} 只基金满足 ≥160 交易日"
                                  f"（测试区间约 {(t1 - t0).days} 天）")})
            continue
        best_key = max(keys, key=lambda k: (_mean_ir(tr, k)
                                            if np.isfinite(_mean_ir(tr, k))
                                            else -9e9))
        oracle_key = max(keys, key=lambda k: (_mean_ir(te, k)
                                              if np.isfinite(_mean_ir(te, k))
                                              else -9e9))
        migrated = _mean_ir(te, best_key)
        oracle = _mean_ir(te, oracle_key)
        rows.append({
            "折": fi,
            "训练区间": f"{s0.date()}~{s1.date()}",
            "测试区间": f"{t0.date()}~{t1.date()}",
            "训练基金数": len(tr), "测试基金数": len(te),
            "训练最优配置": f"{best_key[0]}/{best_key[1]}",
            "训练集IR": round(_mean_ir(tr, best_key), 3),
            "测试集IR(迁移)": round(migrated, 3) if np.isfinite(migrated) else np.nan,
            "测试集IR(oracle)": round(oracle, 3) if np.isfinite(oracle) else np.nan,
            "过拟合代价": (round(oracle - migrated, 3)
                        if np.isfinite(oracle) and np.isfinite(migrated) else np.nan),
        })
        # ★ 每个候选配置在训练期与测试期的 IR 全部留档（P1-03）：
        #   没有这份明细，就无法回答"0.317 是选参技能还是抽签运气"。
        for k in keys:
            tri, tei = _mean_ir(tr, k), _mean_ir(te, k)
            cand_rows.append({
                "折": fi, "配置": f"{k[0]}/{k[1]}",
                "信号窗口": k[0], "再平衡周期": k[1],
                "训练集IR": round(tri, 4) if np.isfinite(tri) else np.nan,
                "测试集IR": round(tei, 4) if np.isfinite(tei) else np.nan,
                "被训练期选中": (k == best_key),
                "被测试期选中": (k == oracle_key),
            })
            # 逐基金 IR（长表）：选择感知 bootstrap 必须在**基金维度**重采样，
            # 而不是对 12 个 IR 标量重采样 —— 后者 argmax 几乎必然命中 oracle，
            # 得到的分布会退化成 oracle 本身，p 值是假的。
            for c in sorted(te):
                if k not in te[c]:
                    continue
                ex = te[c][k]["excess"].dropna()
                if len(ex) < 30:
                    continue
                sd = float(ex.std(ddof=1))
                ir_f = (float(ex.mean() * 252) / (sd * np.sqrt(252))
                        if sd > 1e-9 else np.nan)
                fund_rows.append({
                    "折": fi, "配置": f"{k[0]}/{k[1]}",
                    "基金": c, "测试期IR": ir_f,
                    "被训练期选中": (k == best_key),
                })
        for c in te:
            if best_key in te[c]:
                oos_parts.append(te[c][best_key]["excess"].rename(f"{c}|折{fi}"))

    df = pd.DataFrame(rows)
    # ★ 一折都没跑成就必须报失败，而不是返回 ok=True + 全 NaN 的汇总。
    #   后者会被读成"验证通过但结果为负"，而真相是"根本没验证"。
    if not any("备注" not in r for r in rows):
        return {"ok": False,
                "reason": ("所有折的测试期样本均不足（每只基金可用交易日 < 160），"
                           "时间向前验证无法进行 —— 请放宽分析窗口或减少折数"),
                "folds": df}
    oos = pd.concat(oos_parts, axis=1) if oos_parts else pd.DataFrame()
    mig = df.get("测试集IR(迁移)", pd.Series(dtype=float))
    orc = df.get("测试集IR(oracle)", pd.Series(dtype=float))
    return {
        "ok": True,
        "检验类型": "外层按时间扩展的时间向前验证（内层仅用训练期选参）",
        "folds": df,
        "candidates": pd.DataFrame(cand_rows),
        "candidates_by_fund": pd.DataFrame(fund_rows),
        "oos_excess": oos,
        "summary": {
            "折数": len(df),
            "purge(交易日)": int(purge_days),
            "迁移配置在测试期均值IR": (round(float(np.nanmean(mig)), 3)
                                if len(mig) else np.nan),
            "oracle在测试期均值IR": (round(float(np.nanmean(orc)), 3)
                               if len(orc) else np.nan),
            "参数选择过拟合代价": (round(float(np.nanmean(orc) - np.nanmean(mig)), 3)
                          if len(mig) and len(orc) else np.nan),
        },
    }