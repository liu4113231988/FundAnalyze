# -*- coding: utf-8 -*-
"""
合成基准的**趋势状态判定**（高低点结构口径）。

★ 定位：这是**形态描述**，不是买卖建议、不产出任何持仓含义。
  本项目已用 14 只基金做样本外检验证明：日频方向信号无法超越「恒定仓位对照」
  （RankIC 均值 −0.073，13/14 只为负，见 docs/研究方案.md E0~E4）。
  本模块只回答"合成基准现在长什么样"，不回答"该不该买"。

判据（相邻两组各 N 日，紧邻且不重叠）
--------------------------------------
    高点抬高 且 低点抬高  → 上升
    高点降低 且 低点降低  → 下降
    其余                  → 震荡

拐头描述的是**转折**而不是"处在低位"，因此额外要求四个条件同时成立：
    a) 结构尚未反转（仍是上升/震荡，否则应直接判为下降）
    b) 窗口极值已经发生在**前半段**（高点已过去）
    c) 当前值位于窗口区间的**下 1/3**
    d) 前一日的状态是同方向的（上升/拐头向上）—— 没有"从上升转下来"就不叫拐头
向上对称。0.35 / 0.65 是全文唯一的两个刻度常数，含义是"窗口区间上下各 1/3"。

为什么必须在日频序列上算
------------------------
webapi 下发给前端的曲线是**降采样**的（step = len//400，实测约 4 天一个点）。
若在降采样序列上判"最近 10 日"，实际是"最近 10 个采样点 ≈ 40 个交易日"，
语义完全失真。所以判定一律在日频序列上做，只把**结果**按绘图下标采样下发。

实测（4 只基金合成基准，2368 个交易日）
----------------------------------------
  · 该判据不是纯噪声：真实时间序的"上升+下降"占比比打乱顺序后高 25~28pp
  · 体制态（上升/下降/震荡）平均段长：N=5 → 3.4 天，N=10 → 5.3 天，N=20 → 7.4 天
  · 拐头的段长**恒为 1 天**（它是瞬时标记，不是体制）—— 因此可视化时
    拐头画成曲线上的点，体制态才画成色带。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# 状态编码：单字符，便于把整条序列压成一个短字符串下发
UP, DOWN, RANGE = "U", "D", "R"          # 上升 / 下降 / 震荡
TURN_UP, TURN_DOWN = "u", "d"            # 拐头向上 / 拐头向下
NA = "."                                  # 样本不足
TURN_CODES = (TURN_UP, TURN_DOWN)
REGIME_CODES = (UP, DOWN, RANGE)

# 可选回看窗口（交易日）。1~5 是"几天涨跌"，10/20 才勉强读得出短期形态。
TREND_LEVELS: tuple = (1, 2, 3, 5, 10, 20)
DEFAULT_LEVEL = 10
_POS_LO, _POS_HI = 0.35, 0.65


def _codes_to_state(code: str) -> int:
    return {"U": 1, "D": -1, "R": 0, "u": 2, "d": -2}.get(code, 0)


def classify_at(closes: np.ndarray, t: int, n: int,
                prev: Optional[str]) -> Optional[dict]:
    """第 t 日的判定。返回 None 表示样本不足（需要前 2N 天）。"""
    r0 = t - n + 1
    if r0 < n:
        return None
    rec = closes[r0: t + 1]
    pri = closes[r0 - n: r0]
    if len(rec) != n or len(pri) != n:
        return None
    if not (np.isfinite(rec).all() and np.isfinite(pri).all()):
        return None

    hr, lr = float(rec.max()), float(rec.min())
    hp, lp = float(pri.max()), float(pri.min())
    arg_h = int(np.argmax(rec))          # 窗口内下标，0 = 最早
    arg_l = int(np.argmin(rec))

    if hr > hp and lr > lp:
        st = UP
    elif hr < hp and lr < lp:
        st = DOWN
    else:
        st = RANGE

    rng = hr - lr
    pos = (float(closes[t]) - lr) / rng if rng > 0 else 0.5
    early = max(1, n // 2)               # “前半段”的边界

    if (st != DOWN and arg_h < early and pos <= _POS_LO
            and prev in (UP, TURN_UP)):
        st = TURN_DOWN
    elif (st != UP and arg_l < early and pos >= _POS_HI
            and prev in (DOWN, TURN_DOWN)):
        st = TURN_UP

    return {"code": st, "hr": hr, "lr": lr, "hp": hp, "lp": lp,
            "pos": float(pos), "arg_h": arg_h, "arg_l": arg_l,
            "r0": r0, "t": t}


def classify_series(closes: Sequence[float], n: int) -> List[str]:
    """整段日频状态编码（长度 = len(closes)，样本不足处为 '.'）。"""
    arr = np.asarray(closes, dtype=float)
    out = [NA] * len(arr)
    prev: Optional[str] = None
    for t in range(len(arr)):
        r = classify_at(arr, t, n, prev)
        if r is None:
            continue
        out[t] = r["code"]
        prev = r["code"]
    return out


def run_lengths(codes: Sequence[str]) -> Dict[str, list]:
    """把编码压成连续段，并把体制态与拐头分开统计。

    必须分开：拐头的段长恒为 1 天，混在一起会把中位段长永久拉到 1 天，
    让人误以为整个判据都不可读。分开后体制态的真实段长才显出来。
    """
    regime, turn = [], []
    i, m = 0, len(codes)
    while i < m:
        c = codes[i]
        if c == NA:
            i += 1
            continue
        j = i
        while j + 1 < m and codes[j + 1] == c:
            j += 1
        (turn if c in TURN_CODES else regime).append(j - i + 1)
        i = j + 1
    return {"regime": regime, "turn": turn}


def summarize(codes: Sequence[str]) -> dict:
    """由一个回看窗口的日频编码，给出"这个窗口下判据是否可读"的自证数据。"""
    rl = run_lengths(codes)
    reg = sorted(rl["regime"])
    turn = sorted(rl["turn"])
    days = sum(1 for c in codes if c != NA)
    reg_days = sum(reg)
    return {
        "days": days,
        "regime_runs": len(reg),
        "regime_median": int(reg[len(reg) // 2]) if reg else 0,
        "regime_mean": round(float(np.mean(reg)), 2) if reg else 0.0,
        "regime_p90": int(reg[min(len(reg) - 1, int(len(reg) * 0.9))]) if reg else 0,
        "regime_days": reg_days,
        "turn_runs": len(turn),
        # 拐头平均每多少个交易日出现一次 —— 太频繁说明它只是噪声标记
        "turn_every": round(days / len(turn), 1) if turn else None,
    }


def trend_payload(syn_nav: pd.Series, plot_index: pd.Index,
                  levels: Sequence[int] = TREND_LEVELS) -> dict:
    """
    给单个模型算好全部回看窗口的趋势状态。

    参数
    ----
    syn_nav : 日频合成基准净值（未归一化也行，判定与取值比例无关）
    plot_index : 前端实际绘制的那些日期（降采样后的）
        状态在日频上算完后按它取子集，再压成字符串下发 —— 这样"回看 N 日"
        的语义才真的等于 N 个交易日。
    """
    s = pd.Series(syn_nav).dropna()
    if len(s) < 4:
        return {"levels": list(levels), "codes": {}, "stats": {}, "last": {}}
    base = float(s.iloc[0]) or 1.0
    vals = s.to_numpy(dtype=float)
    # 绘图下标 → 日频位置
    pos = s.index.get_indexer(pd.Index(plot_index))
    pos = np.where(pos < 0, 0, pos)

    codes_out: Dict[str, str] = {}
    stats_out: Dict[str, dict] = {}
    last_out: Dict[str, dict] = {}
    for n in levels:
        n = int(n)
        daily = classify_series(vals, n)
        codes_out[str(n)] = "".join(daily[i] for i in pos)
        stats_out[str(n)] = summarize(daily)
        r = classify_at(vals, len(vals) - 1, n, daily[len(vals) - 2]
                        if len(vals) > 1 else None)
        if r is None:
            last_out[str(n)] = None
            continue
        i0 = r["r0"]
        last_out[str(n)] = {
            "state": r["code"],
            # 与图表同口径（起点归一为 1），避免用户看到两套数字
            "hr": round(r["hr"] / base, 5), "lr": round(r["lr"] / base, 5),
            "hp": round(r["hp"] / base, 5), "lp": round(r["lp"] / base, 5),
            "pos": round(r["pos"], 4),
            "arg_h": r["arg_h"] + 1, "arg_l": r["arg_l"] + 1,
            "win_start": str(s.index[i0].date()),
            "win_end": str(s.index[len(vals) - 1].date()),
            "prior_start": str(s.index[i0 - n].date()),
            "prior_end": str(s.index[i0 - 1].date()),
        }
    return {"levels": list(levels), "codes": codes_out,
            "stats": stats_out, "last": last_out}
