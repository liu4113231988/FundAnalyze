# -*- coding: utf-8 -*-
"""
真实持仓 vs RBSA 拟合持仓：对齐模块的行为回归（可离线运行）。

覆盖 src/holdings.py 与 run_holdings.py 的几个关键口径 —— 这几个地方都出过
**静默错误**（不抛异常、只是数字全错），所以必须有测试守着：

  1. CSV 往返丢类型：股票代码变 int → 与前导零的映射索引对不上，映射率归零
  2. 列名口径：暴露表用行业名、权重表用指数代码 → intersection 静默为空
  3. 单位换算：占净值比例是百分点，不除 100 会让日收益放大近百倍
  4. 披露滞后：持仓基准必须从季末 + lag 起算，否则是前瞻偏差

运行：
    python tests/test_holdings.py        # 退出码 0 = 全部通过
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                             # noqa: BLE001
    pass

import numpy as np
import pandas as pd

from src import config as C
from src import holdings as H

FAILS = []
CODE = "110022"


def chk(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


# ============================================================ T1 行业映射
def t1():
    print("\nT1 · 行业映射：股票 → 申万一级")
    m = H.load_industry_map()
    chk(len(m) > 3000, f"映射覆盖 {len(m)} 只股票（实测约 5221）")
    chk(bool((m.index.astype(str).str.len() == 6).all()),
        "索引均为 6 位代码（前导零保留）")
    chk("食品饮料" in set(m.values) and "汽车" in set(m.values),
        "含食品饮料 / 汽车等行业")
    # 常见股票应能映射
    for c, nm in [("600519", "贵州茅台"), ("000333", "美的集团")]:
        chk(c in m.index, f"{c} {nm} 在映射表中")


# ============================================================ T2 持仓读取
def t2():
    print("\nT2 · 持仓明细：类型规整与'完整'标记")
    hd = H.load_fund_holdings(CODE)
    if hd.empty:
        print("  （跳过：无本地持仓缓存，需先联网跑一次 run_holdings.py）")
        return
    chk(hd["股票代码"].astype(str).str.len().eq(6).all(),
        "股票代码均为 6 位（CSV 往返后前导零未丢）")
    chk(pd.api.types.is_numeric_dtype(hd["占净值比例"]),
        "占净值比例为数值型")
    chk(pd.api.types.is_datetime64_any_dtype(hd["季末"]),
        "季末为日期型")
    periods = sorted(hd["完整"].sum() and hd[hd["完整"]]["期数"].unique())
    chk(len(periods) >= 2, f"完整（半年报/年报）期数 {len(periods)} 个：{periods}")
    chk(all(p.endswith("Q2") or p.endswith("Q4") for p in periods),
        "完整期数只含 Q2/Q4（季报仅前十大，不参与聚合）")


# ============================================================ T3 暴露聚合
def t3():
    print("\nT3 · 半年度暴露：口径与取值范围")
    m = H.load_industry_map()
    hd = H.load_fund_holdings(CODE)
    if hd.empty:
        print("  （跳过：无本地持仓缓存）")
        return
    exp = H.semiannual_exposure(hd, m)
    chk(not exp.empty, f"暴露表 {exp.shape}")
    tot = exp.sum(axis=1)
    chk(bool(((tot > 50) & (tot < 100)).all()),
        f"每期权重合计在 50~100 之间（实测 {tot.min():.1f}~{tot.max():.1f}，"
        f"差额是现金/债券）")
    # Q2 季末是 6/30，Q4 季末是 12/31（不是都 31 日）
    chk(all((d.month == 6 and d.day == 30) or (d.month == 12 and d.day == 31)
            for d in exp.index),
        "索引只含 6/30 与 12/31 两个季末")
    # 列名口径：默认是行业名，转码后应变成 801xxx
    coded = H._exposure_to_codes(exp)
    chk(all(str(c).startswith("801") for c in coded.columns),
        "转码后列为指数代码 801xxx（与 RBSA 权重/panel 同口径）")
    chk(set(coded.columns) == set(exp.columns).intersection(
        {v: k for k, v in H._industry_index_codes().items()}.get(c, c)
        for c in exp.columns) or len(coded.columns) == len(exp.columns),
        "转码不增删列")


# ============================================================ T4 单位与滞后
def t4():
    print("\nT4 · 持仓基准：单位换算与披露滞后")
    m = H.load_industry_map()
    hd = H.load_fund_holdings(CODE)
    if hd.empty:
        print("  （跳过：无本地持仓缓存）")
        return
    exp = H.semiannual_exposure(hd, m)
    panel = None
    try:
        from src import data as D
        panel = D.load_panel(CODE, verbose=False)
    except Exception as e:                                    # noqa: BLE001
        print(f"  （跳过 panel：{type(e).__name__}）")
        return
    hr = H.holding_benchmark(exp, panel)
    if hr.empty:
        print("  （跳过：持仓基准为空）")
        return
    chk(bool(hr.abs().max() < 0.2),
        f"日收益量级合理（|r|max = {hr.abs().max():.4f}；"
        f"若不除 100 会到几十，实测曾出现年化 −95%）")
    # 披露滞后：首个生效日必须晚于最早季末 + lag
    first_q = exp.index[0]
    chk(hr.index[0] >= first_q + pd.Timedelta(days=H.DISCLOSURE_LAG_DAYS)
        or hr.index[0] >= pd.Timestamp(C.START_DATE),
        f"基准起点 {hr.index[0].date()} 不早于 {first_q.date()} + "
        f"{H.DISCLOSURE_LAG_DAYS} 天（或受分析窗口起点约束）")


# ============================================================ T5 空输入
def t5():
    print("\nT5 · 空输入必须返回 ok=False 而不是抛异常")
    for fn, arg in [(H.align_vs_rbsa, (pd.DataFrame(), pd.DataFrame())),
                    (H.holding_benchmark, (pd.DataFrame(), pd.DataFrame()))]:
        try:
            r = fn(*arg)
            bad = (isinstance(r, dict) and r.get("ok") is False) or \
                  (isinstance(r, pd.Series) and r.empty)
            chk(bad, f"{fn.__name__} 空输入安全返回")
        except Exception as e:                                # noqa: BLE001
            chk(False, f"{fn.__name__} 空输入抛异常 {type(e).__name__}")
    r = H.compare_benchmarks(pd.Series(dtype=float),
                             pd.Series(dtype=float),
                             pd.Series(dtype=float))
    chk(r.get("ok") is False, "compare_benchmarks 空输入返回 ok=False")


def main():
    print("=" * 78)
    print("真实持仓 vs RBSA 对齐 · 行为回归")
    print("=" * 78)
    for fn in (t1, t2, t3, t4, t5):
        try:
            fn()
        except Exception as e:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILS.append(f"{fn.__name__} 抛异常 {type(e).__name__}: {e}")
    print("\n" + "=" * 78)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
