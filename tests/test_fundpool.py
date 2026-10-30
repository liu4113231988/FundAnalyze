# -*- coding: utf-8 -*-
"""
扩池筛选规则的行为回归（离线，不需要网络）。

守的是 `src/fundpool.py` 里几个**容易悄悄失效**的规则实现：
份额类别推断、规模解析、成立时间分层、单管理人上限、以及
"规则只执行不修改"这一前提本身。
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                             # noqa: BLE001
    pass

import pandas as pd

import market_research.funds.universe as F
import market_research.config as C

FAILS = []


def chk(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


# ============================================================ F1 份额类别
def f1():
    print("\nF1 · 份额类别推断")
    cases = [
        ("华夏成长混合", None), ("华夏大盘精选混合A", "A"),
        ("中海可转债债券A", "A"), ("华夏成长混合(后端)", "other"),
        ("某基金C", "C"), ("某基金E", "E"), ("某基金H", "H"),
    ]
    for name, want in cases:
        got = F._share_class(name)
        chk(got == want, f"{name!r} → {got}（期望 {want}）")


# ============================================================ F2 规模解析
def f2():
    print("\nF2 · 规模解析（'197.40亿' → 197.4）")
    for s, want in [("197.40亿", 197.40), ("2亿", 2.0), ("5000万", 0.5),
                    ("---", None), ("", None), ("暂无", None)]:
        got = F._parse_scale(s)
        # 未知规模返回 NaN（不是 None）—— 语义等价，但类型是 float
        if want is None:
            chk(bool(pd.isna(got)), f"{s!r} → NaN（未知）")
        else:
            chk(got is not None and not pd.isna(got) and abs(got - want) < 1e-9,
                f"{s!r} → {got}（期望 {want}）")


# ============================================================ F3 分层
def f3():
    print("\nF3 · 成立时间分层")
    s = pd.Series(pd.to_datetime(
        ["2001-12-18", "2014-12-31", "2015-01-01", "2019-12-31",
         "2020-01-01", "2026-02-10", None]))
    got = F._layer(s)
    want = ["2015前", "2015前", "2015-2019", "2015-2019", "2020后",
            "2020后", None]
    for i, (g, w) in enumerate(zip(got, want)):
        ok = (pd.isna(g) and w is None) or (str(g) == str(w))
        chk(ok, f"第 {i + 1} 个 → {g}（期望 {w}）")


# ============================================================ F4 规则常量
def f4():
    print("\nF4 · 冻结的规则常量未被改动")
    chk(F.MAX_PER_MANAGER == 3, f"单管理人上限 = {F.MAX_PER_MANAGER}")
    chk(F.MIN_SCALE_YI == 2.0, f"规模下限 = {F.MIN_SCALE_YI} 亿")
    chk(F.MIN_AGE_DAYS == 730, f"成立满 {F.MIN_AGE_DAYS} 天（≈2 年）")
    chk(set(F.ALLOWED_TYPES) == {"股票型", "混合型-偏股", "混合型-灵活"},
        f"允许类型 = {F.ALLOWED_TYPES}")
    # 主题词表必须真的能挡住行业基金
    for w in ("消费", "医药", "新能源", "半导体", "指数"):
        chk(w in F.NAME_EXCLUDE_WORDS, f"主题词表含「{w}」")
    chk("ETF" in F.NAME_EXCLUDE_WORDS and "FOF" in F.NAME_EXCLUDE_WORDS,
        "主题词表含 ETF / FOF")


# ============================================================ F5 未扩池时回退
def f5():
    print("\nF5 · 未扩池时 load_pool 回退到 C.EXTENDED_FUNDS")
    import os
    fp = C.DATA_DIR / "fund_pool.csv"
    if not fp.exists():
        chk(F.pool_codes() == list(C.EXTENDED_FUNDS.keys()),
            "无 fund_pool.csv 时回退到 EXTENDED_FUNDS")
        return
    codes = F.pool_codes()
    chk(len(codes) > 0 and all(len(c) == 6 and c.isdigit() for c in codes),
        f"已扩池：{len(codes)} 只，代码均为 6 位数字")
    d = F.load_pool()
    chk(d["管理人"].value_counts().max() <= F.MAX_PER_MANAGER,
        f"单管理人最多 {d['管理人'].value_counts().max()} 只（上限 "
        f"{F.MAX_PER_MANAGER}）")
    age = (pd.Timestamp.today() - pd.to_datetime(d["成立时间"])).dt.days
    chk(bool((age >= F.MIN_AGE_DAYS).all()),
        f"全部成立满 {F.MIN_AGE_DAYS} 天（最短 {int(age.min())} 天）")
    chk(bool((d["当前规模_亿"] >= F.MIN_SCALE_YI).all()),
        f"全部规模 ≥ {F.MIN_SCALE_YI} 亿（最小 {d['当前规模_亿'].min()} 亿）")
    chk(set(d["层"]) <= {"2015前", "2015-2019", "2020后"},
        f"分层标签合法：{sorted(set(d['层']))}")


def main():
    print("=" * 78)
    print("扩池筛选规则 · 行为回归")
    print("=" * 78)
    for fn in (f1, f2, f3, f4, f5):
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
