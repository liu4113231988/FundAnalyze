# -*- coding: utf-8 -*-
"""
本轮 TODO 修复的行为回归（可离线运行，不需要网络与 akshare）。

与 tests/test_p0_fixes.py 的分工：那里是历史遗留的 P0 回归；
这里是 2026-10-11 这一轮新增修复的**验收测试**，每条都对应 TODO 里的
一个"行动与验收"。数值断言一律自带最小复现输入（样本长度 / 随机种子 / 参数），
否则无法验证"是否真的修好了"（见 TODO 收尾段）。

运行：
    python tests/test_regressions.py        # 退出码 0 = 全部通过
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
from src.backtest import run_backtest, perf_metrics
from src.signal import SignalEngine
from src.models import _score, _constrained_fit, ModelZoo
from src.data import total_return_index

FAILS = []


def chk(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
        return False
    except ValueError:
        return True


# ============================================================ R1 P0-04 前缀不变性
def r1():
    print("\nR1 · P0-04 禁止从未来回填（前缀不变性）")
    # 最小复现输入：几何随机游走 seed=42，μ=0.04%/日，σ=1.1%/日，200 日
    rng = np.random.default_rng(42)
    r = rng.normal(0.0004, 0.011, 200)
    syn = pd.Series(np.exp(np.cumsum(r)),
                    index=pd.bdate_range("2015-01-01", periods=200))
    eng = SignalEngine()
    full = eng.vol_target_channel(syn)["vol_target_pos"]
    pref = eng.vol_target_channel(syn.iloc[:30])["vol_target_pos"]
    chk(abs(float(full.iloc[0]) - float(pref.iloc[0])) < 1e-12,
        f"同一序列：完整 200 日首日 {float(full.iloc[0]):.6f} "
        f"== 仅前 30 日首日 {float(pref.iloc[0]):.6f}"
        "（修复前是 0.950712 vs 1.000000）")
    # 追加未来数据后，历史仓位不得改变
    syn2 = pd.concat([syn, pd.Series(np.exp(np.cumsum(
        np.concatenate([r, rng.normal(0.0004, 0.011, 100)])))[200:],
        index=pd.bdate_range("2015-01-01", periods=300)[200:])])
    full2 = eng.vol_target_channel(syn2)["vol_target_pos"]
    same = bool(np.allclose(full.values,
                            full2.reindex(syn.index).values, atol=1e-12))
    chk(same, "追加 100 日后，前 200 日的仓位序列保持不变")


# ============================================================ R2 P0-04 重估日历
def r2():
    print("\nR2 · P0-04 固定日历重估点（linspace 会漂移，calendar 不会）")
    n0, n1 = 2600, 2700
    dates0 = pd.bdate_range("2010-01-04", periods=n0)
    dates1 = pd.bdate_range("2010-01-04", periods=n1)
    W, refit, mr = C.RBSA["window"], C.RBSA["refit"], 60
    a = ModelZoo._rebal_idx(n0, W, refit, mr)
    b = ModelZoo._rebal_idx(n1, W, refit, mr)[:len(a)]
    drift = sum(1 for x, y in zip(a, b) if x != y)
    chk(drift > 0, f"（对照）linspace 抽稀下 {drift}/{len(a)} 个历史重估点发生位移")

    ca = ModelZoo._rebal_calendar(n0, W, refit, mr, dates0)
    cb = ModelZoo._rebal_calendar(n1, W, refit, mr, dates1)[:len(ca)]
    chk(len(ca) > 0 and ca == cb,
        f"calendar 模式下 {len(ca)} 个重估点全部保持不变（追加 100 日无位移）")
    chk(all(x > 0 for x in cb), "calendar 返回的都是合法下标")


# ============================================================ R3 P0-05 负 R² 与降级
def r3():
    print("\nR3 · P0-05 负 R² 保留、降级可观测")
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 500)
    pred = 2 * y.mean() - y                       # 真实 R² = −3
    sse = float(((pred - y) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum())
    chk(abs(1 - sse / sst + 3.0) < 1e-9, "构造确认：真实 R² = −3")
    got = _score(pred, y)[0]
    chk(abs(got + 3.0) < 1e-9, f"_score 返回 {got:.4f}（修复前是 0）")

    flat = np.full(300, 0.5)                      # 常数目标 → R² 未定义
    chk(np.isnan(_score(flat, flat)[0]),
        "常数目标 → R² 为 NaN（既不是 0 也不是 −∞）")

    st: dict = {}
    _constrained_fit(np.full((5, 3), np.nan), np.full(5, np.nan), 0.9,
                     status=st)
    chk(st.get("ok") is False and st.get("degraded") == "equal_weight",
        f"约束优化失败时如实上报：ok={st.get('ok')} "
        f"reason={st.get('reason', '')[:28]}")

    st2: dict = {}
    X = rng.normal(0, 0.01, (200, 3))
    w = _constrained_fit(X, X @ np.array([0.3, 0.3, 0.3]), 0.9, status=st2)
    chk(st2.get("ok") is True and abs(float(np.sum(w)) - 0.9) < 1e-6,
        "正常输入仍收敛，且 Σw 满足约束（未误报降级）")


# ============================================================ R4 P1-04 统计边界
def r4():
    print("\nR4 · P1-04 胜率分母 / 回撤起点")
    n, h = 600, C.BACKTEST["fwd_horizon"]
    idx = pd.bdate_range("2015-01-01", periods=n)
    nav = pd.Series(1.001 ** np.arange(1, n + 1), index=idx)   # 持续上涨
    sig = pd.DataFrame({"position_final": 1.0, "score": 1.0}, index=idx)
    bt = run_backtest(nav, sig, fund_name="t")
    win = len(bt["series"]["择时策略(融合)"])
    chk(abs(bt["win_long"] - 1.0) < 1e-12,
        f"持续上涨样本的多头胜率 = {bt['win_long']:.6f}"
        f"（修复前约 {(win - h) / win:.4f}，即尾部无效标签被算进了分母）")
    chk(bt["n_valid_label"] == win - h,
        f"有效前瞻标签数 = {bt['n_valid_label']} = 窗口 {win} − 前瞻 {h}")

    # 最大回撤必须从初始资本起算：首日 −50% 必须被计入
    r = pd.Series(np.concatenate([[-0.5], np.zeros(29)]),
                  index=pd.bdate_range("2015-01-01", periods=30))
    m = perf_metrics(r, name="t")
    chk(abs(m["最大回撤"] + 0.5) < 1e-9,
        f"首日 −50% 计入回撤：MDD = {m['最大回撤']:.4f}（修复前是 0）")
    chk("算术夏普" in m and abs(m["算术夏普"] - m["夏普"]) > 1e-9,
        f"几何夏普 {m['夏普']:.4f} 与算术夏普 {m['算术夏普']:.4f} 分开给出（不再混用）")


# ============================================================ R5 P0-01 总回报口径
def r5():
    print("\nR5 · P0-01 总回报构造的已知答案测试")
    idx = pd.bdate_range("2020-01-01", periods=3)

    # 1) 无分红：总回报 = 单位净值涨幅
    u = pd.Series([1.0, 1.05, 1.10], index=idx)
    tr = total_return_index(u)
    chk(abs(tr.iloc[-1] - 1.10) < 1e-12,
        f"无分红：总回报 {tr.iloc[-1]:.4f} = 单位净值涨幅 1.10")

    # 2) 现金分红：除息日净值下跌的那部分不是亏损
    u2 = pd.Series([2.0, 2.06, 1.96], index=idx)          # 第 3 日除息 0.10
    dv = pd.DataFrame({"per_unit": [0.10]},
                      index=pd.to_datetime([idx[2]]))
    tr2 = total_return_index(u2, dividends=dv)
    # 第 2 日 +3%，第 3 日 (1.96+0.10−2.06)/2.06 = 0
    chk(abs(tr2.iloc[-1] - 1.03) < 1e-9,
        f"现金分红：除息不产生亏损，总回报 {tr2.iloc[-1]:.4f}（应为 1.03）")

    # 3) 拆分：1 拆 2 后净值减半，总回报不变
    u3 = pd.Series([2.0, 2.04, 1.02], index=idx)
    sp = pd.DataFrame({"ratio": [2.0]}, index=pd.to_datetime([idx[2]]))
    tr3 = total_return_index(u3, splits=sp)
    chk(abs(tr3.iloc[-1] - 1.02) < 1e-9,
        f"拆分：份额翻倍后总回报 {tr3.iloc[-1]:.4f}（应为 1.02，即只有 +2% 涨幅）")

    # 4) 组合：分红 + 涨跌
    u4 = pd.Series([1.0, 1.05], index=idx[:2])
    dv4 = pd.DataFrame({"per_unit": [0.05]}, index=pd.to_datetime([idx[1]]))
    tr4 = total_return_index(u4, dividends=dv4)
    chk(abs(tr4.iloc[-1] - 1.10) < 1e-9,
        f"分红+涨跌：总回报 {tr4.iloc[-1]:.4f}（应为 1.10）")

    # 5) 对照：错误口径（累计净值直算）会低估 —— 这正是 P0-01 要修的东西
    cum = pd.Series([2.0, 2.06, 2.06], index=idx)   # 累计净值口径下除息日不变
    chk(abs(cum.pct_change().iloc[-1]) < 1e-12,
        "（对照）累计净值直算在除息日给出 0% 收益，掩盖了真实的基金涨跌")


# ============================================================ R6 P1-06 参数校验
def r6():
    print("\nR6 · P1-06 API 参数：非法输入停在 4xx，语义三态分清")
    from src.webapi import (_parse_max_rebal, _parse_auto_equity,
                            _check_params)

    chk(_parse_max_rebal(None, default=None) is None,
        "省略 max_rebal → 用配置默认（60）")
    chk(_parse_max_rebal(0, default=None) == 0,
        "显式 0 → 不抽稀（「全部 · 最精确」，修复前被折叠成默认 60）")
    chk(_parse_max_rebal("30") == 30, "字符串数字 → 30")
    chk(_raises(_parse_max_rebal, "abc"), "max_rebal='abc' → ValueError（修复前是 HTTP 500）")
    chk(_raises(_parse_max_rebal, -1) and _raises(_parse_max_rebal, 10 ** 9),
        "负数 / 超大值 → ValueError")

    cfg_default = bool(C.RBSA.get("auto_equity", False))
    chk(_parse_auto_equity({}) == cfg_default,
        f"省略 auto_equity → 跟随核心配置 {cfg_default}（修复前硬编码 False）")
    chk(_parse_auto_equity({"auto_equity": False}) is False,
        "显式 False 仍然生效")
    chk(_raises(_parse_auto_equity, {"auto_equity": "maybe"}),
        "auto_equity='maybe' → ValueError")

    chk(_check_params({}) == {} and _check_params(None) == {},
        "空 params 合法")
    chk(_raises(_check_params, {"no_such_model": {"a": 1}}),
        "未知模型名 → ValueError")
    chk(_raises(_check_params, {"enet_slsqp": {"alpha": "x"}}),
        "非数值超参 → ValueError")
    chk(_raises(_check_params, {"enet_slsqp": {"alpha": float("inf")}}),
        "非有限超参 → ValueError")


# ============================================================ R7 P0-03 时间切分
def r7():
    print("\nR7 · P0-03 时间向前验证：训练期严格早于测试期")
    from src.walkforward import time_folds
    days = pd.bdate_range("2010-01-04", periods=1500)
    folds = time_folds(days, n_folds=3, purge_days=20, min_train_frac=0.4)
    chk(len(folds) == 3, f"切出 {len(folds)} 折")
    ok_gap = all(t0 > s1 for s0, s1, t0, t1 in folds)
    chk(ok_gap, "每折的测试期起点都晚于训练期末尾（无时间重叠）")
    ok_purge = all((pd.Timestamp(t0) - pd.Timestamp(s1)).days >= 5
                   for s0, s1, t0, t1 in folds)
    chk(ok_purge, f"训练/测试之间留出了 purge 间隔（{folds[0][1].date()} → "
                  f"{folds[0][2].date()}）")
    ok_order = all(folds[i][3] <= folds[i + 1][3] for i in range(len(folds) - 1))
    chk(ok_order, "各折按时间单调推进（扩展窗口）")
    chk(time_folds(pd.bdate_range("2020-01-01", periods=50)) == [],
        "样本过短时不硬凑折数，直接返回空")


# ============================================================ R8 P2-02 朴素基准
def r8():
    print("\nR8 · P2-02 风格画像必须胜过朴素基准（基准可计算）")
    from src.experiments import naive_benchmarks
    rng = np.random.default_rng(4)
    n, k = 900, 6
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    X = rng.normal(0.0005, 0.012, (n, k))
    b = np.zeros(k)
    b[:3] = [0.45, 0.25, 0.15]
    y = X @ b + rng.normal(0, 0.0015, n)
    px = pd.DataFrame(100 * np.exp(np.cumsum(X, axis=0)), index=idx,
                      columns=[f"i{j}" for j in range(k)])
    px["__FUND__"] = 100 * np.exp(np.cumsum(y))
    out = naive_benchmarks(px)
    chk(not out.empty and len(out) >= 4,
        f"返回 {len(out)} 个基准（单一宽基 / 等权 / 相关Top / 受约束回归）")
    if not out.empty:
        best_single = out[out["基准"].str.startswith("单一")]["样本外R²"].max()
        constrained = out[out["基准"].str.startswith("受约束")]["样本外R²"].iloc[0]
        chk(np.isfinite(best_single) and np.isfinite(constrained),
            f"最优单一宽基 样本外R²={best_single:.3f}，"
            f"受约束回归 样本外R²={constrained:.3f}（两者可直接比较）")
        chk((out["样本外R²"] > 0.5).any(),
            "构造出的信号能被至少一个基准捕捉（基准实现正确）")


# ============================================================ R9 P1-03 多重比较
def r9():
    print("\nR9 · P1-03 多重比较校正")
    from src.robustness import multiplicity_adjust
    ps = [0.04] * 10                       # 10 个"看起来显著"的结果
    df, note = multiplicity_adjust(ps, alpha=0.05, method="bonferroni")
    chk(int(df["拒绝原假设"].sum()) == 0,
        f"Bonferroni：10 个 p=0.04 一个都不显著（{note}）")
    df2, note2 = multiplicity_adjust([0.001] * 10, alpha=0.05, method="fdr")
    chk(int(df2["拒绝原假设"].sum()) == 10,
        f"BH-FDR：10 个 p=0.001 全部通过（{note2}）")
    df3, _ = multiplicity_adjust([0.001, 0.04, 0.9], alpha=0.05, method="fdr")
    chk(int(df3["拒绝原假设"].sum()) >= 1, "混合 p 值时 BH 至少保留最显著的一项")


def main():
    print("=" * 78)
    print("本轮 TODO 修复 · 行为回归")
    print("=" * 78)
    for fn in (r1, r2, r3, r4, r5, r6, r7, r8, r9):
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
