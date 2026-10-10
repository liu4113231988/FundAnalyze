# -*- coding: utf-8 -*-
"""
P0 修复的回归验证（可离线运行，不需要网络与 akshare）。

覆盖三个致命缺陷：
  T1 对照组口径   每个方案各配一份恒定仓位对照（P0-1）
  T2 回测边界     短样本 / 日期区间非法不再崩溃（P0-1 附带）
  T3 健康门控     阈值可达 + 正常样本不误报（P0-2）
  T4 E4 对照组    被评估方案用它自己的平均仓位做对照（P0-3）

运行：
    python tests/test_p0_fixes.py        # 退出码 0 = 全部通过
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
from src.backtest import run_backtest
from src.signal import SignalEngine, build_decision, health_stats

FAILS = []


def chk(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def _synth(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n, name="date")
    r = pd.Series(rng.normal(0.0004, 0.013, n), index=idx)
    nav = (1 + r).cumprod()
    return idx, r, nav


# ============================================================== T1 对照组口径
def t1():
    print("\nT1 · 对照组必须与被比较方案「平均仓位一致」")
    idx, r, nav = _synth()
    score = pd.Series(np.sin(np.arange(len(idx)) / 60.0), index=idx)
    sig = pd.DataFrame({
        "position_final": np.clip(0.5 + 0.5 * score, 0, 1),   # 平均约 50%
        "score": score,
    })
    # 满仓方案：平均仓位 100%，且不调仓 → 应与其对照（也是 100%）完全一致
    variants = {"满仓方案": pd.Series(1.0, index=idx)}
    bt = run_backtest(nav, sig, variants=variants)

    ctrl_main = bt["controls"]["择时策略(融合)"]
    ctrl_full = bt["controls"]["满仓方案"]
    avg_main = bt["avg_position_all"]["择时策略(融合)"]
    chk(ctrl_main == f"恒定仓位({avg_main:.0%})",
        f"主策略对照 = {ctrl_main}，与其自身平均仓位 {avg_main:.1%} 一致")
    chk(ctrl_full == "恒定仓位(100%)", f"满仓方案对照 = {ctrl_full}（平均仓位 100%）")
    chk(ctrl_main != ctrl_full, "两个方案不再共用同一份对照")

    # 核心断言：满仓方案 ≡ 买入持有 ≡ 恒定仓位(100%)，净贡献必须≈0。
    # 修复前它是「满仓方案 − 恒定仓位(50%)」，会被记上一笔纯粹来自仓位差的假超额。
    contrib = bt["timing_contrib_all"]["满仓方案"]
    old_style = float(bt["metrics"].loc["满仓方案", "夏普"]
                      - bt["metrics"].loc[ctrl_main, "夏普"])
    chk(abs(contrib["夏普"]) < 1e-9,
        f"满仓方案 vs 自己的对照：超额夏普 = {contrib['夏普']:+.2e}（应为 0）")
    chk(abs(old_style) > 1e-6,
        f"（对照）若错用主策略对照会得到 {old_style:+.4f} —— 这就是被修复的假超额")

    # 主策略自己的净贡献仍然可读，且沿用旧键名
    chk(set(bt["timing_contrib"]) == {"年化收益", "夏普", "最大回撤", "Calmar"},
        "timing_contrib 向后兼容（仍是主策略的平铺字典）")
    chk(bt["control_name"] == ctrl_main, "control_name 向后兼容 = 主策略对照")


# ============================================================== T2 回测边界
def t2():
    print("\nT2 · 边界不再静默崩溃")
    idx, r, nav = _synth()
    sig = pd.DataFrame({"position_final": pd.Series(0.6, index=idx),
                        "score": pd.Series(0.2, index=idx)})

    # 起始日晚于结束日 → 应给出明确 ValueError，而不是 win_idx[0] 的 IndexError
    try:
        run_backtest(nav, sig, start_date="2023-01-01", end_date="2020-01-01")
        chk(False, "start>end 应抛 ValueError")
    except ValueError as e:
        chk("回测区间过短" in str(e) or "至少需要" in str(e), f"start>end → ValueError：{e}")
    except Exception as e:                                    # noqa: BLE001
        chk(False, f"start>end 抛了非预期异常 {type(e).__name__}: {e}")

    # 极短样本：旧实现 start_i = min(252, n-30) 会得到负索引 idx[-k]
    try:
        run_backtest(nav.iloc[:40], sig.iloc[:40])
        chk(False, "40 天样本应抛 ValueError")
    except ValueError as e:
        chk(True, f"短样本 → ValueError：{str(e)[:60]}…")
    except Exception as e:                                    # noqa: BLE001
        chk(False, f"短样本抛了非预期异常 {type(e).__name__}: {e}")

    # 结束日早于数据起点
    try:
        run_backtest(nav, sig, end_date="1999-01-01")
        chk(False, "end 早于数据应抛 ValueError")
    except ValueError:
        chk(True, "end_date 早于数据起点 → ValueError")


# ============================================================== T3 健康门控
def _health_inputs(n=1500, te_mult=None, alpha_shift=0.0, seed=11):
    """构造 fit_health_channel 的输入。te_mult/alpha_shift 用于注入失效。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n, name="date")
    daily = rng.normal(0.0, 0.008, n)
    syn_ret = pd.Series(daily, index=idx)
    active = pd.Series(rng.normal(alpha_shift / 252.0, 0.004, n), index=idx)
    nav = np.exp((syn_ret + active).cumsum())

    refit = idx[::20]
    te = np.full(len(refit), 0.08)
    if te_mult is not None:
        te[len(refit) // 2:] *= te_mult
    fit_diag = pd.DataFrame({"r2": np.full(len(refit), 0.92), "te": te},
                            index=refit)
    w = np.abs(rng.normal(0.2, 0.02, (len(refit), 4)))
    weights = pd.DataFrame(w / w.sum(1, keepdims=True) * 0.9,
                           index=refit, columns=list("abcd"))
    return nav, syn_ret, fit_diag, weights


def t3():
    print("\nT3 · 健康门控：正常不误报，失效必须触发")
    eng = SignalEngine()

    # --- 3a 健康样本：不应产生任何交易（避免修完变成"频繁误报"）
    h = eng.fit_health_channel(*_health_inputs())
    dec, trades = build_decision(h)
    st = health_stats(h)
    chk(int(dec.sum()) == len(dec), "健康样本：全程持有，无误报")
    chk(len(trades) == 0, "健康样本：零交易")
    chk(st["min"] > C.SIGNAL["health_exit"],
        f"健康样本最低健康分 {st['min']:+.3f} > 离场线 {st['exit']:+.2f}")

    # --- 3b 注入 TE 扩张 3 倍：门控必须触发
    h2 = eng.fit_health_channel(*_health_inputs(te_mult=3.0))
    dec2, trades2 = build_decision(h2)
    st2 = health_stats(h2)
    chk(st2["reachable"], f"TE 扩张 3× → 健康分触及离场线（最低 {st2['min']:+.3f}）")
    chk(len(trades2) > 0, f"TE 扩张 3× → 产生 {len(trades2)} 笔交易")
    chk(int(dec2.sum()) < len(dec2), "TE 扩张 3× → 确实出现过空仓")
    chk(any("跟踪误差" in t["reason"] for t in trades2),
        "卖出原因正确归因到「跟踪误差」分量")

    # --- 3c 单一分量恶化**不应**触发（抗噪）：聚合是「最差项 0.6 + 整体 0.4」，
    # 单独一项抖动不足以动手，否则会被最吵的那个分量牵着走。
    h3 = eng.fit_health_channel(*_health_inputs(alpha_shift=-0.20))
    st3 = health_stats(h3)
    chk(len(build_decision(h3)[1]) == 0,
        f"仅 alpha 年化 -20%（c_alpha={h3['c_alpha'].min():+.2f}）→ 不触发"
        f"（健康分最低 {st3['min']:+.3f}），符合抗噪设计")

    # --- 3c2 多分量共振才触发
    nav4, syn4, diag4, w4 = _health_inputs(alpha_shift=-0.20)
    diag4 = diag4.copy()
    diag4.loc[diag4.index[len(diag4) // 2]:, "te"] *= 2.6      # TE 同时扩张
    h4 = eng.fit_health_channel(nav4, syn4, diag4, w4)
    dec4, trades4 = build_decision(h4)
    st4 = health_stats(h4)
    chk(st4["reachable"] and len(trades4) > 0,
        f"alpha 钝化 + TE 扩张 2.6x 共振 → 触发 {len(trades4)} 笔"
        f"（健康分最低 {st4['min']:+.3f}）")

    # --- 3c3 轻微跑输（噪声量级）绝不应触发
    h3b = eng.fit_health_channel(*_health_inputs(alpha_shift=-0.03))
    chk(len(build_decision(h3b)[1]) == 0,
        f"alpha 年化 -3%（噪声量级）不触发（最低 {health_stats(h3b)['min']:+.3f}）")

    # --- 3c4 拟合不可信（R²≈0，典型成因：固定 0.9 仓位假设对混合基金不成立）
    #      → 门控必须**停用**而不是卖出：R² 低说的是"我的模型差"，不是"基金差"。
    nav5, syn5, diag5, w5 = _health_inputs()
    diag5 = diag5.copy()
    diag5["r2"] = 0.05
    diag5["te"] = 0.30                       # 顺带让 TE 也恶化
    h5 = eng.fit_health_channel(nav5, syn5, diag5, w5)
    dec5, trades5 = build_decision(h5)
    st5 = health_stats(h5)
    chk(st5["unreliable_ratio"] > 0.9,
        f"R²≈0 → 不可信占比 {st5['unreliable_ratio']:.0%}")
    chk(len(trades5) == 0 and int(dec5.sum()) == len(dec5),
        "R²≈0 → 门控停用（全程持有），不因模型失效而卖出基金")

    # --- 3d alpha 分量必须是 t 值，而不是固定刻度
    chk("alpha_t" in h.columns, "fit_health_channel 输出 alpha_t 列")
    tvals = h["alpha_t"].dropna()
    chk(tvals.abs().max() < 8 and np.isfinite(tvals).all(),
        f"alpha_t 数值合理（|t| 最大 {tvals.abs().max():.2f}），不再是饱和的 ±1")

    # --- 3e 与旧的「四分量均值」对比：同样的数据下旧聚合够不到离场线
    comps = h2[["c_r2", "c_te", "c_alpha", "c_drift"]]
    old_mean = comps.fillna(0.0).mean(axis=1)
    new_score = h2["health_score"]
    chk((new_score <= C.SIGNAL["health_exit"]).any()
        and not (old_mean <= C.SIGNAL["health_exit"]).any(),
        f"同样的数据：新聚合能触及离场线（{new_score.min():+.3f}），"
        f"旧均值聚合最低只有 {old_mean.min():+.3f} —— 这就是永不触发的根因")


# ============================================================== T4 E4 对照
def t4():
    print("\nT4 · E4 周期矩阵：对照按被评估方案自身的平均仓位构造")
    from src.experiments import lowfreq_grid

    idx, r, nav = _synth(n=900)
    syn_nav = np.exp(pd.Series(r * 0.9, index=idx).cumsum())
    weights = pd.DataFrame({"a": 0.5, "b": 0.4}, index=idx[::20])
    g = lowfreq_grid(nav, syn_nav, weights, None,
                     refit_days=(60, 120), windows=(60, 120))
    chk(len(g) > 0, f"lowfreq_grid 返回 {len(g)} 行")
    if not g.empty:
        # 每个格子的对照组仓位必须与该格子「平均仓位」一致
        bad = 0
        for row in g.itertuples():
            want = f"恒定仓位({row.平均仓位:.0%})"
            if str(row.对照组) != want:
                bad += 1
        chk(bad == 0, f"全部 {len(g)} 个格子的对照组与自身平均仓位一致（不一致 {bad} 个）")


def main():
    print("=" * 78)
    print("P0 修复回归验证")
    print("=" * 78)
    for fn in (t1, t2, t3, t4):
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
