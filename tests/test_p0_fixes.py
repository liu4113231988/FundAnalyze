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
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                             # noqa: BLE001
    pass

import numpy as np
import pandas as pd

import market_research.config as C
from market_research.modeling.backtest import run_backtest
from market_research.modeling.signals import SignalEngine, build_decision, health_stats

C_RF = C.BACKTEST["cash_rate"]

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
    chk(ctrl_main == f"恒定仓位({avg_main:.1%})",
        f"主策略对照 = {ctrl_main}，与其自身平均仓位 {avg_main:.1%} 一致")
    chk(ctrl_full == "恒定仓位(100.0%)", f"满仓方案对照 = {ctrl_full}（平均仓位 100%）")
    chk(ctrl_main != ctrl_full, "两个方案不再共用同一份对照")

    # ★ P0-02 键碰撞回归：精确仓位不同但四舍五入到 0.1% 相同（或相近）的两个
    #   方案，必须各自配一份**自己的**对照。旧实现用 .0% 作键，
    #   50.1% 与 50.4% 会共用"恒定仓位(50%)"，后者凭空多出虚假超额。
    close = {"近A": pd.Series(0.501, index=idx), "近B": pd.Series(0.504, index=idx)}
    btc = run_backtest(nav, sig, variants=close)
    cA, cB = btc["controls"]["近A"], btc["controls"]["近B"]
    chk(cA != cB, f"相近仓位不再碰撞：{cA} vs {cB}")
    for v in ("近A", "近B"):
        tcv = btc["timing_contrib_all"][v]
        chk(abs(tcv["年化收益"]) < 1e-12 and abs(tcv["夏普"]) < 1e-12,
            f"{v} 对自己的精确对照净贡献 = {tcv['年化收益']:+.2e}（应为 0）")
    # 顺序无关：交换 variants 顺序结果不变
    btc2 = run_backtest(nav, sig, variants={"近B": close["近B"], "近A": close["近A"]})
    chk(abs(btc2["timing_contrib_all"]["近B"]["年化收益"]
            - btc["timing_contrib_all"]["近B"]["年化收益"]) < 1e-15,
        "交换策略顺序不改变净贡献")

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

    # ★ 夏普不可加：净贡献必须由「超额日序列」算（信息比率式），
    #    不能拿 metrics 表里两个几何年化夏普相减。
    # 用「满仓 vs 等仓位对照」构造一个满仓且天天调仓的例子：
    # 若用相减口径，得到的差值与超额序列的真实 IR 完全不同。
    sig3 = pd.DataFrame({"position_final": pd.Series(0.5 + 0.45 * np.sin(
        np.arange(len(idx)) / 40.0), index=idx), "score": pd.Series(0.1, index=idx)})
    bt3 = run_backtest(nav, sig3, main_name="满仓调仓")
    m3 = bt3["metrics"]
    cn3 = bt3["control_name"]
    naive = float(m3.loc["满仓调仓", "夏普"] - m3.loc[cn3, "夏普"])
    proper = bt3["timing_contrib"]["夏普"]
    ex = (bt3["series"]["满仓调仓"] - bt3["series"][cn3])
    ir = float(ex.mean() * 252) / float(ex.std(ddof=1) * np.sqrt(252))
    chk(abs(proper - ir) < 1e-9,
        f"净贡献夏普 == 超额序列 IR（{proper:.4f} vs {ir:.4f}，IR 不扣无风险利率）")
    chk(abs(proper - naive) > 1e-6,
        f"净贡献夏普 {proper:.4f} ≠ 两个几何夏普之差 {naive:.4f} "
        "—— 口径确已切换，不能再退回相减")


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

    # --- 3c 中等程度的单一分量恶化**不应**触发（抗噪）：聚合是
    # 「最差项 0.6 + 整体 0.4」，单项轻微抖动不足以动手。
    h3 = eng.fit_health_channel(*_health_inputs(alpha_shift=-0.08))
    st3 = health_stats(h3)
    chk(len(build_decision(h3)[1]) == 0,
        f"仅 alpha 年化 -8%（c_alpha={h3['c_alpha'].min():+.2f}）→ 不触发"
        f"（健康分最低 {st3['min']:+.3f}），符合抗噪设计")

    # --- 3c2 极端 + 共振：一定触发
    nav4, syn4, diag4, w4 = _health_inputs(alpha_shift=-0.20)
    diag4 = diag4.copy()
    diag4.loc[diag4.index[len(diag4) // 2]:, "te"] *= 2.6      # TE 同时扩张
    h4 = eng.fit_health_channel(nav4, syn4, diag4, w4)
    dec4, trades4 = build_decision(h4)
    st4 = health_stats(h4)
    chk(st4["reachable"] and len(trades4) > 0,
        f"alpha 钝化 + TE 扩张 2.6x 共振 → 触发 {len(trades4)} 笔"
        f"（健康分最低 {st4['min']:+.3f}）")

    # --- 3c2b 分量必须**实际能够参与**决策，而不是文档里的装饰。
    # 实测 14 只基金 × 33152 个观测：刻度取 6 时 c_alpha 从未达到 −0.5（0.00%），
    # 即"能力钝化"这个最有经济含义的失效模式完全无法参与风控。刻度 3 修复。
    s_all = h3["c_alpha"].dropna()
    chk(float(C.SIGNAL["health_alpha_t"]) <= 3.0,
        f"alpha 分量刻度 = {C.SIGNAL['health_alpha_t']}（≤3 才能让该项进入告警区）")

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
    from market_research.research.experiments import lowfreq_grid

    idx, r, nav = _synth(n=900)
    syn_nav = np.exp(pd.Series(r * 0.9, index=idx).cumsum())
    weights = pd.DataFrame({"a": 0.5, "b": 0.4}, index=idx[::20])
    g = lowfreq_grid(nav, syn_nav, weights, None,
                     refit_days=(60, 120), windows=(60, 120))
    chk(len(g) > 0, f"lowfreq_grid 返回 {len(g)} 行")
    if not g.empty:
        # 每个格子的对照组仓位必须与该格子「平均仓位」一致
        import re
        bad = []
        for row in g.itertuples():
            got = str(row.对照组)
            # 直接比对字符串会撞上 0.05% 的格式化边界（如 0.556499 → "55.6%"
            # 而 round(...,4) → "55.7%"），所以解析出百分比按数值比较，
            # 容差取 .1% 精度的一半。展示名撞车时会追加 "#N" 消歧（P0-02）。
            m = re.search(r"\(([\d.]+)%\)", got)
            fm = float(m.group(1)) / 100.0 if m else float("nan")
            if not np.isfinite(fm) or abs(fm - float(row.平均仓位)) > 5e-4:
                bad.append(f"{row.平均仓位:.4f} vs {got}")
        chk(not bad, f"全部 {len(g)} 个格子的对照组与自身平均仓位一致"
                     + (f"（不一致：{bad}）" if bad else ""))


# ============================================================== T5 模型层
def _synth_panel(n=600, k=5, seed=3):
    """列波动率刻意拉开 6 倍，用于检验权重是否从标准化空间正确还原。"""
    rng = np.random.default_rng(seed)
    sd = np.linspace(0.004, 0.024, k)
    X = rng.normal(0.0, 1.0, (n, k)) * sd
    w = np.array([0.30, 0.25, 0.15, 0.12, 0.08])[:k]
    y = X @ w
    cols = [f"a{i}" for i in range(k)]
    return X, y, cols


def t5():
    print("\nT5 · 模型层：权重必须从标准化空间正确还原（除以 sd）")
    from market_research.modeling.models import fit_window

    X, y, cols = _synth_panel()
    X2 = X.copy()
    X2[:, 0] *= 2.0        # 只放大第 0 列的波动

    for key in ("pcr", "krr"):
        try:
            r1 = fit_window(key, X, y, cols, None, 0.9)
            r2 = fit_window(key, X2, y, cols, None, 0.9)
            w1 = r1.weights.reindex(cols).fillna(0.0).values
            w2 = r2.weights.reindex(cols).fillna(0.0).values
            ratio1 = w1[0] / w1[1] if abs(w1[1]) > 1e-9 else np.nan
            ratio2 = w2[0] / w2[1] if abs(w2[1]) > 1e-9 else np.nan
            # 第 0 列波动翻倍 ⇒ 它的权重应约为原来的一半
            want = ratio1 / 2.0
            ok = np.isfinite(ratio2) and abs(ratio2 - want) < 0.15 * abs(want)
            chk(ok, f"{key}: 列0 波动×2 → 权重比 {ratio1:.3f} → {ratio2:.3f}"
                    f"（应≈{want:.3f}）。若不除以 sd 则两者会完全相等")
        except Exception as e:                                # noqa: BLE001
            chk(False, f"{key} 拟合抛异常 {type(e).__name__}: {e}")

    # 评分口径：pred 必须等于 X@w（PCA 早期版本用带截距的 lr.predict，
    # 与样本外用的 X@w 不一致，会让窗口内 R² 虚高并被误判为过拟合）
    try:
        r = fit_window("pcr", X, y, cols, None, 0.9)
        wv = r.weights.reindex(cols).fillna(0.0).values
        chk(np.allclose(r.pred, X @ wv, atol=1e-10),
            "pcr: pred 与 X@w 一致（窗口内与样本外同一口径）")
    except Exception as e:                                    # noqa: BLE001
        chk(False, f"pcr 一致性检查抛异常 {type(e).__name__}: {e}")

    # SLSQP 不收敛时必须兜底，不能返回 NaN
    from market_research.modeling.models import _constrained_fit
    bad = np.full((5, 3), np.nan)          # 全 NaN 的输入会让 SLSQP 直接失败
    w = _constrained_fit(bad, np.full(5, np.nan), 0.9)
    chk(np.all(np.isfinite(w)) and abs(w.sum() - 0.9) < 1e-6,
        f"约束回归在病态输入下返回有限权重（Σw={w.sum():.3f}），不产生 NaN")


# ============================================================== T6 数据缓存
def t6():
    print("\nT6 · 数据缓存：口径可失效、并发写不损坏")
    import market_research.data.marketdata as D

    # 元数据不一致 → 判为过期
    D._cache_meta_write("_t6probe", {"v": 1, "end": None})
    same = D._cache_meta_valid("_t6probe", {"v": 1, "end": None})
    diff = D._cache_meta_valid("_t6probe", {"v": 2, "end": "2020-01-01"})
    chk(same and not diff, "缓存元数据一致则命中、口径变更则判为过期")
    p = D._cache_meta_path("_t6probe")
    if p.exists():
        p.unlink()

    # ---- 分层缓存的行为语义（用计数 builder，不碰网络）
    # 存储重构后 load_panel 不再落盘整块面板，改为「指数池全局一份 + 净值每只一份」，
    # 面板即时重建。所以这里要测的是 _cached 的命中/失效语义，而不是某个键的拼法。
    import market_research.config as C
    calls = {"n": 0}

    def _builder():
        calls["n"] += 1
        idx = pd.bdate_range("2020-01-01", periods=5, name="date")
        return pd.DataFrame({"a": range(5), "b": range(5)}, index=idx)

    import uuid
    name = "_t6layered_" + uuid.uuid4().hex
    for suffix in (".csv", ".meta.json"):
        f = C.DATA_DIR / f"{name}{suffix}"
        if f.exists():
            f.unlink()

    d1 = D._cached(name, _builder, meta={"v": 1})
    n1 = calls["n"]
    d2 = D._cached(name, _builder, meta={"v": 1})
    n2 = calls["n"]
    chk(n1 == 1 and n2 == 1,
        f"二级加载命中缓存（builder 只被调用 {n1} 次，第二次未重算）")
    chk(d1.equals(d2), "命中缓存与首次构建结果一致")

    d3 = D._cached(name, _builder, meta={"v": 2})       # 版本变更 → 应判过期
    chk(calls["n"] == 2, f"口径版本变更后重新构建（builder 累计 {calls['n']} 次）")
    chk(d3.equals(d1), "重建结果与首次一致")

    # 原子写不应留下临时文件
    left = list(C.DATA_DIR.glob(f"{name}*.tmp"))
    chk(not left, f"原子写无残留临时文件（{len(left)} 个）")
    for suffix in (".csv", ".meta.json"):
        f = C.DATA_DIR / f"{name}{suffix}"
        if f.exists():
            f.unlink()

    import market_research.data.storage as storage
    with storage.connection() as db:
        db.execute("DELETE FROM prices WHERE dataset=?", [name])
        db.execute("DELETE FROM datasets WHERE name=?", [name])

    # ---- 分层结构：指数池必须与基金净值分开存放
    import inspect
    sig_lp = inspect.signature(D.load_panel)
    chk("fund_code" in sig_lp.parameters and "end" in sig_lp.parameters,
        "load_panel 签名保持兼容（fund_code / start / end / refresh / verbose）")
    for fn in ("load_index_universe", "load_fund_nav_cached",
               "_atomic_csv_write", "_os_replace_retry"):
        chk(hasattr(D, fn), f"分层缓存函数存在：{fn}")
    # 指数池的缓存名不含基金代码 —— 这是"全局共享、不逐基金重复"的关键
    src_txt = inspect.getsource(D.load_index_universe)
    chk("index_universe_" in src_txt and "fund_code" not in src_txt,
        "指数池缓存键与基金无关（全局一份，避免 N 份重复指数数据）")

    # 并发写同一缓存文件（两个相同请求不同 job_id 会撞同一路径）
    import pickle, threading
    import market_research.config as C
    from market_research.data.files import _atomic_pickle_dump
    fp = C.DATA_DIR / "_t6probe.pkl"
    def w(i):
        for _ in range(30):
            _atomic_pickle_dump({"i": i, "pad": list(range(500))}, fp)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    try:
        obj = pickle.loads(fp.read_bytes())
        ok = isinstance(obj, dict) and "pad" in obj
    except Exception as e:                                    # noqa: BLE001
        ok = False
        print(f"      加载失败 {type(e).__name__}: {e}")
    chk(ok, "4 线程并发写同一路径后文件仍可正常反序列化")
    left = list(C.DATA_DIR.glob("_t6probe*.tmp"))
    chk(not left, f"无残留临时文件（{len(left)} 个）")
    if fp.exists():
        fp.unlink()


# ============================================================== T7 仓位自适应
def _low_equity_panel(n=1400, gamma=0.30, seed=17, k=12):
    """
    构造一只"实际股票仓位只有 gamma"的基金。

    关键：y 的信号必须是**非负且集中的**持仓组合 Σ b_j·u_j（Σb=gamma），
    而不是"k 个指数的等权平均"。后者会让 RBSA 在 Σw 约束下的解与信号
    方向一致但尺度差 sqrt(k) 倍，从而自己造出尺度失真 ——
    那样测的就不是"仓位失配"，而是数据构造的artifact。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    X = rng.normal(0.0005, 0.012, (n, k))
    b = np.zeros(k)
    pick = rng.choice(k, size=3, replace=False)       # 集中持仓，符合真实基金
    b[pick] = gamma * np.array([0.5, 0.3, 0.2])
    y = X @ b + rng.normal(0, 0.001, n)
    cols = [f"i{i}" for i in range(k)]
    px = pd.concat([pd.DataFrame(100 * np.exp(X[:, j].cumsum()), index=idx,
                                 columns=[cols[j]]) for j in range(k)], axis=1)
    px["__FUND__"] = 100 * np.exp(y.cumsum())
    return px


def t7():
    print("\nT7 · 仓位自适应：固定 Σw=0.9 对低仓位基金是灾难性的")
    import market_research.config as C
    from market_research.modeling.models import ModelZoo

    for gamma in (0.30, 0.90):
        px = _low_equity_panel(gamma=gamma)
        assets = [c for c in px.columns if c != "__FUND__"]
        R = np.log(px).diff().dropna()
        dates = R.index
        X, y = R[assets].values, R["__FUND__"].values
        reb = ModelZoo._rebal_idx(len(dates), C.RBSA["window"],
                                  C.RBSA["refit"], 30)
        fixed = ModelZoo({**C.RBSA, "auto_equity": False})._run_one(
            "enet_slsqp", X, y, assets, dates, reb, None)
        auto = ModelZoo({**C.RBSA, "auto_equity": True})._run_one(
            "enet_slsqp", X, y, assets, dates, reb, None)
        rf, ra = fixed.fit_diag["r2"].median(), auto.fit_diag["r2"].median()
        eq = auto.extra.get("equity_used", float("nan"))
        tag = f"真实仓位 {gamma:.0%}"
        if gamma < 0.5:
            chk(rf < 0.5, f"{tag}：固定 Σw=0.9 的窗口 R² 仅 {rf:.3f}（确认失配）")
            chk(ra > 0.78,
                f"{tag}：自适应后 R² 恢复到 {ra:.3f}（估计仓位 {eq:.2f}）")
            chk(abs(eq - gamma) < 0.10, f"{tag}：估计仓位 {eq:.2f} 接近真值")
            chk(auto.oos_te < fixed.oos_te,
                f"{tag}：年化 TE {fixed.oos_te:.1%} → {auto.oos_te:.1%}")
        else:
            chk(ra > 0.85, f"{tag}：自适应后 R² = {ra:.3f}（应仍然很高）")
            chk(abs(eq - gamma) < 0.15, f"{tag}：估计仓位 {eq:.2f} 接近真值")

    # ModelZoo(cfg) 传入的 equity_ratio 必须真的生效（曾经被静默忽略）
    px = _low_equity_panel(gamma=0.9)
    assets = [c for c in px.columns if c != "__FUND__"]
    R = np.log(px).diff().dropna()
    dates = R.index
    X, y = R[assets].values, R["__FUND__"].values
    reb = ModelZoo._rebal_idx(len(dates), C.RBSA["window"], C.RBSA["refit"], 30)
    a = ModelZoo({**C.RBSA, "auto_equity": False, "equity_ratio": 0.90})._run_one(
        "enet_slsqp", X, y, assets, dates, reb, None)
    b = ModelZoo({**C.RBSA, "auto_equity": False, "equity_ratio": 0.30})._run_one(
        "enet_slsqp", X, y, assets, dates, reb, None)
    chk(abs(a.fit_diag["r2"].median() - b.fit_diag["r2"].median()) > 0.05,
        f"ModelZoo(cfg) 的 equity_ratio 真正生效："
        f"0.90→{a.fit_diag['r2'].median():.3f}  0.30→{b.fit_diag['r2'].median():.3f}")


# ============================================================== T8 成本敏感性
def t8():
    print("\nT8 · 成本敏感性：成本影响必须与换手成正比")
    from market_research.research.experiments import cost_sensitivity, cost_durability, COST_TIERS

    rng = np.random.default_rng(5)
    n = 1500
    idx = pd.bdate_range("2018-01-02", periods=n, name="date")
    nav = pd.Series(100 * np.exp(rng.normal(0.0002, 0.012, n).cumsum()),
                    index=idx, name="fund")

    # 方案A：逐日翻转（换手极高）  方案B：恒定 0.5（零换手，≡ 自己的对照）
    alt = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, 0.0), index=idx)
    flat = pd.Series(0.5, index=idx)
    sig = pd.DataFrame({"position_final": alt, "score": pd.Series(0.5, index=idx)})

    df = cost_sensitivity(nav, sig, variants={"零换手": flat}, costs=COST_TIERS)
    rate = {c[0]: c[1] for c in COST_TIERS}
    cols = [c[0] for c in COST_TIERS]          # ★ 必须按费率排序，
    piv = df.pivot_table(index="方案", columns="成本档位",             # 否则
                         values="超额年化").reindex(columns=cols)     # "首末档"
    tur = df.groupby("方案")["年化换手"].first()                     # 会取错
    hi_name = "择时策略(融合)"          # sig["position_final"] 对应的高换手方案
    chk(tur[hi_name] > 100,
        f"逐日翻转方案的年化换手 = {tur[hi_name]:.0f}x（确实是高换手）")

    # 零换手方案：它就是自己的对照，任何成本档下净贡献都应精确为 0
    zero = piv.loc["零换手"].dropna()
    chk(zero.abs().max() < 1e-9,
        f"零换手方案在 3 档成本下净贡献恒为 0（最大 {zero.abs().max():.2e}）")

    # 高换手方案：净贡献随成本单调下降，且降幅≈ 换手 × Δ费率
    hi = piv.loc[hi_name].dropna()
    chk(hi.is_monotonic_decreasing,
        "高换手方案的净贡献随成本单调下降："
        + " → ".join(f"{v:.2%}" for v in hi.values))
    predicted = (rate[cols[-1]] - rate[cols[0]]) * tur[hi_name]
    actual = float(hi[cols[0]] - hi[cols[-1]])
    chk(abs(actual - predicted) < 0.15 * max(abs(predicted), 1e-9),
        f"成本影响量级正确：实测 {actual:.2%} vs 换手×Δ费率 = {predicted:.2%}"
        f"（换手 {tur[hi_name]:.1f}）")

    # 耐久性视图：零换手方案净贡献恰为 0（非正），高换手方案承受不了任何档
    d = cost_durability(df)
    dd = d.set_index("方案")["最高可承受档位"].to_dict()
    chk(dd.get("零换手") == "无正贡献",
        f"零换手方案的最高可承受档位 = {dd.get('零换手')}（净贡献为 0，不算正）")
    chk(dd.get(hi_name) == "无正贡献",
        f"高换手方案（换手 {tur[hi_name]:.0f}）在所有成本档下均无正贡献")


# ============================================================== T9 趋势状态
def t9():
    print("\nT9 · 趋势状态判定（高低点结构）：标签正确性与退化边界")
    from market_research.modeling.trends import classify_series, classify_at, summarize, run_lengths, trend_payload, TREND_LEVELS

    # --- 9a 单调上升 / 下降：结构最清晰的情形
    up = list(range(1, 21))
    dn = list(range(20, 0, -1))
    for n in (1, 2, 3, 5, 10):
        cu = classify_series(up, n)
        cd = classify_series(dn, n)
        body_u = cu[2 * n - 1:]
        body_d = cd[2 * n - 1:]
        chk(all(c == 'U' for c in body_u),
            f"N={n} 单调上升 → 全程 U（{len(body_u)} 天）")
        chk(all(c == 'D' for c in body_d),
            f"N={n} 单调下降 → 全程 D（{len(body_d)} 天）")

    # --- 9b 样本不足：前 2N−1 天必须是占位符，不能凭空判定
    for n in (1, 3, 5):
        c = classify_series(up, n)
        chk(all(x == '.' for x in c[:2 * n - 1]),
            f"N={n}：前 {2 * n - 1} 天为占位符（样本不足）")

    # --- 9c 编码长度必须等于序列长度（前端按下标一一对应取色）
    for n in (1, 2, 5, 10):
        chk(len(classify_series(up, n)) == len(up),
            f"N={n}：编码长度 = 序列长度")

    # --- 9d N=1 退化：只能比单日涨跌，出不了震荡与拐头
    rng = np.random.default_rng(3)
    noisy = list(np.cumsum(rng.normal(0, 1, 400))) 
    c1 = classify_series(noisy, 1)
    kinds1 = set(c1) - {'.'}
    chk(kinds1 <= {'U', 'D'},
        f"N=1 只产生 U/D，实际出现的状态 {sorted(kinds1)}")
    chk('R' in set(classify_series(noisy, 5)) - {'.'},
        "N=5 时同样的序列能判出震荡（说明 N=1 是退化而非数据问题）")

    # --- 9e 震荡：高点抬高但低点同时降低（区间扩张）必须判为 R。
    #     构造要让"两组各 N 日"真的满足：prior=[3,8]（高 8 低 3）、
    #     recent=[1,9]（高 9 低 1）→ 高升但低降，两个方向的条件都不满足。
    seq = [3.0, 8.0, 1.0, 9.0]
    r = classify_at(np.asarray(seq), 3, 2, 'U')
    chk(r is not None and r['hr'] == 9 and r['lr'] == 1
        and r['hp'] == 8 and r['lp'] == 3 and r['code'] == 'R',
        f"高点抬高+低点降低 → 震荡（高 {r['hr']}/{r['hp']} 低 {r['lr']}/{r['lp']}"
        f"，实际 {r['code'] if r else '—'}）")
    # 对照：两组极值都抬高才是上升
    seq2 = [1.0, 4.0, 3.0, 5.0]
    r2 = classify_at(np.asarray(seq2), 3, 2, 'U')
    chk(r2 is not None and r2['code'] == 'U',
        f"高点与低点都抬高 → 上升（实际 {r2['code'] if r2 else '—'}）")

    # --- 9f 拐头向下：上升途中冲高回落。
    #     条件 = 结构未反转 + 极值落在窗口前半段 + 当前值在区间下 1/3
    #            + 前一日同为向上（否则只是"处在低位"，不是"转折"）
    turn_dn = [8, 9, 10, 11, 12, 13, 14, 13, 12, 11]
    cd = classify_series(turn_dn, 4)
    chk('d' in cd, f"上升后冲高回落 → 出现拐头向下（{''.join(cd)}）")
    # 去掉"前一日同向"这个条件就不该出现 —— 用一个持续下降序列验证
    cd2 = classify_series(list(range(20, 0, -1)), 4)
    chk('d' not in cd2 and 'u' not in cd2,
        "持续下降序列不产生拐头（前一日非向上 → 不满足转折条件）")

    # --- 9g 拐头向上：与 9f 镜像
    turn_up = [-x for x in turn_dn]
    cu = classify_series(turn_up, 4)
    chk('u' in cu, f"下降后探底回升 → 出现拐头向上（{''.join(cu)}）")

    # --- 9h 段长统计：拐头段长必须恒为 1，且与体制态分开统计
    codes = classify_series(noisy, 5)
    rl = run_lengths(codes)
    chk(all(x == 1 for x in rl['turn']) if rl['turn'] else True,
        f"拐头段长恒为 1 天（共 {len(rl['turn'])} 段，"
        f"取值 {sorted(set(rl['turn']))}）")
    chk(all(x >= 1 for x in rl['regime']),
        f"体制态段长统计正常（{len(rl['regime'])} 段）")
    st = summarize(codes)
    chk(st['regime_runs'] == len(rl['regime'])
        and st['turn_runs'] == len(rl['turn']),
        "summarize 与 run_lengths 的段数一致")
    chk(st['regime_mean'] > 0 and st['days'] > 0,
        f"summarize 给出均值 {st['regime_mean']} 天 / 有效 {st['days']} 天")

    # --- 9i 打乱时间顺序后方向性标签应显著减少（判据抓的是真结构，不是噪声）
    real_d = sum(1 for c in classify_series(noisy, 5) if c in ('U', 'D'))
    shuf = noisy[:]
    for i in range(len(shuf) - 1, 0, -1):
        j = int(rng.integers(0, i + 1))
        shuf[i], shuf[j] = shuf[j], shuf[i]
    shuf_d = sum(1 for c in classify_series(shuf, 5) if c in ('U', 'D'))
    chk(real_d > shuf_d,
        f"真实序列的方向性标签 {real_d} > 打乱后 {shuf_d}"
        f"（判据确实抓到了时序结构）")

    # --- 9j trend_payload：codes 长度必须等于绘图下标长度
    import pandas as pd
    idx = pd.bdate_range("2020-01-01", periods=300, name="date")
    nav = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    plot_idx = idx[::4]
    pl = trend_payload(nav, plot_idx)
    chk(set(pl['levels']) == set(TREND_LEVELS),
        f"payload 含全部 {len(TREND_LEVELS)} 个回看窗口")
    bad = [k for k, v in pl['codes'].items() if len(v) != len(plot_idx)]
    chk(not bad, f"每个窗口的 codes 长度 = 绘图点数 {len(plot_idx)}"
        + (f"（异常：{bad}）" if bad else ""))
    chk(all(pl['last'][k] is not None for k in map(str, TREND_LEVELS)),
        "每个窗口都给出了最后一天的关键判据数值")
    chk(all(set(pl['stats'][k]) >= {'regime_runs', 'regime_mean',
                                    'regime_median', 'turn_runs'}
            for k in map(str, TREND_LEVELS)),
        "每个窗口都给出了历史可靠性统计")


def main():
    print("=" * 78)
    print("P0 修复回归验证")
    print("=" * 78)
    for fn in (t1, t2, t3, t4, t5, t6, t7, t8, t9):
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
