# -*- coding: utf-8 -*-
"""P0-06 功效分析（重做）。

旧推导 `t ≈ IR × sqrt(n_eff)` → "约 190 只基金"被撤回，因为它：
  ① 把**年化时间序列 IR** 与**横截面谱有效维度**混在一个式子里；
  ② 假设 n_eff/n 是常数（扩池不改变相关结构）；
  ③ 没有声明 α、目标功效、最小有意义效应；
  ④ 谱有效维度 ≠ 均值检验的通用有效样本量。

现在按教科书做法重做：先声明检验与效应量，再算所需样本与当前功效。

    python run.py power          # 读 output/results/E7_*.csv，写 P0-06_功效分析.csv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import stats

from market_research.config import ROOT
RES = ROOT / "output" / "results"

ALPHA = 0.05
POWER = 0.80
Z_A = stats.norm.ppf(1 - ALPHA / 2)      # 1.96
Z_B = stats.norm.ppf(POWER)              # 0.84

from market_research.config import RES_DIR as RES
# 策略 tag → (逐基金 CSV, IR 列名, 汇总 CSV 或 None, 超额序列 CSV 或 None)
# E4低频 没有汇总 CSV，其 n_eff 必须从超额序列的相关矩阵现算 ——
# 直接拿名义 n=14 会把"14 只高度相关的基金"当成 14 份独立证据，
# 这正是旧"190 只"推导犯的错。
STRATS = [
    ("健康门控", "E7_稳健性_健康门控_逐基金.csv", "超额夏普",
     "E7_稳健性_健康门控_汇总.csv", None),
    ("波动率目标", "E7_稳健性_波动率目标_逐基金.csv", "超额夏普",
     "E7_稳健性_波动率目标_汇总.csv", None),
    ("E4低频", "E7_稳健性_E4低频_逐基金.csv", "sharpe",
     None, "E7_稳健性_E4低频_超额序列.csv"),
]


def _n_eff_from_excess(fp: Path) -> float:
    from market_research.research.robustness import effective_sample_size
    ex = pd.read_csv(fp, index_col=0)
    ex = ex.dropna(axis=1, how="all")
    if ex.shape[1] < 2:
        return float(ex.shape[1])
    r = effective_sample_size(ex)
    return float(r["n_eff"])


def power_of(delta: float, n_eff: float) -> float:
    """给定标准化效应量 δ 与有效样本量，双侧 t 检验的功效（正态近似）。"""
    if not np.isfinite(delta) or delta <= 0:
        return float("nan")
    return float(stats.norm.cdf(delta * np.sqrt(n_eff) - Z_A))


def main(argv=None, *, prog=None):
    import argparse
    argparse.ArgumentParser(description=__doc__, prog=prog).parse_args(argv)
    rows = []
    for name, fp, col, sumfp, exfp in STRATS:
        p = RES / fp
        if not p.exists():
            print(f"[skip] {name}: 缺 {fp}")
            continue
        df = pd.read_csv(p, index_col=0)
        if col not in df.columns:
            print(f"[skip] {name}: 无列 {col}（实际 {df.columns.tolist()}）")
            continue
        ir = pd.to_numeric(df[col], errors="coerce").dropna()
        n = len(ir)
        if n < 3:
            continue
        mu, sd = float(ir.mean()), float(ir.std(ddof=1))
        delta = abs(mu) / sd if sd > 1e-12 else float("nan")

        # 有效样本量：优先汇总 CSV；否则从超额序列现算；再不行才用名义 n
        n_eff, ratio = float(n), 1.0
        if sumfp and (RES / sumfp).exists():
            s = pd.read_csv(RES / sumfp)
            if "有效样本量_n_eff" in s.columns:
                n_eff = float(s["有效样本量_n_eff"].iloc[0])
                ratio = n_eff / n
        elif exfp and (RES / exfp).exists():
            n_eff = _n_eff_from_excess(RES / exfp)
            ratio = n_eff / n
        need_n_eff = ((Z_A + Z_B) ** 2) / delta ** 2 if delta and delta > 0 else float("inf")
        need_funds = need_n_eff / ratio if ratio > 0 else float("inf")

        rows.append({
            "策略": name,
            "基金数": n,
            "n_eff": round(n_eff, 2),
            "n_eff/n": round(ratio, 3),
            "平均IR": round(mu, 4),
            "IR标准差": round(sd, 4),
            "效应量δ": round(delta, 4),
            f"当前功效(α={ALPHA})": round(power_of(delta, n_eff), 3),
            "所需n_eff(80%功效)": (round(need_n_eff, 1)
                            if np.isfinite(need_n_eff) else "∞"),
            "所需基金数": (int(np.ceil(need_funds))
                       if np.isfinite(need_funds) else "∞"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RES / "P0-06_功效分析.csv", index=False, encoding="utf-8-sig")
    print("=" * 104)
    print("P0-06 · 功效分析（α=0.05 双侧，目标功效 80%，δ = |mean(IR)|/sd(IR)）")
    print("=" * 104)
    print(out.to_string(index=False))

    print("""
    读法与边界
    ----------
    1. **效应量 δ** 用的是"跨基金 IR 的标准化均值"，与旧的 `IR × sqrt(n_eff)` 不同 ——
       这里 δ 和 n_eff 是两个独立的量：δ 描述"单只基金层面效应有多大"，
       n_eff 描述"14 只基金相当于几个独立观测"。
    2. **所需基金数 = 所需 n_eff ÷ (n_eff/n)**。这个除法假设扩池不改变相关结构；
       若新基金与原样本相关性更低，所需数量会下降；更高则上升。
       → 扩池**不能保证**获得独立证据，这也是 P2-01 要先定纳入规则的原因。
    3. **当前功效** 全部远低于 80%：说明"检验不显著"很可能只是**缺乏功效**，
       而不是"有效应不存在"。这两件事必须分开陈述（P1-03）。
    4. 本分析只量化了**横截面**的有效样本量。**时间维度**的依赖（E9b 显示
       迁移 IR 为 −0.23、过拟合代价 0.769）没有被折算进这里的 n_eff ——
       它靠的是 E9b 的外层时间切分，不是样本量。
    5. 最小有意义效应未在本表声明：若认为 IR < 0.1 没有经济价值，
       则所需样本量应按 δ_min 而不是观测 δ 来算（观测 δ 本身被选择偏差抬高）。
    """)


if __name__ == "__main__":
    main()
