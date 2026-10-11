# -*- coding: utf-8 -*-
"""
E7 · 稳健性检验：E4 那条"250 日窗口 + 120 日再平衡"的线索到底有多可信。

    python run_robustness.py [代码...]      # 默认全部 14 只

回答三个问题：
  1. 14 只基金的有效独立样本量有多少？（横截面相关矩阵）
  2. 超额是市场暴露的另一种写法吗？（对沪深300 回归 + Newey-West）
  3. "14/14 为正"在移动块 bootstrap 下有多意外？（保留横截面相关性）

⚠ 因子回归里 β 接近 1 不代表模型有问题 —— 策略平均仓位只有 ~60%，
   低仓位策略天然就有 β≈0.6。真正的判据是 **α 是否显著为正**。
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from src import config as C, data as D
from src.rbsa import RBSA
from src.robustness import assess_positions
from src.signal import SignalEngine

pd.set_option("display.width", 200)


def _report(res: dict) -> None:
    print("\n" + "=" * 96)
    print(f"E7 · 稳健性检验（{res['label']}）")
    print("=" * 96)
    if not res.get("ok"):
        print("失败：", res.get("diag"))
        return
    print("\n[逐基金样本外超额]")
    print(res["per_fund"].to_string())

    eff = res["effective"]
    print("\n[1] 横截面独立性与有效样本量")
    print(f"  平均两两相关系数      {eff['avg_corr']:+.3f}")
    print(f"  第一主成分解释方差    {eff['pc1_share']:.1%}")
    print(f"  ★ 有效独立样本量 n_eff = {eff['n_eff']:.2f}"
          f"  ({eff['n_eff_ratio']:.0%} of {eff['n_funds']})")

    fac = res.get("factor")
    if fac:
        print("\n[2] 共同因子检查")
        print(f"  市场 beta 范围  {fac['per_fund']['市场beta'].min():+.3f} ~ "
              f"{fac['per_fund']['市场beta'].max():+.3f}"
              f"   市场 R² 最大 {fac['per_fund']['市场R2'].max():.1%}")
        print(f"  年化超额 > 0    {fac['n_pos_alpha']}/{fac['n_total']} 只")
        print(f"  ★ NW t 值 |t|>1.96  {fac['n_sig']}/{fac['n_total']} 只")

    b = res["bootstrap"]
    print("\n[3] 移动块 bootstrap（60 日块 × 2000 次，所有基金共用重采样日期）")
    print(f"  观测到的'净贡献为正'比例  {b['share_pos']:.1%}")
    print(f"  零假设分布                均值 {b['boot_mean']:.1%}"
          f"，90% 区间 [{b['boot_p05']:.1%}, {b['boot_p95']:.1%}]")
    print(f"  ★ p 值 = {b['p_value']:.3f}")
    v = ("显著" if b["p_value"] < 0.05
         else "边缘" if b["p_value"] < 0.10 else "不显著")
    print(f"  判定：{v}")

    pf = res["per_fund"]
    print("\n[结论]")
    print(f"  · 有效独立样本量只有 {eff['n_eff']:.1f} 个"
          f"（第一主成分解释 {eff['pc1_share']:.0%}）")
    if fac:
        print(f"  · NW t 显著的只有 {fac['n_sig']}/{fac['n_total']} 只")
    print(f"  · bootstrap p = {b['p_value']:.3f}（{v}）")
    print(f"  · 平均仓位 {pf['平均仓位'].mean():.1%}"
          f"，年化换手 {pf['换手'].mean():.2f}x")


def main():
    codes = sys.argv[1:] or list(C.EXTENDED_FUNDS.keys())
    t0 = time.time()
    nav_by, syn_by, pos_vt = {}, {}, {}
    for c in codes:
        panel = D.load_panel(c, verbose=False)
        fit = RBSA().fit_rolling(panel, verbose=False)
        syn = RBSA.synth_nav(fit["syn_ret"], 1.0)
        nav_by[c] = panel["__FUND__"].reindex(syn.index)
        syn_by[c] = syn
        pos_vt[c] = SignalEngine().vol_target_channel(syn)["vol_target_pos"]
        print(f"[fit] {c} {C.EXTENDED_FUNDS[c]:16s} "
              f"R²中位={fit['diag']['r2'].median():.3f}")

    panel = D.load_panel(codes[0], verbose=False)
    res_vt = assess_positions(nav_by, pos_vt, panel, label="波动率目标(逆波动)")
    _report(res_vt)

    out = C.RES_DIR
    for res, tag in ((res_vt, "波动率目标"),):
        if not res.get("ok"):
            continue
        res["per_fund"].to_csv(out / f"E7_稳健性_{tag}_逐基金.csv",
                               encoding="utf-8-sig")
        res["excess"].to_csv(out / f"E7_稳健性_{tag}_超额序列.csv",
                             encoding="utf-8-sig")
        b, eff = res["bootstrap"], res["effective"]
        pd.DataFrame([{
            "策略": res["label"],
            "有效样本量_n_eff": round(eff["n_eff"], 2),
            "第一主成分解释方差": round(eff["pc1_share"], 4),
            "平均两两相关": round(eff["avg_corr"], 4),
            "为正比例": b["share_pos"],
            "bootstrap_p05": b["boot_p05"],
            "bootstrap_p95": b["boot_p95"],
            "bootstrap_p值": round(b["p_value"], 4),
            "NW显著只数": res["factor"]["n_sig"] if res.get("factor") else None,
        }]).to_csv(out / f"E7_稳健性_{tag}_汇总.csv", index=False,
                   encoding="utf-8-sig")
    print(f"\n总耗时 {time.time()-t0:.0f}s  结果写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())