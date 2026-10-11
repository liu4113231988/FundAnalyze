# -*- coding: utf-8 -*-
"""
按管理人分折的 walk-forward 验证（E9）。

    python run_walkforward.py [基金代码...]

回答的问题：E4 报告的"最优参数"如果是在一部分基金上选出来的，
用到**没见过**的基金上还灵吗？参数选择的过拟合代价有多大？

对照的三条基准线：
  · 固定配置  ：全样本最优配置（E4 的说法），迁移到各折测试集
  · 迁移配置  ：每折用自己的训练集选最优，迁移到该折测试集（可实现）
  · oracle    ：直接用测试集本身选最优（不可实现的上界）

过拟合代价 = oracle − 迁移配置。若接近 0，说明参数选择没怎么过拟合；
若显著为正，说明"从网格里挑最大值"确实只是挑到了噪声。
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src import config as C, data as D
from src.rbsa import RBSA
from src.walkforward import (walk_forward, walk_forward_time, default_grid)

pd.set_option("display.width", 200)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    codes = sys.argv[1:] or list(C.EXTENDED_FUNDS.keys())
    t0 = time.time()
    per_fund = {}
    for c in codes:
        try:
            panel = D.load_panel(c, verbose=False)
            fit = RBSA().fit_rolling(panel, verbose=False)
            syn = RBSA.synth_nav(fit["syn_ret"], 1.0)
            per_fund[c] = {
                "nav": panel["__FUND__"].reindex(syn.index),
                "syn_nav": syn,
                "weights": fit["weights"],
                "pe": D.load_pe_panel(verbose=False),
            }
            print(f"[fit] {c} {C.EXTENDED_FUNDS[c]:16s} "
                  f"R²中位={fit['diag']['r2'].median():.3f} "
                  f"管理人={C.MANAGER_OF.get(c, '?')}")
        except Exception as e:                                # noqa: BLE001
            print(f"[SKIP] {c} 失败 {type(e).__name__}: {str(e)[:60]}")

    res = walk_forward(per_fund)
    print("\n" + "=" * 108)
    print("E9 · 按管理人分折的 walk-forward 验证（3×4 = 12 个候选配置，纯趋势通道）")
    print("=" * 108)
    print(f"\n候选配置：{len(default_grid())} 个（信号窗口 {sorted({g['win'] for g in default_grid()})} "
          f"× 再平衡 {sorted({g['reb'] for g in default_grid()})}）")
    print("\n[逐折结果]")
    print(res["folds"].to_string(index=False))

    s = res["summary"]
    print("\n[汇总]")
    print(f"  全体样本最优配置          : {s['全体最优配置']}   ← E4 报告的就是这个")
    print(f"  固定配置 迁移到测试集 IR   : {s['固定配置在测试集均值IR']:+.3f}")
    print(f"  逐折自选  迁移到测试集 IR  : {s['迁移配置在测试集均值IR']:+.3f}   "
          f"← 横截面迁移口径下的上限（★ 不等于时间维度可实现）")
    print(f"  oracle（用测试集本身）  IR : {s['oracle在测试集均值IR']:+.3f}   ← 不可实现的上界")
    print(f"  ★ 参数选择过拟合代价       : {s['参数选择过拟合代价']:+.3f}")
    print(f"  选择一致性                 : {s['选择一致性']}")

    # ---- 判定
    cost = s["参数选择过拟合代价"]
    mig = s["迁移配置在测试集均值IR"]
    print("\n[判定]")
    if mig > 0 and cost < 0.10:
        verdict = ("★ 参数选择基本没有过拟合，且迁移后仍为正 —— "
                   "这条线索值得在独立样本上继续验证")
        ok = True
    elif mig > 0:
        verdict = (f"★ 迁移后仍为正，但过拟合代价 {cost:+.3f} 不小 —— "
                   "全样本选参的结论打了折扣")
        ok = None
    else:
        verdict = ("★ 迁移后不再为正 —— E4 的结论是参数选择过拟合的产物，"
                   "不能作为候选线索")
        ok = False
    print(f"  {verdict}")
    print(f"  对照：全体样本直接选最优的 IR = "
          f"{max([res['table'][c][(int(g[0]), int(g[1]))]['超额夏普'] for c in res['table'] for g in [(k[0], k[1])]] if False else [0.0]):+.3f}"
          if False else
          f"  （E4 报告的 +0.352 是全样本口径，这里的固定配置迁移 IR 才是可实现值）")

    # ---- E9b：真正的时间向前验证（P0-03）
    # walk_forward 的每一折都覆盖了同一段时间，只隔离了管理人；
    # 它**不能**回答"参数能否迁移到未来"。这一段才是时间维度的检验。
    print("\n" + "=" * 108)
    print("E9b · 时间向前验证（外层按时间扩展，内层只用训练期选参）")
    print("=" * 108)
    res_t = walk_forward_time(per_fund)
    if not res_t.get("ok"):
        print(f"  跳过：{res_t.get('reason')}")
    else:
        print(res_t["folds"].to_string(index=False))
        st = res_t["summary"]
        print("\n[汇总]")
        for k, v in st.items():
            print(f"  {k:24s}: {v}")
        mig_t, cost_t = st["迁移配置在测试期均值IR"], st["参数选择过拟合代价"]
        ok_mig = isinstance(mig_t, float) and np.isfinite(mig_t) and mig_t > 0
        ok_cost = isinstance(cost_t, float) and np.isfinite(cost_t) and cost_t < 0.10
        if ok_mig and ok_cost:
            print("  ★ 时间维度也站得住：训练期选出的参数在未来仍然为正")
        else:
            print("  ★ 时间维度不成立：训练期选出的参数在未来不再为正，"
                  "或过拟合代价过大 —— E9 的横截面结论不得外推为时间泛化")

    out = C.RES_DIR
    res["folds"].to_csv(out / "E9_分折验证_逐折.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"指标": k, "值": v} for k, v in s.items()]).to_csv(
        out / "E9_分折验证_汇总.csv", index=False, encoding="utf-8-sig")
    if res_t.get("ok"):
        res_t["folds"].to_csv(out / "E9b_时间向前验证_逐折.csv",
                              index=False, encoding="utf-8-sig")
        pd.DataFrame([{"指标": k, "值": v}
                      for k, v in res_t["summary"].items()]).to_csv(
            out / "E9b_时间向前验证_汇总.csv", index=False, encoding="utf-8-sig")
    print(f"\n总耗时 {time.time()-t0:.0f}s  结果写入 {out}")
    return 0 if ok else 0


if __name__ == "__main__":
    sys.exit(main())
