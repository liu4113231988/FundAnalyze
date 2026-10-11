# -*- coding: utf-8 -*-
"""诊断实验入口：IC 期限结构 / 通道消融 / 参数敏感性。"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src import config as C, data as D
from src.pipeline import run_fund
from src.experiments import (ic_term_structure, channel_ablation,
                             param_sensitivity, summarize_funds)

pd.set_option("display.width", 200)


def main():
    codes = sys.argv[1:] or list(C.DEMO_FUNDS.keys())
    t0 = time.time()
    results, ics, abl, sens = {}, [], [], []

    pe = D.load_pe_panel(verbose=False)

    for code in codes:
        res = run_fund(code, C.DEMO_FUNDS.get(code, code), verbose=False)
        results[code] = res
        sig, nav = res["signal"], res["nav"]

        # E1
        t = ic_term_structure(sig, nav)
        t.insert(0, "基金", res["name"])
        ics.append(t)

        # E2
        a = channel_ablation(sig, nav, syn_nav=res["syn_nav"])
        a.insert(0, "基金", res["name"])
        abl.append(a)

        # E3
        s = param_sensitivity(nav.reindex(res["syn_nav"].index), res["syn_nav"],
                              res["weights_hist"], pe)
        s.insert(0, "基金", res["name"])
        sens.append(s)
        print(f"[exp] {code} {res['name']} 完成  ({time.time()-t0:.0f}s)")

    out = C.RES_DIR
    pd.concat(ics).to_csv(out / "E1_IC期限结构.csv", index=False, encoding="utf-8-sig")
    pd.concat(abl).to_csv(out / "E2_通道消融.csv", encoding="utf-8-sig")
    pd.concat(sens).to_csv(out / "E3_参数敏感性.csv", index=False, encoding="utf-8-sig")
    summarize_funds(results).to_csv(out / "E0_跨基金汇总.csv",
                                    index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("E1 · IC 期限结构（RankIC，正=信号方向正确）")
    print("=" * 96)
    e1 = pd.concat(ics)
    piv = e1.pivot_table(index=["基金", "信号"], columns="前瞻期",
                         values="RankIC")[["5日", "20日", "60日", "120日", "250日"]]
    print(piv.round(4).to_string())
    print(f"\n按信号汇总（{e1['基金'].nunique()} 只基金平均）：")
    print(e1.pivot_table(index="信号", columns="前瞻期", values="RankIC")
          [["5日", "20日", "60日", "120日", "250日"]].round(4).to_string())

    print("\n" + "=" * 96)
    print("E2 · 通道消融（超额 = 相对恒定仓位对照）")
    print("=" * 96)
    e2 = pd.concat(abl)
    print(e2[["基金", "年化收益", "夏普", "最大回撤", "超额夏普_vs对照",
              "超额年化_vs对照"]].round(4).to_string(index=False))

    print("\n" + "=" * 96)
    print("E3 · 参数敏感性（超额夏普 > 0 才算有效）")
    print("=" * 96)
    e3 = pd.concat(sens)
    print(f"共 {len(e3)} 组参数，超额夏普 > 0 的比例："
          f"{(e3['超额夏普'] > 0).mean():.1%}")
    print(f"超额夏普 均值 {e3['超额夏普'].mean():.4f}  中位数 {e3['超额夏普'].median():.4f}  "
          f"最大 {e3['超额夏普'].max():.4f}")
    print("\n按趋势权重分组：")
    print(e3.groupby("趋势权重")[["超额夏普", "RankIC"]].mean().round(4).to_string())

    print(f"\n总耗时 {time.time()-t0:.0f}s  结果已写入 {out}")


if __name__ == "__main__":
    main()
