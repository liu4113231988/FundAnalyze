# -*- coding: utf-8 -*-
"""
扩展样本验证：14 只不同风格 A 股主动权益基金，检验「周期匹配」结论的稳健性。

输出
----
  output/results/X1_扩展_逐基金.csv   每只基金的最优周期配置与净贡献
  output/results/X2_扩展_周期矩阵.csv 信号窗口 × 再平衡周期 的超额夏普矩阵（跨基金平均）
  output/results/X3_扩展_IC期限结构.csv
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
from src.pipeline import run_fund, save_result
from src.experiments import ic_term_structure, lowfreq_grid

pd.set_option("display.width", 220)


def main():
    codes = sys.argv[1:] or list(C.EXTENDED_FUNDS.keys())
    t0 = time.time()
    pe = D.load_pe_panel(verbose=False)
    per_fund, matrices, ics = [], [], []

    for code in codes:
        name = C.EXTENDED_FUNDS.get(code, code)
        try:
            res = run_fund(code, name, verbose=False)
            nav = res["nav"].reindex(res["syn_nav"].index)

            # 默认参数下的结论
            m = res["backtest"]["metrics"]
            ctrl = res["backtest"]["control_name"]
            per_fund.append({
                "代码": code, "名称": name,
                "样本天数": int(m.loc["择时策略(融合)", "样本天数"]),
                "默认_策略夏普": m.loc["择时策略(融合)", "夏普"],
                "默认_对照夏普": m.loc[ctrl, "夏普"],
                "默认_超额夏普": res["backtest"]["timing_contrib"]["夏普"],
                "默认_RankIC": res["backtest"]["ic"],
                "默认_平均仓位": res["backtest"]["avg_position"],
            })

            # E1
            t = ic_term_structure(res["signal"], nav)
            t.insert(0, "基金", name)
            ics.append(t)

            # E4 周期矩阵
            g = lowfreq_grid(nav, res["syn_nav"], res["weights_hist"], pe)
            g.insert(0, "基金", name)
            matrices.append(g)

            best = g.loc[g["超额夏普"].idxmax()]
            per_fund[-1].update({
                "最优窗口": int(best["信号窗口"]),
                "最优周期": int(best["再平衡周期"]),
                "最优_超额夏普": best["超额夏普"],
                "最优_策略夏普": best["策略夏普"],
                "最优_对照夏普": best["对照夏普"],
                "最优_策略年化": best["策略年化"],
                "最优_对照年化": best["对照年化"],
            })
            print(f"[{time.time()-t0:6.0f}s] {code} {name:16s} "
                  f"默认超额夏普={per_fund[-1]['默认_超额夏普']:+.3f}  "
                  f"最优={best['信号窗口']}日/{best['再平衡周期']}日 "
                  f"超额夏普={best['超额夏普']:+.3f}")
        except Exception as e:      # noqa: BLE001
            print(f"[SKIP] {code} {name} 失败: {type(e).__name__}: {str(e)[:60]}")

    out = C.RES_DIR
    pf = pd.DataFrame(per_fund)
    pf.to_csv(out / "X1_扩展_逐基金.csv", index=False, encoding="utf-8-sig")
    mat = pd.concat(matrices)
    mat.to_csv(out / "X2_扩展_周期矩阵.csv", index=False, encoding="utf-8-sig")
    e1 = pd.concat(ics)
    e1.to_csv(out / "X3_扩展_IC期限结构.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 108)
    print(f"扩展样本 {len(pf)} 只基金 · 逐基金结果")
    print("=" * 108)
    print(pf.round(4).to_string(index=False))

    # 样本量不足的基金（样本外 < 500 天）不参与汇总统计，避免噪声主导结论
    MIN_DAYS = 500
    short = pf[pf["样本天数"] < MIN_DAYS]
    valid = pf[pf["样本天数"] >= MIN_DAYS]
    if len(short):
        print(f"\n[note] 样本外不足 {MIN_DAYS} 天的基金不参与汇总统计："
              + "、".join(f"{r.名称}({r.样本天数}天)" for r in short.itertuples()))

    print("\n" + "=" * 108)
    print(f"默认参数（趋势60/偏离252/估值）下的择时净贡献（相对恒定仓位对照）"
          f"—— 基于 {len(valid)} 只有效样本")
    print("=" * 108)
    n_pos = int((valid["默认_超额夏普"] > 0).sum())
    print(f"超额夏普 > 0 的基金数：{n_pos}/{len(valid)}  ({n_pos/len(valid):.0%})")
    print(f"超额夏普 均值 {valid['默认_超额夏普'].mean():+.4f}  "
          f"中位数 {valid['默认_超额夏普'].median():+.4f}")
    print(f"RankIC 均值 {valid['默认_RankIC'].mean():+.4f}  "
          f"为正的比例 {(valid['默认_RankIC']>0).mean():.0%}")

    # 周期矩阵也只在有效样本上统计
    valid_names = set(valid["名称"])
    mat_v = mat[mat["基金"].isin(valid_names)]
    print("\n" + "=" * 108)
    print(f"周期矩阵：跨基金平均的超额夏普（{len(valid_names)} 只有效样本，"
          f"行=信号窗口，列=再平衡周期）")
    print("=" * 108)
    piv = mat_v.pivot_table(index="信号窗口", columns="再平衡周期",
                            values="超额夏普", aggfunc="mean")
    print(piv.round(3).to_string())
    print("\n各格子上超额夏普 > 0 的基金占比：")
    piv2 = mat_v.pivot_table(index="信号窗口", columns="再平衡周期", values="超额夏普",
                             aggfunc=lambda s: (s > 0).mean())
    print((piv2 * 100).round(0).to_string())
    print("\n按信号窗口汇总（跨再平衡周期平均）：")
    print(mat_v.groupby("信号窗口")[["超额夏普"]].mean().round(4).to_string())

    print("\n" + "=" * 108)
    print("IC 期限结构（跨基金平均 RankIC）")
    print("=" * 108)
    print(e1.pivot_table(index="信号", columns="前瞻期", values="RankIC")
          [["5日", "20日", "60日", "120日", "250日"]].round(4).to_string())

    print(f"\n总耗时 {time.time()-t0:.0f}s  结果写入 {out}")


if __name__ == "__main__":
    main()
