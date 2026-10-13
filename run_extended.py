# -*- coding: utf-8 -*-
"""
扩展样本验证：14 只不同风格 A 股主动权益基金，检验「周期匹配」结论的稳健性。

输出
----
  output/results/X1_扩展_逐基金.csv   每只基金的最优周期配置与净贡献
  output/results/X2_扩展_周期矩阵.csv 信号窗口 × 再平衡周期 的超额夏普矩阵（跨基金平均）
  output/results/X3_扩展_IC期限结构.csv

    python run_extended.py                  # 全部 EXTENDED_FUNDS，默认近 5 年
    python run_extended.py 110022 000001    # 指定基金
    python run_extended.py --window-years 2 # 改用近 2 年窗口（下限 2 年）
"""
import argparse
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
from src.window import add_window_args, apply_window
from src.experiments import (ic_term_structure, lowfreq_grid,
                             cost_sensitivity, cost_durability,
                             naive_benchmarks)

pd.set_option("display.width", 220)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_window_args(ap)
    ap.add_argument("codes", nargs="*",
                    help="基金代码，留空则用 EXTENDED_FUNDS")
    args = ap.parse_args()
    apply_window(args)
    codes = args.codes or list(C.EXTENDED_FUNDS.keys())
    t0 = time.time()
    pe = D.load_pe_panel(verbose=False)
    per_fund, matrices, ics, costs, nbs = [], [], [], [], []

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

            # P2-02 朴素基准：风格画像必须证明自己**胜过**"只用沪深300"
            # / "所有指数等权"，而不只是"窗口内 R² 看起来挺高"
            nb = naive_benchmarks(D.load_panel(code, verbose=False))
            if not nb.empty:
                nb.insert(0, "基金", name)
                nbs.append(nb)

            # E10 成本敏感性：日频择时（高换手）与健康门控（低换手）对比
            cs = cost_sensitivity(
                nav, res["signal"],
                variants={"拟合健康决策": res["decision_pos"]})
            cs.insert(0, "基金", name)
            costs.append(cs)
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
    cdf = pd.concat(costs, ignore_index=True) if costs else pd.DataFrame()
    if not cdf.empty:
        cdf.to_csv(out / "E10_成本敏感性.csv", index=False, encoding="utf-8-sig")
    nbdf = pd.concat(nbs, ignore_index=True) if nbs else pd.DataFrame()
    if not nbdf.empty:
        nbdf.to_csv(out / "X4_扩展_朴素基准.csv", index=False,
                    encoding="utf-8-sig")

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

    # ---------------- P2-02 朴素基准
    if not nbdf.empty:
        print("\n" + "=" * 108)
        print("P2-02 · 朴素基准对照：风格画像必须胜过简单基准")
        print("  R² 保留负值（P0-05）：负的样本外 R² = 连'常数基准'都不如")
        print("=" * 108)
        best_single = (nbdf[nbdf["基准"].str.startswith("单一")]
                       .groupby("基金")["样本外R²"].max().rename("最优单一宽基"))
        constrained = (nbdf[nbdf["基准"].str.startswith("受约束")]
                       .groupby("基金")["样本外R²"].max().rename("受约束回归"))
        eq_w = (nbdf[nbdf["基准"].str.startswith("等权")]
                .groupby("基金")["样本外R²"].max().rename("等权混合"))
        cmp_ = pd.concat([best_single, eq_w, constrained], axis=1)
        cmp_["受约束−最优单一"] = cmp_["受约束回归"] - cmp_["最优单一宽基"]
        print(cmp_.round(4).to_string())
        print(f"\n  跨基金均值：最优单一宽基 {best_single.mean():+.4f}，"
              f"等权混合 {eq_w.mean():+.4f}，"
              f"受约束回归 {constrained.mean():+.4f}")
        win = int((cmp_["受约束−最优单一"] > 0).sum())
        print(f"  受约束回归胜过最优单一宽基的基金数：{win}/{len(cmp_)}")
        if win <= len(cmp_) / 2:
            print("  ★ 警告：风格画像在过半基金上并没有超过'只用单一宽基' ——"
                  "此时 R² 高低主要来自指数数量而不是解释力（P2-02 的核心判据）")

    # ---------------- E10 成本敏感性
    if not cdf.empty:
        cv = cdf[cdf["基金"].isin(valid_names)]
        print("\n" + "=" * 108)
        print("E10 · 成本敏感性：同一批方案在 3 档交易成本下的净贡献")
        print("  本项目默认单边 0.15% 对应的是**场内 ETF**；场外主动基金的真实申赎")
        print("  是申购 1.5% / 赎回 0.5%，差 10 倍以上。换手越高，越吃这一档差。")
        print("=" * 108)
        agg = cv.groupby(["方案", "成本档位"], as_index=False)[
            ["年化换手", "超额夏普", "超额年化"]].mean()
        agg["超额年化"] = agg["超额年化"]
        order = {"场内ETF": 0, "场外C类": 1, "场外A类": 2}
        agg = agg.sort_values(["方案", "成本档位"],
                              key=lambda s: s.map(order) if s.name == "成本档位" else s)
        show = agg.copy()
        show["年化换手"] = show["年化换手"].round(2)
        show["超额夏普"] = show["超额夏普"].round(3)
        show["超额年化"] = (show["超额年化"] * 100).round(2).astype(str) + "%"
        print(show.to_string(index=False))

        print("\n各方案的成本耐久性（判据：超额年化 > 0）：")
        dur = []
        for name_ in cv["基金"].unique():
            d = cost_durability(cv[cv["基金"] == name_])
            if not d.empty:
                d.insert(0, "基金", name_)
                dur.append(d)
        if dur:
            dd = pd.concat(dur, ignore_index=True)
            # ★ 三档数值是**跨基金均值**，而「最高可承受档位」是逐基金判定的
            #   **众数** —— 两者口径不同，均值会被少数极端基金拉高，
            #   于是会出现"三档均为正、却判为无正贡献"的表观矛盾。
            #   修正办法不是改判据（判据本身正确），而是把两个口径都摆出来：
            #   补一列"各档为正的基金数"，让均值被谁抬高一目了然。
            def _pos(s):
                return f"{int((s > 0).sum())}/{len(s)}"

            print(dd.groupby("方案").agg(
                场内ETF=("场内ETF", "mean"),
                场外C类=("场外C类", "mean"),
                场外A类=("场外A类", "mean"),
                场内ETF为正=("场内ETF", _pos),
                场外C类为正=("场外C类", _pos),
                场外A类为正=("场外A类", _pos),
                最高可承受档位=("最高可承受档位",
                              lambda s: s.value_counts().index[0]),
            ).round(4).to_string())
            print("\n  读法：")
            print("   · 三档数值 = 跨基金**平均**超额年化，为正 = 扣掉该档成本后仍创造价值")
            print("   · 「为正」列 = 该档下超额为正的**基金数 / 总数**（均值会被少数基金拉高，")
            print("     判断策略是否普适要看这一列，不能只看均值）")
            print("   · 「最高可承受档位」= 逐基金判定的**众数**，口径是逐基金而非平均，")
            print("     与均值列不一致属正常 —— 它直接决定该策略只能用场内 ETF 还是也能用场外基金")

    print(f"\n总耗时 {time.time()-t0:.0f}s  结果写入 {out}")


if __name__ == "__main__":
    main()
