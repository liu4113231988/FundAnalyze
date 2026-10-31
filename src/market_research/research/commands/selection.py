# -*- coding: utf-8 -*-
"""
P1-03 · 选择过程感知的显著性检验。

    python run.py selection            # 读 E9b_候选配置明细.csv

回答：E9b 的"过拟合代价 0.769"里，多少是**真实过拟合**，多少只是
"从 12 个候选里挑最大值"的**抽签运气**。

旧 bootstrap 只检验"已选策略的均值是否不为 0"，把选择过程留在零假设之外；
本脚本把选择过程放进零假设里。
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from market_research.config import ROOT
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from market_research.research.selection import rank_percentile, selection_aware_bootstrap, decompose_overfit_cost, selection_frequency

RES = ROOT / "output" / "results"


def main(argv=None, *, prog=None):
    import argparse
    argparse.ArgumentParser(description=__doc__, prog=prog).parse_args(argv)
    fp = RES / "E9b_候选配置明细.csv"
    if not fp.exists():
        print(f"缺 {fp} —— 先跑 python run.py walkforward")
        return 1
    cand = pd.read_csv(fp)
    if cand.empty or "被训练期选中" not in cand.columns:
        print("候选明细为空或列不全，先重跑 python run.py walkforward")
        return 1
    fbf = RES / "E9b_候选逐基金IR.csv"
    by_fund = pd.read_csv(fbf) if fbf.exists() else pd.DataFrame()

    # ---- 指标 1：选中配置的测试期排名
    rk = rank_percentile(cand)
    print("=" * 100)
    print("[1] 被训练期选中的配置，其测试期 IR 在候选中的排名百分位")
    print("=" * 100)
    print(rk.to_string(index=False))
    if not rk.empty:
        mp = float(rk["排名百分位%"].mean())
        print(f"\n  平均排名百分位 {mp:.1f}%"
              + ("  → 接近 50%，**选参没有技能**（与随机挑一个差不多）"
                 if mp < 65 else
                 "  → 明显高于 50%，选参有一定技能"))

    # ---- 指标 2：选择感知 bootstrap（必须在基金维度重采样）
    print("\n" + "=" * 100)
    print("[2] 选择感知 bootstrap：把'从候选里选最优'放进零假设")
    print("=" * 100)
    if by_fund.empty:
        print("  跳过：缺 E9b_候选逐基金IR.csv，重跑 python run.py walkforward")
        sa = {"ok": False}
    else:
        sa = selection_aware_bootstrap(by_fund, block=60, n_boot=2000)
    if sa.get("ok"):
        print(sa["per_fold"].to_string(index=False))
        s = sa["summary"]
        print("\n[汇总]")
        for k, v in s.items():
            print(f"  {k:24s}: {v}")
        if s["p值(上尾)均值"] > 0.05:
            print("  ★ 真实迁移 IR 没有超过'纯选择运气'的分布 —— "
                  "没有证据支持这条线索")
    else:
        print(f"  跳过：{sa.get('reason', '无可用数据')}")
    # 被跳过的折必须可见：静默跳过会让"只算了 2/3 折"看起来像"算了全部"
    for msg in sa.get("skipped") or []:
        print(f"  ⚠ {msg}")

    # ---- 拆解过拟合代价
    dec = decompose_overfit_cost(cand)
    print("\n" + "=" * 100)
    print("[3] 过拟合代价拆解：总代价 = 运气部分 + 过拟合部分")
    print("=" * 100)
    print(dec.round(4).to_string(index=False))
    if not dec.empty:
        tot = float(dec["总代价(oracle−迁移)"].mean())
        luck = float(dec["运气部分(oracle−均值)"].mean())
        over = float(dec["过拟合部分(均值−迁移)"].mean())
        print(f"\n  总代价均值     {tot:+.4f}")
        print(f"    运气部分     {luck:+.4f}  （挑最大值本身就会偏高，**不是**过拟合）")
        print(f"    过拟合部分   {over:+.4f}  （训练期选择相对平均水平的真实损失）")
        if abs(tot) > 1e-9:
            print(f"  → 过拟合占总代价 {over / tot:.0%}"
                  f"，运气占 {luck / tot:.0%}")

    # ---- 选参一致性
    sf = selection_frequency(cand)
    print("\n" + "=" * 100)
    print("[4] 选参一致性")
    print("=" * 100)
    print(sf.to_string(index=False))
    n_same = int(sf["是否同一配置"].sum()) if not sf.empty else 0
    print(f"\n  训练期与测试期(oracle)选中同一配置的折数：{n_same}/{len(sf)}")

    if not rk.empty:
        rk.to_csv(RES / "P1-03_选中配置排名.csv", index=False,
                  encoding="utf-8-sig")
    if sa.get("ok"):
        sa["per_fold"].to_csv(RES / "P1-03_选择感知bootstrap.csv",
                              index=False, encoding="utf-8-sig")
    if not dec.empty:
        dec.to_csv(RES / "P1-03_过拟合代价拆解.csv", index=False,
                   encoding="utf-8-sig")
    print(f"\n结果写入 {RES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
