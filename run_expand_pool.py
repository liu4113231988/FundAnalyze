# -*- coding: utf-8 -*-
"""
基金池扩充入口：按 `docs/扩池预注册规则.md` 冻结的规则筛选样本。

    python run_expand_pool.py                  # 默认预抽 1200 只取详情，每层 20 只
    python run_expand_pool.py --per-layer 25   # 每层 25 只（共约 75 只）
    python run_expand_pool.py --pre-sample 300 # 缩小预抽（调试，快但不全）

输出：
  output/data/fund_pool.csv      入选清单（其他脚本用 src.fundpool.load_pool() 读）
  output/results/P1_扩池_入选清单.csv
  output/results/P1_扩池_排除记录.csv   ★ 逐条留痕：代码 + 原因

**规则已在拉数据前冻结，本脚本只执行不改规则。**

两个做不到的规则项（免费数据源限制，已写进 src/fundpool.py 的 docstring，
也会写进最终报告）：
  1. 存续偏差无法消除 —— fund_name_em 是在场列表，已清盘/合并的不在其中；
  2. 规模口径降级 —— 只能取"当前规模"，规则要求的是"成立满 2 年时规模"。
"""
import argparse
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

import pandas as pd

from src import config as C
from src.fundpool import build_pool, load_pool

pd.set_option("display.width", 220)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-layer", type=int, default=20,
                    help="每个成立时间层纳入多少只（默认 20，三层共约 60 只）")
    ap.add_argument("--pre-sample", type=int, default=1200,
                    help="从粗筛选出的候选里预抽多少只取详情（默认 1200）")
    ap.add_argument("--seed", type=int, default=None,
                    help="抽样种子（默认用 fundpool.DEFAULT_SEED，保证可复现）")
    args = ap.parse_args()

    t0 = time.time()
    kw = {} if args.seed is None else {"seed": args.seed}
    pool, excl = build_pool(target_per_layer=args.per_layer,
                            pre_sample=args.pre_sample, **kw)

    if pool.empty:
        print("✗ 未选出任何基金")
        return 1

    print("\n" + "=" * 96)
    print(f"入选 {len(pool)} 只")
    print(pool[["基金代码", "基金简称", "成立时间", "当前规模_亿",
                "管理人", "层"]].to_string(index=False))
    print("\n各层只数：")
    print(pool["层"].value_counts().to_string())
    print(f"\n排除记录合计 {len(excl)} 条（已逐条留痕：代码 + 原因）")
    print("排除原因 Top 15：")
    print(excl["排除原因"].str.replace(r"（.*）", "", regex=True)
          .value_counts().head(15).to_string())

    # 落盘：入选清单同时放一份到 data 目录，供其他脚本复用
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(C.DATA_DIR / "fund_pool.csv", index=False, encoding="utf-8-sig")
    pool.to_csv(C.RES_DIR / "P1_扩池_入选清单.csv", index=False,
                encoding="utf-8-sig")
    excl.to_csv(C.RES_DIR / "P1_扩池_排除记录.csv", index=False,
                encoding="utf-8-sig")
    print(f"\n总耗时 {time.time() - t0:.0f}s")
    print(f"入选清单 → {C.DATA_DIR / 'fund_pool.csv'}")
    print(f"排除记录 → {C.RES_DIR / 'P1_扩池_排除记录.csv'}")
    print("\n下一步：")
    print("  python warm_cache.py $(python -c \"from src.fundpool import load_pool; "
          "print(' '.join(load_pool()['基金代码']))\")")
    print("  python run_holdings.py $(python -c \"from src.fundpool import load_pool; "
          "print(' '.join(load_pool()['基金代码']))\")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
