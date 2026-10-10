# -*- coding: utf-8 -*-
"""
主入口。

    python run.py                 # 跑全部演示基金
    python run.py 110022          # 只跑一只
    python run.py --refresh       # 强制重新抓取数据
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src import config as C
from src.pipeline import run_fund, save_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", help="基金代码，留空则跑全部演示基金")
    ap.add_argument("--refresh", action="store_true", help="强制刷新数据缓存")
    ap.add_argument("--start", default=None, help="回测起始日 YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="回测结束日 YYYY-MM-DD")
    args = ap.parse_args()

    codes = args.codes or list(C.DEMO_FUNDS.keys())
    t0 = time.time()
    results, rows = [], []

    for code in codes:
        try:
            res = run_fund(code, C.DEMO_FUNDS.get(code, code),
                           refresh=args.refresh, verbose=True,
                           start_date=args.start, end_date=args.end)
            outdir = save_result(res)
            results.append(res)
            m = res["backtest"]["metrics"]
            ctrl = res["backtest"]["control_name"]
            tc = res["backtest"]["timing_contrib"]
            rows.append({
                "代码": code, "名称": res["name"],
                "样本外天数": int(m["样本天数"].iloc[0]),
                "策略年化": m.loc["择时策略(融合)", "年化收益"],
                "持有年化": m.loc["买入持有(100%)", "年化收益"],
                "对照年化": m.loc[ctrl, "年化收益"] if ctrl in m.index else None,
                "策略夏普": m.loc["择时策略(融合)", "夏普"],
                "持有夏普": m.loc["买入持有(100%)", "夏普"],
                "对照夏普": m.loc[ctrl, "夏普"] if ctrl in m.index else None,
                "策略回撤": m.loc["择时策略(融合)", "最大回撤"],
                "持有回撤": m.loc["买入持有(100%)", "最大回撤"],
                "择时净贡献_夏普": tc.get("夏普"),
                "择时净贡献_年化": tc.get("年化收益"),
                # 健康决策 vs 它自己的对照（平均仓位通常 100%，对照即买入持有）
                "健康决策_净贡献_夏普": (res["backtest"]["timing_contrib_all"]
                                   .get("拟合健康决策", {}).get("夏普")),
                "健康决策_交易数": len(res["backtest"]["trades"]),
                "RankIC": res["backtest"]["ic"], "ICIR": res["backtest"]["icir"],
                "多头胜率": res["backtest"]["win_long"],
                "年化换手": res["backtest"]["turnover"],
                "平均仓位": res["backtest"]["avg_position"],
                "输出目录": str(outdir.name),
            })
            print(f"[save] {outdir}")
        except Exception as e:      # noqa: BLE001
            import traceback
            print(f"[ERROR] {code} 失败: {type(e).__name__}: {e}")
            traceback.print_exc()

    if rows:
        summary = pd.DataFrame(rows)
        fp = C.RES_DIR / "summary.csv"
        summary.to_csv(fp, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 100)
        print("汇总")
        print("=" * 100)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n汇总已保存: {fp}")

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
