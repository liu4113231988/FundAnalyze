# -*- coding: utf-8 -*-
"""
数据预热：把基金净值抓到本地缓存，不做任何分析。

    python warm_cache.py                # 预热默认扩展样本（EXTENDED_FUNDS）
    python warm_cache.py 110022 000001  # 预热指定基金
    python warm_cache.py --refresh      # 强制重抓

用途：
    1. 扩池前批量下载数据（网络波动时可以反复跑到全部成功为止）
    2. 离线分析前的准备 —— 缓存齐全后 run.py / run_extended.py / webapp 全部离线可用

存储结构（src/data.py 的分层缓存）：
    output/data/index_universe_{start}.csv   指数池，全局一份，与基金无关
    output/data/fund_nav_{code}_{start}.csv  基金净值，每只一份（~50 KB）
    面板 = 两者对齐后即时重建，不落盘。
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

from src import config as C, data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", help="基金代码，留空则用 EXTENDED_FUNDS")
    ap.add_argument("--refresh", action="store_true", help="强制重抓")
    # ★ 默认必须走 fetch_start_for()，不能用 C.START_DATE（滚动近 5 年）。
    #   缓存名里带 start：若预热用滚动日期，而 load_panel 读取时用的是
    #   fetch_start_for() 返回的固定起点（2016-01-01），两边键永远对不上 ——
    #   预热等于白做，而且每天跑一次就多一份缓存。
    ap.add_argument("--start", default=None,
                    help="抓取起点，默认 C.FETCH_START_DATE（分析窗口在分析时裁剪）")
    args = ap.parse_args()

    start = D.fetch_start_for(args.start)
    if args.start and args.start != start:
        print(f"  注：--start {args.start} 早于固定抓取起点，按 {start} 抓取")

    codes = args.codes or list(C.EXTENDED_FUNDS.keys())
    print(f"预热 {len(codes)} 只基金的数据缓存（抓取起点 {start}）")

    # 指数池全局一份 —— 只需一次
    t0 = time.time()
    px = D.load_index_universe(start, refresh=args.refresh, verbose=False)
    print(f"[指数池] {px.shape[0]} 交易日 × {px.shape[1]} 指数  "
          f"({px.index[0].date()} ~ {px.index[-1].date()})  {time.time()-t0:.1f}s")

    ok, fail = [], []
    for i, c in enumerate(codes, 1):
        name = (C.EXTENDED_FUNDS.get(c) or C.DEMO_FUNDS.get(c) or c)
        try:
            t = time.time()
            nav = D.load_fund_nav_cached(c, start, refresh=args.refresh)
            ok.append(c)
            print(f"  [{i}/{len(codes)}] {c} {name:16s} {len(nav)} 行  "
                  f"{nav.index[0].date()} ~ {nav.index[-1].date()}  "
                  f"{time.time()-t:.1f}s")
        except Exception as e:                                # noqa: BLE001
            fail.append((c, f"{type(e).__name__}: {str(e)[:60]}"))
            print(f"  [{i}/{len(codes)}] {c} {name:16s} 失败 {type(e).__name__}")

    print(f"\n完成：成功 {len(ok)}，失败 {len(fail)}")
    if fail:
        for c, e in fail:
            print(f"  {c}: {e}")
        print("可重跑本命令补齐（已成功的会命中缓存，不会重复请求）")
        # ★ P1-09：必需步骤失败必须有非零退出码。旧实现一律 return 0，
        #   于是 CI 里"只预热了一半"和"全部成功"完全无法区分。
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
