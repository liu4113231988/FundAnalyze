# -*- coding: utf-8 -*-
"""逐模型计时，定位滚动评估的性能瓶颈。"""
import sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import market_research.config as C
import market_research.data.marketdata as D
from market_research.modeling.models import ModelZoo, MODEL_ORDER, fit_window

code = sys.argv[1] if len(sys.argv) > 1 else "110022"
panel = D.load_panel(code, verbose=False)
R = np.log(panel.astype(float)).diff().dropna()
assets = [c for c in panel.columns if c != "__FUND__"]
X_all = R[assets].values
y_all = R["__FUND__"].values
rebal = list(range(C.RBSA["window"], len(R), C.RBSA["refit"]))
print(f"[bench] {code} 滚动次数={len(rebal)} 窗口={C.RBSA['window']} 重估={C.RBSA['refit']}\n")

print(f"{'模型':<14}{'全滚动耗时':>12}{'单次拟合':>10}{'OOS R²':>10}")
print("-" * 50)
rows = []
for k in MODEL_ORDER:
    t0 = time.time()
    try:
        rep = ModelZoo()._run_one(k, X_all, y_all, assets, R.index, rebal, None)
        total = time.time() - t0
        # 单次
        t1 = time.time()
        fit_window(k, X_all[-C.RBSA["window"]:], y_all[-C.RBSA["window"]:], assets)
        one = time.time() - t1
        rows.append((k, total, one, rep.oos_r2))
        print(f"{k:<14}{total:>11.1f}s{one:>9.3f}s{rep.oos_r2:>10.4f}")
    except Exception as e:
        print(f"{k:<14}{'ERR':>12}  {type(e).__name__}: {str(e)[:50]}")

print("-" * 50)
tot = sum(r[1] for r in rows)
print(f"串行总计 {tot:.1f}s")
print("\n按耗时排序：")
for k, t, one, r2 in sorted(rows, key=lambda x: -x[1]):
    print(f"  {k:<14}{t:>8.1f}s  占比 {t/tot:>5.1%}  单次 {one:.3f}s")
