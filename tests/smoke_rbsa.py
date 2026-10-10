# -*- coding: utf-8 -*-
"""冒烟测试：跑通 数据 → RBSA 全链路，检查输出合理性。"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config as C
from src import data as D
from src.rbsa import RBSA

CODE = sys.argv[1] if len(sys.argv) > 1 else "110022"
t0 = time.time()
panel = D.load_panel(CODE, verbose=True)
print(f"\n面板：{panel.shape}")
print("列：", list(panel.columns)[:8], "...")

res = RBSA().fit_rolling(panel, verbose=True)
print("\n" + "=" * 78)
print("最新一期风格暴露：")
print(RBSA.latest_exposure(res["weights"].iloc[-1]).to_string(index=False))
print("\n诊断统计：")
print(res["diag"].describe().loc[["mean", "50%", "min", "max"]].to_string())
print(f"\n合成基准样本外区间：{res['syn_ret'].index[0].date()} ~ {res['syn_ret'].index[-1].date()}"
      f"  共 {len(res['syn_ret'])} 天")
print(f"耗时 {time.time()-t0:.1f}s")
