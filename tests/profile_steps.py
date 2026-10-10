# -*- coding: utf-8 -*-
"""分步计时，定位性能瓶颈。"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd, numpy as np
from src import config as C, data as D
from src.rbsa import RBSA
from src.signal import SignalEngine
from src.backtest import run_backtest, _fwd_ret

t = {}
def tick(k, t0):
    t[k] = time.time() - t0
    print(f"  [{t[k]:7.2f}s] {k}")

t0 = time.time(); panel = D.load_panel("110022", verbose=False); tick("load_panel", t0)
t0 = time.time(); fit = RBSA().fit_rolling(panel, verbose=False); tick("rbsa", t0)
syn_nav = RBSA.synth_nav(fit["syn_ret"], 1.0)
nav = panel["__FUND__"].reindex(syn_nav.index)

t0 = time.time(); pe = D.load_pe_panel(verbose=False); tick("load_pe_panel", t0)
print("   pe shape:", pe.shape, "| cpu时间已排除网络" if False else "")

eng = SignalEngine()
t0 = time.time(); tc = eng.trend_channel(syn_nav); tick("trend_channel", t0)
t0 = time.time(); dc = eng.deviation_channel(nav, syn_nav); tick("deviation_channel", t0)
t0 = time.time(); vc = eng.valuation_channel(fit["weights"], pe, syn_nav.index); tick("valuation_channel", t0)
t0 = time.time(); sig = eng.compute(nav, syn_nav, fit["weights"], pe); tick("compute(total)", t0)

t0 = time.time(); bt = run_backtest(nav, sig, fund_name="t"); tick("run_backtest", t0)

print("\n总耗时:", round(sum(t.values()), 1), "s")
