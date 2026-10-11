# -*- coding: utf-8 -*-
"""单只基金的端到端流水线：数据 → 风格拟合 → 信号 → 回测。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .backtest import run_backtest, score_to_position
from .rbsa import RBSA
from .signal import SignalEngine, build_decision, health_stats


def run_fund(code: str, name: str = "", refresh: bool = False,
             verbose: bool = True, start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> dict:
    name = name or C.DEMO_FUNDS.get(code, code)
    if verbose:
        print(f"\n{'='*78}\n>>> {code} {name}\n{'='*78}")

    panel = D.load_panel(code, refresh=refresh, verbose=False)
    nav = panel["__FUND__"]
    if verbose:
        print(f"[data] {panel.shape[0]} 交易日 × {panel.shape[1]-1} 条指数  "
              f"{panel.index[0].date()} ~ {panel.index[-1].date()}")

    # ---------- 1. 风格拟合
    rbsa = RBSA()
    fit = rbsa.fit_rolling(panel, verbose=verbose)

    # 合成基准净值：与基金在样本外起点对齐（便于价差比较）
    syn_ret = fit["syn_ret"]
    syn_nav = RBSA.synth_nav(syn_ret, start_value=1.0)
    nav_oos = nav.reindex(syn_nav.index)

    # ---------- 2. 信号
    pe_pct = D.load_pe_panel(verbose=False)
    engine = SignalEngine()
    sig = engine.compute(nav_oos, syn_nav, fit["weights"], pe_pct)

    # 拟合健康度通道 + 二值买卖决策
    health = engine.fit_health_channel(nav_oos, syn_ret, fit["diag"],
                                       fit["weights"])
    dec_pos, dec_trades = build_decision(health)

    # ---------- 3. 回测（含单通道归因 + 对照实验）
    vt = engine.vol_target_channel(syn_nav)
    variants = {
        "仅趋势通道": score_to_position(sig["trend_score"]),
        "仅相对偏离通道": score_to_position(sig["dev_score"]),
        "仅赛道估值通道": score_to_position(sig["val_score"]),
        "拟合健康决策": dec_pos,
        "波动率目标": vt["vol_target_pos"].reindex(nav_oos.index).ffill().fillna(1.0),
    }
    bt = run_backtest(nav_oos, sig, fund_name=name, variants=variants,
                      start_date=start_date, end_date=end_date,
                      decision_trades=dec_trades)

    # ---------- 4. 打包
    latest_w = fit["weights"].iloc[-1]
    latest_w = latest_w[latest_w > 0].sort_values(ascending=False)
    exposure = RBSA.latest_exposure(latest_w)
    snap = SignalEngine.snapshot(sig, name)

    if verbose:
        print(f"\n[结果] 最新风格暴露：")
        print(exposure.to_string(index=False))
        print(f"\n[结果] 当前决策：{json.dumps(snap, ensure_ascii=False, indent=2)}")
        print(f"\n[结果] 绩效：")
        print(bt["metrics"].to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"       RankIC={bt['ic']:.4f}  ICIR={bt['icir']:.3f}  "
              f"多头胜率={bt['win_long']:.1%}  年化换手={bt['turnover']:.1f}x  "
              f"平均仓位={bt['avg_position']:.0%}")
        if len(bt["groups"]):
            print("\n       分组前瞻收益：")
            print(bt["groups"].to_string(float_format=lambda x: f"{x:.4%}"))
        # 各方案 vs 它自己的恒定仓位对照（不再共用主策略那一份）
        print("\n       逐方案择时净贡献（各自对照，仓位已中性化）：")
        for v, tv in bt["timing_contrib_all"].items():
            if not tv:
                continue
            print(f"         {v:14s} 对照={bt['controls'].get(v)}  "
                  f"超额夏普={tv['夏普']:+.3f}  超额年化={tv['年化收益']:+.2%}  "
                  f"平均仓位={bt['avg_position_all'].get(v, float('nan')):.0%}")

        if bt["trades"]:
            print(f"\n       健康门控交易（{len(bt['trades'])} 笔，次日生效）：")
            for t in bt["trades"]:
                print(f"         {t['exec_date'].date()}  {t['action']}  "
                      f"health={t['health']:+.2f}  {t['reason']}")
        else:
            # 零交易时必须明确告警：该方案已退化为买入持有，
            # 它与对照的差额只是仓位差，不是择时能力。
            st = health_stats(health)
            print(f"\n       [警告] 健康门控全程未触发（{len(bt['trades'])} 笔交易）"
                  f"→ 「拟合健康决策」≡ 买入持有。")
            print(f"               健康分 最低={st['min']:+.3f} "
                  f"5%分位={st['p05']:+.3f} 中位={st['median']:+.3f} "
                  f"离场线={st['exit']:+.2f}  距离场线还差 {st['gap_to_exit']:+.3f}")
            print("               该方案的绩效差异仅来自仓位水平，"
                  "不得作为择时能力的证据。")

    return {
        "code": code, "name": name,
        "panel": panel, "nav": nav,
        "weights_hist": fit["weights"], "fit_diag": fit["diag"],
        "syn_nav": syn_nav, "syn_ret": syn_ret,
        "signal": sig, "health": health,
        # 健康门控的仓位列：下游要做成本敏感性（E10）等需要原始仓位序列，
        # 而 backtest 只返回收益率序列，拿不到仓位。
        "decision_pos": dec_pos, "decision_trades": dec_trades,
        "backtest": bt,
        "exposure": exposure, "snapshot": snap,
    }


def save_result(res: dict, outdir: Optional[Path] = None) -> Path:
    outdir = Path(outdir or C.RES_DIR / f"{res['code']}_{res['name']}")
    outdir.mkdir(parents=True, exist_ok=True)
    res["weights_hist"].to_csv(outdir / "style_weights.csv", encoding="utf-8-sig")
    res["signal"].to_csv(outdir / "signals.csv", encoding="utf-8-sig")
    res["backtest"]["equity"].to_csv(outdir / "equity.csv", encoding="utf-8-sig")
    res["backtest"]["metrics"].to_csv(outdir / "metrics.csv", encoding="utf-8-sig")
    if len(res["backtest"]["groups"]):
        res["backtest"]["groups"].to_csv(outdir / "ic_groups.csv", encoding="utf-8-sig")
    pd.DataFrame([res["snapshot"]]).to_csv(outdir / "decision.csv",
                                           index=False, encoding="utf-8-sig")
    meta = {
        "code": res["code"], "name": res["name"],
        "日期区间": [str(res["nav"].index[0].date()), str(res["nav"].index[-1].date())],
        "样本外天数": int(res["backtest"]["metrics"]["样本天数"].iloc[0]),
        "RankIC": None if pd.isna(res["backtest"]["ic"]) else round(res["backtest"]["ic"], 4),
        "ICIR": None if pd.isna(res["backtest"]["icir"]) else round(res["backtest"]["icir"], 3),
        "多头胜率": round(res["backtest"]["win_long"], 4),
        "年化换手": round(res["backtest"]["turnover"], 2),
        "平均仓位": round(res["backtest"]["avg_position"], 4),
        "健康门控交易数": len(res["backtest"]["trades"]),
        "健康分最低": round(health_stats(res["health"])["min"], 4),
        "离场线": C.SIGNAL["health_exit"],
    }
    (outdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return outdir
