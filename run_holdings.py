# -*- coding: utf-8 -*-
"""
真实持仓 vs RBSA 拟合持仓的对齐入口。

    python run_holdings.py                    # 全部 EXTENDED_FUNDS
    python run_holdings.py 110022 000001      # 指定基金
    python run_holdings.py --refresh          # 强制重抓持仓与行业映射
    python run_holdings.py --window-years 2   # 近 2 年窗口（下限 2 年）

做两件事（对应 TODO 的"验证"与"替代"）：

  1. **验证**：逐行业对比 RBSA 拟合权重与半年报/年报的真实持仓暴露，
     回答"RBSA 说的 37% 食品饮料，实际是不是 37%"。
  2. **替代**：只用真实持仓构造基准组合（按披露滞后起算），
     与 RBSA 合成基准比收益，回答"如果直接拿真实持仓当基准，还要 ML 干什么"。

输出（output/results/）：
  H0_持仓对齐汇总.csv        跨基金汇总（每只基金的核心指标）
  H1_持仓对齐_逐期_{代码}.csv 逐期的偏差明细
  H2_持仓对齐_按行业_{代码}.csv 按行业的平均偏差排序
  H3_持仓基准对比_{代码}.csv   基金 / RBSA / 持仓 三条基准的统计

★ 关键限制（每条都会打印，不能隐瞒）：
  · 季报（Q1/Q3）只披露**前十大重仓**，本模块**只用半年报/年报**（完整持仓）；
  · 持仓有**披露滞后**（Q2 约 7 月、Q4 约次年 1 月），持仓基准从披露后起算，
    否则是用未来信息构造历史基准；
  · RBSA 的宽基暴露无法与持仓对应，对齐只覆盖行业部分。
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

import numpy as np
import pandas as pd

from src import config as C, data as D
from src.rbsa import RBSA
from src import holdings as H
from src.window import add_window_args, apply_window

pd.set_option("display.width", 220)


def _corr_series(al: dict) -> pd.Series:
    """逐期权重相关的非空序列（末期可能因共同行业不足而为 NaN）。"""
    return al["periods"]["权重相关"].dropna()


def run_one(code: str, industry_map: pd.Series, refresh: bool = False) -> dict:
    """单只基金：数据 → RBSA → 对齐 → 持仓基准。"""
    t0 = time.time()
    panel = D.load_panel(code, verbose=False)
    hd = H.load_fund_holdings(code, refresh=refresh, verbose=False)
    if hd.empty:
        return {"code": code, "ok": False, "reason": "未取到持仓明细"}
    exp = H.semiannual_exposure(hd, industry_map)
    if exp.empty:
        return {"code": code, "ok": False, "reason": "无完整（半年报/年报）持仓"}

    fit = RBSA().fit_rolling(panel, verbose=False)
    al = H.align_vs_rbsa(fit["weights"], exp)
    if not al.get("ok"):
        return {"code": code, "ok": False, "reason": al.get("reason", "对齐失败")}

    hr = H.holding_benchmark(exp, panel)
    syn_nav = np.exp(fit["syn_ret"].cumsum())
    cmp_ = H.compare_benchmarks(panel["__FUND__"], syn_nav, hr)

    out = {
        "code": code,
        "ok": True,
        "name": C.EXTENDED_FUNDS.get(code, C.DEMO_FUNDS.get(code, code)),
        "n_periods": al["n_periods"],
        "mad_mean": al["mad_mean"],
        "corr_mean": al["corr_mean"],
        # ★ 取最后一个**有效**相关，不是最后一期：末期共同行业不足时相关为 NaN，
        #   直接 iloc[-1] 会把 NaN 带进汇总，"相关变化"随之全变 NaN。
        "corr_first": float(_corr_series(al).iloc[0]) if len(_corr_series(al)) else np.nan,
        "corr_last": float(_corr_series(al).iloc[-1]) if len(_corr_series(al)) else np.nan,
        "corr_valid": int(len(_corr_series(al))),
        "broad_exposure": al["broad_exposure"],
        "periods": al["periods"],
        "by_industry": al["by_industry"],
        "detail": al["detail"],
        "compare": cmp_,
        "elapsed": round(time.time() - t0, 1),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_window_args(ap)
    ap.add_argument("codes", nargs="*",
                    help="基金代码，留空则用 EXTENDED_FUNDS")
    ap.add_argument("--refresh", action="store_true",
                    help="强制重抓持仓与行业映射")
    args = ap.parse_args()
    win_start, _ = apply_window(args)
    codes = args.codes or list(C.EXTENDED_FUNDS.keys())

    print(f"[holdings] 分析窗口 {win_start} 起 · 只用半年报/年报（完整持仓）· "
          f"披露滞后 {H.DISCLOSURE_LAG_DAYS} 天")

    industry_map = H.load_industry_map(refresh=args.refresh)
    if industry_map.empty:
        print("[holdings] ✗ 行业映射构造失败（无法取得申万成分股）")
        return 1

    out_dir = C.RES_DIR
    rows, ok_n = [], 0
    for i, code in enumerate(codes, 1):
        print(f"\n[{i}/{len(codes)}] {code} ...")
        r = run_one(code, industry_map, refresh=args.refresh)
        if not r.get("ok"):
            print(f"  跳过：{r.get('reason')}")
            rows.append({"代码": code, "状态": "跳过", "原因": r.get("reason")})
            continue
        ok_n += 1
        c = r["compare"]
        h_ann = (float(c["stats"].loc[c["stats"]["基准"] == "持仓基准",
                                     "年化收益"].iloc[0]) if c.get("ok") else np.nan)
        s_ann = (float(c["stats"].loc[c["stats"]["基准"] == "RBSA合成基准",
                                      "年化收益"].iloc[0]) if c.get("ok") else np.nan)
        rows.append({
            "代码": code, "名称": r["name"], "状态": "完成",
            "可对比期数": r["n_periods"],
            "平均绝对偏差": round(r["mad_mean"], 4),
            "权重相关_首期": round(r["corr_first"], 3),
            "权重相关_末期": round(r["corr_last"], 3),
            "有效相关期数": r["corr_valid"],
            "相关变化": round(r["corr_last"] - r["corr_first"], 3),
            "RBSA宽基暴露": round(r["broad_exposure"], 3),
            "RBSA基准年化": round(s_ann, 4) if np.isfinite(s_ann) else np.nan,
            "持仓基准年化": round(h_ann, 4) if np.isfinite(h_ann) else np.nan,
            "耗时s": r["elapsed"],
        })
        # 落盘
        r["periods"].to_csv(out_dir / f"H1_持仓对齐_逐期_{code}.csv",
                            index=False, encoding="utf-8-sig")
        r["by_industry"].to_csv(out_dir / f"H2_持仓对齐_按行业_{code}.csv",
                                encoding="utf-8-sig")
        if c.get("ok"):
            c["stats"].to_csv(out_dir / f"H3_持仓基准对比_{code}.csv",
                              index=False, encoding="utf-8-sig")
        note = (f"（仅 {r['corr_valid']}/{r['n_periods']} 期能算相关，"
                f"其余共同行业不足）" if r["corr_valid"] < r["n_periods"] else "")
        print(f"  期数 {r['n_periods']} · 平均绝对偏差 {r['mad_mean']:.4f} · "
              f"权重相关 {r['corr_first']:.3f} → {r['corr_last']:.3f}"
              f"（{'下降' if r['corr_last'] < r['corr_first'] else '上升'}）{note}")
        if c.get("ok"):
            st = c["stats"].set_index("基准")
            print(f"  年化：基金 {st.loc['基金实际','年化收益']:+.2%} · "
                  f"RBSA {st.loc['RBSA合成基准','年化收益']:+.2%} · "
                  f"持仓 {st.loc['持仓基准','年化收益']:+.2%}"
                  f"（{c['n_days']} 个交易日）")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "H0_持仓对齐汇总.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 96)
    print(f"完成 {ok_n}/{len(codes)} 只")
    if ok_n:
        sub = df[df["状态"] == "完成"]
        print(f"  平均绝对偏差 均值 {sub['平均绝对偏差'].mean():.4f}"
              f"  中位 {sub['平均绝对偏差'].median():.4f}")
        print(f"  权重相关 首期均值 {sub['权重相关_首期'].mean():.3f} → "
              f"末期均值 {sub['权重相关_末期'].mean():.3f}"
              f"（变化 {sub['相关变化'].mean():+.3f}）")
        d = (sub["持仓基准年化"] - sub["RBSA基准年化"]).dropna()
        if len(d):
            print(f"  持仓基准 − RBSA基准 年化差：均值 {d.mean():+.2%}，"
                  f"持仓更优 {int((d > 0).sum())}/{len(d)} 只")
    print(f"\n结果写入 {out_dir}")
    print("\n读法：")
    print("  · 平均绝对偏差 = 两边归一到 100% 后，逐行业 |RBSA权重 − 真实暴露| 的均值")
    print("  · 权重相关下降 = RBSA 与真实持仓的一致性在变差（风格漂移的直接证据）")
    print("  · 持仓基准年化更高 ≠ 持仓基准更好：它只有半年频调仓、且含现金拖累，")
    print("    两者是不同频率/不同信息集的基准，差值只说明'ML 拟合多带来了/少带来了什么'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
