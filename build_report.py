# -*- coding: utf-8 -*-
"""生成 HTML 研究报告。"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src import config as C
from src.charts import build_all
from src.pipeline import run_fund
from src.report import build_report
from src.experiments import summarize_funds
from src.runinfo import write_fingerprint, fingerprint_lines


def read_csv_safe(name: str):
    fp = C.RES_DIR / name
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception as e:      # noqa: BLE001
            print(f"[warn] 读取 {name} 失败: {e}")
    return None


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else C.MAIN_FUND
    name = C.DEMO_FUNDS.get(code, C.EXTENDED_FUNDS.get(code, code))

    print(f"[report] 运行主案例 {code} {name} ...")
    main_res = run_fund(code, name, verbose=False)
    figs = build_all(main_res)
    print(f"[report] 生成 {len(figs)} 张图")

    # 跨基金汇总：优先用扩展样本结果
    summary = read_csv_safe("X1_扩展_逐基金.csv")
    if summary is None:
        summary = read_csv_safe("E0_跨基金汇总.csv")
    if summary is None:
        summary = pd.DataFrame()

    # 转成报告表格需要的列名
    if "净贡献_夏普" not in summary.columns and "默认_超额夏普" in summary.columns:
        summary = summary.rename(columns={
            "默认_策略夏普": "策略夏普", "默认_对照夏普": "对照夏普",
            "默认_超额夏普": "净贡献_夏普", "默认_RankIC": "RankIC",
            "默认_平均仓位": "平均仓位",
            "最优_超额夏普": "最优_超额夏普", "最优窗口": "最优窗口",
            "最优周期": "最优周期",
        })
        cols = [c for c in ["代码", "名称", "样本天数", "策略夏普", "对照夏普",
                            "净贡献_夏普", "RankIC", "平均仓位",
                            "最优窗口", "最优周期", "最优_超额夏普"]
                if c in summary.columns]
        summary = summary[cols]

    experiments = {
        # E1 优先用 14 只基金的全样本结果（X3）
        "E1_IC期限结构": (read_csv_safe("X3_扩展_IC期限结构.csv")
                       if (C.RES_DIR / "X3_扩展_IC期限结构.csv").exists()
                       else read_csv_safe("E1_IC期限结构.csv")),
        "E2_通道消融": read_csv_safe("E2_通道消融.csv"),
        "E3_参数敏感性": read_csv_safe("E3_参数敏感性.csv"),
        # E4 优先用 14 只基金的全样本结果（X2），退化到 4 只基金的 E4
        "E4_周期匹配": (read_csv_safe("X2_扩展_周期矩阵.csv")
                       if (C.RES_DIR / "X2_扩展_周期矩阵.csv").exists()
                       else read_csv_safe("E4_周期匹配.csv")),
    }

    # E1：跨基金平均后转成「信号 × 前瞻期」矩阵，按时间顺序排列
    e1 = experiments["E1_IC期限结构"]
    if e1 is not None and "基金" in e1.columns:
        order = ["5日", "20日", "60日", "120日", "250日"]
        agg = (e1.groupby(["信号", "前瞻期"], as_index=False)["RankIC"].mean()
               .round(4))
        piv = agg.pivot(index="信号", columns="前瞻期", values="RankIC")
        piv = piv.reindex(columns=[c for c in order if c in piv.columns])
        # 调整信号顺序，便于阅读
        sorder = ["趋势通道", "相对偏离通道", "赛道估值通道", "综合分"]
        piv = piv.reindex([s for s in sorder if s in piv.index])
        experiments["E1_IC期限结构"] = piv.reset_index()
        experiments["_e1_n"] = e1["基金"].nunique()

    # E2：跨基金平均；剔除各基金各自的「恒定仓位」对照行
    #     （超额列已经表达了"相对该基金自身对照"的含义）
    e2 = read_csv_safe("E2_通道消融.csv")
    if e2 is not None and "基金" in e2.columns:
        e2 = e2[~e2["方案"].astype(str).str.startswith("恒定仓位")]
        cols = [c for c in ["方案", "年化收益", "夏普", "最大回撤",
                            "超额夏普_vs对照", "超额年化_vs对照"]
                if c in e2.columns]
        agg = (e2.groupby("方案", as_index=False)[cols[1:]].mean()
               .sort_values("超额夏普_vs对照", ascending=False).round(4))
        # 买入持有不是"相对恒定仓位"的方案，超额列留空以免误读
        bh = agg["方案"].astype(str).str.startswith("买入持有")
        agg.loc[bh, ["超额夏普_vs对照", "超额年化_vs对照"]] = np.nan
        experiments["E2_通道消融"] = agg
        experiments["_e2_n"] = e2["基金"].nunique()

    # E3：不能对参数取平均——改为「按趋势权重分组的分布统计」
    e3 = experiments["E3_参数敏感性"]
    if e3 is not None and not e3.empty:
        g = e3.groupby("趋势权重").agg(
            参数组合数=("超额夏普", "size"),
            正贡献比例=("超额夏普", lambda s: (s > 0).mean()),
            平均超额夏普=("超额夏普", "mean"),
            中位超额夏普=("超额夏普", "median"),
            平均RankIC=("RankIC", "mean"),
        ).round(4).reset_index()
        g["参数组合数"] = g["参数组合数"].astype(int)
        experiments["E3_参数敏感性"] = g
        experiments["_e3_stat"] = {
            "n": len(e3),
            "pos": float((e3["超额夏普"] > 0).mean()),
            "mean": float(e3["超额夏普"].mean()),
            "median": float(e3["超额夏普"].median()),
        }

    # ---- 新证据纳入报告（复查发现：P0-06/P2-02/P1-03 的产物完全没进 HTML，
    #      E9b 的数字还是硬编码 —— 又一次"代码已修、文档引用旧数"）。
    # ★ 必须在 verdict 构造**之前**读取，否则 verdict 只能继续硬编码。
    def _read_json_cols(name, keys):
        fp = C.RES_DIR / name
        if not fp.exists():
            return None
        try:
            d = pd.read_csv(fp)
            out = {}
            for k in keys:
                if k in d.columns:
                    out[k] = d[k].iloc[0]
            return out or None
        except Exception as e:                                # noqa: BLE001
            print(f"[warn] 读取 {name} 失败: {e}")
            return None

    # E9b 汇总是**长格式**（指标/值 两列），不是宽格式
    _e9b_df = read_csv_safe("E9b_时间向前验证_汇总.csv")
    e9b = None
    if _e9b_df is not None and {"指标", "值"}.issubset(_e9b_df.columns):
        e9b = dict(zip(_e9b_df["指标"], pd.to_numeric(_e9b_df["值"],
                                                      errors="coerce")))
    mc = read_csv_safe("E7_稳健性_多重比较.csv")
    p006 = read_csv_safe("P0-06_功效分析.csv")
    p202 = read_csv_safe("P2-02_增量价值分析.csv")
    p103 = read_csv_safe("P1-03_过拟合代价拆解.csv")
    evidence = {
        "e9b": e9b, "multiplicity": mc, "power": p006,
        "increment": p202, "overfit_split": p103,
    }

    # 跨基金结论文字（E9b 数字从上面的 evidence 取，不再硬编码）
    if not summary.empty and "净贡献_夏普" in summary.columns:
        v = summary["净贡献_夏普"].astype(float)
        n = len(v)
        pos = int((v > 0).sum())
        rk = float(summary["RankIC"].astype(float).mean()) if "RankIC" in summary.columns else float("nan")
        _mig = e9b.get("迁移配置在测试期均值IR") if e9b else None
        _cost = e9b.get("参数选择过拟合代价") if e9b else None
        _mig_s = (f"{float(_mig):+.3f}"
                  if _mig is not None and np.isfinite(_mig) else "负值")
        _cost_s = (f"{float(_cost):+.3f}"
                   if _cost is not None and np.isfinite(_cost) else "")
        verdict = (f"14 只不同风格基金的检验结果：默认参数下超额夏普为正的仅 {pos}/{n} 只"
                   f"（{pos/n:.0%}），均值 {v.mean():+.4f}，中位数 {v.median():+.4f}，"
                   f"RankIC 均值 {rk:+.4f}。"
                   f"<b>结论高度一致——这套日频信号不创造择时价值，且方向轻微为负。</b>"
                   f"长周期格子虽出现正超额夏普，但存在选择偏差、缺少 IC 层面支撑，"
                   f"且 <b>E9b 时间向前验证的迁移 IR 为 {_mig_s}"
                   + (f"、过拟合代价 {_cost_s}" if _cost_s else "")
                   + "</b>——训练期选出的参数在未来不再为正，"
                     "只能作为<b>待验证的候选假设</b>，不能视为已证实的效应。")
    else:
        verdict = ""

    meta_extra = {
        "n_funds": f"{len(summary)}" if not summary.empty else "4",
        "period": f"{main_res['nav'].index[0].date()} ~ {main_res['nav'].index[-1].date()}",
        "cross_fund_verdict": verdict,
        "evidence": evidence,
    }

    # ---- 运行指纹（P1-09 / N-09）：结论必须能回答"是在哪套环境/数据/参数下跑的"
    fp = write_fingerprint(
        C.RES_DIR / "run_fingerprint.json",
        extra={"fund": code, "models": sorted(main_res.get("models", {}).keys())
               if isinstance(main_res.get("models"), dict) else None,
               "max_rebal": C.RBSA.get("max_rebal"),
               "rebal_mode": C.RBSA.get("rebal_mode")})
    print("[report] 运行指纹：")
    for ln in fingerprint_lines(fp):
        print(ln)

    path = build_report(main_res, summary, experiments, figs,
                        extra_meta=meta_extra, fingerprint=fp)
    print(f"[report] 完成：{path}")


if __name__ == "__main__":
    main()
