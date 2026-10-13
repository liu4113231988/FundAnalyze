# -*- coding: utf-8 -*-
"""
增量价值分析：风格画像（受约束回归）相对「单一宽基」到底多带来了什么。

为什么必须重做一遍
----------------
`naive_benchmarks` 里的「最优单一宽基」是**事后**从 5 个候选里挑最好的，
而受约束回归只用训练段拟合 —— 两者不对称，前者占了选择偏差的便宜。
本脚本把比较改成**事前公平**：

    同一训练/测试切分；"选哪个宽基"这个动作也只在训练段上完成；
    所有方案在同一个测试段上评估。

方案（全部只用训练段信息定参）
  A 单一宽基(MSE)   训练段 MSE 最低的一个指数 + 无约束 β
  B 单一宽基(相关)  训练段 |corr| 最高的一个指数 + 无约束 β
  C 等权混合        全部指数等权
  E Top3 相关等权   训练段 |corr| 前 3 个等权
  D 受约束回归      非负 + Σw = equity_ratio（= RBSA 的口径）

除 R² / TE 外再加一个朴素基准没有的维度：**权重稳定性** ——
把训练段再切 3 段分别拟合，看相邻段权重的 L1 变动。若风格画像只是
"多拟合了一些噪声"，它的稳定性应该**更差**而不是更好。

    python run_increment.py [基金代码...]
    python run_increment.py --window-years 2      # 近 2 年窗口（下限 2 年）

注意：本脚本用**面板内切分**做训练/测试（TRAIN_FRAC），窗口越短可切段越少；
2 年窗口下 `sp < 200` 会直接跳过并返回 None。
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from src import config as C, data as D
from src.models import _constrained_fit
from src.window import add_window_args, apply_window

TRAIN_FRAC = 0.70
EQUITY = 0.90


def _r2(pred, y):
    sse = float(((np.asarray(pred) - y) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return float("nan") if sst <= 1e-18 else 1.0 - sse / sst


def _pick_single(Xtr, ytr, by="mse"):
    """训练段内选一个宽基（事前），返回 (j, beta)。"""
    best, bj, bb = (np.inf, -1, 0.0) if by == "mse" else (-np.inf, -1, 0.0)
    for j in range(Xtr.shape[1]):
        v = float(np.var(Xtr[:, j]))
        if v <= 1e-18:
            continue
        b = float(np.cov(ytr, Xtr[:, j])[0, 1] / v)
        if by == "mse":
            e = float(np.mean((b * Xtr[:, j] - ytr) ** 2))
            if e < best:
                best, bj, bb = e, j, b
        else:
            c = abs(float(np.corrcoef(Xtr[:, j], ytr)[0, 1]))
            if c > best:
                best, bj, bb = c, j, b
    return bj, bb


def _stability(Xtr, ytr, fit, n_seg=3):
    """训练段分 n_seg 段分别拟合，相邻段权重 L1 距离的均值。"""
    segs = np.array_split(np.arange(len(ytr)), n_seg)
    ws = [np.asarray(fit(Xtr[s], ytr[s]), dtype=float) for s in segs
          if len(s) >= 60]
    if len(ws) < 2:
        return float("nan")
    return float(np.mean([np.abs(ws[i + 1] - ws[i]).sum()
                          for i in range(len(ws) - 1)]))


def run_one(code):
    px = D.load_panel(code, verbose=False)
    R = np.log(px.astype(float)).diff().dropna()
    assets = [c for c in px.columns if c != "__FUND__"]
    X, y = R[assets].values, R["__FUND__"].values
    n = len(y)
    sp = int(n * TRAIN_FRAC)
    if sp < 200 or n - sp < 100:
        return None
    Xtr, ytr, Xte, yte = X[:sp], y[:sp], X[sp:], y[sp:]

    def w_eq(Xa, ya):
        return np.full(Xa.shape[1], EQUITY / Xa.shape[1])

    def w_top3(Xa, ya):
        corr = np.abs(np.array([np.corrcoef(Xa[:, j], ya)[0, 1]
                                for j in range(Xa.shape[1])]))
        corr = np.nan_to_num(corr)
        top = np.argsort(-corr)[:3]
        w = np.zeros(Xa.shape[1])
        w[top] = EQUITY / len(top)
        return w

    def w_single_mse(Xa, ya):
        j, b = _pick_single(Xa, ya, "mse")
        w = np.zeros(Xa.shape[1])
        if j >= 0:
            w[j] = b
        return w

    def w_single_corr(Xa, ya):
        j, b = _pick_single(Xa, ya, "corr")
        w = np.zeros(Xa.shape[1])
        if j >= 0:
            w[j] = b
        return w

    def w_constrained(Xa, ya):
        return _constrained_fit(Xa, ya, EQUITY)

    schemes = [
        ("A_单一宽基MSE", w_single_mse),
        ("B_单一宽基相关", w_single_corr),
        ("C_等权混合", w_eq),
        ("E_Top3相关等权", w_top3),
        ("D_受约束回归", w_constrained),
    ]
    out = {"基金": code, "名称": C.EXTENDED_FUNDS.get(code, code),
           "训练天数": sp, "测试天数": n - sp}
    for name, fit in schemes:
        w = np.asarray(fit(Xtr, ytr), dtype=float)
        out[f"{name}_R2"] = _r2(Xte @ w, yte)
        out[f"{name}_TE"] = float(np.std(Xte @ w - yte, ddof=1) * np.sqrt(252))
        out[f"{name}_稳定"] = _stability(Xtr, ytr, fit)
    a = out["A_单一宽基MSE_R2"]
    d = out["D_受约束回归_R2"]
    out["R2增量_D减A"] = (d - a) if np.isfinite(a) and np.isfinite(d) else np.nan
    out["TE变化_D减A"] = (out["D_受约束回归_TE"] - out["A_单一宽基MSE_TE"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_window_args(ap)
    ap.add_argument("codes", nargs="*",
                    help="基金代码，留空则用 EXTENDED_FUNDS")
    args = ap.parse_args()
    apply_window(args)
    codes = args.codes or list(C.EXTENDED_FUNDS.keys())
    rows = [r for r in (run_one(c) for c in codes) if r]
    if not rows:
        print("无有效结果")
        return 1
    df = pd.DataFrame(rows)
    RES = ROOT / "output" / "results"
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "P2-02_增量价值分析.csv", index=False,
              encoding="utf-8-sig")

    print("=" * 112)
    print("P2-02 · 增量价值：风格画像 vs 朴素基准（事前公平比较，训练 70% / 测试 30%）")
    print("=" * 112)
    show = ["基金", "名称", "A_单一宽基MSE_R2", "B_单一宽基相关_R2",
            "C_等权混合_R2", "E_Top3相关等权_R2", "D_受约束回归_R2",
            "R2增量_D减A", "A_单一宽基MSE_TE", "D_受约束回归_TE",
            "TE变化_D减A"]
    print(df[show].round(4).to_string(index=False))

    inc = df["R2增量_D减A"].dropna()
    print("\n[汇总]")
    print(f"  基金数                    {len(df)}")
    print(f"  R2 增量 > 0.05 的基金数   {int((inc > 0.05).sum())}/{len(inc)}")
    print(f"  R2 增量 > 0 的基金数      {int((inc > 0).sum())}/{len(inc)}")
    print(f"  R2 增量 均值 / 中位数     {inc.mean():+.4f} / {inc.median():+.4f}")
    print(f"  A（单一宽基）R2 均值      {df['A_单一宽基MSE_R2'].mean():.4f}")
    print(f"  D（受约束回归）R2 均值    {df['D_受约束回归_R2'].mean():.4f}")
    print(f"  E（Top3 等权）R2 均值     {df['E_Top3相关等权_R2'].mean():.4f}")

    sa = df["A_单一宽基MSE_稳定"].mean()
    se = df["E_Top3相关等权_稳定"].mean()
    sd = df["D_受约束回归_稳定"].mean()
    print("\n  权重稳定性（相邻训练段 L1，越小越稳）")
    print(f"    A 单一宽基   {sa:.4f}")
    print(f"    E Top3 等权  {se:.4f}")
    print(f"    D 受约束回归 {sd:.4f}"
          + ("  <- 比单一宽基更稳（复杂度买到了稳定性）"
             if sd < sa else "  <- 比单一宽基更不稳（复杂度没买到稳定性）"))

    print("""
怎么读
----
1. **增量 > 0 才算有增量价值**：若 D 只比 A 高零点几个 R²，说明 44 个指数的
   复杂度并不比"挑一个宽基"更值 —— 那应该简化模型而不是扩池。
2. **稳定性是第二个判据**：D 若同时更不稳，说明它把噪声也拟合进去了；
   D 若更稳，说明多出来的指数确实在刻画一个持续的结构（风格漂移、行业轮动）。
3. **本表只回答"解释力"**，不回答"预测力"：样本外 R² 是同期解释，
   不是提前预测收益。后者由 E1 的 IC 期限结构与 E9b 的时间向前验证负责。
4. A/B 是**事前**选的宽基，因此没有"最优单一宽基"那种选择偏差 ——
   这是它与 naive_benchmarks 的关键区别。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
