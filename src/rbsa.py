# -*- coding: utf-8 -*-
"""
RBSA 风格分析入口（Returns-Based Style Analysis）。

历史上本模块自带一套「ElasticNet 筛选 + SLSQP 精修」实现，与 models.py 中
的 enet_slsqp 完全同构。为消除双轨，本模块现在只保留：

  · RBSA.fit_rolling : 构造滚动重估点，委托 ModelZoo 的 enet_slsqp 完成拟合，
                       返回与旧版相同的 {weights, syn_ret, diag, assets}
  · synth_nav        : 合成收益 → 合成净值
  · latest_exposure  : 最新权重的名称/层级明细表

算法细节（两阶段、非负、Σw=股票仓位）见 src/models.py 中 enet_slsqp 的文档。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .models import ModelZoo


class RBSA:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or C.RBSA

    def fit_rolling(self, px: pd.DataFrame, fund_col: str = "__FUND__",
                    verbose: bool = True) -> Dict[str, object]:
        cfg = self.cfg
        assets = [c for c in px.columns if c != fund_col]
        R = np.log(px.astype(float)).diff().dropna()
        dates = R.index
        X_all = R[assets].values
        y_all = R[fund_col].values

        # 与 ModelZoo.run 一致：可选的指数股息调整，用于剥离"价格指数 vs
        # 基金总回报"带来的伪 alpha（默认 0，不改变行为）。
        div = float(cfg.get("index_dividend_annual", 0.0) or 0.0)
        if abs(div) > 1e-9:
            X_all = X_all + div / 252.0

        n = len(dates)
        W = cfg["window"]
        if n < W + 30:
            raise ValueError(f"样本不足：仅 {n} 行，窗口需要 {W}。")

        # RBSA 默认使用全部重估点（不抽稀）；抽稀是 Web 首屏的耗时控制
        rebal_idx = ModelZoo._rebal_idx(n, W, cfg["refit"], None)
        if verbose:
            print(f"[rbsa] 滚动拟合：{n} 交易日，窗口 {W}，重估 {cfg['refit']}，"
                  f"共 {len(rebal_idx)} 次估算")

        rep = ModelZoo(cfg)._run_one("enet_slsqp", X_all, y_all, assets,
                                     dates, rebal_idx, None)

        if verbose:
            diag = rep.fit_diag
            steps = list(range(0, len(diag), 12)) + [len(diag) - 1]
            for k in dict.fromkeys(steps):
                row = diag.iloc[k]
                print(f"   {diag.index[k].date()}  R²={row['r2']:.3f} "
                      f"TE={row['te']:.1%}")

        self.weights_hist_ = rep.weights_hist
        self.syn_ret_ = rep.syn_ret
        self.diag_ = rep.fit_diag
        self.assets_ = assets
        return {
            "weights": rep.weights_hist,
            "syn_ret": rep.syn_ret,
            "diag": rep.fit_diag,
            "assets": assets,
        }

    # ---------------------------------------------------- 合成基准净值
    @staticmethod
    def synth_nav(syn_ret: pd.Series, start_value: float = 1.0) -> pd.Series:
        return start_value * np.exp(syn_ret.cumsum())

    # ---------------------------------------------------- 自由 alpha 诊断
    @staticmethod
    def alpha_diagnose(px: pd.DataFrame, weights: pd.Series,
                       fund_col: str = "__FUND__",
                       train_frac: float = 0.7) -> dict:
        """
        用**允许截距**的 OLS 估计"真实 alpha"，作为无截距约束拟合的对照。

        为什么需要它
        ------------
        约束拟合（Σw = equity_ratio、无截距）在数学上**无法表达 alpha**：
        指数组合被强制过原点，基金持续跑赢的那部分只能变成残差漂移。
        于是图上出现"基金起飞、合成基准横盘"，看起来像拟合失败。

        实测（`output/results/诊断_合成基准背离.csv` 的对照实验）：
          110013  约束拟合累计 1.40 / OLS(含 alpha) 1.83 / 基金实际 1.71
          519697  约束拟合 1.11 / OLS 1.49 / 基金实际 1.33
        即落后部分主要是 alpha，不是仓位错配 —— 且放开 Σw 上限**更差**
        （R² 从 0.83 掉到 0.77），所以正确的解法是承认 alpha，不是加仓位。

        返回：年化 alpha、t 值、无截距 vs 有截距的 R² 对比。
        """
        R = np.log(px.astype(float)).diff().dropna()
        assets = [c for c in px.columns if c != fund_col]
        keep = [a for a in assets if a in weights.index]
        X, y = R[keep].values, R[fund_col].values
        n = len(y)
        sp = int(n * float(np.clip(train_frac, 0.3, 0.9)))
        if sp < 100 or n - sp < 60:
            return {}
        Xtr, ytr, Xte, yte = X[:sp], y[:sp], X[sp:], y[sp:]

        def _r2(pred, yy):
            sse = float(((np.asarray(pred) - yy) ** 2).sum())
            sst = float(((yy - yy.mean()) ** 2).sum())
            return float("nan") if sst <= 1e-18 else 1.0 - sse / sst

        # 有截距 OLS（训练段估参，测试段评估）
        A = np.column_stack([np.ones(len(Xtr)), Xtr])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        alpha_d, beta = float(coef[0]), coef[1:]
        pred_ols = Xte @ beta + alpha_d
        # 同一组 beta 但强制过原点（用于对比"去掉 alpha 后掉多少"）
        pred_noint = Xte @ beta

        d = (np.log(px[fund_col]).diff().dropna().iloc[sp:]
             - pd.Series(Xte @ beta, index=R.index[sp:]))
        d = d.dropna()
        t = None
        if len(d) > 30:
            try:
                from src.robustness import _nw_mean_t
                t = float(_nw_mean_t(d.values, lags=5))
            except Exception:                                # noqa: BLE001
                t = None
        return {
            "年化alpha": float(alpha_d * 252),
            "alpha_t值": t,
            "测试段R2_含alpha": _r2(pred_ols, yte),
            "测试段R2_去alpha": _r2(pred_noint, yte),
            "beta和": float(np.sum(beta)),
            "训练天数": sp, "测试天数": n - sp,
        }

    # ---------------------------------------------------- 风格暴露明细
    @staticmethod
    def latest_exposure(weights: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({
            "名称": [C.NAME_OF.get(c, c) for c in weights.index],
            "层级": [C.LAYER_OF.get(c, "其他") for c in weights.index],
            "权重": weights.values,
        }).sort_values("权重", ascending=False).reset_index(drop=True)
        return df
