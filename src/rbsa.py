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

    # ---------------------------------------------------- 风格暴露明细
    @staticmethod
    def latest_exposure(weights: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({
            "名称": [C.NAME_OF.get(c, c) for c in weights.index],
            "层级": [C.LAYER_OF.get(c, "其他") for c in weights.index],
            "权重": weights.values,
        }).sort_values("权重", ascending=False).reset_index(drop=True)
        return df
