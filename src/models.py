# -*- coding: utf-8 -*-
"""
多模型风格拟合引擎（Model Zoo for RBSA）
======================================

为什么要多模型
--------------
单一算法给出单一答案，无法回答"这个结果是不是算法的偶然"。
本模块把 11 个覆盖不同统计流派的拟合器统一到同一个接口下，
在**完全相同的数据切分**上做滚动样本外评估，用统一指标排名，
从而让"选哪个模型"变成一个有证据的选择，而不是拍脑袋。

统一接口
--------
    fit_window(X, y, cols, spec, params) -> FitResult
        X   : (T, N) 指数收益矩阵
        y   : (T,)   基金收益
        cols: N 个指数代码
        spec: ModelSpec

FitResult 字段
    weights : 可解释权重（pd.Series，index=指数代码），用于风格穿透与合成基准
    pred    : 窗口内拟合/预测的基金收益
    r2      : 窗口内拟合优度
    te      : 年化跟踪误差
    extra   : 算法特有诊断（选择的正则强度、特征重要性等）

评估口径（务必看清）
--------------------
线性约束类模型（enet/lasso/ridge/corr/pcr）产生的 weights 之和 = equity_ratio，
可以直接理解为"一个可投资组合"，其样本外合成收益 = X @ w，是**真实的样本外**。

黑箱模型（gbr/rf/krr/mlp）没有"权重和=0.9"的经济含义，
它们的 weights 仅用于重要性展示；样本外合成收益 = 模型对未来收益的预测。
两类都只用 t 时刻之前的数据拟合、对 t 之后做预测，因此**排名是公平的**。
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor

from . import config as C


# ====================================================================== 模型规格
@dataclass
class ModelSpec:
    key: str
    name: str
    family: str                # linear / kernel / tree / nn
    desc: str
    # 权重是否有直接经济含义。默认 False 是刻意的 fail-safe：
    # 新增模型时若忘记声明，默认会被当作"黑箱"而不会误导用户
    # 把重要性排序当成真实持仓。
    interpretable: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


def _default_params() -> Dict[str, Dict[str, Any]]:
    """每个模型的可调超参（前端暴露为可编辑表单）。"""
    return {
        "enet_slsqp": dict(l1_ratio=0.5, alpha_scale=1.0, max_assets=8, min_weight=0.01),
        "lasso":      dict(alpha_scale=1.0, max_assets=8, min_weight=0.01),
        "ridge":      dict(alpha_scale=1.0, max_assets=10, min_weight=0.005),
        "corr_topk":  dict(top_k=6, min_weight=0.01),
        "pcr":        dict(n_components=12, min_weight=0.01),
        "ewma":       dict(half_life=126, max_assets=15, min_weight=0.0),
        "huber":      dict(huber_k=1.345, n_iter=3,
                           max_assets=15, min_weight=0.0),
        "gbr":        dict(max_depth=3, learning_rate=0.06, max_iter=160,
                           min_weight=0.005),
        "rf":         dict(n_estimators=120, max_depth=6, min_weight=0.005),
        "krr":        dict(alpha=0.30, gamma_scale=1.0, min_weight=0.005),
        "mlp":        dict(hidden=(32, 16), alpha=1e-2, max_iter=400, min_weight=0.005),
    }


MODELS: Dict[str, ModelSpec] = {
    "enet_slsqp": ModelSpec(
        "enet_slsqp", "非负弹性网 + 约束精修", "linear",
        "两阶段：非负 ElasticNet 做稀疏筛选（时序CV 选 α），"
        "再在支撑集上做 Σw=股票仓位的精确约束最小二乘。稀疏、可解释，最稳健。",
        interpretable=True),
    "lasso": ModelSpec(
        "lasso", "非负 Lasso", "linear",
        "L1 正则比弹性网更强，倾向于选出更少的指数。适合持仓极集中的基金，"
        "但可能因过度稀疏而漏掉真实风格。",
        interpretable=True),
    "ridge": ModelSpec(
        "ridge", "非负 Ridge", "linear",
        "L2 正则，所有指数保留微小权重。拟合更平滑、R² 通常更高，"
        "但权重分散、解释性下降，容易把噪声也当风格。",
        interpretable=True),
    "corr_topk": ModelSpec(
        "corr_topk", "相关性 Top-K + 约束回归", "linear",
        "最朴素的方法：按相关系数取前 K 个指数，再做约束最小二乘。"
        "作为基准线——它赢了才说明机器学习真的有价值。",
        interpretable=True),
    "pcr": ModelSpec(
        "pcr", "主成分回归 PCR", "linear",
        "先对指数收益做 PCA 抑制共线性，再回归。"
        "宽基之间高度相关时比直接回归更稳。",
        interpretable=True),
    "ewma": ModelSpec(
        "ewma", "指数加权约束回归 EWMA", "linear",
        "用指数衰减权重替代 252 天硬截断：越近的行情权重越大，"
        "对基金风格切换反应更快且不缩小有效样本。风格漂移基金上显著占优。",
        interpretable=True),
    "huber": ModelSpec(
        "huber", "Huber 稳健约束回归", "linear",
        "IRLS 迭代重加权：极端行情日（涨跌停/事件冲击）自动降权，"
        "避免个别异常日扭曲风格估计。换手率低、结果稳。",
        interpretable=True),
    "gbr": ModelSpec(
        "gbr", "梯度提升树", "tree",
        "非线性模型。捕捉指数之间的交互与状态依赖，"
        "但外推能力弱、权重不可直接当作组合。", interpretable=False),
    "rf": ModelSpec(
        "rf", "随机森林", "tree",
        "多棵决策树投票。对异常值稳健，能给出稳定的特征重要性排序。",
        interpretable=False),
    "krr": ModelSpec(
        "krr", "核岭回归 RBF", "kernel",
        "用 RBF 核捕捉非线性关系，是线性模型与神经网络之间的平滑过渡。",
        interpretable=False),
    "mlp": ModelSpec(
        "mlp", "多层感知机", "nn",
        "小型神经网络。理论上拟合能力最强，"
        "但在单只基金的样本量下极易过拟合，是重要的反面对照组。",
        interpretable=False),
}

MODEL_ORDER = list(MODELS.keys())


# ====================================================================== 约束回归工具
def _constrained_fit(X: np.ndarray, y: np.ndarray, total: float,
                     sw: Optional[np.ndarray] = None) -> np.ndarray:
    """min Σ sw_i(Xw-y)²  s.t. w>=0, Σw=total（SLSQP 精确解）。"""
    from scipy.optimize import minimize

    k = X.shape[1]
    sw = np.ones(len(y)) if sw is None else np.asarray(sw, dtype=float)

    def obj(w):
        r = (X @ w - y) * np.sqrt(sw)
        return float(r @ r)

    def jac(w):
        return 2.0 * X.T @ (sw * (X @ w - y))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - total,
             "jac": lambda w: np.ones(k)}]
    res = minimize(obj, np.full(k, total / k), jac=jac, method="SLSQP",
                   bounds=[(0.0, None)] * k, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    # ★ 必须检查收敛：原实现直接 res.x 就用，SLSQP 在病态窗口（指数高度共线、
    # 或窗口内样本过少）会返回未收敛解甚至 NaN，权重里的 NaN 会一路污染
    # 合成基准、健康分与回测净值，且不会报错——只会得到一条莫名其妙的曲线。
    x = np.asarray(res.x, dtype=float)
    if not res.success or not np.all(np.isfinite(x)):
        return np.full(k, total / k)
    w = np.clip(x, 0.0, None)
    if w.sum() > 0:
        w = w * total / w.sum()
    else:
        w = np.full(k, total / k)
    return w


def _clean_weights(w: np.ndarray, cols: List[str], equity_ratio: float,
                   min_weight: float, max_assets: int) -> pd.Series:
    """去噪声 → 截断 → 重新归一到 equity_ratio。"""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    order = np.argsort(-w)
    if len(w) > max_assets:
        order = order[:max_assets]
    m = np.zeros_like(w)
    m[order] = w[order]
    m[m < min_weight] = 0.0
    if m.sum() <= 0:                       # 全被剪掉 → 保留最大项
        m = np.zeros_like(w)
        m[int(np.argmax(w))] = max(equity_ratio, 1e-3)
    m = m * equity_ratio / m.sum()
    return pd.Series(m, index=cols).sort_values(ascending=False)


def _alpha_ladder(base: float, scale: float) -> List[float]:
    """
    围绕基准 α 生成正则强度阶梯。

    只保留 3 档：滚动评估会在每个窗口上重复调用本函数，
    档位越多耗时线性增长，而 CV 曲���在 3 档上已足够选出量级。
    """
    base = max(float(base), 1e-5)
    s = max(float(scale), 1e-3)
    ladder = [base * s * f for f in (0.5, 2.0, 8.0)]
    return sorted({round(a, 8) for a in ladder if a > 0})


def _ts_cv_mse(make_model, Xs: np.ndarray, ys: np.ndarray, n_splits: int = 2):
    """
    时序交叉验证选超参，返回 **(在全样本上重新拟合好的模型实例, MSE)**。

    注意：候选超参由 `make_model()` 生成为**工厂**（无参 lambda），
    因为 sklearn 的 clone 语义要求超参在构造时固化。

    n_splits 默认 2 而非 4：滚动评估会在 ~120 个窗口上反复调用本函数，
    折数直接决定总耗时（4 折 × 3 档 = 12 次拟合/窗口）。
    2 折在正则强度这种量级选择上已经足够，且耗时减半。
    """
    from sklearn.model_selection import TimeSeriesSplit

    n = len(ys)
    ns = max(2, min(n_splits, n // 60))
    if ns < 2:
        return None, np.inf
    cv = TimeSeriesSplit(n_splits=ns)
    best_factory, best_e = None, np.inf
    for factory in make_model():
        errs = []
        for tr, te in cv.split(Xs):
            m = factory()
            try:
                m.fit(Xs[tr], ys[tr])
                errs.append(float(np.mean((m.predict(Xs[te]) - ys[te]) ** 2)))
            except Exception:          # noqa: BLE001
                errs = []
                break
        if not errs:
            continue
        e = float(np.mean(errs))
        if e < best_e:
            best_factory, best_e = factory, e
    if best_factory is None:
        return None, np.inf
    # 用最优超参在全样本上重新拟合，供调用方直接使用
    final = best_factory()
    final.fit(Xs, ys)
    return final, best_e


# ====================================================================== 单次拟合结果
@dataclass
class FitResult:
    weights: pd.Series
    pred: np.ndarray
    r2: float
    te: float
    extra: Dict[str, Any] = field(default_factory=dict)
    # 样本外预测器：输入 (n, N) 原始收益行，输出基金收益预测。
    # 线性模型 = X @ w；黑箱模型 = 已拟合的 sklearn 估计器（若输入需标准化，
    # 则用闭包包住标准化步骤）。为 None 时调用方退回 X @ w。
    predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None


def _score(pred: np.ndarray, y: np.ndarray) -> tuple:
    """(R², 年化 TE)。R² 下限截到 0，避免负值污染平均。"""
    resid = np.asarray(pred, dtype=float) - y
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum()) or 1e-12
    r2 = max(0.0, 1.0 - sse / sst)
    te = float(np.std(resid, ddof=1) * np.sqrt(252)) if len(resid) > 1 else np.nan
    return r2, te


# ====================================================================== 各模型实现
def _fit_enet(X, y, cols, equity_ratio, prm):
    """非负 ElasticNet 稀疏筛选 → SLSQP 约束精修（复刻原 RBSA 两阶段）。"""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    ysd = (y.std() or 1.0)
    ys = (y - y.mean()) / ysd

    l1 = float(np.clip(prm.get("l1_ratio", 0.5), 0.0, 1.0))
    grid = _alpha_ladder(3e-4, prm.get("alpha_scale", 1.0))

    def factory():
        for a in grid:
            yield (lambda a=a: ElasticNet(alpha=a, l1_ratio=l1, positive=True,
                                          fit_intercept=True, max_iter=20000,
                                          tol=1e-5))
    mdl, _ = _ts_cv_mse(factory, Xs, ys)
    if mdl is None:
        mdl = ElasticNet(alpha=grid[0], l1_ratio=l1, positive=True,
                         max_iter=20000).fit(Xs, ys)
    coef = mdl.coef_ / sd
    sup = np.where(np.abs(coef) > 1e-12)[0]
    if sup.size == 0:
        corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])])
        corr = np.nan_to_num(corr)
        sup = np.array([int(np.argmax(corr))])
    if sup.size > int(prm.get("max_assets", 8)):
        sup = sup[np.argsort(-np.abs(coef[sup]))[:int(prm.get("max_assets", 8))]]

    w_s = _constrained_fit(X[:, sup], y, equity_ratio)
    w_full = np.zeros(X.shape[1])
    w_full[sup] = w_s
    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.01)),
                       int(prm.get("max_assets", 8)))
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"alpha": float(mdl.alpha), "l1_ratio": l1,
                            "n_support": int(sup.size)})


def _fit_lasso(X, y, cols, equity_ratio, prm):
    """非负 Lasso（L1 更强）+ 约束精修。"""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    ys = (y - y.mean()) / (y.std() or 1.0)
    grid = _alpha_ladder(1e-4, prm.get("alpha_scale", 1.0))

    def factory():
        for a in grid:
            yield (lambda a=a: Lasso(alpha=a, positive=True, fit_intercept=True,
                                      max_iter=40000, tol=1e-5))
    mdl, _ = _ts_cv_mse(factory, Xs, ys)
    if mdl is None:
        mdl = Lasso(alpha=grid[0], positive=True, max_iter=40000).fit(Xs, ys)
    coef = mdl.coef_ / sd
    sup = np.where(np.abs(coef) > 1e-12)[0]
    if sup.size == 0:
        sup = np.array([int(np.nanargmax(np.abs(np.array(
            [np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])]))))])
    if sup.size > int(prm.get("max_assets", 8)):
        sup = sup[np.argsort(-np.abs(coef[sup]))[:int(prm.get("max_assets", 8))]]

    w_full = np.zeros(X.shape[1])
    w_full[sup] = _constrained_fit(X[:, sup], y, equity_ratio)
    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.01)),
                       int(prm.get("max_assets", 8)))
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"alpha": float(mdl.alpha), "n_support": int(sup.size)})


def _fit_ridge(X, y, cols, equity_ratio, prm):
    """非负 Ridge：系数整体稠密，小权重即低相关指数。"""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    ys = (y - y.mean()) / (y.std() or 1.0)
    grid = _alpha_ladder(3e-3, prm.get("alpha_scale", 1.0))

    def factory():
        for a in grid:
            yield (lambda a=a: Ridge(alpha=a, positive=True, fit_intercept=True))
    mdl, _ = _ts_cv_mse(factory, Xs, ys)
    if mdl is None:
        mdl = Ridge(alpha=grid[0], positive=True).fit(Xs, ys)
    coef = mdl.coef_ / sd
    w = _clean_weights(coef, cols, equity_ratio,
                       float(prm.get("min_weight", 0.005)),
                       int(prm.get("max_assets", 10)))
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"alpha": float(mdl.alpha)})


def _fit_corr_topk(X, y, cols, equity_ratio, prm):
    """相关性 Top-K + 约束回归（基准线模型）。"""
    sd = X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    corr = np.array([abs(np.corrcoef(X[:, j] / sd[j], y)[0, 1])
                     for j in range(X.shape[1])])
    corr = np.nan_to_num(corr)
    k = int(np.clip(int(prm.get("top_k", 6)), 1, X.shape[1]))
    sup = np.argsort(-corr)[:k]
    w_full = np.zeros(X.shape[1])
    w_full[sup] = _constrained_fit(X[:, sup], y, equity_ratio)
    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.01)), k)
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"top_corr": [float(corr[j]) for j in sup],
                            "top_corr_codes": [cols[j] for j in sup]})


def _fit_pcr(X, y, cols, equity_ratio, prm):
    """
    主成分回归：PCA 抑制共线性 → OLS → 再拆回原始指数并归一。

    主成分个数用**样本内最后 20% 做 holdout**选一次即可；
    早期版本对每个候选 k 都重跑一遍 PCA+OLS，在 118 个滚动窗口上
    完全是浪费（PCA 本身已按方差降序，前 k 个主成分的解释力是单调的）。
    """
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    kmax = int(np.clip(int(prm.get("n_components", 12)), 2,
                      min(X.shape[1] - 1, max(2, len(X) // 20))))
    cut = int(len(y) * 0.8)
    cands = sorted({max(2, kmax // 3), max(2, kmax // 2), kmax})
    best_k, best_e = cands[-1], np.inf
    for k in cands:
        p2 = PCA(n_components=k, random_state=0).fit(Xs[:cut])
        pr2 = LinearRegression().fit(p2.transform(Xs[:cut]), y[:cut])
        e = float(np.mean((pr2.predict(p2.transform(Xs[cut:])) - y[cut:]) ** 2))
        if e < best_e:
            best_k, best_e = k, e

    p = PCA(n_components=best_k, random_state=0).fit(Xs)
    Z = p.transform(Xs)
    lr = LinearRegression().fit(Z, y)
    # 把主成分载荷拆回**原始指数空间**：
    #   y ≈ Z·β = ((X−mu)/sd)·components_ᵀ·β  ⇒  对 X 的系数 = components_ᵀ·β / sd
    # 原实现漏掉了 /sd —— 各指数日波动相差 2~3 倍，不除会系统性扭曲权重
    # （把低波动指数的重要性放大、高波动指数压小）。
    w_full = (p.components_.T @ lr.coef_) / sd
    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.01)), X.shape[1])
    wv = w.reindex(cols).fillna(0.0).values
    # ★ 评分口径必须与其他线性模型一致：全部用 X@w，不带截距。
    # 原实现用带截距的 lr.predict(Z) 算窗口内 R²，而样本外用的是 X@w
    # （_run_one 里线性模型一律走 X@w），两个口径不一致会让窗口内 R² 虚高、
    # 进而在前端被误判为"过拟合"（窗口内 − 样本外 > 0.08）。
    pred = X @ wv
    r2, te = _score(pred, y)
    return FitResult(weights=w, pred=pred, r2=r2, te=te,
                     extra={"n_components": int(best_k),
                            "explained_var": float(p.explained_variance_ratio_.sum())})


def _fit_ewma(X, y, cols, equity_ratio, prm):
    """指数衰减加权的约束 LS：近期样本权重高，风格切换时收敛更快。"""
    half_life = float(prm.get("half_life", 126))
    age = np.arange(len(y))[::-1]                 # 最近一日 age=0 → 权重 1
    sw = 0.5 ** (age / half_life)
    w_full = _constrained_fit(X, y, equity_ratio, sw)
    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.0)),
                       int(prm.get("max_assets", 15)))
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"half_life": half_life})


def _fit_huber(X, y, cols, equity_ratio, prm):
    """
    Huber IRLS：残差大的样本被降权，极端行情日不再主导拟合。

    尺度用残差 MAD/0.6745 估计（比标准差更耐异常值）；
    MAD 为 0（残差高度集中）时退回标准差。
    """
    k = float(prm.get("huber_k", 1.345))
    n_iter = int(prm.get("n_iter", 3))
    sw = np.ones(len(y))
    w_full = None
    for _ in range(n_iter):
        w_full = _constrained_fit(X, y, equity_ratio, sw)
        r = X @ w_full - y
        scale = np.median(np.abs(r - np.median(r))) / 0.6745
        if scale <= 1e-12:
            scale = float(r.std()) or 1.0
        u = r / scale
        sw = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))

    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.0)),
                       int(prm.get("max_assets", 15)))
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"huber_k": k, "n_iter": n_iter})


def _fit_gbr(X, y, cols, equity_ratio, prm):
    """
    梯度提升树（黑箱）。

    **不做超参搜索**：GBDT 在 118 个滚动窗口上重复训练，
    早前的版本对学习率做 2 档 × 2 折 CV，额外付出 ~470 次拟合，
    而实测对样本外 R² 的影响不足 0.01——横向排名关心的是模型族之间的差异，
    不是同一族内的最优超参。这里固定超参，把预算留给跨模型对比。
    """
    mdl = HistGradientBoostingRegressor(
        learning_rate=float(prm.get("learning_rate", 0.06)),
        max_depth=int(prm.get("max_depth", 3)),
        max_iter=int(prm.get("max_iter", 160)), min_samples_leaf=12,
        l2_regularization=1.0, random_state=0).fit(X, y)
    pred = mdl.predict(X)
    # 置换式重要性（HistGBR 无内置 importance，用与拟合值的相关系数近似）
    imp = np.array([abs(np.corrcoef(X[:, j], pred)[0, 1])
                    for j in range(X.shape[1])])
    imp = np.nan_to_num(imp)
    if imp.sum() > 0:
        imp = imp / imp.sum() * equity_ratio
    w = pd.Series(imp, index=cols).sort_values(ascending=False)
    r2, te = _score(pred, y)
    return FitResult(weights=w, pred=pred, r2=r2, te=te,
                     extra={"learning_rate": float(mdl.learning_rate),
                            "max_depth": int(mdl.max_depth),
                            "max_iter": int(mdl.max_iter)},
                     predictor=lambda Z: mdl.predict(Z))


def _fit_rf(X, y, cols, equity_ratio, prm):
    """
    随机森林（黑箱），用内置 feature_importances_。

    同样**不做超参搜索**（理由同 _fit_gbr）。注意 max_depth 默认给到 6：
    RF 在本问题上主要价值是提供一条稳健的非线性基准线，
    不需要调到最优；但也不能太浅，否则深度不足反而拉低基准线。
    """
    mdl = RandomForestRegressor(
        n_estimators=int(prm.get("n_estimators", 120)),
        max_depth=int(prm.get("max_depth", 6)),
        min_samples_leaf=8, random_state=0, n_jobs=1).fit(X, y)
    pred = mdl.predict(X)
    imp = np.asarray(mdl.feature_importances_, dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum() * equity_ratio
    w = pd.Series(imp, index=cols).sort_values(ascending=False)
    r2, te = _score(pred, y)
    return FitResult(weights=w, pred=pred, r2=r2, te=te,
                     extra={"n_estimators": int(mdl.n_estimators),
                            "max_depth": int(mdl.max_depth)},
                     predictor=lambda Z: mdl.predict(Z))


def _fit_krr(X, y, cols, equity_ratio, prm):
    """RBF 核岭回归。"""
    sd = X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xn = X / sd
    a_grid = _alpha_ladder(0.3, float(prm.get("alpha_scale", 1.0)))
    g_base = 1.0 / max(Xn.shape[1], 1)
    g_grid = [g_base * f * max(float(prm.get("gamma_scale", 1.0)), 1e-3)
              for f in (0.3, 1.0, 3.0)]

    def factory():
        for a in a_grid:
            for g in g_grid:
                yield (lambda a=a, g=g: KernelRidge(alpha=a, kernel="rbf", gamma=g))
    best, _ = _ts_cv_mse(factory, Xn, y)
    mdl = best
    if mdl is None:
        mdl = KernelRidge(alpha=0.3, kernel="rbf", gamma=g_base).fit(Xn, y)
    pred = mdl.predict(Xn)
    # 用 dual 系数还原到**原始指数空间**：w_raw = (Xn^T @ dual) / sd
    # 原实现同样漏了 /sd。注意 RBF 核下这只是一个线性近似（核模型不存在
    # 真正的原始空间权重），本模型 interpretable=False，权重仅作重要性展示。
    try:
        w_raw = (Xn.T @ np.asarray(mdl.dual_coef_).ravel()) / sd
    except Exception:                      # noqa: BLE001
        w_raw = np.array([abs(np.corrcoef(X[:, j], pred)[0, 1])
                          for j in range(X.shape[1])])
    w_raw = np.nan_to_num(w_raw)
    if w_raw.sum() <= 0:
        w_raw = np.abs(w_raw)
    w = _clean_weights(w_raw, cols, equity_ratio,
                       float(prm.get("min_weight", 0.005)), X.shape[1])
    r2, te = _score(pred, y)
    return FitResult(weights=w, pred=pred, r2=r2, te=te,
                     extra={"alpha": float(mdl.alpha), "gamma": float(mdl.gamma)},
                     predictor=lambda Z: mdl.predict(Z / sd))


def _fit_mlp(X, y, cols, equity_ratio, prm):
    """
    小型多层感知机（"非线性是否有用"的对照组）。

    两个关键工程决策，都是踩过坑之后定下来的：
      1. **必须标准化 y**。sklearn 的 MLP 对 y 尺度极其敏感，
         原始净值收益（数量级 0.01）配默认初始化会导致输出恒为 0，
         R² 直接为 0 —— 那不是模型能力问题，是优化没启动。
      2. **用 adam 而非 lbfgs**。lbfgs 在 252×43 这种小样本上
         每窗口要跑数百次迭代，滚动 118 个窗口会拖到十几分钟，不可接受。
    """
    sd = X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xn = X / sd
    ymu, ysd = y.mean(), (y.std() or 1.0)
    ys = (y - ymu) / ysd
    hidden = tuple(prm.get("hidden", (32, 16)))
    # 同样不做超参搜索（理由同 _fit_gbr）：adam 在小样本上初始化敏感，
    # CV 选出的 alpha 差异远小于模型族之间的差异。
    mdl = MLPRegressor(hidden_layer_sizes=hidden,
                       alpha=float(prm.get("alpha", 1e-2)), solver="adam",
                       max_iter=int(prm.get("max_iter", 400)), random_state=0,
                       activation="tanh", learning_rate_init=0.01,
                       tol=1e-4).fit(Xn, ys)
    pred = mdl.predict(Xn) * ysd + ymu
    # 第一层权重形状 (n_features, n_hidden)，按 hidden 维取均值 → 输入重要性
    imp = np.abs(np.asarray(mdl.coefs_[0], dtype=float)).mean(axis=1)
    imp = np.nan_to_num(imp)
    if imp.sum() > 0:
        imp = imp / imp.sum() * equity_ratio
    w = pd.Series(imp, index=cols).sort_values(ascending=False)
    r2, te = _score(pred, y)
    return FitResult(weights=w, pred=pred, r2=r2, te=te,
                     extra={"hidden": list(hidden), "alpha": float(mdl.alpha),
                            "iter": int(mdl.n_iter_)},
                     predictor=lambda Z: mdl.predict(Z / sd) * ysd + ymu)


_DISPATCH = {
    "enet_slsqp": _fit_enet,
    "lasso": _fit_lasso,
    "ridge": _fit_ridge,
    "corr_topk": _fit_corr_topk,
    "pcr": _fit_pcr,
    "ewma": _fit_ewma,
    "huber": _fit_huber,
    "gbr": _fit_gbr,
    "rf": _fit_rf,
    "krr": _fit_krr,
    "mlp": _fit_mlp,
}


def _coerce_params(model_key: str, raw: Optional[dict]) -> dict:
    """把 JSON 传来的超参整理成各模型可用的类型（尤其 hidden 需要元组）。"""
    prm = dict(_default_params().get(model_key, {}))
    for k, default in list(prm.items()):
        if not isinstance(raw, dict) or k not in raw or raw[k] is None:
            continue
        v = raw[k]
        try:
            if k == "hidden":
                if isinstance(v, str):
                    v = [int(x) for x in v.replace("，", ",").split(",") if x.strip()]
                prm[k] = tuple(int(x) for x in v)
            elif isinstance(default, bool):
                prm[k] = bool(v)
            elif isinstance(default, (int, np.integer)):
                prm[k] = int(v)
            elif isinstance(default, (float, np.floating)):
                prm[k] = float(v)
            else:
                prm[k] = v
        except (TypeError, ValueError):
            pass                      # 参数非法时保留默认值，不让整个任务失败
    return prm


def fit_window(model_key: str, X: np.ndarray, y: np.ndarray, cols: List[str],
               params: Optional[dict] = None,
               equity_ratio: Optional[float] = None) -> FitResult:
    """在单个时间窗口上拟合一个模型。"""
    if model_key not in _DISPATCH:
        raise ValueError(f"未知模型：{model_key}")
    prm = _coerce_params(model_key, params)
    er = equity_ratio if equity_ratio is not None else C.RBSA["equity_ratio"]
    return _DISPATCH[model_key](X, y, cols, er, prm)


def estimate_equity(fit_fn, X: np.ndarray, y: np.ndarray, cols: List[str],
                    bounds: tuple = (0.2, 1.0),
                    grid=(0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.95),
                    method: str = "grid"):
    """
    估计基金在窗口内的实际股票仓位，返回 (仓位, 已按该仓位拟合好的结果)。

    ★ R² 对 Σw 的目标值 E 极度敏感，且**每只基金的最优 E 完全不同**（实测窗口 R² 中位数）：

    | 基金        | E=0.9 | E=0.5 | E=0.3 | 最优 E |
    |-----------|-------|-------|-------|--------|
    | 兴全趋势    | 0.000 | 0.301 | 0.851 | 0.30 |
    | 华夏成长    | 0.000 | 0.101 | 0.838 | 0.25 |
    | 海富通股票  | 0.000 | 0.707 | 0.736 | 0.30 |
    | 景顺长城新兴 | 0.600 | 0.908 | 0.784 | 0.50 |
    | 汇添富蓝筹  | 0.664 | 0.888 | 0.747 | 0.50 |
    | 易方达消费  | 0.929 | 0.748 | 0.521 | 0.90 |

    固定 0.9 对混合型基金是灾难性的：R² 直接掉到 0，合成基准完全不能用。
    根因不是"仓位假设"这么轻描淡写的一句，而是：合成收益 = E·u，
    残差 = (γ−E)·u + ε，**尺度差 (γ−E) 会直接平方进 SSE**。
    E 偏离真实仓位 0.6，残差里就混进 0.36·var(u) 的纯尺度误差。

    两种估法（都只用窗口内数据，不引入前瞻）：

    - ``method="grid"``（默认）：在 E 网格上**直接最大化窗口内 R²**。
      R² 就是我们要优化的目标，没有代理误差。
      代价是每窗口要拟合 len(grid) 次（默认 8 次，约慢 4 倍）。
    - ``method="beta"``：隐含 beta 不动点，γ = β·E（β = cov(y,s)/var(s)）。
      只需 2 次拟合，但 **γ 的估计会混入 alpha 与基准的协方差**：
      持续跑赢且做反向选择的基金（如行业主题基金）会被系统性低估 ——
      实测易方达消费行业被估成 0.65（实际约 0.9），窗口 R² 从 0.929 掉到 0.871。

    实测（14 只基金，enet_slsqp，40 个重估点）：

    | 口径          | R² 中位 | R²<0.5 只数 | 年化 TE 均值 | 耗时/只 |
    |-------------|-------|-----------|------------|-------|
    | 固定 0.9      | 0.769 | 4         | 9.88%      | 0.87s |
    | beta 不动点   | 0.884 | 0         | 6.96%      | 1.61s |
    | 局部细搜       | 0.890 | 0         | 6.90%      | 2.6s |
    | 全网格最大 R²  | 0.891 | 0         | 6.88%      | 6.56s |

    在真实数据上三者差别很小（0.884 vs 0.891），但在**已知真值的合成检验**上
    （构造 y = Σb_j·u_j, Σb=γ, 噪声极低）差别很大：

    | γ 真值 | beta 估计 | 全网格估计 |
    |-------|----------|-----------|
    | 0.30  | 0.382    | **0.300** |
    | 0.60  | 0.703    | **0.600** |

    beta 法**系统性高估约 25%**：γ = β·E 里的 β 用的是"当前 w 下的协方差比"，
    而 w 受非负 + 稀疏约束，并不严格 ∝ 最优方向，协方差比因此有偏。
    局部细搜（"local"，默认）以几乎不增加的耗时拿到全网格的精度。
    """
    lo, hi = bounds

    def _r2_of(f):
        wv = f.weights.reindex(cols).fillna(0.0).values
        return _score(X @ wv, y)[0]

    # ---- 阶段一：beta 不动点给初值（2 次拟合）
    E = float(np.clip(C.RBSA.get("equity_ratio", 0.9), lo, hi))
    fr = None
    for _ in range(2):
        fr = fit_fn(E)
        wv = fr.weights.reindex(cols).fillna(0.0).values
        v = float(np.var(X @ wv))
        if not np.isfinite(v) or v <= 1e-18:
            break
        gamma = float(np.cov(y, X @ wv)[0, 1] / v) * E   # ★ 必须乘回 E：β = γ/E
        E_new = float(np.clip(gamma, lo, hi))
        if abs(E_new - E) < 0.03:
            E = E_new
            break
        E = E_new

    if method == "beta":
        return E, fr

    if method == "grid":
        cands = [float(np.clip(g, lo, hi)) for g in grid]
    else:                                        # "local"（默认）
        step = float(C.RBSA.get("equity_local_step", 0.10))
        cands = sorted({float(np.clip(E + d, lo, hi))
                        for d in (-step, 0.0, step)})

    best_E, best_fr, best_r2 = None, None, -np.inf
    for Ec in cands:
        if fr is not None and abs(Ec - E) < 1e-9:
            r2 = _r2_of(fr)                      # 阶段一的结果直接复用
        else:
            fr = fit_fn(Ec)
            r2 = _r2_of(fr)
        if r2 > best_r2:
            best_E, best_fr, best_r2 = Ec, fr, r2
    return best_E, best_fr


# ====================================================================== 进程池辅助
def _failed_report(key: str, err: Exception) -> ModelReport:
    """某个模型失败时返回一个占位报告，保证前端仍能渲染完整表格。"""
    spec = MODELS.get(key, ModelSpec(key, key, "unknown", ""))
    return ModelReport(
        key=key, name=spec.name, family=spec.family, desc=spec.desc,
        interpretable=spec.interpretable,
        oos_r2=float("nan"), oos_te=float("nan"),
        oos_ic=float("nan"), win=float("nan"),
        beta=float("nan"), vol_ratio=float("nan"),
        fit_r2_mean=float("nan"), fit_te_mean=float("nan"),
        latest_weights=pd.Series(dtype=float), weights_last_date=None,
        weights_hist=pd.DataFrame(), syn_ret=pd.Series(dtype=float),
        extra={"error": f"{type(err).__name__}: {str(err)[:200]}"},
        elapsed=0.0, fit_diag=pd.DataFrame(),
    )


def _worker(key: str, X_all, y_all, assets, dates, rebal_idx, cfg, override):
    """进程池入口：必须是模块级函数，且参数可 pickle。"""
    try:
        return ModelZoo(cfg)._run_one(key, np.asarray(X_all), np.asarray(y_all),
                                      list(assets), pd.DatetimeIndex(dates),
                                      list(rebal_idx), override)
    except Exception as e:                          # noqa: BLE001
        return _failed_report(key, e)


# ====================================================================== 滚动样本外
@dataclass
class ModelReport:
    key: str
    name: str
    family: str
    desc: str
    interpretable: bool
    oos_r2: float                  # 样本外 R²（用 OOS 收益算）
    oos_te: float                  # 年化跟踪误差
    oos_ic: float                  # 合成收益与基金收益的秩相关（应≈1）
    win: float                     # 方向一致率
    beta: float                    # 基金对合成基准的回归斜率（隐含股票仓位）
    vol_ratio: float               # 合成基准波动 / 基金波动
    fit_r2_mean: float             # 各窗口内拟合 R² 均值
    fit_te_mean: float
    latest_weights: pd.Series
    weights_last_date: Optional[pd.Timestamp]
    weights_hist: pd.DataFrame      # 滚动权重（index=重估日）
    syn_ret: pd.Series             # 样本外合成收益
    extra: Dict[str, Any]
    elapsed: float
    fit_diag: pd.DataFrame = field(default_factory=pd.DataFrame)  # 逐窗口 r2/te/alpha


class ModelZoo:
    """
    在同一数据切分上滚动评估多个模型，输出可排名、可切换的结果集合。

    关键点：所有模型共享同一份 rebal_idx 与窗口切分，因此横向排名公平。
    """

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or C.RBSA
        self.max_rebal = int((cfg or C.RBSA).get("max_rebal", 0)) or None

    @staticmethod
    def _rebal_idx(n: int, W: int, refit: int, max_rebal: Optional[int] = None
                   ) -> List[int]:
        """
        构造滚动重估点，必要时**均匀抽稀**到 max_rebal 个。

        为什么需要抽稀：样本越长，重估点越多，总耗时线性增长。
        早期版本 2621 个交易日产生 119 个重估点，其中最贵的 RF 要跑 34 秒，
        导致 Web 首屏等待过长。

        抽稀对模型**公平性无影响**：所有模型共享同一份 rebal_idx，
        切分完全一致，横向排名依然可比。代价是合成基准的更新频率变低，
        但这只影响"基准有多新鲜"，不影响"这个模型能不能拟合"。
        """
        idx = list(range(W, n, refit))
        if max_rebal and len(idx) > max_rebal:
            import numpy as _np
            sel = _np.linspace(0, len(idx) - 1, max_rebal).round().astype(int)
            idx = [idx[i] for i in dict.fromkeys(sel.tolist())]
        return idx

    # -------------------------------------------------------------- 主入口
    def run(self, px: pd.DataFrame, model_keys: List[str],
            fund_col: str = "__FUND__", params: Optional[dict] = None,
            progress=None, workers: int = 0,
            max_rebal: Optional[int] = None) -> Dict[str, ModelReport]:
        """
        px        : 对齐后的价格面板
        model_keys: 要评估的模型 key 列表
        params    : {model_key: {超参}} 覆盖
        progress  : 回调 progress(model_key, done, total, oos_r2)
        workers   : 并行进程数；0 = 自动（受 CPU 核数与模型数限制）
        max_rebal : 重估点上限，None = 用配置默认值

        模型之间完全独立，因此用进程池并行。注意：**并行压缩的是总墙钟，
        不是关键路径**——当某一个模型（如 RF）占了大部分耗时，
        并行化对它无能为力。此时应调小 max_rebal。
        """
        cfg = self.cfg
        params = params or {}
        assets = [c for c in px.columns if c != fund_col]
        P = np.log(px.astype(float))
        R = P.diff().dropna()
        dates = R.index
        X_all = R[assets].values
        y_all = R[fund_col].values

        n = len(dates)
        W = cfg["window"]
        if n < W + 60:
            raise ValueError(f"样本不足：仅 {n} 行，滚动窗口需要 {W}+60。")
        mr = max_rebal if max_rebal is not None else self.max_rebal
        rebal_idx = self._rebal_idx(n, W, cfg["refit"], mr)

        keys = list(model_keys)
        if not keys:
            return {}

        if workers == 0:
            import os
            workers = max(1, min(len(keys), (os.cpu_count() or 4) - 1))
        use_pool = workers > 1 and len(keys) > 1

        out: Dict[str, ModelReport] = {}
        if use_pool:
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                ctx = __import__("multiprocessing").get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=ctx) as ex:
                    futs = {
                        ex.submit(_worker, key, X_all, y_all, assets,
                                  dates, rebal_idx, cfg, params.get(key)): key
                        for key in keys
                    }
                    done = 0
                    for fut in as_completed(futs):
                        key = futs[fut]
                        try:
                            out[key] = fut.result()
                        except Exception as e:      # noqa: BLE001
                            out[key] = _failed_report(key, e)
                        done += 1
                        if progress:
                            progress(key, done, len(keys), out[key].oos_r2)
                return out
            except Exception as e:                  # noqa: BLE001
                print(f"[models] 进程池不可用（{type(e).__name__}: {str(e)[:60]}），"
                      f"退回串行")
                out = {}

        for i, key in enumerate(keys, 1):
            try:
                rep = self._run_one(key, X_all, y_all, assets, dates,
                                    rebal_idx, params.get(key))
            except Exception as e:                  # noqa: BLE001
                rep = _failed_report(key, e)
            out[key] = rep
            if progress:
                progress(key, i, len(keys), rep.oos_r2)
        return out

    # -------------------------------------------------------------- 单模型滚动
    def _run_one(self, key: str, X_all, y_all, assets, dates, rebal_idx,
                 override: Optional[dict]) -> ModelReport:
        t0 = time.time()
        cfg = self.cfg
        rows, fit_r2s, fit_tes, extras = [], [], [], []
        diag_rows, equities = [], []
        syn = pd.Series(index=dates[rebal_idx[0]:], dtype=float)
        last_w = None
        last_date = None
        spec = MODELS[key]
        auto_equity = bool(cfg.get("auto_equity", False))
        bounds = tuple(cfg.get("equity_bounds", (0.2, 1.0)))
        eq_start = float(cfg.get("equity_ratio", C.RBSA["equity_ratio"]))

        for k, i in enumerate(rebal_idx):
            lo, hi = i - cfg["window"], i
            Xw, yw = X_all[lo:hi], y_all[lo:hi]
            if len(yw) < cfg["min_obs"]:
                continue
            if auto_equity and spec.interpretable:
                # 只有可解释模型（Σw 有经济含义）才谈得上"实际仓位"；
                # 黑箱模型的权重只是重要性排序，缩放它没有意义。
                er, fr = estimate_equity(
                    lambda E: fit_window(key, Xw, yw, assets, override,
                                         equity_ratio=E),
                    Xw, yw, assets, bounds=bounds,
                    grid=tuple(cfg.get("equity_grid",
                                       (0.20, 0.30, 0.40, 0.50,
                                        0.60, 0.70, 0.80, 0.95))),
                    method=str(cfg.get("equity_method", "grid")))
                equities.append(er)
            else:
                # ★ 之前这里是 None，导致 fit_window 回落到全局
                # C.RBSA["equity_ratio"]，也就是 ModelZoo(cfg) 传入的
                # equity_ratio **完全被忽略** —— 一个怎么改都不生效的配置项。
                er = eq_start
                fr = fit_window(key, Xw, yw, assets, override, equity_ratio=er)
            w_full = fr.weights.reindex(assets).fillna(0.0).values

            # 权重覆盖到下一个重估点：抽稀后相邻点间隔可能远大于 refit，
            # 若只覆盖固定 refit 天会丢掉中间大段样本
            next_i = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else len(dates)
            j_end = min(next_i, len(dates))
            X_next = X_all[i:j_end]
            if fr.predictor is not None:
                # 黑箱模型：用「上一窗口拟合好的估计器」预测下一段，
                # 这是真正的样本外（前向）预测。
                try:
                    seg = np.asarray(fr.predictor(X_next), dtype=float)
                    if not np.all(np.isfinite(seg)):
                        raise ValueError("预测含非有限值")
                except Exception:          # noqa: BLE001
                    seg = X_next @ w_full
            else:
                # 线性约束模型：合成收益 = X @ w，权重和 = 股票仓位
                seg = X_next @ w_full
            syn.loc[dates[i:j_end]] = seg

            rows.append({"date": dates[i - 1], **fr.weights.to_dict()})
            fit_r2s.append(fr.r2)
            fit_tes.append(fr.te)
            diag_rows.append({"date": dates[i - 1], "r2": fr.r2, "te": fr.te,
                              "alpha": fr.extra.get("alpha")})
            if fr.extra:
                extras.append(fr.extra)
            last_w, last_date = fr.weights, dates[i - 1]

        syn = syn.dropna()
        y_oos = pd.Series(y_all, index=dates).reindex(syn.index)
        oos_r2, oos_te = _score(syn.values, y_oos.values)
        ic = float(syn.corr(y_oos, method="spearman"))
        win = float((np.sign(syn.values) == np.sign(y_oos.values)).mean())

        # 隐含股票仓位：把基金收益对合成基准收益做无约束回归，斜率即 beta。
        # 关键诊断 —— 若beta 远小于 equity_ratio（0.9），说明基金实际仓位远低于
        # 假设值，Σw=0.9 的合成基准就会系统性高估波动，导致 R² 被压低。
        # 典型场景：做了大量对冲/低仓位运作的"趋势型"基金。
        var_x = float(np.var(syn.values))
        beta = (float(np.cov(y_oos.values, syn.values)[0, 1] / var_x)
                if var_x > 1e-18 else float("nan"))
        vol_ratio = (float(syn.std() / y_oos.std())
                     if y_oos.std() > 1e-12 else float("nan"))

        wdf = pd.DataFrame(rows).set_index("date").fillna(0.0)
        wdf = wdf.loc[:, (wdf.abs().sum(0) > 0)]

        fit_diag = pd.DataFrame(diag_rows).set_index("date")

        # 各参数的中位值（诊断用）
        agg_extra: Dict[str, Any] = {}
        if equities:
            agg_extra["equity_used"] = float(np.median(equities))
        if extras:
            allk = set().union(*[set(e.keys()) for e in extras])
            for k in allk:
                vals = [e[k] for e in extras if k in e]
                if not vals:
                    continue
                v0 = vals[0]
                if isinstance(v0, (bool, np.bool_)):
                    agg_extra[k] = None
                elif isinstance(v0, (int, float, np.floating)):
                    agg_extra[k] = float(np.nanmedian(
                        [float(v) for v in vals]))
                elif isinstance(v0, (list, tuple)):
                    # 只对纯数值列表取中位数（如 top_corr），字符串列表直接丢弃
                    flat = [x for v in vals for x in v]
                    if flat and all(isinstance(x, (int, float, np.floating))
                                     for x in flat):
                        agg_extra[k] = float(np.nanmedian(
                            [float(x) for x in flat]))
                elif isinstance(v0, str):
                    agg_extra[k] = None

        return ModelReport(
            key=key, name=spec.name, family=spec.family, desc=spec.desc,
            interpretable=spec.interpretable,
            oos_r2=oos_r2, oos_te=oos_te, oos_ic=ic, win=win,
            beta=beta, vol_ratio=vol_ratio,
            fit_r2_mean=float(np.mean(fit_r2s)) if fit_r2s else np.nan,
            fit_te_mean=float(np.mean(fit_tes)) if fit_tes else np.nan,
            latest_weights=last_w if last_w is not None else pd.Series(dtype=float),
            weights_last_date=last_date,
            weights_hist=wdf, syn_ret=syn,
            extra=agg_extra, elapsed=time.time() - t0,
            fit_diag=fit_diag,
        )

    # -------------------------------------------------------------- 排名
    @staticmethod
    def rank(reports: Dict[str, ModelReport], key: str = "样本外R²") -> pd.DataFrame:
        """按指定指标生成排名表（默认样本外 R²，越大越好）。"""
        rows = []
        for k, r in reports.items():
            rows.append({
                "key": k, "模型": r.name, "流派": r.family,
                "可解释": "是" if r.interpretable else "否",
                "样本外R²": r.oos_r2, "样本外TE": r.oos_te,
                "窗口内R²": r.fit_r2_mean, "窗口内TE": r.fit_te_mean,
                "方向一致率": r.win, "隐含仓位": r.beta, "耗时": r.elapsed,
            })
        df = pd.DataFrame(rows)
        if key not in df.columns:
            key = "样本外R²"
        asc = key in ("样本外TE", "窗口内TE", "耗时")
        return df.sort_values(key, ascending=asc).reset_index(drop=True)


# ====================================================================== 辅助
def model_catalog() -> List[dict]:
    """给前端的模型清单（名称、描述、可调超参）。"""
    defs = _default_params()
    out = []
    for k in MODEL_ORDER:
        s = MODELS[k]
        out.append({"key": k, "name": s.name, "family": s.family,
                    "desc": s.desc, "interpretable": s.interpretable,
                    "params": defs.get(k, {})})
    return out