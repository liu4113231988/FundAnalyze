from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import market_research.config as C
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from market_research.modeling.catalog import _default_params
def _constrained_fit(X: np.ndarray, y: np.ndarray, total: float,
                     sw: Optional[np.ndarray] = None,
                     status: Optional[dict] = None) -> np.ndarray:
    """
    min Σ sw_i(Xw-y)²  s.t. w>=0, Σw=total（SLSQP 精确解）。

    status 为可选的可变字典：传入后会把 `ok`（是否收敛）与 `reason`（失败原因）
    写进去，供调用方把"降级"如实上报（P0-05）。不传则行为与旧版完全一致。
    """
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
    if status is not None:
        status["ok"] = bool(res.success) and bool(np.all(np.isfinite(x)))
        status["reason"] = ("" if status["ok"]
                            else (f"SLSQP 未收敛({getattr(res, 'message', '')})"
                                  if not res.success else "解含非有限值"))
    if not res.success or not np.all(np.isfinite(x)):
        if status is not None:
            status["degraded"] = "equal_weight"
        return np.full(k, total / k)
    w = np.clip(x, 0.0, None)
    if w.sum() > 0:
        w = w * total / w.sum()
    else:
        if status is not None:
            status["degraded"] = "equal_weight"
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


def _ts_cv_mse(make_model, Xs: np.ndarray, ys: np.ndarray, n_splits: int = 2,
               transform=None):
    """
    时序交叉验证选超参，返回 **(在全样本上重新拟合好的模型实例, MSE)**。

    注意：候选超参由 `make_model()` 生成为**工厂**（无参 lambda），
    因为 sklearn 的 clone 语义要求超参在构造时固化。

    n_splits 默认 2 而非 4：滚动评估会在 ~120 个窗口上反复调用本函数，
    折数直接决定总耗时（4 折 × 3 档 = 12 次拟合/窗口）。
    2 折在正则强度这种量级选择上已经足够，且耗时减半。

    transform(X_tr, y_tr, X_te, y_te) → (X_tr', y_tr', X_te', y_te')
    ---------------------------------------------------------------
    ★ P1-02：标准化/PCA/特征选择这类**预处理必须只在同一折的训练段上拟合**，
    再用它变换该折的验证段。旧实现先用**整个滚动窗口**算出 mu/sd 再切折，
    于是内层验证段的均值和方差被提前写进了预处理 —— 它不等于使用了外层的
    未来收益，但会让内层选参偏乐观，且这个偏差无从察觉。
    传 None 则保持旧行为（调用方若不需要预处理）。
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
                if transform is not None:
                    Xtr, ytr, Xte, yte = transform(Xs[tr], ys[tr], Xs[te], ys[te])
                else:
                    Xtr, ytr, Xte, yte = Xs[tr], ys[tr], Xs[te], ys[te]
                m.fit(Xtr, ytr)
                errs.append(float(np.mean((m.predict(Xte) - yte) ** 2)))
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
    """
    (R², 年化 TE)。

    ★ R² **保留原始值，包括负数**（P0-05）。旧实现用 max(0.0, ·) 把负值截到 0，
    效果是把"模型彻底失败"伪装成"模型恰好没有解释力"——真实 R² = −3 会显示成 0。
    负 R² 恰恰是识别失效窗口最直接的诊断量，截断等于销毁证据。

    y 为常数（sst ≈ 0）时 R² 在数学上无定义，返回 NaN 而不是 0 或 −∞：
    NaN 会被中位数/均值自动跳过，也不会被误当成"极差的拟合"。
    """
    resid = np.asarray(pred, dtype=float) - y
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    te = float(np.std(resid, ddof=1) * np.sqrt(252)) if len(resid) > 1 else np.nan
    if not np.isfinite(sst) or sst <= 1e-18:
        return float("nan"), te          # 目标无变异 → R² 未定义
    return 1.0 - sse / sst, te


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

    # ★ P1-02：内层 CV 的标准化必须**只用该折的训练段**拟合，
    #   再用它变换同一折的验证段。上面 Xs/ys 的全窗口标准化只用于
    #   "最后在全样本上重新拟合"那一步，不参与选参。
    def _fold_standardize(Xtr, ytr, Xte, yte):
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        ym, ysd = ytr.mean(), (ytr.std() or 1.0)
        return ((Xtr - mu) / sd, (ytr - ym) / ysd,
                (Xte - mu) / sd, (yte - ym) / ysd)

    mdl, _ = _ts_cv_mse(factory, Xs, ys, transform=_fold_standardize)
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

    st: Dict[str, Any] = {}
    w_s = _constrained_fit(X[:, sup], y, equity_ratio, status=st)
    w_full = np.zeros(X.shape[1])
    w_full[sup] = w_s
    w = _clean_weights(w_full, cols, equity_ratio,
                       float(prm.get("min_weight", 0.01)),
                       int(prm.get("max_assets", 8)))
    wv = w.reindex(cols).fillna(0.0).values
    r2, te = _score(X @ wv, y)
    return FitResult(weights=w, pred=X @ wv, r2=r2, te=te,
                     extra={"alpha": float(mdl.alpha), "l1_ratio": l1,
                            "n_support": int(sup.size),
                            # ★ 优化失败会静默退回等权（P0-05），这里如实上报：
                            #   1 = SLSQP 收敛，0 = 已降级。聚合后是"收敛率"。
                            "fit_ok": 1.0 if st.get("ok", True) else 0.0})


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
        v = _score(X @ wv, y)[0]
        # ★ P0-05 的连带修复：R² 不再被截到 0 之后，它会是 NaN（目标无变异）
        #   或 −inf。若不归一成一个**有限**的极小值，`r2 > best_r2` 对 NaN
        #   恒为 False，于是所有候选都选不中 → 返回 (None, None)，
        #   自适应仓位（默认开启）会拿到 None 并一路污染下去。
        return -1e9 if not np.isfinite(v) else float(v)

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
    if best_E is None:                       # 兜底：至少退回阶段一的 beta 不动点
        return E, fr
    return best_E, best_fr
