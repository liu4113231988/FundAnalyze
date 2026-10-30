from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
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
