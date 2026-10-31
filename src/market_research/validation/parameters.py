"""Web 请求的模型参数校验；不依赖 Flask 或任务调度。"""
import market_research.config as C
from market_research.modeling.catalog import MODEL_ORDER, _default_params
from market_research.validation.window import parse_window as _parse_window

MAX_REBAL_HARD = 2000
_INF = float("inf")


def _finite(x) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (_INF, -_INF)


def _parse_max_rebal(raw, default=None):
    """
    重估点上限。三态必须分清（P1-06）：

      · 键缺失 / None / ""  → default（"用配置默认值"，当前为 60）
      · **0**（或 "0"）     → **不抽稀**，对应 UI 的「全部 · 最精确」
      · 正整数              → 该值（≤ MAX_REBAL_HARD）

    旧实现是 `None if mr in (None, "", 0) else int(mr)`：0 被折叠成 None，
    于是 UI 上选「全部 · 最精确」实际只跑 60 个重估点，与界面承诺不符；
    而传 "abc" 会让 int() 抛异常，直接变成 HTTP 500。
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    if isinstance(raw, bool):
        raise ValueError("max_rebal 必须是非负整数")
    if isinstance(raw, int):
        v = int(raw)
    elif isinstance(raw, float):
        if not float(raw).is_integer():
            raise ValueError(f"max_rebal 必须是整数，收到 {raw!r}")
        v = int(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s.lstrip("+-").isdigit():
            raise ValueError(f"max_rebal 必须是非负整数，收到 {raw!r}")
        v = int(s)
    else:
        raise ValueError(f"max_rebal 类型不支持：{type(raw).__name__}")
    if v < 0:
        raise ValueError("max_rebal 不能为负")
    if v > MAX_REBAL_HARD:
        raise ValueError(f"max_rebal 过大（硬上限 {MAX_REBAL_HARD}）")
    return v


def _parse_auto_equity(body: dict) -> bool:
    """
    auto_equity 缺省时**跟随核心配置**（C.RBSA['auto_equity'] 现为 True），
    而不是硬编码 False —— 旧实现让 API、核心配置与 UI 默认值三者不一致。
    """
    if "auto_equity" not in body or body.get("auto_equity") is None:
        return bool(C.RBSA.get("auto_equity", False))
    v = body.get("auto_equity")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"auto_equity 必须是布尔值，收到 {v!r}")


def _check_params(params):
    """按公开模型目录校验参数，拒绝被估计器静默截断或忽略的输入。"""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("params 必须是对象")
    defaults = _default_params()
    integers = {'max_assets', 'top_k', 'n_components', 'half_life',
                'n_iter', 'max_depth', 'max_iter', 'n_estimators'}
    for m, p in params.items():
        if m not in MODEL_ORDER:
            raise ValueError(f"未知模型：{m}")
        if not isinstance(p, dict):
            raise ValueError(f"params[{m}] 必须是对象")
        for k, v in p.items():
            label = f"params[{m}][{k}]"
            if k not in defaults[m]:
                raise ValueError(f"{label} 是未知参数")
            if k == 'hidden':
                if (not isinstance(v, (list, tuple)) or not v
                        or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in v)):
                    raise ValueError(f"{label} 必须是非空的正整数层宽列表")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{label} 必须是数值")
            if not _finite(v):
                raise ValueError(f"{label} 必须是有限数值")
            if k in integers and (v <= 0 or int(v) != v):
                raise ValueError(f"{label} 必须是正整数")
            if k == 'l1_ratio' and not 0 <= v <= 1:
                raise ValueError(f"{label} 必须在 0 到 1 之间")
            if k == 'min_weight' and not 0 <= v < 1:
                raise ValueError(f"{label} 必须大于等于 0 且小于 1")
            if k == 'alpha' and m == 'mlp' and v == 0:
                continue
            if k not in integers | {'l1_ratio', 'min_weight'} and v <= 0:
                raise ValueError(f"{label} 必须大于 0")
    return params
