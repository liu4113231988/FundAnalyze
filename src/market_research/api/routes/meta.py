"""Health and catalogue routes."""
from fastapi import APIRouter
import market_research.config as C
from market_research.modeling.catalog import MODEL_ORDER, model_catalog
from market_research.market.analysis import STOCKS, INDICES
router=APIRouter()

@router.get('/api/health')
def health():
    return {"ok":True,"version":"0.3.0","framework":"fastapi","models":len(MODEL_ORDER)}

@router.get('/api/catalog')
def catalog():
    known = []
    for d in (C.DEMO_FUNDS, C.EXTENDED_FUNDS):
        for k, v in d.items():
            if not any(x["code"] == k for x in known):
                known.append({"code": k, "name": v})
    return {
        "assets": {"stocks": [{"code": k, "name": v} for k, v in STOCKS.items()],
                   "indices": [{"code": k, "name": v} for k, v in INDICES.items()]},
        "models": model_catalog(),
        "default_models": ["enet_slsqp", "lasso", "ridge", "corr_topk", "pcr"],
        "quick_models": ["enet_slsqp", "corr_topk", "pcr", "rf", "krr"],
        "full_models": MODEL_ORDER,
        "known_funds": known,
        "candidate_pool": {
            "宽基": C.BROAD, "市值风格": C.SIZE, "价值成长": C.STYLE,
            "行业": len(C.SW_INDUSTRY),
        },
        "rbsa": {"window": C.RBSA["window"], "refit": C.RBSA["refit"],
                 "equity_ratio": C.RBSA["equity_ratio"],
                 "min_weight": C.RBSA["min_weight"],
                 "max_rebal": C.RBSA.get("max_rebal"),
                 "auto_equity": C.RBSA.get("auto_equity", False),
                 "equity_bounds": C.RBSA.get("equity_bounds", (0.5, 1.0))},
    }
