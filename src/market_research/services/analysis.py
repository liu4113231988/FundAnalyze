"""Application orchestration, shared by HTTP adapters and worker processes."""
import hashlib
import json
import logging
import pickle

import market_research.config as C
import market_research.data.marketdata as D
import market_research.tasks.sync as sync
from market_research.tasks.jobs import JOBS
from market_research.market.analysis import analyze_market
CACHE_DIR = C.DATA_DIR / "webcache"

def analyze(**arguments):
    from market_research.funds.analysis import analyze as calculate
    return calculate(**arguments)

def backtest_fund(**arguments):
    from market_research.funds.analysis import backtest_fund as calculate
    return calculate(**arguments)

def refit_single(**arguments):
    from market_research.funds.analysis import refit_single as calculate
    return calculate(**arguments)

from market_research.data.files import _atomic_pickle_dump

LOG = logging.getLogger(__name__)
RUNNER = JOBS


def result_cache_path(code, models, params, max_rebal=None, auto_equity=False, data_stamp="", start="", end=""):
    key = json.dumps({"v":C.CACHE_VERSION,"c":code,"m":sorted(models or []),"p":params or {},
                      "r":C.RBSA.get("max_rebal") if max_rebal is None else max_rebal,
                      "a":bool(auto_equity),"d":data_stamp,"s":start or "","e":end or ""},sort_keys=True,ensure_ascii=False)
    return CACHE_DIR / f"result_v{C.CACHE_VERSION}_{code}_{hashlib.md5(key.encode()).hexdigest()[:10]}.pkl"


def fund_cache_path(params):
    return result_cache_path(params["code"],params["models"],params["params"],params.get("max_rebal"),params.get("auto_equity",False),D.data_stamp(exclude_stocks=True,fund_code=params["code"]),params.get("start"),params.get("end"))


def submit(operation, params):
    if operation == "analyze":
        sync.register("fund", params["code"], D.NAV_BASIS_TOTAL)
        path = fund_cache_path(params)
        if not params.get("refresh") and path.exists():
            try:
                with path.open("rb") as file:
                    payload=pickle.load(file)
                payload.setdefault("meta",{})["from_cache"]=True
                return {"ok":True,"result":payload,"cached":True}
            except Exception:
                LOG.exception("结果缓存读取失败，将重新分析")
        identity = path.name
    else:
        identity = operation
    key=json.dumps([identity,params],sort_keys=True,ensure_ascii=False)
    models=params.get("models") or [params.get("model") or params.get("kind",operation)]
    jid,merged=RUNNER.submit(params["code"],models,operation,params,key=key)
    return {"ok":True,"job_id":jid,"cached":False,"merged":merged}


def execute(operation, params, jid):
    if operation == "analyze":
        initial_path = fund_cache_path(params)
        result=analyze(**params,job_id=jid)
        try:
            if fund_cache_path(params) == initial_path:
                _atomic_pickle_dump(result,initial_path)
            else:
                LOG.info("分析期间行情发生变化，跳过结果缓存写入")
        except Exception:
            LOG.exception("分析完成，但缓存写入失败")
        return result
    if operation == "market":
        return analyze_market(**params,job_id=jid)
    if operation == "refit":
        JOBS.update(jid,stage="重新学习模型",progress=.2)
        arguments=dict(params)
        arguments["model_key"]=arguments.pop("model")
        return refit_single(**arguments)
    if operation == "backtest":
        arguments=dict(params)
        arguments["model_key"]=arguments.pop("model")
        arguments["start_date"]=arguments.pop("start",None)
        arguments["end_date"]=arguments.pop("end",None)
        return backtest_fund(**arguments,job_id=jid)
    raise ValueError("未知任务操作")
