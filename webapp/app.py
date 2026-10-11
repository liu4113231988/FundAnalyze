# -*- coding: utf-8 -*-
"""
Fund Lens · Web 服务入口

启动：
    python webapp/app.py            # 默认 http://127.0.0.1:8760
    python webapp/app.py --port 9000 --host 0.0.0.0

设计
----
* 前端是单页应用（SPA），所有交互通过 /api/* 完成。
* 分析是分钟级操作，因此一律走「异步任务 + 轮询进度」：
      POST /api/analyze      → 立刻返回 job_id
      GET  /api/job/<id>     → 轮询状态与进度
      GET  /api/result/<id>  → 取完整结果
* 结果按 (基金, 模型集合, 参数) 落盘缓存，命中则秒回。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings                                            # noqa: E402
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS                                # noqa: E402
import pandas as pd                                        # noqa: E402

from src import config as C                                # noqa: E402
from src import data as D                                  # noqa: E402
from src.models import MODEL_ORDER, model_catalog         # noqa: E402
from src.webapi import (CACHE_DIR, JOBS, analyze, backtest_fund,  # noqa: E402
                         refit_single, _parse_max_rebal,
                         _parse_auto_equity, _check_params,
                         _finite)

STATIC = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
try:
    CORS(app)
except Exception:                                         # noqa: BLE001
    pass

# 允许并行任务（每个任务内部已用进程池）
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# ====================================================================== 静态资源
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


# 注意：前端引用路径是 /static/app.js，而 STATIC 本身已指向 webapp/static，
# 因此这里直接以 STATIC 为根，不要再拼一层 "static"。
@app.route("/static/<path:fn>")
def static_files(fn: str):
    return send_from_directory(STATIC, fn)


@app.route("/vendor/<path:fn>")
def vendor_files(fn: str):
    return send_from_directory(STATIC / "vendor", fn)


# ====================================================================== 元信息
@app.route("/api/health")
def health():
    return jsonify({"ok": True, "version": "1.0",
                    "models": len(MODEL_ORDER)})


@app.route("/api/catalog")
def catalog():
    """模型清单 + 可调超参 + 已知基金列表。"""
    known = []
    for d in (C.DEMO_FUNDS, C.EXTENDED_FUNDS):
        for k, v in d.items():
            if not any(x["code"] == k for x in known):
                known.append({"code": k, "name": v})
    return jsonify({
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
    })


# ====================================================================== 分析任务
def _result_cache_path(code: str, models, params, max_rebal=None,
                       auto_equity=False, data_stamp: str = ""):
    import hashlib
    # ★ P1-05：缓存键必须绑定**底层数据摘要**，否则行情追加后旧结果不会失效，
    #   表现为"刷新无效"。data_stamp 由 src.data.data_stamp() 给出。
    key = json.dumps({"v": C.CACHE_VERSION, "c": code,
                      "m": sorted(models or []),
                      "p": params or {}, "r": max_rebal or 0,
                      "a": bool(auto_equity), "d": data_stamp},
                     sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    # 版本号写进文件名，便于启动时清扫旧版本缓存
    return CACHE_DIR / f"result_v{C.CACHE_VERSION}_{code}_{h}.pkl"


# ------------------------------------------------------------------ 参数校验（P1-06）
# 硬上限：再大只是把请求拖到超时，没有任何额外精度收益
MAX_REBAL_HARD = 2000
_INF = float("inf")

# 并发闸门（P1-07）：分析任务内部还会开进程池，不设上限会把 CPU/内存打满。
# 单实例即可，不需要引入分布式系统；超限时返回 429 而不是排队到超时。
MAX_CONCURRENT_JOBS = int(os.environ.get("FIL_MAX_JOBS", "2"))
_JOB_SEM = threading.BoundedSemaphore(max(1, MAX_CONCURRENT_JOBS))


def _bad(field: str, msg: str, status: int = 400):
    """结构化 4xx。非法输入必须停在 4xx —— 旧实现里 int("abc") 直接抛成 500。"""
    return jsonify({"ok": False, "error": msg, "field": field}), status


# ------------------------------------------------------------------ 参数解析
# _finite / _parse_max_rebal / _parse_auto_equity / _check_params 定义在
# src/webapi.py —— 放在那里是为了让离线回归测试能直接导入它们
# （导入本模块会触发 Flask 初始化与旧缓存清扫，不适合在测试里做）。


def _atomic_pickle_dump(obj, path: Path, retries: int = 6):
    """
    原子写盘：先写同目录临时文件再 os.replace。

    为什么必须这样：结果缓存路径只由 (基金, 模型集, 参数) 决定，而任务 id 带时间戳，
    所以**两个相同的并发请求会持有不同的 job_id 却写同一个文件**。
    直接 open(path,'wb') 会让两个线程交错写入，产生截断/半截的 pickle，
    而读取侧只是 try/except 吞掉异常 → 表现为"缓存偶尔失效、偶尔白屏"。

    两个踩过的坑：
      1. 临时文件名必须**每次调用唯一**。只加 pid 是不够的——Flask 是多线程，
         同进程内的并发请求会撞同一个 tmp 名，互相覆盖。
      2. Windows 下 os.replace 在目标文件被另一线程短暂占用时会抛 WinError 32，
         必须重试；重试仍失败就放弃写缓存（缓存本身是 best-effort，不该让请求失败）。
    """
    import os
    import threading
    import time
    import uuid

    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump(obj, f)
        for i in range(retries):
            try:
                os.replace(tmp, path)
                return
            except OSError:
                if i == retries - 1:
                    raise
                time.sleep(0.05 * (i + 1))
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _sweep_old_caches():
    """删除非当前版本的结果缓存，避免旧口径文件无限堆积。"""
    prefix = f"result_v{C.CACHE_VERSION}_"
    for p in CACHE_DIR.glob("result_*.pkl"):
        if not p.name.startswith(prefix):
            try:
                p.unlink()
            except OSError:
                pass


_sweep_old_caches()


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True) or {}
    code = str(body.get("code", "")).strip()
    models = body.get("models") or MODEL_ORDER
    params = body.get("params") or {}
    refresh = bool(body.get("refresh", False))
    # 重估点上限：0 = 不抽稀（「全部 · 最精确」），省略 = 用配置默认 60。
    try:
        max_rebal = _parse_max_rebal(body.get("max_rebal", None), default=None)
        auto_equity = _parse_auto_equity(body)
        params = _check_params(params)
    except ValueError as e:
        return _bad("params", str(e))

    if not (len(code) == 6 and code.isdigit()):
        return _bad("code", "请输入 6 位基金代码，例如 110022")
    if not isinstance(models, (list, tuple)):
        return _bad("models", "models 必须是模型名列表")
    bad = [m for m in models if m not in MODEL_ORDER]
    if bad:
        return _bad("models", f"未知模型：{','.join(map(str, bad))}")

    # ---- 并发闸门（P1-07）：每个任务内部还会开进程池，不设上限会把 CPU 打满
    if not _JOB_SEM.acquire(blocking=False):
        return _bad("concurrency",
                    f"同时运行的分析任务已达上限 {MAX_CONCURRENT_JOBS}，请稍后重试",
                    429)

    # ---- 缓存命中：直接返回
    try:
        stamp = D.data_stamp()
    except Exception:                             # noqa: BLE001
        stamp = "unknown"
    cp = _result_cache_path(code, models, params, max_rebal, auto_equity, stamp)
    if cp.exists() and not refresh:
        try:
            with cp.open("rb") as f:
                payload = pickle.load(f)
            payload.setdefault("meta", {})["from_cache"] = True
            return jsonify({"ok": True, "result": payload, "cached": True})
        except Exception:                             # noqa: BLE001
            pass

    jid = JOBS.create(code, models)

    def worker():
        try:
            JOBS.update(jid, stage="开始", progress=0.02)
            res = analyze(code, models=models, params=params,
                          refresh=refresh, job_id=jid, max_rebal=max_rebal,
                          auto_equity=auto_equity)
            try:
                _atomic_pickle_dump(res, cp)
            except Exception:                         # noqa: BLE001
                pass      # 缓存写失败不影响本次返回结果，下次重算即可
            JOBS.finish(jid, res)
        except Exception as e:                        # noqa: BLE001
            traceback.print_exc()
            JOBS.fail(jid, f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            _JOB_SEM.release()

    threading.Thread(target=worker, name=f"job-{jid}",
                     daemon=True).start()
    return jsonify({"ok": True, "job_id": jid, "cached": False})


@app.route("/api/job/<jid>")
def api_job(jid: str):
    j = JOBS.public(jid)
    if not j:
        return jsonify({"error": "任务不存在或已过期"}), 404
    return jsonify(j)


@app.route("/api/result/<jid>")
def api_result(jid: str):
    j = JOBS.get(jid)
    if not j:
        return jsonify({"error": "任务不存在"}), 404
    if j["status"] == "error":
        return jsonify({"error": j["error"]}), 500
    if j["status"] != "done":
        return jsonify({"error": "尚未完成", "status": j["status"]}), 202
    return jsonify({"ok": True, "result": j["result"]})


# ====================================================================== 重新学习
@app.route("/api/refit", methods=["POST"])
def api_refit():
    """对单个模型重新学习（可带新超参），用于「换参数重试」。"""
    body = request.get_json(silent=True) or {}
    code = str(body.get("code", "")).strip()
    model = str(body.get("model", "")).strip()
    params = body.get("params") or {}
    try:
        max_rebal = _parse_max_rebal(body.get("max_rebal", None), default=None)
        auto_equity = _parse_auto_equity(body)
        params = _check_params(params)
    except ValueError as e:
        return _bad("params", str(e))
    if model not in MODEL_ORDER:
        return _bad("model", f"未知模型：{model}")
    if not (len(code) == 6 and code.isdigit()):
        return _bad("code", "请输入 6 位基金代码")

    jid = JOBS.create(code, [model])

    def worker():
        try:
            JOBS.update(jid, stage=f"重新学习 {model}", progress=0.2)
            payload = refit_single(code, model, params, max_rebal=max_rebal,
                                   auto_equity=auto_equity)
            JOBS.update(jid, progress=0.95, stage="组装结果")
            JOBS.finish(jid, payload)
        except Exception as e:                        # noqa: BLE001
            traceback.print_exc()
            JOBS.fail(jid, f"{type(e).__name__}: {str(e)[:300]}")

    threading.Thread(target=worker, name=f"refit-{jid}",
                     daemon=True).start()
    return jsonify({"ok": True, "job_id": jid})


# ====================================================================== 决策回测
@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """基于拟合健康度的买卖决策，在用户指定日期区间上做回测。"""
    body = request.get_json(silent=True) or {}
    code = str(body.get("code", "")).strip()
    model_key = str(body.get("model", "enet_slsqp")).strip()
    auto_equity = bool(body.get("auto_equity", False))
    start_date = body.get("start") or None
    end_date = body.get("end") or None
    cost_rate = body.get("cost_rate", None)

    if not (len(code) == 6 and code.isdigit()):
        return jsonify({"error": "请输入 6 位基金代码"}), 400
    if model_key not in MODEL_ORDER:
        return jsonify({"error": f"未知模型：{model_key}"}), 400
    for d, lbl in ((start_date, "起始"), (end_date, "结束")):
        if d:
            try:
                pd.Timestamp(d)
            except (ValueError, TypeError):
                return jsonify({"error": f"{lbl}日期格式应为 YYYY-MM-DD"}), 400
    if start_date and end_date and start_date >= end_date:
        return _bad("start", "起始日需早于结束日")
    # 单边成本：负数或大到离谱的值必须挡在 4xx，不能让它在回测里静默生效
    if cost_rate is not None:
        if isinstance(cost_rate, bool) or not isinstance(cost_rate, (int, float)):
            return _bad("cost_rate", "cost_rate 必须是数值")
        if not _finite(cost_rate) or float(cost_rate) < 0 or float(cost_rate) > 0.05:
            return _bad("cost_rate", "cost_rate 应在 [0, 0.05] 区间内")
        cost_rate = float(cost_rate)

    jid = JOBS.create(code, [model_key])

    def worker():
        try:
            payload = backtest_fund(code, model_key=model_key,
                                    auto_equity=auto_equity,
                                    start_date=start_date, end_date=end_date,
                                    cost_rate=cost_rate, job_id=jid)
            JOBS.finish(jid, payload)
        except Exception as e:                        # noqa: BLE001
            traceback.print_exc()
            JOBS.fail(jid, f"{type(e).__name__}: {str(e)[:300]}")

    threading.Thread(target=worker, name=f"bt-{jid}",
                     daemon=True).start()
    return jsonify({"ok": True, "job_id": jid})


@app.route("/api/name/<code>")
def api_name(code: str):
    """查基金中文名；found=False 表示名录里查不到（可能代码有误，也可能接口不可用）。"""
    from src.webapi import _guess_name
    try:
        nm = _guess_name(code)
    except Exception as e:                            # noqa: BLE001
        return jsonify({"code": code, "name": code, "found": False,
                        "error": str(e)[:80]})
    return jsonify({"code": code, "name": nm or code, "found": nm is not None})


@app.route("/api/corr")
def api_corr():
    """
    只重算相关性区块（滚动窗口可调）。

    改滚动窗口不需要重跑拟合：面板走缓存，这里只做滚动相关，几百毫秒。
    没有这个端点时，前端那个"滚动窗口"输入框是个纯装饰的死控件。
    """
    from src.webapi import correlation_block
    code = str(request.args.get("code", "")).strip()
    try:
        win = int(request.args.get("win", 120))
    except (TypeError, ValueError):
        return jsonify({"error": "win 必须是整数"}), 400
    if not (len(code) == 6 and code.isdigit()):
        return jsonify({"error": "请输入 6 位基金代码"}), 400
    win = max(20, min(win, 750))
    try:
        panel = D.load_panel(code, verbose=False)
        corr = correlation_block(panel, panel["__FUND__"], roll_n=win)
    except Exception as e:                            # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500
    return jsonify({"ok": True, "code": code, "correlation": corr})


# ====================================================================== 启动
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"\n  Fund Lens · 基金风格穿透分析")
    print(f"  → http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()