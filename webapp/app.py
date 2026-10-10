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
from src.models import MODEL_ORDER, model_catalog         # noqa: E402
from src.webapi import (CACHE_DIR, JOBS, analyze, backtest_fund,  # noqa: E402
                         refit_single)

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
                       auto_equity=False):
    import hashlib
    key = json.dumps({"v": C.CACHE_VERSION, "c": code,
                      "m": sorted(models or []),
                      "p": params or {}, "r": max_rebal or 0,
                      "a": bool(auto_equity)},
                     sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    # 版本号写进文件名，便于启动时清扫旧版本缓存
    return CACHE_DIR / f"result_v{C.CACHE_VERSION}_{code}_{h}.pkl"


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
    # 重估点上限：越大越准越慢。前端可传 0 表示"不抽稀"（最精确但最慢）。
    mr = body.get("max_rebal", None)
    max_rebal = None if mr in (None, "", 0) else int(mr)
    auto_equity = bool(body.get("auto_equity", False))

    if not (len(code) == 6 and code.isdigit()):
        return jsonify({"error": "请输入 6 位基金代码，例如 110022"}), 400
    bad = [m for m in models if m not in MODEL_ORDER]
    if bad:
        return jsonify({"error": f"未知模型：{','.join(bad)}"}), 400

    # ---- 缓存命中：直接返回
    cp = _result_cache_path(code, models, params, max_rebal, auto_equity)
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
                with cp.open("wb") as f:
                    pickle.dump(res, f)
            except Exception:                         # noqa: BLE001
                pass
            JOBS.finish(jid, res)
        except Exception as e:                        # noqa: BLE001
            traceback.print_exc()
            JOBS.fail(jid, f"{type(e).__name__}: {str(e)[:300]}")

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
    mr = body.get("max_rebal", None)
    max_rebal = None if mr in (None, "", 0) else int(mr)
    auto_equity = bool(body.get("auto_equity", False))
    if model not in MODEL_ORDER:
        return jsonify({"error": f"未知模型：{model}"}), 400
    if not (len(code) == 6 and code.isdigit()):
        return jsonify({"error": "请输入 6 位基金代码"}), 400

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
        return jsonify({"error": "起始日需早于结束日"}), 400

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
    from src.webapi import get_fund_name
    try:
        return jsonify({"code": code, "name": get_fund_name(code)})
    except Exception as e:                            # noqa: BLE001
        return jsonify({"code": code, "name": code, "error": str(e)[:80]})


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