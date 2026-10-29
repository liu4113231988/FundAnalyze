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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings                                            # noqa: E402
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS                                # noqa: E402

from src import config as C                                # noqa: E402
from src import data as D                                  # noqa: E402
from src.models import MODEL_ORDER, model_catalog         # noqa: E402
from src.webapi import CACHE_DIR, analyze, backtest_fund, refit_single
from src.jobs import JOBS, JobRunner, JobsBusy
from src.webparams import (_parse_max_rebal, _parse_auto_equity,
                           _check_params, _finite, _parse_window)
from src.fileio import _atomic_pickle_dump


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
                       auto_equity=False, data_stamp: str = "",
                       start: str = "", end: str = ""):
    import hashlib
    # ★ P1-05：缓存键必须绑定**底层数据摘要**，否则行情追加后旧结果不会失效，
    #   表现为"刷新无效"。data_stamp 由 src.data.data_stamp() 给出。
    # ★ 2026-10-12：缓存键还必须绑定**分析窗口**。窗口默认近 5 年且可由前端
    #   改变，若不加进键里，换窗口会命中旧窗口的结果（表现为"改了没反应"）。
    key = json.dumps({"v": C.CACHE_VERSION, "c": code,
                      "m": sorted(models or []),
                      "p": params or {}, "r": (C.RBSA.get("max_rebal") if max_rebal is None else max_rebal),
                      "a": bool(auto_equity), "d": data_stamp,
                      "s": start or "", "e": end or ""},
                     sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    # 版本号写进文件名，便于启动时清扫旧版本缓存
    return CACHE_DIR / f"result_v{C.CACHE_VERSION}_{code}_{h}.pkl"


RUNNER = JobRunner(JOBS, max_jobs=int(os.environ.get("FIL_MAX_JOBS", "2")))


def _bad(field: str, msg: str, status: int = 400):
    """结构化 4xx。非法输入必须停在 4xx —— 旧实现里 int("abc") 直接抛成 500。"""
    return jsonify({"ok": False, "error": msg, "field": field}), status


def _start_job(code, models, work, key=None):
    """HTTP 只负责把调度结果转换为响应。"""
    try:
        jid, merged = RUNNER.submit(code, models, work, key=key)
    except JobsBusy as error:
        return _bad("concurrency", str(error), 429)
    return jsonify({"ok": True, "job_id": jid, "cached": False, "merged": merged})


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
    if not isinstance(body, dict):
        return _bad("body", "请求体必须是对象")
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
    # 分析窗口（默认近 5 年，下限 2 年）。单独 try 是为了给出正确的 field。
    try:
        start, end = _parse_window(body.get("start"), body.get("end"))
    except ValueError as e:
        return _bad("start", str(e))

    if not (len(code) == 6 and code.isdigit()):
        return _bad("code", "请输入 6 位基金代码，例如 110022")
    if not isinstance(models, (list, tuple)):
        return _bad("models", "models 必须是模型名列表")
    bad = [m for m in models if m not in MODEL_ORDER]
    if bad:
        return _bad("models", f"未知模型：{','.join(map(str, bad))}")

    # ---- 缓存命中：直接返回
    try:
        stamp = D.data_stamp()
    except Exception:                             # noqa: BLE001
        stamp = "unknown"
    cp = _result_cache_path(code, models, params, max_rebal, auto_equity,
                            stamp, start, end)
    if cp.exists() and not refresh:
        try:
            with cp.open("rb") as f:
                payload = pickle.load(f)
            payload.setdefault("meta", {})["from_cache"] = True
            return jsonify({"ok": True, "result": payload, "cached": True})
        except Exception:                             # noqa: BLE001
            pass

    def work(jid):
        res = analyze(code, models=models, params=params,
                      refresh=refresh, job_id=jid, max_rebal=max_rebal,
                      auto_equity=auto_equity, start=start, end=end)
        try:
            final_cp = _result_cache_path(code, models, params, max_rebal,
                                          auto_equity, D.data_stamp(), start, end)
            _atomic_pickle_dump(res, final_cp)
        except Exception:
            app.logger.exception("分析成功，但结果缓存写入失败")
        return res

    return _start_job(code, models, work, key=f"{cp.name}|refresh={refresh}")


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
    if not isinstance(body, dict):
        return _bad("body", "请求体必须是对象")
    code = str(body.get("code", "")).strip()
    model = str(body.get("model", "")).strip()
    params = body.get("params") or {}
    try:
        max_rebal = _parse_max_rebal(body.get("max_rebal", None), default=None)
        auto_equity = _parse_auto_equity(body)
        params = _check_params({model: params})[model]
    except ValueError as e:
        return _bad("params", str(e))
    # ★ 窗口必须与主分析一致，否则重算结果落在默认窗口上
    try:
        start, end = _parse_window(body.get("start"), body.get("end"))
    except ValueError as e:
        return _bad("start", str(e))
    if model not in MODEL_ORDER:
        return _bad("model", f"未知模型：{model}")
    if not (len(code) == 6 and code.isdigit()):
        return _bad("code", "请输入 6 位基金代码")

    def work(jid):
        JOBS.update(jid, stage=f"重新学习 {model}", progress=0.2)
        return refit_single(code, model, params, max_rebal=max_rebal,
                            auto_equity=auto_equity, start=start, end=end)

    return _start_job(code, [model], work)


# ====================================================================== 决策回测
@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """基于拟合健康度的买卖决策，在用户指定日期区间上做回测。"""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return _bad("body", "请求体必须是对象")
    code = str(body.get("code", "")).strip()
    model_key = str(body.get("model", "enet_slsqp")).strip()
    try:
        auto_equity = _parse_auto_equity(body)
    except ValueError as e:
        return _bad("auto_equity", str(e))
    start_date = body.get("start") or None
    end_date = body.get("end") or None
    cost_rate = body.get("cost_rate", None)

    if not (len(code) == 6 and code.isdigit()):
        return jsonify({"error": "请输入 6 位基金代码"}), 400
    if model_key not in MODEL_ORDER:
        return jsonify({"error": f"未知模型：{model_key}"}), 400
    # ★ 区间校验统一交给 _parse_window：格式、先后、以及 2 年下限
    #   （旧实现只查格式与先后，1 个月的区间也能提交，回测结果全是噪声）
    try:
        start_date, end_date = _parse_window(start_date, end_date)
    except ValueError as e:
        return _bad("start", str(e))
    # 单边成本：负数或大到离谱的值必须挡在 4xx，不能让它在回测里静默生效
    if cost_rate is not None:
        if isinstance(cost_rate, bool) or not isinstance(cost_rate, (int, float)):
            return _bad("cost_rate", "cost_rate 必须是数值")
        if not _finite(cost_rate) or float(cost_rate) < 0 or float(cost_rate) > 0.05:
            return _bad("cost_rate", "cost_rate 应在 [0, 0.05] 区间内")
        cost_rate = float(cost_rate)

    def work(jid):
        return backtest_fund(code, model_key=model_key,
                             auto_equity=auto_equity, start_date=start_date,
                             end_date=end_date, cost_rate=cost_rate, job_id=jid)

    return _start_job(code, [model_key], work)


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
    # ★ 必须与主分析同窗口，否则相关性区块与拟合结果来自不同区间
    try:
        w_start, w_end = _parse_window(request.args.get("start"),
                                       request.args.get("end"))
    except ValueError as e:
        return _bad("start", str(e))
    try:
        panel = D.load_panel(code, start=(w_start or C.START_DATE),
                             end=w_end, verbose=False)
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