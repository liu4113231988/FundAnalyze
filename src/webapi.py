# -*- coding: utf-8 -*-
"""
分析引擎：为 Web 前端组装一份完整的、可序列化的分析结果。

一次 analyze() 调用产出四组内容：
    1. K 线/净值序列（基金 + 各模型合成基准）
    2. 与候选指数的相关性分析（矩阵 + Top 榜 + 滚动相关性）
    3. 各机器学习模型的拟合结果（样本外评估 + 滚动权重）
    4. 基于上述证据生成的自然语言分析结论

设计要点
--------
* 结果以 JSON 友好的字典返回，数值统一转成原生类型（NaN → None）。
* 数据与拟合结果都可缓存，避免每次请求都重算（重算是分钟级操作）。
* 允许只跑部分模型（前端勾选），也允许对单个模型重算并替换（重学习）。
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from . import config as C
from . import data as D
from .backtest import run_backtest, score_to_position
from .models import (MODELS, MODEL_ORDER, ModelReport, ModelZoo,
                      model_catalog)
from .signal import SignalEngine, build_decision, health_stats

CACHE_DIR = C.DATA_DIR / "webcache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 已完成/失败任务在内存中的保留时间；过期后整任务清除（分析结果磁盘缓存仍在）
JOB_TTL_SEC = 30 * 60


# ====================================================================== 工具
def _jnum(v, nd: int = 6):
    """数值 → JSON 安全值。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _jsafe(obj):
    """递归把 numpy / pandas 转成 JSON 可序列化结构。"""
    if isinstance(obj, dict):
        return {str(k): _jsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsafe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _jnum(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return _jnum(obj)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pd.Series):
        return [_jsafe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return _jsafe(obj.to_dict("list"))
    return str(obj)


def _dates(idx) -> List[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(idx)]


# ====================================================================== 任务管理
class JobStore:
    """
    极简任务表：分析是分钟级操作，必须异步 + 可轮询进度。

    每个任务有唯一 id，状态在内存；同时把结果落盘到 CACHE_DIR，
    这样进程重启后历史结果仍可直接打开。
    """

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def _prune_locked(self):
        """惰性清理：done/error 超过 TTL 的任务整条删除，避免结果常驻内存。"""
        now = time.time()
        expired = [jid for jid, j in self._jobs.items()
                   if j["status"] in ("done", "error")
                   and now - j["created"] >= JOB_TTL_SEC]
        for jid in expired:
            del self._jobs[jid]

    def create(self, code: str, models: List[str]) -> str:
        jid = hashlib.md5(
            f"{code}|{','.join(sorted(models))}|{time.time()}".encode()
        ).hexdigest()[:12]
        with self._lock:
            self._prune_locked()
            self._jobs[jid] = {
                "id": jid, "code": code, "models": models,
                "status": "running", "stage": "准备数据", "progress": 0.0,
                "detail": [], "result": None, "error": None,
                "created": time.time(),
            }
        return jid

    def update(self, jid: str, **kw):
        with self._lock:
            if jid in self._jobs:
                self._jobs[jid].update(kw)

    def get(self, jid: str) -> Optional[dict]:
        with self._lock:
            self._prune_locked()
            return self._jobs.get(jid)

    def finish(self, jid: str, result: dict):
        self.update(jid, status="done", progress=1.0, stage="完成", result=result)

    def fail(self, jid: str, err: str):
        self.update(jid, status="error", stage="失败", error=err)

    def public(self, jid: str) -> Optional[dict]:
        j = self.get(jid)
        if not j:
            return None
        return {k: v for k, v in j.items() if k != "result"}


JOBS = JobStore()


# ====================================================================== 缓存
def _cache_path(code: str, tag: str) -> Path:
    return CACHE_DIR / f"{code}_{tag}.pkl"


def _cache_load(code: str, tag: str):
    p = _cache_path(code, tag)
    if not p.exists():
        return None
    try:
        import pickle
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:                      # noqa: BLE001
        return None


def _cache_save(code: str, tag: str, obj):
    try:
        import pickle
        with _cache_path(code, tag).open("wb") as f:
            pickle.dump(obj, f)
    except Exception:                      # noqa: BLE001
        pass


# ====================================================================== 相关性分析
def correlation_block(panel: pd.DataFrame, nav: pd.Series,
                      top_n: int = 15, roll_n: int = 120) -> dict:
    """
    基金与候选指数的相关性分析。

    三个层次：
      1. 全样本相关系数矩阵（基金 × 全部指数）
      2. Top-N 排行榜（按 |相关系数| 排序）
      3. 滚动 roll_n 日相关系数（看风格是否漂移）

    roll_n 可由前端传入（/api/corr 会用它做低成本重算）。
    """
    R = np.log(panel.astype(float)).diff().dropna()
    fund = R["__FUND__"]
    corr = R.drop(columns=["__FUND__"]).corrwith(fund)

    assets = list(corr.index)
    mat = R.drop(columns=["__FUND__"]).corr()
    # 只保留与基金相关的前 top_n 个指数做矩阵展示，避免 43×43 糊成一团
    keep = [c for c in corr.abs().sort_values(ascending=False).index[:top_n]]
    sub = mat.loc[keep, keep].copy()
    # 加上基金这一行/列，方便一眼看出谁最相关
    sub.insert(0, "【基金】", corr.reindex(keep))
    sub.loc["【基金】"] = np.concatenate([[1.0], corr.reindex(keep).values])

    # 滚动 roll_n 日相关系数（对 Top8 指数）
    roll_n = int(max(20, min(int(roll_n), len(R) - 1)))
    roll = {}
    for c in keep[:8]:
        rc = R[c].rolling(roll_n).corr(fund).dropna()
        if len(rc):
            roll[C.NAME_OF.get(c, c)] = {
                "dates": _dates(rc.index),
                "values": [_jnum(v, 4) for v in rc.values],
            }

    top_rows = []
    for code_, v in corr.abs().sort_values(ascending=False).head(top_n).items():
        top_rows.append({
            "code": code_,
            "name": C.NAME_OF.get(code_, code_),
            "layer": C.LAYER_OF.get(code_, "其他"),
            "corr": _jnum(corr[code_], 4),
            "abs": _jnum(abs(corr[code_]), 4),
        })

    # 按大类汇总平均相关性
    layer_corr = {}
    for code_, v in corr.items():
        layer_corr.setdefault(C.LAYER_OF.get(code_, "其他"), []).append(v)
    layer_rows = [{"layer": k,
                   "avg_corr": _jnum(float(np.mean(v)), 4),
                   "n": len(v)}
                  for k, v in layer_corr.items()]
    layer_rows.sort(key=lambda r: -abs(r["avg_corr"] or 0))

    # 风格漂移：基金与 Top3 指数的滚动相关是否发生明显变化
    drift = []
    for c in keep[:3]:
        rc = R[c].rolling(roll_n).corr(fund).dropna()
        if len(rc) < 200:
            continue
        drift.append({
            "name": C.NAME_OF.get(c, c),
            "early": _jnum(rc.iloc[:len(rc) // 3].mean(), 4),
            "late": _jnum(rc.iloc[-len(rc) // 3:].mean(), 4),
        })
    for d in drift:
        d["delta"] = _jnum((d["late"] or 0) - (d["early"] or 0), 4)

    return {
        "matrix": {"cols": [str(c) for c in sub.columns],
                   "rows": [str(i) for i in sub.index],
                   "values": [[_jnum(v, 4) for v in row]
                              for row in sub.values]},
        "top": top_rows,
        "layers": layer_rows,
        "rolling": roll,
        "drift": drift,
        "roll_window": roll_n,
        "n_assets": len(assets),
    }


# ====================================================================== 净值 / K 线
def nav_block(panel: pd.DataFrame, nav: pd.Series) -> dict:
    """
    净值区块：基金净值曲线、月度收益热力图、回撤曲线。

    注意：开放式基金只有净值没有成交量，因此这里给的是净值 K 线
    （以每日净值构造 开=高=低=收 的退化形式），不是交易所 K 线。
    前端会用「收盘净值 + 涨跌幅柱」的方式呈现，更符合净值语义。
    """
    nav = nav.dropna()
    rets = nav.pct_change().fillna(0.0)
    dd = nav / nav.cummax() - 1.0

    # 月度收益热力图（年 × 月）
    m = nav.resample("ME").last()
    mret = m.pct_change()
    mret.iloc[0] = np.nan
    years_list = sorted({d.year for d in mret.index})
    monthly = {}
    for d, v in mret.items():
        if pd.isna(v):
            continue
        monthly.setdefault(str(d.year), {})[str(d.month)] = _jnum(v, 5)

    # 滚动 60 日波动 vs 全样本
    vol60 = rets.rolling(60).std() * np.sqrt(252)
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    ann = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    avol = float(rets.std() * np.sqrt(252))
    return {
        "dates": _dates(nav.index),
        "nav": [_jnum(v, 5) for v in nav.values],
        "ret": [_jnum(v, 5) for v in rets.values],
        "drawdown": [_jnum(v, 5) for v in dd.values],
        "vol60": [_jnum(v, 5) for v in vol60.values],
        "monthly": monthly,
        "years": [str(y) for y in years_list],
        "stats": {
            "start": nav.index[0].strftime("%Y-%m-%d"),
            "end": nav.index[-1].strftime("%Y-%m-%d"),
            "n_days": int(len(nav)),
            "latest": _jnum(nav.iloc[-1], 5),
            "total_return": _jnum(nav.iloc[-1] / nav.iloc[0] - 1, 5),
            "years_span": _jnum(years, 2),
            "ann_return": _jnum(ann, 5),
            "ann_vol": _jnum(avol, 5),
            "max_drawdown": _jnum(dd.min(), 5),
            # 夏普比率 = (年化收益 - 无风险) / 年化波动，无风险取 0
            "sharpe": _jnum(ann / avol if avol > 1e-9 else np.nan, 5),
            "calmar": _jnum(ann / abs(dd.min()) if dd.min() < -1e-9 else np.nan, 5),
            "max_vol60": _jnum(vol60.max(), 5),
            "current_vol60": _jnum(vol60.iloc[-1], 5),
        },
    }


# ====================================================================== 模型结果序列化
def _weights_payload(w: pd.Series, min_w: float = 0.005,
                     top_n: int = 15) -> dict:
    """
    序列化权重用于条形图。

    **必须过滤掉权重≈0 的项**：线性约束模型会把落选指数置零，
    但 Series 里仍保留这些列。若不过滤，43 个指数会挤进图表，
    大部分是零高度的柱，前端完全读不出重点。
    """
    if w is None or len(w) == 0:
        return {"items": [], "sum": 0.0, "n": 0, "n_zero": 0}
    w = w[w > 0]
    if w.empty:
        return {"items": [], "sum": 0.0, "n": 0, "n_zero": 0}
    dropped = int((w <= min_w).sum())
    wv = w[w > min_w].head(top_n)
    items = []
    for code_, v in wv.items():
        items.append({
            "code": str(code_),
            "name": C.NAME_OF.get(code_, str(code_)),
            "layer": C.LAYER_OF.get(code_, "其他"),
            "weight": _jnum(v, 5),
        })
    return {"items": items, "sum": _jnum(w.sum(), 4), "n": int(len(w)),
            "n_zero": dropped}


def _weights_hist_payload(wh: pd.DataFrame, assets_limit: int = 8,
                         max_points: int = 120) -> dict:
    """滚动权重历史：Top-N 资产的权重随时间变化（堆叠面积图）。"""
    if wh is None or wh.empty:
        return {"dates": [], "series": [], "assets": []}
    total_w = wh.abs().sum(axis=0).sort_values(ascending=False)
    top = list(total_w.head(assets_limit).index)
    step = max(1, len(wh) // max_points)
    sl = wh.iloc[::step][top].tail(max_points)
    return {
        "dates": _dates(sl.index),
        "assets": [C.NAME_OF.get(c, str(c)) for c in top],
        "series": [[_jnum(v, 5) for v in sl[c].values] for c in top],
    }


def model_payload(rep: ModelReport, nav_oos: pd.Series,
                  max_points: int = 400) -> dict:
    """单个模型的完整可视化数据。"""
    syn = rep.syn_ret.dropna()
    syn_nav = np.exp(syn.cumsum())
    # 把基金净值与合成基准都归一到起点=1，才能在同一坐标轴上比较形状。
    # 注意：绝不能跨各自起点归一后再比，否则会抹掉真实的收益差。
    fund_aligned = nav_oos.reindex(syn_nav.index)
    fund_line = fund_aligned / fund_aligned.dropna().iloc[0] if \
        len(fund_aligned.dropna()) else fund_aligned
    syn_line = syn_nav / syn_nav.iloc[0] if len(syn_nav) else syn_nav

    step = max(1, len(syn_nav) // max_points)
    idx = syn_nav.index[::step]

    return {
        "key": rep.key, "name": rep.name, "family": rep.family,
        "desc": rep.desc, "interpretable": rep.interpretable,
        "metrics": {
            "oos_r2": _jnum(rep.oos_r2, 4),
            "oos_te": _jnum(rep.oos_te, 4),
            "oos_ic": _jnum(rep.oos_ic, 4),
            "win": _jnum(rep.win, 4),
            "beta": _jnum(rep.beta, 4),
            "vol_ratio": _jnum(rep.vol_ratio, 4),
            "fit_r2_mean": _jnum(rep.fit_r2_mean, 4),
            "fit_te_mean": _jnum(rep.fit_te_mean, 4),
            "elapsed": _jnum(rep.elapsed, 2),
            "n_weights": int((rep.latest_weights > 0).sum())
            if rep.latest_weights is not None else 0,
        },
        "extra": _jsafe({k: v for k, v in rep.extra.items() if v is not None}),
        "latest_weights": _weights_payload(rep.latest_weights),
        "weights_last_date": (rep.weights_last_date.strftime("%Y-%m-%d")
                              if rep.weights_last_date is not None else None),
        "weights_hist": _weights_hist_payload(rep.weights_hist),
        "curve": {
            "dates": _dates(idx),
            "fund": [_jnum(v, 5) for v in fund_line.reindex(idx)],
            "synth": [_jnum(v, 5) for v in syn_line.reindex(idx)],
        },
        "residual": {
            "dates": _dates(syn.index[::step]),
            "value": [_jnum(v, 5) for v in
                      (fund_aligned - syn_nav).reindex(
                          syn.index[::step]).values],
        },
    }


# ====================================================================== 结论生成
def build_insights(code: str, name: str, navb: dict, corr: dict,
                   models: Dict[str, ModelReport], rank: pd.DataFrame,
                   payload: Dict[str, dict]) -> dict:
    """
    基于真实数据生成结构化结论。分「可靠结论」与「不可靠结论」两侧写，
    避免把拟合优度当成预测能力。
    """
    st = navb["stats"]
    out: Dict[str, Any] = {}

    # ---------- 1. 风格穿透（拟合层面，成立）
    if not rank.empty:
        best_row = rank.iloc[0]
        bm = payload.get(best_row["key"], {})
        lw = (bm.get("latest_weights") or {}).get("items", [])
        top_txt = "、".join(f"{r['name']} {r['weight']:.0%}" for r in lw[:3]
                           if r["weight"] and r["weight"] > 0.02)
        # 若最佳模型是黑箱，其权重只是重要性排序，必须降级表述
        blackbox = not bm.get("interpretable", True)
        if blackbox:
            tail = ("该模型是<b>黑箱模型</b>，下面权重是特征重要性排序，"
                    "<b>不能当作真实持仓</b>；它胜出说明非线性成分重要，"
                    "但风格暴露仍应以可解释模型（线性约束类）为准。")
        else:
            tail = "这意味着它的收益来源高度集中，可被上述赛道指数直接复制。"
        out["penetration"] = {
            "ok": True,
            "model": best_row["模型"],
            "text": (f"滚动样本外拟合显示，{name} 可被少数几个指数解释："
                     f"<b>{top_txt}</b>。该模型样本外 R² = "
                     f"<b>{best_row['样本外R²']:.3f}</b>，"
                     f"年化跟踪误差 {best_row['样本外TE']:.1%}。" + tail),
        }
        # 结论：低 TE = 好解释
        iqr = rank["样本外TE"].quantile([0.25, 0.75])
        out["te_spread"] = {
            "text": (f"模型间的跟踪误差从 {rank['样本外TE'].min():.1%} 到 "
                     f"{rank['样本外TE'].max():.1%}，"
                     f"四分位区间 {iqr[0.25]:.1%}~{iqr[0.75]:.1%}。"
                     f"差异主要来自模型的正则化强度与非线性程度，"
                     f"而非数据差异——因为所有模型用的是同一份切分。"),
        }

    # ---------- 2. 风格漂移（相关性层面）
    if corr.get("drift"):
        d = corr["drift"]
        shifted = [r for r in d if r["delta"] is not None and abs(r["delta"]) > 0.10]
        if shifted:
            txt = "；".join(f"与{r['name']}的 120 日滚动相关从 "
                           f"{r['early']:+.2f} 变为 {r['late']:+.2f}"
                           for r in d)
            out["drift"] = {
                "warn": bool(shifted),
                "text": (f"相关性并非恒定：{txt}。"
                         + ("<b>存在风格漂移，说明基金经理近期调仓改变了赛道暴露，"
                            "固定权重的合成基准会滞后。</b>" if shifted else
                            "<b>漂移幅度均在 0.10 以内，风格暴露相对稳定。</b>")),
            }

    # ---------- 3. 拟合 ≠ 预测（关键提醒）
    if not rank.empty:
        lin = rank[rank["可解释"] == "是"]
        blk = rank[rank["可解释"] == "否"]
        parts = []
        if len(lin):
            parts.append(f"线性约束类模型（{len(lin)} 个）样本外 R² "
                         f"{lin['样本外R²'].min():.3f}~{lin['样本外R²'].max():.3f}")
        if len(blk):
            parts.append(f"黑箱模型（{len(blk)} 个）样本外 R² "
                         f"{blk['样本外R²'].min():.3f}~{blk['样本外R²'].max():.3f}")
        overfit = []
        if len(blk):
            for k in blk["key"]:
                p = payload.get(k, {})
                fi = p.get("metrics", {}).get("fit_r2_mean")
                oo = p.get("metrics", {}).get("oos_r2")
                if fi is None or oo is None:
                    continue
                gap = fi - oo
                if gap > 0.08:
                    overfit.append((gap, f"{p['name']}（窗口内 {fi:.3f} → 样本外 "
                                         f"{oo:.3f}，差 {gap:+.3f}）"))
        # 可解释模型也可能过拟合，一并检查
        if len(lin):
            for k in lin["key"]:
                p = payload.get(k, {})
                fi = p.get("metrics", {}).get("fit_r2_mean")
                oo = p.get("metrics", {}).get("oos_r2")
                if fi is None or oo is None:
                    continue
                gap = fi - oo
                if gap > 0.08:
                    overfit.append((gap, f"{p['name']}（窗口内 {fi:.3f} → 样本外 "
                                         f"{oo:.3f}，差 {gap:+.3f}）"))
        overfit.sort(reverse=True)
        top_of = [t for _, t in overfit[:2]]
        out["overfit"] = {
            "warn": bool(overfit),
            "text": ("；".join(parts) + "。"
                     + ("<b>" + "；".join(top_of) + " 存在过拟合："
                        "训练集拟合度高但样本外塌陷，这类模型的权重"
                        "不能当作真实持仓看。</b>" if overfit else
                        "<b>各模型训练/样本外差距均在 0.08 以内，控制良好。</b>")),
        }

    # ---------- 4. 仓位假设失配（最容易误读的一类低 R²）
    # 关键：线性约束模型强制 Σw=equity_ratio，因此它的隐含 beta 直接反映
    # 「假设与实际的偏离」；黑箱模型不做仓位约束，beta 恒等于 1，无诊断价值。
    # 所以只用**可解释模型**的 beta 做判断。
    lin_keys = [x["key"] for x in rank.to_dict("records")
                if x.get("可解释") == "是" and x["key"] in models]
    lin_betas = [float(models[k].beta) for k in lin_keys
                 if models[k].beta is not None and np.isfinite(models[k].beta)]
    if lin_betas:
        b = float(np.median(lin_betas))
        eq = C.RBSA["equity_ratio"]
        vr = next((models[k].vol_ratio for k in lin_keys
                   if models[k].vol_ratio is not None
                   and np.isfinite(models[k].vol_ratio)), None)
        vr = float(vr) if vr is not None else None
        if b < eq * 0.7:
            out["exposure_gap"] = {
                "warn": True,
                "text": (
                    f"<b>该基金的实际股票仓位约 {b:.0%}</b>"
                    f"（用线性约束模型的回归斜率估计），"
                    f"远低于框架假设的 {eq:.0%}。"
                    f"合成基准因此承担了过多波动"
                    + (f"（是基金自身波动的 {vr:.1f} 倍）" if vr else "") + "，"
                    f"<b>这会系统性压低线性模型的 R²</b>。"
                    f"因此下面的低 R² <b>不代表风格识别失败</b>，"
                    f"而是说明该基金大概率做了对冲、分档或仓位择时。"),
            }
            out["exposure_fix"] = {
                "text": (
                    f"要让 R² 反映「风格识别能力」而非「仓位差异」，"
                    f"应把约束改成 <code>Σw ≈ {b:.2f}</code>（按隐含 beta 缩放），"
                    f"或改用<b>不约束仓位</b>的纯风格回归。"
                    f"本页面为保持跨基金可比，仍统一用 {eq:.0%}；"
                    f"此时<b>建议直接参考排名靠前的黑箱模型</b>——"
                    f"它们不受仓位假设影响。"),
            }
        else:
            out["exposure_gap"] = {
                "warn": False,
                "text": (f"线性约束模型反推出的实际股票仓位约 {b:.0%}，"
                         f"与框架假设的 {eq:.0%} 基本吻合，"
                         f"说明该基金是常规满仓运作，R² 具备横向可比性。"),
            }

    # ---------- 5. 候选池覆盖度
    out["coverage"] = {
        "text": (f"候选池共 {corr.get('n_assets', 0)} 个指数，"
                 f"覆盖宽基 / 市值风格 / 价值成长 / 申万一级行业四层。"
                 f"相关度最高的层级是 "
                 f"{corr['layers'][0]['layer']}（平均相关 "
                 f"{corr['layers'][0]['avg_corr']:+.3f}），"
                 f"这直接决定了拟合出的风格权重的可信区间。"),
    }

    # ---------- 5. 走势描述（只描述，不预测）
    nav = np.array([v for v in navb["nav"] if v is not None], dtype=float)
    if len(nav) > 250:
        r = np.diff(np.log(nav))
        s: List[str] = []
        for lbl, n in (("近1月", 21), ("近3月", 63), ("近6月", 126),
                       ("近1年", 252)):
            if len(r) > n:
                s.append(f"{lbl} {float(np.exp(r[-n:].sum()) - 1):+.1%}")
        out["trend"] = {
            "text": (f"近期动量：{'，'.join(s)}。"
                     f"当前 60 日年化波动 {st.get('current_vol60')}，"
                     f"处于历史区间"
                     f"{'偏高' if (st.get('current_vol60') or 0) > (st.get('max_vol60') or 1) * 0.7 else '中性偏低'}"
                     f"（历史峰值 {st.get('max_vol60')}）。"
                     f"<b>此处只做状态描述，不给出买卖建议。</b>"),
        }
    return out


# ====================================================================== 主流程
def analyze(code: str, models: Optional[List[str]] = None,
            params: Optional[dict] = None, refresh: bool = False,
            job_id: Optional[str] = None,
            max_rebal: Optional[int] = None,
            auto_equity: bool = False) -> dict:
    """完整分析流程。models 为空表示全部 9 个模型。"""
    models = models or MODEL_ORDER
    models = [m for m in models if m in MODELS]
    t0 = time.time()

    def prog(stage, pct, detail=""):
        if job_id:
            JOBS.update(job_id, stage=stage, progress=pct, detail=detail)

    # ---------- 0. 数据
    prog("抓取基金与指数数据", 0.05)
    panel = D.load_panel(code, refresh=refresh, verbose=False)
    nav = panel["__FUND__"]
    name = (C.DEMO_FUNDS.get(code) or C.EXTENDED_FUNDS.get(code) or
            _guess_name(code))
    prog("数据就绪", 0.15, f"{panel.shape[0]} 交易日 × {panel.shape[1]-1} 指数")

    # ---------- 1. 净值
    prog("计算净值序列与回撤", 0.22)
    navb = nav_block(panel, nav)

    # ---------- 2. 相关性
    prog("计算与指数的相关性", 0.32)
    corr = correlation_block(panel, nav)

    # ---------- 3. 多模型拟合（耗时主体，占 0.35~0.9）
    prog("机器学习拟合中", 0.35, f"{len(models)} 个模型")

    def on_progress(key, done, total, r2):
        JOBS.update(job_id, progress=0.35 + 0.5 * done / max(total, 1),
                    stage=f"拟合 {MODELS[key].name}",
                    detail=f"({done}/{total}) 样本外R²="
                           f"{'—' if r2 is None or (isinstance(r2, float) and not math.isfinite(r2)) else f'{r2:.3f}'}")

    zoo_cfg = {**C.RBSA, "auto_equity": True} if auto_equity else None
    zoo = ModelZoo(zoo_cfg)
    reps = zoo.run(panel, models, params=params, max_rebal=max_rebal,
                   progress=on_progress if job_id else None)

    # ---------- 4. 序列化
    prog("组装结果", 0.9)
    R = np.log(panel.astype(float)).diff().dropna()
    nav_oos = nav.reindex(R.index[C.RBSA["window"]:])
    payload = {k: model_payload(r, nav_oos) for k, r in reps.items()}
    rank = ModelZoo.rank(reps)

    prog("生成分析结论", 0.96)
    insights = build_insights(code, name, navb, corr, reps, rank, payload)

    result = {
        "code": code, "name": name,
        "models": payload,
        "rank": _jsafe(rank.to_dict("records")),
        "nav": navb,
        "correlation": corr,
        "insights": insights,
        "meta": {
            "n_days": int(panel.shape[0]),
            "n_assets": int(panel.shape[1] - 1),
            "start": panel.index[0].strftime("%Y-%m-%d"),
            "end": panel.index[-1].strftime("%Y-%m-%d"),
            "models": models,
            "params": _jsafe(params or {}),
            "elapsed": round(time.time() - t0, 1),
            "window": C.RBSA["window"],
            "refit": C.RBSA["refit"],
            "max_rebal": (max_rebal if max_rebal is not None
                          else C.RBSA.get("max_rebal")),
            "n_rebal": int(len(reps[next(iter(reps))].weights_hist))
            if reps else 0,
            "equity_ratio": C.RBSA["equity_ratio"],
            "auto_equity": bool(auto_equity),
            "best_model": (rank.iloc[0]["key"] if not rank.empty and
                           pd.notna(rank.iloc[0]["样本外R²"]) else None),
        },
    }
    prog("完成", 1.0)
    return result


def _guess_name(code: str) -> Optional[str]:
    """
    尝试解析基金中文名；**查不到时返回 None**，不再用代码冒充名字。

    早期实现失败时回退成 code，调用方无法区分"这只基金就叫 110022"和
    "名录里查不到"，于是前端对"代码输错了"完全静默，只能等任务跑完才发现。
    """
    hit = _cache_load(code, "name")
    if hit is not None:
        return hit if hit != "" else None
    nm = None
    try:
        import akshare as ak
        info = ak.fund_name_em()
        col = "基金代码" if "基金代码" in info.columns else info.columns[0]
        namecol = next((c for c in ("基金简称", "名称", "基金名称")
                        if c in info.columns), None)
        m = info[info[col].astype(str).str.zfill(6) == code]
        if len(m) and namecol:
            nm = str(m[namecol].iloc[0])
        elif len(m) and info.shape[1] > 1:
            nm = str(m.iloc[0, 1])
    except Exception:                      # noqa: BLE001
        nm = None
    _cache_save(code, "name", nm or "")
    return nm


def get_fund_name(code: str) -> str:
    return _guess_name(code) or code


# ====================================================================== 局部重算
def refit_single(code: str, model_key: str,
                 params: Optional[dict] = None,
                 max_rebal: Optional[int] = None,
                 auto_equity: bool = False) -> dict:
    """
    只重跑一个模型（"重新学习"按钮）。

    数据走缓存，因此耗时 = 单模型滚动时间，远快于全量重算。
    """
    panel = D.load_panel(code, verbose=False)
    zoo_cfg = {**C.RBSA, "auto_equity": True} if auto_equity else None
    zoo = ModelZoo(zoo_cfg)
    rep = zoo.run(panel, [model_key], params={model_key: params or {}},
                  workers=1, max_rebal=max_rebal)[model_key]
    R = np.log(panel.astype(float)).diff().dropna()
    nav_oos = panel["__FUND__"].reindex(R.index[C.RBSA["window"]:])
    return model_payload(rep, nav_oos)


# ====================================================================== 买卖决策回测
def _backtest_payload(code: str, name: str, model_key: str, auto_equity: bool,
                      bt: dict, health: pd.DataFrame, decision: pd.Series,
                      elapsed: float) -> dict:
    eq = bt["equity"]
    return {
        "meta": {
            "code": code, "name": name,
            "model": model_key, "model_name": MODELS[model_key].name,
            "auto_equity": bool(auto_equity),
            "start": bt["window_start"].strftime("%Y-%m-%d"),
            "end": bt["window_end"].strftime("%Y-%m-%d"),
            "elapsed": round(elapsed, 2),
            "thresholds": {"exit": C.SIGNAL["health_exit"],
                           "enter": C.SIGNAL["health_enter"]},
            # 门控诊断：零交易时必须让用户看到"健康分离离场线还有多远"，
            # 否则"无触发交易"会被误读成"信号判断为一直持有"。
            "no_trades": len(bt["trades"]) == 0,
            "health_stats": _jsafe(health_stats(health)),
        },
        "equity": {
            "dates": _dates(eq.index),
            "series": {k: [_jnum(v, 5) for v in eq[k]] for k in eq.columns},
        },
        "metrics": _jsafe(bt["metrics"].reset_index().to_dict("records")),
        "health": {
            "dates": _dates(health.index),
            "score": [_jnum(v, 4) for v in health["health_score"]],
            "r2": [_jnum(v, 4) for v in health["fit_r2"]],
            "te_ratio": [_jnum(v, 4) for v in health["te_ratio"]],
            "alpha": [_jnum(v, 4) for v in health["active_alpha"]],
            # alpha 的 t 值才是真正进入健康分的分量（原始 alpha 噪声太大）
            "alpha_t": [_jnum(v, 4) for v in health["alpha_t"]],
            "position": [int(v) for v in decision.tolist()],
        },
        "trades": [{
            "date": t["date"].strftime("%Y-%m-%d"),
            "exec_date": t["exec_date"].strftime("%Y-%m-%d"),
            "action": t["action"], "health": round(t["health"], 3),
            "reason": t["reason"],
        } for t in bt["trades"]],
        "summary": {
            "ic": _jnum(bt["ic"]), "icir": _jnum(bt["icir"]),
            "turnover": _jnum(bt["turnover"], 2),
            "avg_position": _jnum(bt["avg_position"], 3),
            "win_long": _jnum(bt["win_long"], 3),
            "control_name": bt["control_name"],
            "timing_contrib": _jsafe(bt["timing_contrib"]),
            # 逐方案的对照名与净贡献：只有方案 vs 它自己的对照才是有效比较
            "controls": _jsafe(bt["controls"]),
            "timing_contrib_all": _jsafe(bt["timing_contrib_all"]),
            "avg_position_all": _jsafe(bt["avg_position_all"]),
        },
    }


def backtest_fund(code: str, model_key: str = "enet_slsqp",
                  auto_equity: bool = False,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  cost_rate: Optional[float] = None,
                  job_id: Optional[str] = None) -> dict:
    """
    拟合 → 健康度买卖决策 → 指定日期区间的回测。

    线性模型滚动拟合约 1~3 秒（max_rebal=60），因此整个任务通常在数秒内完成；
    依然走任务系统以便前端统一轮询。
    """
    if model_key not in MODELS:
        raise ValueError(f"未知模型：{model_key}")
    t0 = time.time()

    def prog(stage, pct):
        if job_id:
            JOBS.update(job_id, stage=stage, progress=pct)

    prog("加载数据", 0.1)
    panel = D.load_panel(code, verbose=False)
    nav = panel["__FUND__"]
    name = (C.DEMO_FUNDS.get(code) or C.EXTENDED_FUNDS.get(code)
            or _guess_name(code))

    prog(f"滚动拟合 {MODELS[model_key].name}", 0.25)
    zoo_cfg = {**C.RBSA, "auto_equity": True} if auto_equity else None
    zoo = ModelZoo(zoo_cfg)
    rep = zoo.run(panel, [model_key], workers=1,
                  max_rebal=C.RBSA["max_rebal"])[model_key]

    syn_ret = rep.syn_ret
    syn_nav = np.exp(syn_ret.cumsum())
    nav_oos = nav.reindex(syn_ret.index)

    prog("计算健康度与买卖决策", 0.7)
    engine = SignalEngine()
    trend = engine.trend_channel(syn_nav)
    health = engine.fit_health_channel(nav_oos, syn_ret, rep.fit_diag,
                                       rep.weights_hist)
    decision, trades = build_decision(health)

    # run_backtest 从 sig 取主策略与打分；趋势通道作为对照变体
    sig = pd.DataFrame({"position_final": decision,
                        "score": health["health_score"]})
    variants = {"仅趋势通道": score_to_position(trend["trend_score"])}
    bt_cfg = None if cost_rate is None else {"cost_rate": float(cost_rate)}

    prog("执行回测", 0.85)
    bt = run_backtest(nav_oos, sig, cfg=bt_cfg, fund_name=name,
                      variants=variants, main_name="拟合健康策略",
                      start_date=start_date, end_date=end_date,
                      decision_trades=trades)

    return _backtest_payload(code, name, model_key, auto_equity, bt,
                             health, decision, time.time() - t0)