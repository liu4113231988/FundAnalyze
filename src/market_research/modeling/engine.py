from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import market_research.config as C
from market_research.modeling.catalog import MODELS, ModelSpec
from market_research.modeling.estimators import fit_window, estimate_equity, _score
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
        # "linspace" = 按样本总长度均匀抽稀（快，但历史点会随追加数据漂移）
        # "calendar" = 固定日历锚点（前缀稳定，研究结论应使用它）
        self.rebal_mode = str((cfg or C.RBSA).get("rebal_mode", "linspace"))

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

    @staticmethod
    def _rebal_calendar(n: int, W: int, refit: int,
                        max_rebal: Optional[int] = None,
                        dates=None, freq: str = "QE") -> List[int]:
        """
        **固定日历**重估点（P0-04 修复路径①）。

        `_rebal_idx` 用 `linspace(0, len-1, max_rebal)` 均匀抽稀，采样位置由
        **样本总长度**决定 —— 追加 100 行数据，历史重估点里就有 52/60 个发生
        位移，历史输出被未来数据改写（实测）。

        这里改为按**自然日历锚点**（默认每季度末）取"锚点之后的第一个 refit 点"。
        锚点只由日期决定，追加未来数据不会移动历史锚点 → 前缀稳定。
        代价：每季一个点的密度低于 linspace 的 60 点，长样本下更粗。

        dates 为完整日期索引；freq 传 pandas 偏移别名（"QE" 季、"ME" 月）。
        """
        idx = list(range(W, n, refit))
        if not idx or dates is None:
            return idx
        di = pd.DatetimeIndex(dates)
        if len(di) < n:                       # 保险：dates 与 n 不匹配则退回原法
            return ModelZoo._rebal_idx(n, W, refit, max_rebal)
        try:
            anchors = pd.date_range(di[0], di[-1], freq=freq)
        except Exception:                     # noqa: BLE001
            return ModelZoo._rebal_idx(n, W, refit, max_rebal)
        out = []
        for a in anchors:
            j = int(np.searchsorted(di.values, np.datetime64(a), side="right"))
            k = int(np.searchsorted(idx, j, side="left"))
            if k < len(idx):
                out.append(idx[k])
        out = sorted(set(out))
        # 仍需限流时，从尾部**丢弃**而不是重新抽稀（丢弃不改变已选点的位置）
        if max_rebal and len(out) > max_rebal:
            step = int(np.ceil(len(out) / max_rebal))
            out = out[::step]
        return out

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

        # ★ 指数股息调整（P0-01 遗留 / alpha 分解配套）
        #   基金侧现在是**总回报**（含分红再投），而候选指数是**价格指数**
        #   （不含成分股股息）。这个口径差全部落进残差、被当成 alpha，
        #   实测约 2~3%/年，复利 9 年就是 20~30% 的"伪 alpha"。
        #   填 `RBSA["index_dividend_annual"]` 即可把它还回指数侧；
        #   默认 0 = 保持原行为，不改变任何现有结果。
        div = float(cfg.get("index_dividend_annual", 0.0) or 0.0)
        if abs(div) > 1e-9:
            X_all = X_all + div / 252.0

        n = len(dates)
        W = cfg["window"]
        if n < W + 60:
            raise ValueError(f"样本不足：仅 {n} 行，滚动窗口需要 {W}+60。")
        mr = max_rebal if max_rebal is not None else self.max_rebal
        # rebal_mode="calendar" → 固定日历锚点，追加数据不改变历史重估点（P0-04）
        if str(cfg.get("rebal_mode", self.rebal_mode)) == "calendar":
            rebal_idx = self._rebal_calendar(n, W, cfg["refit"], mr, dates)
        else:
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
        n_fallback = 0          # 黑箱预测失败、回退成重要性权重的窗口数
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
            degraded = ""
            if fr.predictor is not None:
                # 黑箱模型：用「上一窗口拟合好的估计器」预测下一段，
                # 这是真正的样本外（前向）预测。
                try:
                    seg = np.asarray(fr.predictor(X_next), dtype=float)
                    if not np.all(np.isfinite(seg)):
                        raise ValueError("预测含非有限值")
                except Exception as e:      # noqa: BLE001
                    # ★ 旧实现在这里静默回退成 X_next @ w_full，而 w_full 是
                    #   **重要性权重**（非负、Σ=equity_ratio），把它当线性持仓
                    #   得到的收益会被记成该黑箱模型的样本外结果 ——
                    #   失败被伪装成成功，且没有任何痕迹。现在如实记录（P0-05）。
                    seg = X_next @ w_full
                    degraded = f"blackbox_fallback:{type(e).__name__}"
                    n_fallback += 1
            else:
                # 线性约束模型：合成收益 = X @ w，权重和 = 股票仓位
                seg = X_next @ w_full
            syn.loc[dates[i:j_end]] = seg

            rows.append({"date": dates[i - 1], **fr.weights.to_dict()})
            fit_r2s.append(fr.r2)
            fit_tes.append(fr.te)
            diag_rows.append({"date": dates[i - 1], "r2": fr.r2, "te": fr.te,
                              "alpha": fr.extra.get("alpha"),
                              "degraded": degraded or None})
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
        # ★ 降级必须可观测：黑箱回退窗口数 / 有效窗口数（P0-05）
        if diag_rows:
            agg_extra["fallback_windows"] = int(n_fallback)
            agg_extra["fallback_ratio"] = float(n_fallback) / len(diag_rows)
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
