"""股票、指数日线分析；与基金风格拟合独立，复用 DuckDB 和绩效指标。"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import market_research.config as C
import market_research.data.marketdata as D
from market_research.modeling.backtest import perf_metrics
from market_research.serialization import _jsafe
from market_research.validation.window import parse_window

STOCKS = {"sh600519": "贵州茅台", "sz000001": "平安银行", "sz300750": "宁德时代", "sh601318": "中国平安"}
INDICES = {k: v for k, v in C.NAME_OF.items() if re.fullmatch(r"(?:sh|sz)\d{6}", k)}


def normalize_code(kind, code):
    if kind not in ("stock", "index"):
        raise ValueError("资产类型必须为 stock 或 index")
    code = str(code).strip().lower()
    if re.fullmatch(r"\d{6}", code):
        if kind == "stock":
            prefix = "sh" if code[0] == "6" else "sz" if code[0] in "03" else "bj" if code[0] in "489" else None
            if prefix is None:
                raise ValueError("请输入沪深京 A 股代码，例如 600519 或 sz000001")
            code = prefix + code
        else:
            matches = [k for k in INDICES if k.endswith(code)]
            if len(matches) != 1:
                raise ValueError("指数代码需带市场，例如 sh000001、sh000300 或 sz399006")
            code = matches[0]
    if not re.fullmatch(r"(?:sh|sz|bj)\d{6}", code):
        raise ValueError("代码格式错误，例如 sh600519 或 sh000300")
    if kind == "index" and not re.fullmatch(r"(?:sh000|sz399)\d{3}", code):
        raise ValueError("目前支持 sh000 开头的沪市指数和 sz399 开头的深市指数")
    if kind == "stock":
        digit = code[2]
        expected = "sh" if digit == "6" else "sz" if digit in "03" else "bj" if digit in "489" else None
        if code[:2] != expected or not re.fullmatch(r"(?:sh(?:60|68)\d{4}|sz(?:00|30)\d{4}|bj[489]\d{5})", code):
            raise ValueError("股票代码与市场不匹配；000001 股票属于深圳市场")
    return code


def clean_bars(raw, start, end, volume_unit="shares", allow_nonpositive=False):
    """统一日线与成交量单位；非法价格拒绝写入缓存。"""
    frame = raw.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover"}).copy()
    required = ["open", "close", "high", "low"]
    if frame.empty or not {"date", *required}.issubset(frame.columns):
        raise ValueError("数据源未返回有效日线，请检查代码或稍后刷新")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last") & frame.index.notna()]
    frame = frame.loc[start:end, [c for c in required + ["volume", "amount", "turnover"] if c in frame]]
    frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    if frame.empty or not np.isfinite(frame[required].to_numpy()).all() or (not allow_nonpositive and (frame[required] <= 0).any().any()):
        raise ValueError("日线价格为空、缺失或非正；该复权区间暂不可分析")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) | (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("数据源 OHLC 高低价关系异常")
    for column in ("volume", "amount", "turnover"):
        if column in frame and ((frame[column].dropna() < 0).any() or np.isinf(frame[column]).any()):
            raise ValueError(f"数据源 {column} 含异常值")
    if "volume" in frame and volume_unit == "lots":
        frame["volume"] *= 100
    return frame


def load_bars(kind, code, start, end, adjust="qfq", refresh=False, _fetch_only=False):
    code = normalize_code(kind, code)
    if adjust not in ("qfq", "", "hfq"):
        raise ValueError("复权方式必须为 qfq、hfq 或空字符串")
    if kind == "index":
        adjust = ""
    if not _fetch_only:
        import market_research.tasks.sync as sync
        sync.register(kind, code, adjust)
        if not refresh:
            history = sync.cached(kind, code, adjust, start, end)
            if history is not None:
                return history
    name = f"market_v1_{kind}_{code}_{adjust or 'raw'}_{start}"
    meta = {"v": 1, "kind": kind, "code": code, "start": start, "end": end, "adjust": adjust, "volume_unit": "shares" if kind == "stock" else "source"}

    def build():
        import akshare as ak
        if kind == "stock":
            try:
                raw = ak.stock_zh_a_hist(symbol=code[2:], period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust=adjust, timeout=12)
                result = clean_bars(raw, start, end, "lots", allow_nonpositive=_fetch_only and adjust == "qfq")
            except Exception as primary:
                try:
                    raw = ak.stock_zh_a_hist_tx(symbol=code, start_date=start, end_date=end, adjust=adjust, timeout=12)
                    result = clean_bars(raw, start, end, allow_nonpositive=_fetch_only and adjust == "qfq")
                except Exception as fallback:
                    raise ValueError(f"股票日线暂不可用，请检查代码或稍后重试（{type(primary).__name__} / {type(fallback).__name__}）") from fallback
            info = meta.setdefault("info", {})
            info["source"] = "AKShare / 东方财富" if "日期" in raw.columns else "AKShare / 腾讯证券"
            info["name"] = STOCKS.get(code)
            if not info["name"]:
                try:
                    profile = ak.stock_individual_info_em(symbol=code[2:], timeout=6)
                    names = profile.loc[profile["item"] == "股票简称", "value"]
                    if len(names):
                        info["name"] = str(names.iloc[0])[:60]
                except Exception:
                    pass
            return result
        meta.setdefault("info", {})["source"] = "AKShare / 新浪指数"
        return clean_bars(D._retry(ak.stock_zh_index_daily, tries=2, symbol=code), start, end)

    if _fetch_only:
        frame = build()
        frame.attrs.update(meta.get("info", {}))
        return frame
    frame = D._cached(name, build, meta=meta, refresh=refresh)
    frame.attrs.update(D.cache_info(name))
    return frame


def analyze_market(kind, code, start=None, end=None, adjust="qfq", benchmark="sh000300", refresh=False, job_id=None):
    from market_research.tasks.jobs import JOBS
    code = normalize_code(kind, code)
    benchmark = normalize_code("index", benchmark)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    start = start or (pd.Timestamp(end) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    start, end = parse_window(start, end, min_days=30)
    if job_id:
        JOBS.update(job_id, stage="读取日线行情", progress=0.1)
    bars = load_bars(kind, code, start, end, adjust, refresh)
    if (bars[["open", "close", "high", "low"]] <= 0).any().any():
        raise ValueError("所选区间包含非正复权价，无法计算比例收益；请改用后复权或缩短区间")
    if len(bars) < 21:
        raise ValueError(f"区间仅有 {len(bars)} 根日线，至少需要 21 根；请扩大区间")
    warnings = []
    reference = None
    if benchmark == code:
        reference = bars.close
    else:
        try:
            reference = load_bars("index", benchmark, start, end, refresh=refresh).close
        except Exception:
            warnings.append("基准指数暂不可用，已保留价格与风险分析；可点击刷新重试。")
    if job_id:
        JOBS.update(job_id, stage="计算收益、风险与基准比较", progress=0.8)
    technical = None
    if kind == "stock":
        from market_research.market.technical import analyze_technical
        import market_research.tasks.sync as sync
        if job_id:
            JOBS.update(job_id, stage="计算技术指标与滚动方向验证", progress=0.85)
        warm_start = min(start, (pd.Timestamp(end) - pd.DateOffset(years=6)).strftime("%Y-%m-%d"))
        context = sync.cached(kind, code, adjust, warm_start, end)
        if context is None:
            context = bars
        else:
            overlap = context.index.intersection(bars.index)
            price_columns = ["open", "close", "high", "low"]
            if len(overlap) != len(bars) or not np.allclose(context.loc[overlap, price_columns], bars.loc[overlap, price_columns], rtol=1e-7, atol=1e-6):
                # Never combine a freshly revised adjustment basis with old
                # training history. Show short-sample state until full sync.
                context = bars
        # Very early forward-adjusted prices can be nonpositive. Only use the
        # continuous valid tail, never form returns across that boundary.
        invalid = (context[["open", "close", "high", "low"]] <= 0).any(axis=1)
        if invalid.any():
            context = context.loc[context.index > context.index[invalid][-1]]
        technical = analyze_technical(context, display_start=bars.index[0], adjust=adjust)
    payload = market_payload(kind, code, bars, reference, benchmark, adjust, warnings)
    if technical is not None:
        payload["technical"] = _jsafe(technical)
    return payload


def market_payload(kind, code, bars, reference=None, benchmark="sh000300", adjust="qfq", warnings=None):
    close = bars.close
    returns = close.pct_change().dropna()
    metrics = perf_metrics(returns, rf=0)
    ma = {str(n): close.rolling(n).mean() for n in (20, 60, 120)}
    comparison = {"ok": False, "name": INDICES.get(benchmark, benchmark)}
    if reference is not None:
        pair = pd.concat([close.rename("asset"), reference.rename("benchmark")], axis=1).dropna()
        if len(pair) >= 21:
            growth = pair / pair.iloc[0]
            paired_returns = pair.pct_change().dropna()
            variance = paired_returns.benchmark.var()
            comparison.update(ok=True, dates=pair.index.strftime("%Y-%m-%d").tolist(), asset=growth.asset.tolist(), benchmark=growth.benchmark.tolist(),
                excess_return=float(growth.asset.iloc[-1] - growth.benchmark.iloc[-1]),
                beta=float(paired_returns.asset.cov(paired_returns.benchmark) / variance) if variance > 1e-16 else None,
                correlation=paired_returns.asset.corr(paired_returns.benchmark), n_days=len(pair))
    notices = list(warnings or [])
    if not comparison["ok"] and not notices:
        notices.append("共同交易日不足 21 天，暂不计算基准比较。")
    if kind == "stock":
        notices.append("收益与均线基于所选复权价格，属于历史价格表现；前复权历史可能随分红拆分调整，不是严格时点可得回测。")
    else:
        notices.append("指数价格不含成分股分红，区间收益不是全收益指数回报。成交量使用数据源原始单位。")
    last = float(close.iloc[-1])
    return _jsafe({"kind": kind, "code": code, "name": (bars.attrs.get("name") or STOCKS.get(code, code)) if kind == "stock" else INDICES.get(code, code),
        "meta": {"start": close.index[0].strftime("%Y-%m-%d"), "end": close.index[-1].strftime("%Y-%m-%d"), "n_days": len(bars), "adjust": adjust if kind == "stock" else "", "volume_unit": "股" if kind == "stock" else "源单位", "benchmark": benchmark, "source": bars.attrs.get("source", "AKShare 免费行情")},
        "stats": {"latest": last, "change": float(returns.iloc[-1]), "total_return": float(close.iloc[-1] / close.iloc[0] - 1), "ann_return": metrics.get("年化收益"), "volatility": metrics.get("年化波动"), "max_drawdown": metrics.get("最大回撤"), "ma20_gap": last / ma["20"].iloc[-1] - 1, "ma60_gap": last / ma["60"].iloc[-1] - 1 if pd.notna(ma["60"].iloc[-1]) else None, "latest_volume": bars.volume.iloc[-1] if "volume" in bars else None},
        "series": {"dates": close.index.strftime("%Y-%m-%d").tolist(), "candles": bars[["open", "close", "low", "high"]].values.tolist(), "close": close.tolist(), "ma": {k: v.tolist() for k, v in ma.items()}, "volume": bars.volume.tolist() if "volume" in bars else None, "drawdown": (close / close.cummax() - 1).tolist(), "volatility": (returns.reindex(close.index).rolling(20).std() * np.sqrt(252)).tolist()},
        "comparison": comparison, "warnings": notices})
