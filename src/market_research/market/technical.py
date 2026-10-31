"""Causal daily indicators, explicit trading conditions and purged direction models.

Signals describe the close; they do not assume execution at that same close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_TRAIN = 252
MAX_TRAIN = 756
REFIT = 21


def wilder(series, period=14):
    """Wilder smoothing seeded by the first complete arithmetic average."""
    seeded = series.copy().astype(float)
    valid = np.flatnonzero(seeded.notna())
    if not len(valid) or valid[0] + period > len(seeded):
        return seeded * np.nan
    first = valid[0]
    seed_at = first + period - 1
    seed = seeded.iloc[first:seed_at + 1].mean()
    seeded.iloc[:seed_at] = np.nan
    seeded.iloc[seed_at] = seed
    return seeded.ewm(alpha=1 / period, adjust=False).mean()


def indicators(bars):
    close = bars.close
    out = pd.DataFrame(index=bars.index)
    for n in (20, 60):
        out[f"ma{n}"] = close.rolling(n).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["dif"] = ema12 - ema26
    out["dea"] = out.dif.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"] = 2 * (out.dif - out.dea)
    delta = close.diff()
    gains, losses = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    out["rsi"] = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    out.loc[(losses == 0) & (gains > 0), "rsi"] = 100
    out.loc[(losses == 0) & (gains == 0), "rsi"] = 50
    true_range = pd.concat([bars.high - bars.low, (bars.high - close.shift()).abs(), (bars.low - close.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = wilder(true_range)
    deviation = close.rolling(20).std(ddof=0)
    out["boll_upper"] = out.ma20 + 2 * deviation
    out["boll_lower"] = out.ma20 - 2 * deviation
    out["support"] = bars.low.shift().rolling(20).min()
    out["resistance"] = bars.high.shift().rolling(20).max()
    out["volume_ratio"] = bars.volume / bars.volume.shift().rolling(20).mean().replace(0, np.nan) if "volume" in bars else np.nan
    differences = pd.concat([close - out.ma20, out.ma20 - out.ma60, out.ma60 - out.ma60.shift(5), out.dif - out.dea, close - close.shift(20)], axis=1)
    votes = np.sign(differences.where(differences.abs().gt(close.abs() * 1e-10, axis=0), 0))
    ready = out.ma60.shift(5).notna() & out.dea.notna()
    out["score"] = (votes.sum(axis=1) * 20).where(ready)
    out["entry"] = ready & (out.score >= 60) & out.rsi.between(45, 70) & (close <= out.boll_upper)
    out["exit"] = ready & ((out.score <= -20) | ((close < out.ma20) & (out.macd < 0)))
    out["buy_event"] = out.entry & ~out.entry.shift(fill_value=False)
    out["sell_event"] = out.exit & ~out.exit.shift(fill_value=False)
    return out


def features_for(bars, ind):
    close = bars.close
    features = pd.DataFrame(index=bars.index)
    for n in (1, 5, 20):
        features[f"return{n}"] = close.pct_change(n, fill_method=None)
    for n in (20, 60):
        features[f"gap{n}"] = close / ind[f"ma{n}"] - 1
    features["slope60"] = ind.ma60.pct_change(5, fill_method=None)
    features["macd"] = ind.macd / close
    features["dif"] = ind.dif / close
    features["rsi"] = ind.rsi / 100
    features["atr"] = ind.atr / close
    features["bandwidth"] = (ind.boll_upper - ind.boll_lower) / ind.ma20
    return features.replace([np.inf, -np.inf], np.nan)


def direction_forecast(bars, features, horizon, display_start, training_end=None):
    """Fixed model; train labels must have matured by each prediction close.

    Scaler is fitted only on training rows. No shuffled CV or full-sample tuning.
    Evaluation observations are spaced horizon days apart to avoid label overlap.
    """
    future = bars.close.shift(-horizon) / bars.close - 1
    labels = (future > 0).astype(float).where(future.notna())
    probability = pd.Series(np.nan, index=bars.index)
    baseline = probability.copy()
    model, fitted_at, training_rows = None, -REFIT, 0
    cutoff = bars.index.searchsorted(pd.Timestamp(training_end), side='right') - 1 if training_end else None
    last_train_end = None
    for i in range(len(bars)):
        if features.iloc[i].isna().any():
            continue
        train_end = max(0, min(i, cutoff) - horizon + 1) if cutoff is not None else max(0, i - horizon + 1)
        freeze_due = cutoff is not None and i >= cutoff and last_train_end != train_end
        if (i - fitted_at >= REFIT or freeze_due) and train_end != last_train_end:
            # Last eligible feature at i-h has target ending at i (known now).
            train = features.iloc[:train_end].dropna().tail(MAX_TRAIN)
            target = labels.reindex(train.index).dropna()
            train = train.reindex(target.index)
            if len(train) >= MIN_TRAIN:
                training_rows = len(train)
                if target.nunique() >= 2:
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=0))
                    model.fit(train, target)
                else:
                    model = None
                prior = float(target.mean())
                fitted_at = i
                last_train_end = train_end
        if model is not None:
            probability.iloc[i] = model.predict_proba(features.iloc[[i]])[0, 1]
            baseline.iloc[i] = prior
    valid = pd.DataFrame({"probability": probability, "baseline": baseline, "actual": labels}).loc[display_start:].dropna()
    # Use fixed, non-overlapping starting points, not selected for performance.
    indices = bars.index.get_indexer(valid.index)
    keep, last = [], -horizon
    for position, original in enumerate(indices):
        if original >= last + horizon:
            keep.append(position)
            last = original
    evaluated = valid.iloc[keep]
    n = len(evaluated)
    evaluation = {"n": n, "ok": n >= 20}
    if n:
        actual = evaluated.actual
        evaluation.update(accuracy=float(((evaluated.probability >= .5) == actual).mean()),
                          baseline_accuracy=float(((evaluated.baseline >= .5) == actual).mean()),
                          brier=float(((evaluated.probability - actual) ** 2).mean()),
                          baseline_brier=float(((evaluated.baseline - actual) ** 2).mean()),
                          start=evaluated.index[0].date().isoformat(), end=evaluated.index[-1].date().isoformat())
        evaluation["advantage"] = bool(n >= 20 and evaluation["brier"] < evaluation["baseline_brier"])
    latest = probability.iloc[-1]
    return {"horizon": horizon, "ok": bool(pd.notna(latest)), "probability": latest,
            "training_rows": training_rows, "minimum_training_rows": MIN_TRAIN,
            "evaluation": evaluation, "series": probability.loc[display_start:].tolist(),
            "reason": "至少需要 252 个已成熟训练样本及约 65 日指标预热；单一涨跌类别无法训练。" if pd.isna(latest) else None}


def compare_forecasts(bars, base, candidate, horizon, display_start, context=None):
    """Compare fixed candidates on identical mature non-overlapping observations."""
    prices = bars.close.loc[display_start:]
    actual = (prices.shift(-horizon) / prices - 1).where(prices.shift(-horizon).notna())
    frame = pd.DataFrame({'base': base['series'], 'candidate': candidate['series'],
                          'return': actual}, index=prices.index).dropna()
    positions = bars.index.get_indexer(frame.index)
    keep, last = [], -horizon
    for i, position in enumerate(positions):
        if position >= last + horizon:
            keep.append(i); last = position
    frame = frame.iloc[keep]
    n = len(frame)
    result = {'n': n, 'ok': n >= 20, 'start': None, 'end': None,
              'note': '共同成熟日期比较；候选固定，未据此自动选模型。Brier改善不代表显著性或扣费收益。'}
    if not n:
        return result
    result.update(start=frame.index[0].date().isoformat(), end=frame.index[-1].date().isoformat())
    target = (frame['return'] > 0).astype(float)
    for key in ('base', 'candidate'):
        probability = frame[key]
        bins = []
        for i in range(5):
            group = (probability >= i / 5) & ((probability < (i+1) / 5) if i < 4 else (probability <= 1))
            if group.any():
                bins.append({'lower': i/5, 'upper': (i+1)/5, 'n': int(group.sum()),
                             'probability': float(probability[group].mean()),
                             'frequency': float(target[group].mean())})
        result[key] = {'accuracy': float(((probability >= .5) == target).mean()),
                       'brier': float(((probability - target)**2).mean()), 'calibration': bins}
    result['brier_improvement'] = result['base']['brier'] - result['candidate']['brier']
    if context is not None:
        from market_research.market.context import market_regimes
        regimes = market_regimes(context).reindex(frame.index).fillna('unknown')
        result['regimes'] = []
        for name in ('strong', 'weak', 'mixed', 'unknown'):
            mask = regimes == name
            count = int(mask.sum())
            item = {'regime': name, 'n': count, 'ok': count >= 20}
            if count:
                for key in ('base', 'candidate'):
                    p, y = frame.loc[mask, key], target[mask]
                    item[key] = {'accuracy': float(((p >= .5) == y).mean()),
                                 'brier': float(((p-y)**2).mean())}
            result['regimes'].append(item)
    return result


def analyze_technical(bars, display_start=None, adjust="qfq", reference=None, benchmark=""):
    display_start = display_start or bars.index[0]
    # Bound CPU and memory while keeping ~5 years for rolling model warm-up.
    truncated = len(bars) > 1500
    bars = bars.tail(1500).copy()
    ind = indicators(bars)
    latest, close = ind.iloc[-1], float(bars.close.iloc[-1])
    features = features_for(bars, ind)
    from market_research.market.context import context_features, context_payload
    extra = context_features(bars, ind, reference)
    available = extra.columns.tolist()
    # Schema is fixed by available data fields, never selected by future coverage.
    expanded = features.join(extra[available])
    ready = pd.notna(latest.score)
    if not ready:
        action, trend = "等待样本", "样本不足"
    elif latest.exit:
        action, trend = "减仓警戒", "偏弱" if latest.score <= -60 else "走弱"
    elif latest.entry:
        action, trend = "关注买入条件", "偏强"
    elif latest.score >= 60:
        action, trend = "持有观察", "偏强"
    else:
        action, trend = "等待确认", "震荡"
    reasons = []
    if ready:
        reasons = [f"收盘价{'高于' if close > latest.ma20 else '低于'} MA20",
                   f"MA20 {'高于' if latest.ma20 > latest.ma60 else '低于或等于'} MA60",
                   f"MACD 动能{'偏强' if latest.dif > latest.dea else '偏弱'}",
                   f"RSI14 = {latest.rsi:.1f}" + ("，偏热，避免仅凭强趋势追价" if latest.rsi > 70 else "，偏冷，不单独构成买入理由" if latest.rsi < 30 else "，未触及 30/70 观察阈值")]
        if close > latest.boll_upper:
            reasons.append("价格高于布林上轨，未满足买入关注条件，等待回落确认")
    else:
        reasons = ["至少 65 根日线后才合成趋势条件；已有单项指标仍可查看。"]
    records = []
    for date, row in ind.loc[display_start:].iterrows():
        if row.buy_event or row.sell_event:
            records.append({"date": date.date().isoformat(), "type": "buy" if row.buy_event else "sell",
                            "label": "买入条件首次满足" if row.buy_event else "减仓条件首次满足",
                            "close": bars.close.loc[date], "score": row.score})
    visible = ind.loc[display_start:]
    notices = ["日线收盘后更新，条件首次满足是观察信号，不假设能在信号日收盘价成交。",
               "预测周期按该证券的有效日线计数，停牌或缺失日线会延长实际日期间隔。",
               "模型估计未来收盘相对当前收盘上涨的概率；不预测精确价格，不含交易费用，未单独校准。",
               "历史验证采用当前数据版本；复权历史可能修订，未模拟停牌、涨跌停、滑点和实际成交。"]
    if truncated:
        notices.append("技术指标与模型最多使用最近 1500 根日线；更早价格仍可查看，技术叠加与评估不覆盖该段。")
    if adjust == "":
        notices.append("当前为不复权：分红除息可能产生机械性跳空，影响指标与方向标签；建议切换前复权观察趋势。")
    elif adjust == "hfq":
        notices.append("当前为后复权：所有价位均为后复权坐标，不能直接用作下单价格。")
    else:
        notices.append("价位按前复权坐标显示；历史截止日分析时尤其需核对真实报价后再考虑执行。")
    stop = close - 2 * latest.atr if pd.notna(latest.atr) else np.nan
    forecasts = [direction_forecast(bars, features, h, display_start) for h in (5, 20)]
    for forecast in forecasts:
        candidate = direction_forecast(bars, expanded, forecast['horizon'], display_start) if available else None
        forecast['comparison'] = (compare_forecasts(bars, forecast, candidate, forecast['horizon'], display_start, extra)
                                  if candidate else {'n': 0, 'ok': False, 'note': '新增特征覆盖不足，保留原模型。'})
        forecast['candidate'] = ({'ok': candidate['ok'], 'probability': candidate['probability'],
                                  'features': available} if candidate else None)
    disagreements = [str(f["horizon"]) for f in forecasts if f["ok"] and ready and
                     ((latest.score >= 60 and f["probability"] < .4) or (latest.score <= -60 and f["probability"] > .6))]
    conflict = ("规则趋势与 " + " / ".join(disagreements) + " 日模型方向存在分歧，请结合验证表现等待确认。") if disagreements else None
    return {"as_of": bars.index[-1].date().isoformat(), "ready": ready, "trend": trend, "action": action, "conflict": conflict,
            'context': context_payload(extra, benchmark),
            "score": latest.score, "reasons": reasons,
            "latest": latest.to_dict(),
            "levels": {"support": latest.support, "resistance": latest.resistance,
                       "atr_stop_reference": stop if stop > 0 else None, "atr_distance": latest.atr / close},
            "rules": "趋势分 = 五项方向判断等权合成（收盘/MA20、MA20/MA60、MA60近5日斜率、DIF/DEA、20日动量），范围 −100～100。买入关注：分数≥60、RSI在45～70且未突破布林上轨；减仓警戒：分数≤−20，或收盘低于MA20且MACD柱为负。均为规则条件。",
            "series": {"dates": visible.index.strftime("%Y-%m-%d").tolist(), **{c: visible[c].tolist() for c in ("dif", "dea", "macd", "rsi", "boll_upper", "boll_lower", "atr", "score")}},
            "forecasts": forecasts,
            "signals": records[-12:][::-1], "notices": notices}
