"""固定清单的离线方向验证；仅读取本地CSV快照或同步缓存，不抓取行情。"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm

from market_research.market.analysis import normalize_code
from market_research.market.context import context_features
from market_research.market.technical import indicators, features_for, direction_forecast, compare_forecasts
from market_research.serialization import _jsafe
from market_research.validation.window import _iso

VARIANTS = ('base', 'volatility', 'volume', 'relative', 'all')
VERSION = 2


def validate_plan(plan):
    for key in ('start', 'development_start', 'holdout_start', 'end'):
        if _iso(plan[key], key) != plan[key]:
            raise ValueError(f'{key}需使用规范日期')
    if not plan['start'] < plan['development_start'] < plan['holdout_start'] < plan['end']:
        raise ValueError('需满足 start < development_start < holdout_start < end')
    if not plan.get('assets'):
        raise ValueError('固定样本清单不能为空')
    keys = []
    for asset in plan['assets']:
        code = normalize_code(asset['kind'], asset['code'])
        if code != asset['code']:
            raise ValueError('清单代码需使用规范市场前缀')
        normalize_code('index', asset.get('benchmark', 'sh000300'))
        keys.append((asset['kind'], code))
    if len(set(keys)) != len(keys):
        raise ValueError('固定样本清单有重复资产')
    return plan


def feature_variants(bars, reference):
    ind = indicators(bars)
    base, extra = features_for(bars, ind), context_features(bars, ind, reference)
    groups = {'volatility': ['atr_percentile', 'bandwidth_percentile', 'adx'],
              'volume': ['volume_ratio', 'amount_ratio', 'turnover_ratio'],
              'relative': ['relative20', 'relative60', 'benchmark_gap60', 'benchmark_return20']}
    out = {'base': base}
    for group, columns in groups.items():
        # Source schema, not future coverage, decides optional feature presence.
        present = [c for c in columns if c in extra]
        # An entirely absent group remains unavailable, never a baseline clone.
        out[group] = base.join(extra[present] if present else extra.reindex(columns=columns))
    out['all'] = base.join(extra)
    return out, extra


def paired_evidence(bars, base, candidate, horizon, family_size):
    """Exploratory one-sided HAC normal approximation, with planned-family Bonferroni."""
    target = (bars.close.shift(-horizon) > bars.close).astype(float).where(bars.close.shift(-horizon).notna())
    frame = pd.DataFrame({'base': base, 'candidate': candidate, 'y': target}, index=bars.index).dropna()
    positions = bars.index.get_indexer(frame.index)
    keep, last = [], -horizon
    for i, position in enumerate(positions):
        if position >= last + horizon:
            keep.append(i); last = position
    frame = frame.iloc[keep]
    n = len(frame)
    result = {'n': n, 'family_size': family_size, 'p_one_sided_hac_approx': None,
              'p_bonferroni_approx': None, 'lags': min(5, max(0, n-1))}
    if n < 20:
        return result
    d = ((frame.base-frame.y)**2 - (frame.candidate-frame.y)**2).to_numpy()
    mean = float(d.mean()); centered = d-mean
    variance = float(centered @ centered / n)
    for lag in range(1, result['lags']+1):
        variance += 2*(1-lag/(result['lags']+1))*float(centered[lag:] @ centered[:-lag]/n)
    if variance <= 1e-15:
        return result
    p = float(norm.sf(mean / np.sqrt(variance/n)))
    result.update(p_one_sided_hac_approx=p, p_bonferroni_approx=min(1., p*family_size))
    return result


def evaluate_asset(bars, reference, plan, family_size):
    bars = bars.loc[plan['start']:plan['end']]
    if bars.empty or not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError('行情为空、日期重复或未升序排列')
    if not np.isfinite(bars[['open','high','low','close']].to_numpy()).all() or (bars[['open','high','low','close']] <= 0).any().any():
        raise ValueError('需要有限且正的OHLC价格')
    variants, context = feature_variants(bars, reference)
    cutoff = (pd.Timestamp(plan['holdout_start'])-pd.Timedelta(days=1)).date().isoformat()
    output = {'training_cutoff': cutoff, 'features': {k: list(v.columns) for k,v in variants.items()}, 'horizons': {}}
    for horizon in (5, 20):
        forecasts = {key: direction_forecast(bars, values, horizon, bars.index[0], training_end=cutoff)
                     for key, values in variants.items()}
        probability = pd.DataFrame({k:v['series'] for k,v in forecasts.items()}, index=bars.index)
        stages = {}
        for stage, start, end in (
            ('development', plan['development_start'], cutoff),
            ('holdout', plan['holdout_start'], plan['end'])):
            window = bars.loc[start:end]
            rows = []
            for variant in VARIANTS[1:]:
                pair = probability.loc[window.index, ['base',variant]]
                common = pair.notna().all(axis=1)
                baseline = pair['base'].where(common).tolist()
                candidate = pair[variant].where(common).tolist()
                result = compare_forecasts(window, {'series': baseline}, {'series': candidate}, horizon, start, context) if len(window) else {'n': 0, 'ok': False}
                result['variant'] = variant
                result['evidence'] = paired_evidence(window, baseline, candidate, horizon, family_size)
                rows.append(result)
            stages[stage] = rows
        output['horizons'][str(horizon)] = stages
    output['note'] = '开发期滚动训练，最终测试期固定训练截止日；测试期标签及缩放统计不进入训练。各候选与基线使用各自共同成熟日期，不同候选样本可能不同，不能直接按Brier排名。'
    return output


def read_bars(asset, plan, root):
    if asset.get('csv'):
        frame = pd.read_csv(root / asset['csv'], parse_dates=['date']).set_index('date')
    else:
        from market_research.tasks.sync import cached
        frame = cached(asset['kind'], asset['code'], asset.get('adjust', 'qfq') if asset['kind']=='stock' else '', plan['start'], plan['end'])
    if frame is None or frame.empty:
        raise ValueError('本地行情缺失或缓存未覆盖清单截止日；请先同步或提供CSV快照')
    return frame


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes() + '|'.join(frame.columns).encode()).hexdigest()


def run_plan(plan, root, progress=None):
    validate_plan(plan)
    family_size = len(plan['assets']) * 4 * 2 * 2
    report = {'version': VERSION, 'manifest': plan, 'manifest_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
              'runtime': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'planned_comparisons': family_size, 'assets': [],
              'limitations': ['当前数据版本回放，不是时点可得数据证明。', '同一测试区间反复查看后不再是未触碰的最终测试集。',
                              'HAC正态近似及Bonferroni为探索性诊断，小样本不报p值；不能证明可交易收益。']}
    from market_research.market import technical, context
    report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), Path(technical.__file__), Path(context.__file__))}
    for asset in plan['assets']:
        if progress:
            progress(asset['code'])
        row = {'kind':asset['kind'], 'code':asset['code']}
        try:
            bars = read_bars(asset, plan, root).loc[plan['start']:plan['end']]
            benchmark = asset.get('benchmark', 'sh000300')
            ref_asset = {'kind':'index','code':benchmark}
            if asset.get('benchmark_csv'):
                ref_asset['csv'] = asset['benchmark_csv']
            reference = read_bars(ref_asset, plan, root).loc[plan['start']:plan['end']]
            if bars.empty or reference.empty:
                raise ValueError('清单区间内没有本地资产或基准行情')
            row.update(data_sha256=fingerprint(bars), benchmark_sha256=fingerprint(reference),
                       actual_start=bars.index[0].date().isoformat(), actual_end=bars.index[-1].date().isoformat(),
                       result=evaluate_asset(bars, reference.close, plan, family_size))
            row['status'] = 'evaluated' if any(r['n'] for h in row['result']['horizons'].values() for r in h['holdout']) else 'insufficient'
        except (ValueError, KeyError, OSError) as error:
            row.update(status='unavailable', error=str(error))
        report['assets'].append(row)
    report['complete'] = all(a['status']=='evaluated' for a in report['assets'])
    return _jsafe(report)


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    if Path(args.output).exists():
        parser.error('输出已存在，请使用新的报告文件名以保留既有试验记录')
    plan = json.loads(path.read_text(encoding='utf-8'))
    try:
        report = run_plan(plan, path.parent, progress=lambda code: print(f'评估 {code}…', flush=True))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f"报告已保存：{output}；完整样本覆盖：{report['complete']}")
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
