from __future__ import annotations
from typing import Iterable
import pandas as pd
from market_research.data.http import _retry

def fetch_pe_panel(symbols: Iterable[str], lookback_years: int = 8,
                   verbose: bool = True) -> pd.DataFrame:
    """
    拉取宽基指数的滚动市盈率历史（月度），并转为 **滚动历史分位**（0~1）。

    用途：判断风格基准的"绝对估值水位"。分位=1 表示处于历史最贵档。
    返回列名为指数中文名。
    """
    import akshare as ak

    out = {}
    for sym in symbols:
        try:
            df = _retry(ak.stock_index_pe_lg, tries=2, wait=1.0, symbol=sym)
            df["日期"] = pd.to_datetime(df["日期"])
            s = df.set_index("日期")["滚动市盈率"].astype(float).sort_index()
            s = s[s > 0]
            win = 12 * lookback_years
            pct = s.rolling(win, min_periods=win // 3).apply(
                lambda a: float((a <= a[-1]).mean()), raw=True)
            out[sym] = pct
            if verbose:
                print(f"  [pe] {sym:8s} {len(s):4d} 期  {s.index[0].date()} ~ "
                      f"{s.index[-1].date()}  当前分位={pct.iloc[-1]:.0%}")
        except Exception as e:      # noqa: BLE001
            if verbose:
                print(f"  [pe] {sym:8s} 失败 {type(e).__name__} {str(e)[:40]}")
    return pd.DataFrame(out).sort_index()
