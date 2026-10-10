# -*- coding: utf-8 -*-
"""数据源可用性探针：逐一测试候选接口，输出可用/不可用清单。"""
import warnings, traceback, json
warnings.filterwarnings("ignore")

RESULTS = []


def probe(name, fn):
    try:
        r = fn()
        if r is None:
            RESULTS.append((name, "EMPTY", "返回 None"))
        else:
            RESULTS.append((name, "OK", f"shape={getattr(r, 'shape', len(r))}"))
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:90]}"))


import akshare as ak
import pandas as pd
print("akshare", ak.__version__, "| pandas", pd.__version__)

# ---------- 基金净值 ----------
probe("ak.fund_open_fund_info_em(单位净值)",
      lambda: ak.fund_open_fund_info_em(symbol="000001", indicator="单位净值走势"))
probe("ak.fund_open_fund_info_em(累计净值)",
      lambda: ak.fund_open_fund_info_em(symbol="000001", indicator="累计净值走势"))

# ---------- 指数行情 ----------
probe("ak.index_zh_a_hist(000300)",
      lambda: ak.index_zh_a_hist(symbol="000300", period="daily",
                                 start_date="20230101", end_date="20240110"))
probe("ak.stock_zh_index_daily_em(sh000300)",
      lambda: ak.stock_zh_index_daily_em(symbol="sh000300"))
probe("ak.stock_zh_index_daily(sh000300)",
      lambda: ak.stock_zh_index_daily(symbol="sh000300"))
probe("ak.stock_zh_index_daily(sz399006)",
      lambda: ak.stock_zh_index_daily(symbol="sz399006"))
probe("ak.stock_zh_index_spot_em",
      lambda: ak.stock_zh_index_spot_em(symbol="沪深重要指数"))
probe("ak.index_stock_info",
      lambda: ak.index_stock_info())

# ---------- 备用通道 ----------
def sohu():
    import requests
    r = requests.get("https://q.stock.sohu.com/hisHq",
                     params={"code": "zs_000300", "start": "20230101", "end": "20240110"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    j = r.json()
    assert j and j[0].get("hq"), "empty hq"
    return pd.DataFrame(j[0]["hq"])


probe("sohu hisHq(zs_000300)", sohu)

# ---------- 风格/行业指数（中证系列） ----------
for code in ["000905", "000852", "000016", "399006", "000922"]:
    probe(f"sohu hisHq(zs_{code})",
          lambda c=code: pd.DataFrame(__import__("requests").get(
              "https://q.stock.sohu.com/hisHq",
              params={"code": f"zs_{c}", "start": "20230101", "end": "20240110"},
              headers={"User-Agent": "Mozilla/5.0"}, timeout=25).json()[0]["hq"]))

print("\n" + "=" * 78)
for n, s, d in RESULTS:
    print(f"[{s:5s}] {n:45s} {d}")
print("=" * 78)
ok = sum(1 for _, s, _ in RESULTS if s == "OK")
print(f"可用: {ok}/{len(RESULTS)}")
