# -*- coding: utf-8 -*-
"""基金净值与申万行业数据探针：挑选适合做演示的样本基金。"""
import warnings
warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

FUNDS = {
    "110022": "易方达消费行业(萧楠)",
    "163402": "兴全趋势投资",
    "000083": "汇添富消费行业",
    "519066": "汇添富蓝筹稳健",
    "260108": "景顺长城新兴成长",
    "001102": "前海开源国家比较优势",
    "519697": "交银优势行业",
    "110011": "易方达优质精选",
    "000001": "华夏成长",
    "005827": "易方达蓝筹精选",
}

print("=" * 88)
for code, name in FUNDS.items():
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")
        col = [c for c in df.columns if "累计" in c][0]
        print(f"[OK  ] {code} {name:22s} rows={len(df):5d}  "
              f"{df.iloc[0,0]} ~ {df.iloc[-1,0]}  末值={df[col].iloc[-1]}")
    except Exception as e:
        print(f"[FAIL] {code} {name:22s} {type(e).__name__}: {str(e)[:50]}")

print("\n--- 申万一级行业数据时效性（抽样） ---")
for code, name in [("801010", "农林牧渔"), ("801120", "食品饮料"), ("801080", "电子"),
                   ("801150", "医药生物"), ("801780", "银行"), ("801730", "电力设备")]:
    try:
        df = ak.index_hist_sw(symbol=code, period="day")
        print(f"  [OK  ] {code} {name:6s} rows={len(df):5d} "
              f"{df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")
    except Exception as e:
        print(f"  [FAIL] {code} {name:6s} {type(e).__name__}: {str(e)[:50]}")

print("\n--- 申万一级行业完整清单 ---")
try:
    info = ak.sw_index_first_info()
    print(info.to_string())
except Exception as e:
    print("sw_index_first_info FAIL:", type(e).__name__, str(e)[:80])
    try:
        info = ak.sw_index_third_info()
        print(info.head(5).to_string())
    except Exception as e2:
        print("sw_index_third_info FAIL:", type(e2).__name__, str(e2)[:80])
