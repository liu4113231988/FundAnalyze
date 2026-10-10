# -*- coding: utf-8 -*-
"""候选指数池可用性探针（主力通道：新浪 stock_zh_index_daily）。"""
import warnings
warnings.filterwarnings("ignore")
import akshare as ak

POOL = {
    "宽基": {
        "sh000016": "上证50", "sh000300": "沪深300", "sh000905": "中证500",
        "sh000852": "中证1000", "sz399006": "创业板指", "sh000688": "科创50",
        "sh000903": "中证100", "sz399330": "深证100", "sh000985": "中证全指",
        "sz399303": "国证2000",
    },
    "市值风格": {
        "sz399314": "巨潮大盘", "sz399315": "巨潮中盘", "sz399316": "巨潮小盘",
        "sz399311": "国证1000", "sh000009": "上证380",
    },
    "价值成长/红利": {
        "sz399370": "国证成长", "sz399371": "国证价值",
        "sh000922": "中证红利", "sh000919": "300价值", "sh000918": "300成长",
    },
    "中证全指行业": {
        "sh000986": "全指能源", "sh000987": "全指材料", "sh000988": "全指工业",
        "sh000989": "全指可选", "sh000990": "全指消费", "sh000991": "全指医药",
        "sh000992": "全指金融", "sh000993": "全指信息", "sh000994": "全指电信",
        "sh000995": "全指公用",
    },
}

ok, bad = [], []
for grp, d in POOL.items():
    print(f"\n--- {grp} ---")
    for code, name in d.items():
        try:
            df = ak.stock_zh_index_daily(symbol=code)
            print(f"  [OK  ] {code} {name:8s} rows={len(df):5d} "
                  f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            ok.append(code)
        except Exception as e:
            print(f"  [FAIL] {code} {name:8s} {type(e).__name__}: {str(e)[:50]}")
            bad.append(code)

print("\n" + "=" * 70)
print(f"可用 {len(ok)} / {len(ok)+len(bad)}")
print("不可用：", bad)

# 申万一级行业
print("\n--- 申万一级行业 (index_hist_sw) ---")
try:
    df = ak.index_hist_sw(symbol="801010", period="day")
    print("  [OK  ] 801010 农林牧渔", df.shape, df.columns.tolist())
except Exception as e:
    print("  [FAIL]", type(e).__name__, str(e)[:80])

# 港股指数
print("\n--- 港股指数 ---")
for f, a in [("stock_hk_index_daily_sina", dict(symbol="HSI")),
             ("stock_hk_index_daily_em", dict(symbol="HSI"))]:
    try:
        df = getattr(ak, f)(**a)
        print(f"  [OK  ] {f}", df.shape, df.columns.tolist())
    except Exception as e:
        print(f"  [FAIL] {f}", type(e).__name__, str(e)[:60])
