# -*- coding: utf-8 -*-
"""Web 后端端到端测试：analyze → 各区块 payload 完整性。"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import json
import pandas as pd
from src.webapi import analyze, refit_single, correlation_block, nav_block, get_fund_name
from src import data as D
from src.models import MODEL_ORDER

code = sys.argv[1] if len(sys.argv) > 1 else "110022"
QUICK = ["enet_slsqp", "corr_topk", "pcr", "rf", "krr"]
REBAL = int(sys.argv[2]) if len(sys.argv) > 2 else 60

t0 = time.time()
res = analyze(code, models=QUICK, max_rebal=REBAL)
print(f"\n[analyze] 用时 {time.time()-t0:.1f}s  max_rebal={REBAL}")

# ---------- 完整性校验
def chk(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    return cond

fails = 0
print("\n[meta]")
m = res["meta"]
for k in ["code", "name", "n_days", "n_assets", "start", "end", "best_model", "elapsed"]:
    print(f"  {k:14s} = {m.get(k)}")
fails += not chk(res["code"] == code, "code 正确")
fails += not chk(bool(res["name"]) and res["name"] != "—", f"基金名已解析：{res['name']}")
fails += not chk(m["best_model"] in res["models"], f"best_model={m['best_model']} 存在于 models")

print("\n[nav]")
nv = res["nav"]
for k in ["dates", "nav", "ret", "drawdown", "vol60"]:
    fails += not chk(len(nv[k]) > 1000, f"nav.{k} 长度 {len(nv[k])}")
fails += not chk(len(nv["dates"]) == len(nv["nav"]) == len(nv["drawdown"]), "各序列长度一致")
fails += not chk(len(nv["monthly"]) > 5, f"月度收益年数 {len(nv['monthly'])}")
print("  stats:", json.dumps(nv["stats"], ensure_ascii=False)[:300])

print("\n[correlation]")
cr = res["correlation"]
fails += not chk(len(cr["top"]) == 15, f"Top 榜 {len(cr['top'])} 条")
fails += not chk(len(cr["matrix"]["rows"]) == len(cr["matrix"]["cols"]), "矩阵方阵")
fails += not chk(len(cr["layers"]) >= 3, f"大类 {len(cr['layers'])} 个")
fails += not chk(len(cr["rolling"]) >= 3, f"滚动相关 {len(cr['rolling'])} 条")
print("  Top5:", [(r['name'], r['corr']) for r in cr['top'][:5]])
print("  大类:", [(r['layer'], r['avg_corr']) for r in cr['layers']])
print("  漂移:", cr["drift"])

print("\n[models]")
for k in QUICK:
    mm = res["models"][k]
    ok = (mm["metrics"]["oos_r2"] is not None
          and len(mm["curve"]["dates"]) > 50
          and len(mm["latest_weights"]["items"]) > 0)
    fails += not chk(ok, f"{k:12s} R²={mm['metrics']['oos_r2']} TE={mm['metrics']['oos_te']} "
                         f"权重{len(mm['latest_weights']['items'])}项 "
                         f"曲线{len(mm['curve']['dates'])}点")
    # 权重和校验
    ws = mm["latest_weights"]["sum"]
    if mm["interpretable"]:
        fails += not chk(abs(ws - 0.90) < 0.02, f"  └ 可解释权重和={ws:.3f} 应≈0.90")

print("\n[rank]")
rk = pd.DataFrame(res["rank"])
print(rk[["模型", "流派", "可解释", "样本外R²", "样本外TE", "窗口内R²", "方向一致率"]]
      .round(4).to_string(index=False))
fails += not chk(len(rk) == len(QUICK), f"排名 {len(rk)} 行")
fails += not chk(rk["样本外R²"].is_monotonic_decreasing, "按样本外R²降序")
fails += not chk(m["n_rebal"] <= max(REBAL, 1), f"实际重估点数 {m['n_rebal']}")

print("\n[insights]")
for k, v in res["insights"].items():
    print(f"  · {k}: {str(v.get('text',''))[:150]}...")

print("\n[JSON 序列化]")
try:
    s = json.dumps(res, ensure_ascii=False)
    fails += not chk(len(s) > 100000, f"JSON 长度 {len(s)/1024:.0f} KB")
    json.loads(s)
    fails += not chk(True, "JSON 可往返")
except Exception as e:
    fails += not chk(False, f"JSON 序列化失败: {e}")

print("\n[refit 单模型重学]")
t1 = time.time()
rf = refit_single(code, "rf", {"n_estimators": 60, "max_depth": 4}, max_rebal=REBAL)
print(f"  用时 {time.time()-t1:.1f}s  R²={rf['metrics']['oos_r2']}")
fails += not chk(rf["metrics"]["oos_r2"] is not None, "refit 返回有效结果")

print("\n[name 解析]")
for c in ["110022", "000001"]:
    print(f"  {c} → {get_fund_name(c)}")

print(f"\n{'='*60}")
print(f"失败项：{fails}")
print("=" * 60)