# -*- coding: utf-8 -*-
"""
Web 后端端到端测试：analyze → 各区块 payload 完整性。

    python tests/test_webapi.py [代码] [max_rebal]      # 退出码 0 = 全部通过

★★ 为什么必须包在 main() 里：
ModelZoo 的进程池用 **spawn** 上下文，spawn 会在子进程中重新 import `__main__`。
本文件原来是顶层脚本代码，没有 `if __name__ == "__main__"` 保护，于是每个子进程
都会把整个测试脚本重跑一遍 —— 结果是 5 个模型全部返回 R²=None，
而这个测试恰恰又没有 sys.exit()，所以"全 FAIL"一直没人发现。
任何调用 analyze() / ModelZoo.run() 的脚本都必须有 main 保护。

注意：脚本里用 analyze() 直连，**不经过 Flask**，因此不覆盖路由层的参数校验。
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import json
import pandas as pd
from src.webapi import (analyze, refit_single, correlation_block,
                        nav_block, get_fund_name)
from src import data as D
from src.models import MODEL_ORDER

QUICK = ["enet_slsqp", "corr_topk", "pcr", "rf", "krr"]

fails = 0


def chk(cond, msg):
    global fails
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails += 1


def main():
    global fails
    code = sys.argv[1] if len(sys.argv) > 1 else "110022"
    rebal = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    t0 = time.time()
    res = analyze(code, models=QUICK, max_rebal=rebal)
    print(f"\n[analyze] 用时 {time.time()-t0:.1f}s  max_rebal={rebal}")

    print("\n[meta]")
    m = res["meta"]
    for k in ["code", "name", "n_days", "n_assets", "start", "end",
              "best_model", "elapsed", "n_rebal"]:
        print(f"  {k:14s} = {m.get(k)}")
    chk(res["code"] == code, "code 正确")
    chk(bool(res["name"]) and res["name"] != "—", f"基金名已解析：{res['name']}")
    chk(m["best_model"] in res["models"], f"best_model={m['best_model']} 存在于 models")

    print("\n[nav]")
    nv = res["nav"]
    for k in ["dates", "nav", "ret", "drawdown", "vol60"]:
        chk(len(nv[k]) > 1000, f"nav.{k} 长度 {len(nv[k])}")
    chk(len(nv["dates"]) == len(nv["nav"]) == len(nv["drawdown"]), "各序列长度一致")
    chk(len(nv["monthly"]) > 5, f"月度收益年数 {len(nv['monthly'])}")
    print("  stats:", json.dumps(nv["stats"], ensure_ascii=False)[:300])

    print("\n[correlation]")
    cr = res["correlation"]
    chk(len(cr["top"]) == 15, f"Top 榜 {len(cr['top'])} 条")
    chk(len(cr["matrix"]["rows"]) == len(cr["matrix"]["cols"]), "矩阵方阵")
    chk(len(cr["layers"]) >= 3, f"大类 {len(cr['layers'])} 个")
    chk(len(cr["rolling"]) >= 3, f"滚动相关 {len(cr['rolling'])} 条")
    print("  Top5:", [(r['name'], r['corr']) for r in cr['top'][:5]])
    print("  大类:", [(r['layer'], r['avg_corr']) for r in cr['layers']])
    print("  漂移:", cr["drift"])

    print("\n[models]")
    for k in QUICK:
        mm = res["models"][k]
        ok = (mm["metrics"]["oos_r2"] is not None
              and len(mm["curve"]["dates"]) > 50
              and len(mm["latest_weights"]["items"]) > 0)
        chk(ok, f"{k:12s} R²={mm['metrics']['oos_r2']} TE={mm['metrics']['oos_te']} "
                f"权重{len(mm['latest_weights']['items'])}项 "
                f"曲线{len(mm['curve']['dates'])}点"
                + ("" if ok else f"  extra={mm.get('extra')}"))
        ws = mm["latest_weights"]["sum"]
        if mm["interpretable"] and ok:
            chk(abs(ws - 0.90) < 0.02, f"  └ 可解释权重和={ws:.3f} 应≈0.90")

    print("\n[rank]")
    rk = pd.DataFrame(res["rank"])
    print(rk[["模型", "流派", "可解释", "样本外R²", "样本外TE", "窗口内R²", "方向一致率"]]
          .round(4).to_string(index=False))
    chk(len(rk) == len(QUICK), f"排名 {len(rk)} 行")
    chk(bool(rk["样本外R²"].is_monotonic_decreasing), "按样本外R²降序")
    chk(m["n_rebal"] <= max(rebal, 1), f"实际重估点数 {m['n_rebal']}")

    print("\n[insights]")
    for k, v in res["insights"].items():
        print(f"  · {k}: {str(v.get('text', ''))[:150]}...")

    print("\n[JSON 序列化]")
    try:
        s = json.dumps(res, ensure_ascii=False)
        chk(len(s) > 100000, f"JSON 长度 {len(s)/1024:.0f} KB")
        json.loads(s)
        chk(True, "JSON 可往返")
    except Exception as e:                                    # noqa: BLE001
        chk(False, f"JSON 序列化失败: {e}")

    print("\n[refit 单模型重学]")
    t1 = time.time()
    rf = refit_single(code, "rf", {"n_estimators": 60, "max_depth": 4},
                      max_rebal=rebal)
    print(f"  用时 {time.time()-t1:.1f}s  R²={rf['metrics']['oos_r2']}")
    chk(rf["metrics"]["oos_r2"] is not None, "refit 返回有效结果")

    print("\n[name 解析]")
    for c in ["110022", "000001"]:
        print(f"  {c} → {get_fund_name(c)}")

    print("\n[相关性滚动窗口可调]")
    try:
        panel = D.load_panel(code, verbose=False)
        c120 = correlation_block(panel, panel["__FUND__"], roll_n=120)
        c250 = correlation_block(panel, panel["__FUND__"], roll_n=250)
        chk(c120["roll_window"] == 120 and c250["roll_window"] == 250,
            "roll_window 随入参变化（前端滚动窗口控件生效）")
        kk = next(iter(c120["rolling"]))
        chk(len(c120["rolling"][kk]["values"]) != len(c250["rolling"][kk]["values"]),
            f"不同窗口产出不同长度的滚动序列 "
            f"({len(c120['rolling'][kk]['values'])} vs {len(c250['rolling'][kk]['values'])})")
    except Exception as e:                                    # noqa: BLE001
        chk(False, f"correlation_block(roll_n=) 失败：{type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"失败项：{fails}")
    print("=" * 60)
    return 1 if fails else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
