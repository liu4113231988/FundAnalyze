# -*- coding: utf-8 -*-
"""
多模型引擎冒烟测试：单窗口 + 滚动样本外排名。

★ P1-09：进程池用 `spawn`（见 src/models.py），子进程会**重新导入本模块**。
若业务代码留在模块顶层，每个子进程都会把整套流程再跑一遍 —— 包括再次启动
进程池 —— 形成递归。因此除 import 与常量之外的一切都必须放进 main()。
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src import config as C, data as D
from src.models import ModelZoo, fit_window, MODELS, MODEL_ORDER, model_catalog

pd.set_option("display.width", 200)


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "110022"
    panel = D.load_panel(code, verbose=False)
    nav = panel["__FUND__"]
    print(f"[data] {code} panel={panel.shape} "
          f"{panel.index[0].date()} ~ {panel.index[-1].date()}")

    # ---------------------------------------------------- 1. 单窗口逐模型拟合
    W = C.RBSA["window"]
    R = np.log(panel.astype(float)).diff().dropna()
    assets = [c for c in panel.columns if c != "__FUND__"]
    X = R[assets].values[-W:]
    y = R["__FUND__"].values[-W:]

    print(f"\n{'='*96}\n[1] 单窗口拟合（{W} 日窗口，样本末段）\n{'='*96}")
    rows = []
    for k in MODEL_ORDER:
        t0 = time.time()
        try:
            fr = fit_window(k, X, y, assets)
            top = ", ".join(f"{C.NAME_OF.get(i, i)}={v:.0%}"
                            for i, v in fr.weights.head(3).items())
            rows.append({"模型": k, "R²": fr.r2, "TE": fr.te,
                         "非零权重": int((fr.weights > 0).sum()),
                         "耗时s": time.time() - t0, "Top3": top,
                         # ★ 负 R² 不再被截到 0（P0-05），这里如实显示
                         "收敛率": fr.extra.get("fit_ok")})
            print(f"[OK  ] {k:12s} R²={fr.r2:.4f} TE={fr.te:6.2%} "
                  f"非零={int((fr.weights > 0).sum()):2d} {time.time()-t0:5.2f}s  {top}")
        except Exception as e:
            import traceback
            print(f"[FAIL] {k:12s} {type(e).__name__}: {str(e)[:80]}")
            traceback.print_exc()
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))

    # ---------------------------------------------------- 2. 滚动样本外
    print(f"\n{'='*96}\n[2] 滚动样本外评估（全 {len(MODEL_ORDER)} 模型）\n{'='*96}")
    t0 = time.time()
    zoo = ModelZoo()
    reps = zoo.run(panel, MODEL_ORDER,
                   progress=lambda k, d, t, r: print(f"   ({d}/{t}) {k:12s} OOS R²={r:.4f}"))
    print(f"总耗时 {time.time()-t0:.1f}s")

    rank = ModelZoo.rank(reps)
    print("\n" + rank.round(4).to_string(index=False))
    print("\n按流派分组：")
    print(rank.groupby("流派")[["样本外R²", "样本外TE"]].mean().round(4).to_string())

    # 降级可见性（P0-05）：黑箱模型若发生"回退成重要性权重"，必须能看出来
    fb = [(k, r.extra.get("fallback_ratio"))
          for k, r in reps.items() if r.extra.get("fallback_ratio")]
    if fb:
        print("\n★ 发生黑箱回退的模型：")
        for k, v in fb:
            print(f"   {k:12s} 回退窗口占比 {v:.1%}")

    # ---------------------------------------------------- 3. 一致性校验
    best = rank.iloc[0]
    print(f"\n最佳模型：{best['模型']}  OOS R²={best['样本外R²']:.4f}  TE={best['样本外TE']:.2%}")
    br = reps[best["key"]]
    print("最新权重：")
    print(pd.DataFrame({"名称": [C.NAME_OF.get(i, i) for i in br.latest_weights.index],
                        "权重": br.latest_weights.values}).head(8).to_string(index=False))
    print(f"合成基准收益长度 {len(br.syn_ret)}，滚动权重矩阵 {br.weights_hist.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
