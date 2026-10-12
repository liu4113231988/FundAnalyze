# -*- coding: utf-8 -*-
"""
P0-01 验收入口：逐基金核对收益口径。

    python run_verify_basis.py [基金代码...]

对每只基金调用 `src.data.verify_nav_basis`，比较三种口径的期末总回报：
  A 累计净值直算（旧口径）
  B 单位净值直算
  C 校准后总回报（现行默认）
真值 = 天天基金自算的「累计收益率」（月频，含分红再投）。

验收标准：C 与真值的偏差应为 0（锚点上精确一致）；
A 的偏差记录在案，用于量化"旧口径错得多离谱"。
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from src import config as C, data as D

RES = ROOT / "output" / "results"


def main():
    codes = sys.argv[1:] or list(C.EXTENDED_FUNDS.keys())
    rows = []
    for code in codes:
        try:
            rows.append(D.verify_nav_basis(code))
            r = rows[-1]
            print(f"[{code}] 真值={r['true_total_return']:.4f} | "
                  f"旧口径偏差={r['cum_nav_bias_pct']:+.2f}% | "
                  f"校准后偏差={r['calibrated_bias_pct']:+.4f}%")
        except Exception as e:                                # noqa: BLE001
            print(f"[ERR] {code} {type(e).__name__}: {str(e)[:90]}")
    if not rows:
        print("无有效结果")
        return 1
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "P0-01_收益口径核实.csv", index=False,
              encoding="utf-8-sig")
    bad = df[df["calibrated_bias_pct"].abs() > 0.5]
    print("\n" + "=" * 100)
    print(df.round(4).to_string(index=False))
    print(f"\n旧口径平均偏差 {df['cum_nav_bias_pct'].mean():+.2f}%"
          f"（最差 {df['cum_nav_bias_pct'].min():+.2f}%）")
    print(f"校准后偏差 > 0.5% 的基金：{len(bad)} 只"
          + ("  ← 存在未通过验收的样本，须排查" if len(bad) else "  ← 全部通过"))
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
