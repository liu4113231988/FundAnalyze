# -*- coding: utf-8 -*-
"""
基金真实持仓 vs RBSA 拟合持仓的对齐（2026-10-13）。

为什么要有这个模块
----------------
RBSA 用指数收益回归推断"这只基金在赌哪些行业"，权重可解释，但它终究是
**推断** —— 没有直接证据说拟合出的 37% 食品饮料就是真实的 37%。
季报 / 半年报的持仓明细是唯一的**直接 ground truth**。

本模块做两件事：

  1. `align_vs_rbsa()`    **验证**：逐行业对比 RBSA 权重与真实持仓暴露，
                         量化偏差。回答"RBSA 说的和真实的一致吗"。
  2. `holding_benchmark()` **替代**：只用真实持仓构造基准组合，
                         与 RBSA 合成基准比收益。回答"如果直接拿真实持仓
                         当基准，还 ML 拟合干什么"。

数据源与关键限制（每条都会写进输出，不能隐瞒）
------------------------------------------
* `ak.fund_portfolio_hold_em(symbol, date)` 按**年**取，返回该年各季度明细。
* **季报（Q1/Q3）只披露前十大重仓**：实测 110022 每期仅 11~16 只、
  占净值 77~87%；**半年报（Q2）与年报（Q4）才是完整持仓**（48~100 只、
  87~91%）。因此本模块**只做半年度对齐**（`semiannual_exposure` 只取
  Q2/Q4），季度数据一律不参与 —— 用不完整的前十大做聚合会系统性低估
  小仓位行业，得出的"偏差"是披露规则造成的，不是模型造成的。
* **披露滞后**：Q2 持仓 7 月才公布、Q4 持仓次年 1 月才公布。
  事后验证不受影响，但 `holding_benchmark` 必须从**披露之后**起算，
  否则等于用未来信息构造历史基准（前瞻偏差），比出来的收益不可信。
* 行业口径为**申万一级**（`801xxx`），与项目的 31 个候选行业指数同源，
  不需要任何口径转换。映射表由 31 个指数的成分股并集构造（实测 5221 只）。
* RBSA 权重里还有 16 个**宽基**指数（沪深300 等），持仓只有行业、
  宽基是跨行业的，两者无法一一对应。因此对齐只覆盖**行业部分**，
  宽基暴露单独报告（见 `align_vs_rbsa` 的 `broad_note`）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .fileio import _atomic_csv_write

# 披露滞后的保守取值（自然日）。季末 → 公布：Q2 约 7 月下旬、Q4 约 1 月下旬，
# 法定披露截止分别是 7/31 与 1/31。取 45 天留出余量。
DISCLOSURE_LAG_DAYS = 45

# 缓存版本：改行业口径或持仓列解析方式时必须 +1
HOLDINGS_CACHE_VERSION = 1


# ====================================================================== 行业映射
def _industry_index_codes() -> Dict[str, str]:
    """候选池里的申万一级行业指数（代码 801xxx）。"""
    return {k: v for k, v in C.NAME_OF.items() if str(k).startswith("801")}


def load_industry_map(refresh: bool = False,
                      verbose: bool = True) -> pd.Series:
    """
    股票代码 → 申万一级行业名 的映射（Series，index=6 位代码，value=行业名）。

    构造方式：遍历候选池的 31 个申万一级行业指数，取各自成分股，取并集。
    实测 5221 只股票、约 14 秒（一次性，之后走缓存）。

    ★ 为什么不用东财/中信行业：口径与候选指数（申万）不一致，转换表要人工
      维护且容易错。申万成分股与候选指数同源，零转换成本。
    """
    fp = C.DATA_DIR / f"industry_map_v{HOLDINGS_CACHE_VERSION}.csv"
    if fp.exists() and not refresh:
        try:
            s = pd.read_csv(fp, index_col=0, dtype=str).iloc[:, 0]
            s = s.dropna()
            if len(s):
                return s
        except Exception:                                     # noqa: BLE001
            pass                                    # 缓存损坏则重建

    import akshare as ak
    inds = _industry_index_codes()
    rows = []
    t0 = time.time()
    for code, name in inds.items():
        try:
            d = ak.index_component_sw(symbol=code)
            col = "证券代码" if "证券代码" in d.columns else d.columns[1]
            for c in d[col].astype(str).str.zfill(6):
                rows.append((c, name))
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print(f"[holdings] ⚠ 行业 {code} {name} 成分股获取失败："
                      f"{type(e).__name__}")

    m = (pd.DataFrame(rows, columns=["code", "industry"])
         .drop_duplicates("code", keep="first")
         .set_index("code")["industry"])
    m.name = "industry"
    _atomic_csv_write(m.to_frame(), fp)
    if verbose:
        print(f"[holdings] 行业映射：{len(m)} 只股票 / {len(inds)} 个行业"
              f"（{time.time() - t0:.1f}s）")
    return m


# ====================================================================== 基金持仓
def load_fund_holdings(code: str, refresh: bool = False,
                       verbose: bool = True,
                       start_year: Optional[int] = None) -> pd.DataFrame:
    """
    基金全部持仓明细（按年抓取后合并）。

    返回列：股票代码 / 股票名称 / 占净值比例 / 持股数 / 持仓市值 / 季度 /
            期数(如 2026Q2) / 季末(date) / 完整(bool)

    `完整` = 该期是半年报/年报（全部持仓）还是季报（仅前十大）。
    下游**只应使用 `完整 == True`** 的期数做聚合。
    """
    fp = C.DATA_DIR / f"fund_holdings_{code}_v{HOLDINGS_CACHE_VERSION}.csv"
    if fp.exists() and not refresh:
        try:
            df = pd.read_csv(fp)
            if len(df):
                # ★ 必须重新规整类型：CSV 没有类型信息，"600519" 会被读成 int，
                #   丢掉前导零后与 industry_map 的字符串索引完全对不上 ——
                #   实测症状是"781 条持仓全部未映射"，而联网直取时 100% 命中。
                return _retype_holdings(df)
        except Exception:                                     # noqa: BLE001
            pass

    import akshare as ak
    if start_year is None:
        start_year = int(pd.Timestamp(C.START_DATE).year)
    end_year = int(pd.Timestamp.today().year)
    frames = []
    for yr in range(start_year, end_year + 1):
        try:
            d = ak.fund_portfolio_hold_em(symbol=code, date=str(yr))
            if d is None or len(d):
                frames.append(d)
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print(f"[holdings] ⚠ {code} {yr} 年持仓获取失败："
                      f"{type(e).__name__}")
        time.sleep(0.15)                      # 轻微限速，避免被限流

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = _normalize_holdings(df)
    _atomic_csv_write(df, fp)
    return _retype_holdings(df)


def _retype_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """
    把（缓存读回或新建的）持仓表规整成统一类型。

    CSV 往返会丢类型：股票代码变 int（丢前导零）、比例列变 str、季末变 str。
    不规整的后果是 `股票代码.map(industry_map)` 全军覆没，
    而且**不报错** —— 只是静默地什么都映射不上。
    """
    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    for c in ("占净值比例", "持股数", "持仓市值"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "季末" in df.columns:
        df["季末"] = pd.to_datetime(df["季末"], errors="coerce")
    if "完整" in df.columns:
        df["完整"] = df["完整"].astype(bool)
    return df


def _normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """把原始明细规整成统一列，并解析季度标签 → 期数 / 季末 / 完整标记。"""
    out = pd.DataFrame(index=df.index)
    out["股票代码"] = df.get("股票代码", pd.Series(index=df.index,
                                                  dtype=str)).astype(str).str.zfill(6)
    out["股票名称"] = df.get("股票名称", pd.Series(index=df.index, dtype=str))
    # 占净值比例可能是 "9.90" 或 9.9；脏值一律转 NaN，绝不静默当 0
    pct = pd.to_numeric(df.get("占净值比例"), errors="coerce")
    out["占净值比例"] = pct
    out["持股数"] = pd.to_numeric(df.get("持股数"), errors="coerce")
    out["持仓市值"] = pd.to_numeric(df.get("持仓市值"), errors="coerce")
    q = df.get("季度", pd.Series(index=df.index, dtype=str)).astype(str)
    out["季度"] = q
    out["期数"] = q.str.extract(r"(\d{4})年(\d)季度")[0] + \
                 "Q" + q.str.extract(r"(\d{4})年(\d)季度")[1]
    ye = pd.to_numeric(q.str.extract(r"(\d{4})年(\d)季度")[0], errors="coerce")
    qq = pd.to_numeric(q.str.extract(r"(\d{4})年(\d)季度")[1], errors="coerce")
    # 季末：Q2=6/30、Q4=12/31（只保留这两期，其它在下游过滤）
    out["季末"] = [
        pd.Timestamp(int(y), 6, 30) if k == 2 else
        (pd.Timestamp(int(y), 12, 31) if k == 4 else pd.NaT)
        for y, k in zip(ye, qq)
    ]
    out["季末"] = pd.to_datetime(out["季末"])
    # 完整持仓：半年报(Q2)与年报(Q4)披露全部，季报(Q1/Q3)只有前十大
    out["完整"] = qq.isin([2, 4])
    return out.dropna(subset=["季末"])


def semiannual_exposure(holdings: pd.DataFrame,
                        industry_map: pd.Series,
                        only_full: bool = True) -> pd.DataFrame:
    """
    按申万一级行业聚合的**半年度**暴露（index=季末, columns=行业, values=占净值比例%）。

    only_full=True 时只取 Q2/Q4（完整持仓）。这是唯一诚实的口径 ——
    Q1/Q3 只有前十大，聚合出来的小仓位行业会被系统性低估。
    """
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    h = holdings[holdings["完整"]] if only_full else holdings
    h = h[h["占净值比例"].notna()].copy()
    if h.empty:
        return pd.DataFrame()
    # 防御性再规整一次：调用方可能直接传入未经 _retype_holdings 的表
    h["股票代码"] = h["股票代码"].astype(str).str.zfill(6)
    h["industry"] = h["股票代码"].map(industry_map)
    unmapped = h[h["industry"].isna()]
    if len(unmapped):
        print(f"[holdings] ⚠ {len(unmapped)} 条持仓未映射到申万一级行业"
              f"（权重合计 {unmapped['占净值比例'].sum():.2f}%），"
              f"多为新股/北交所/停牌股")
    h = h[h["industry"].notna()]
    exp = (h.groupby(["季末", "industry"])["占净值比例"].sum()
           .unstack("industry").sort_index())
    return exp


# ====================================================================== 对齐
def _rbsa_at(weights: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    """取 `when` 当日或之前最近一次重估的权重。"""
    idx = weights.index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    pos = idx.searchsorted(when, side="right") - 1
    if pos < 0:
        return pd.Series(dtype=float)
    return weights.iloc[pos]


def _exposure_to_codes(exposure: pd.DataFrame) -> pd.DataFrame:
    """
    把暴露表的列从**行业名**统一成**指数代码**（已是代码则原样返回）。

    ★ 为什么必须转：`semiannual_exposure` 产出的是行业名（"食品饮料"），
      而 RBSA 权重与 panel 的列都是指数代码（"801120"）。两者直接
      `intersection` / `in` 会**静默得到空集** —— 不报错，只是什么都对不上。
    """
    name2code = {v: k for k, v in _industry_index_codes().items()}
    return exposure.rename(columns=lambda c: name2code.get(str(c), str(c)))


def align_vs_rbsa(weights: pd.DataFrame, exposure: pd.DataFrame,
                  industry_map: Optional[pd.Series] = None
                  ) -> dict:
    """
    逐行业对比 RBSA 权重与真实持仓暴露。

    口径说明（必须看清，否则数字会被误读）：
      · RBSA 权重含 16 个宽基指数，这里**只取行业部分**（801xxx）并归一到 100%；
      · 真实暴露同样归一到 100%（原始合计约 87~91%，差额是现金/债券）；
      · 两边都归一化后比的是**结构**（"37% 的股票仓位给了食品饮料"），
        不是绝对仓位。绝对仓位的对比另见 `exposure_raw`。

    返回 dict，`ok=False` 表示样本不足以判断（不抛异常）。
    """
    inds = _industry_index_codes()
    if weights is None or weights.empty or exposure is None or exposure.empty:
        return {"ok": False, "reason": "RBSA 权重或真实持仓为空"}
    exposure = _exposure_to_codes(exposure)

    rows, detail = [], []
    for qend in exposure.index:
        w = _rbsa_at(weights, qend)
        if w.empty:
            continue
        w_ind = w[[c for c in w.index if str(c).startswith("801")]]
        w_broad = w[[c for c in w.index if not str(c).startswith("801")]]
        e = exposure.loc[qend].dropna()
        common = w_ind.index.intersection(e.index)
        if len(common) < 2:
            continue
        ws = w_ind[common] / w_ind[common].sum() if w_ind[common].sum() > 0 else w_ind[common]
        es = e[common] / e[common].sum()
        dev = (ws - es)
        rows.append({
            "季末": qend,
            "披露约": qend + pd.Timedelta(days=DISCLOSURE_LAG_DAYS),
            "RBSA重估日": weights.index[weights.index.searchsorted(
                qend, side="right") - 1],
            "共同行业数": len(common),
            "平均绝对偏差": float(dev.abs().mean()),
            "最大绝对偏差": float(dev.abs().max()),
            "最大偏差行业": str(dev.abs().idxmax()),
            "权重相关": float(ws.corr(es)) if len(common) > 2 else np.nan,
            "RBSA宽基暴露": float(w_broad.sum()),
            "真实暴露合计": float(e.sum()),
        })
        for c in common:
            detail.append({"季末": qend, "行业": inds.get(c, c),
                           "RBSA权重": float(ws[c]), "真实暴露": float(es[c]),
                           "偏差": float(ws[c] - es[c])})

    if not rows:
        return {"ok": False,
                "reason": "没有可对比的期数（RBSA 权重重估日晚于所有持仓季末，"
                          "或共同行业不足）"}

    df = pd.DataFrame(rows).sort_values("季末")
    dd = pd.DataFrame(detail)
    # 汇总：跨期平均
    agg = (dd.groupby("行业")
           .agg(RBSA权重=("RBSA权重", "mean"), 真实暴露=("真实暴露", "mean"),
                偏差=("偏差", "mean"), 平均绝对偏差=("偏差", lambda s: s.abs().mean()))
           .sort_values("平均绝对偏差", ascending=False))
    broad = float(df["RBSA宽基暴露"].mean())
    broad_note = (
        f"RBSA 拟合出的宽基暴露平均 {broad:.1%} —— 这部分**无法与持仓对应**"
        f"（宽基跨行业），对齐只覆盖行业部分。若该值很大，说明模型把一部分"
        f"行业暴露记在了宽基上，行业层面的偏差会被低估。"
    ) if broad > 1e-6 else "RBSA 未拟合出宽基暴露，行业层面可直接对比。"
    return {
        "ok": True,
        "periods": df,
        "detail": dd,
        "by_industry": agg,
        "n_periods": int(len(df)),
        "mad_mean": float(df["平均绝对偏差"].mean()),
        "corr_mean": float(df["权重相关"].mean()),
        "broad_exposure": broad,
        "broad_note": broad_note,
    }


# ====================================================================== 持仓基准
def holding_benchmark(exposure: pd.DataFrame, panel: pd.DataFrame,
                      lag_days: int = DISCLOSURE_LAG_DAYS) -> pd.Series:
    """
    只用真实持仓构造的基准组合**日收益**序列（index=交易日）。

    做法：每期行业暴露 × 对应行业指数日收益，加权求和。
    暴露是"占净值比例"（含现金），所以未覆盖部分按现金（0 收益）处理 ——
    这让它天然比"满仓指数组合"低一部分收益，与基金的真实收益口径一致。

    ★ **披露滞后必须处理**：Q2 持仓季末 6/30，但要到 7 月中下旬才公布。
      因此该期暴露从 `季末 + lag_days` 起生效，而不是 6/30。
      否则等于用未来信息构造历史基准，比出来的收益不可信。
    """
    if exposure is None or exposure.empty or panel is None or panel.empty:
        return pd.Series(dtype=float)
    exposure = _exposure_to_codes(exposure)
    cols = [c for c in exposure.columns if c in panel.columns]
    if not cols:
        return pd.Series(dtype=float)

    rets = np.log(panel[cols].astype(float)).diff()
    parts = []
    for qend in exposure.index:
        start = qend + pd.Timedelta(days=int(lag_days))
        # ★ 单位换算：exposure 的"占净值比例"是**百分点**（37.03 = 37.03%），
        #   而 rets 是小数日收益。不除 100 会让每日收益被放大近百倍
        #   （实测症状：年化 −95%、年化波动 1650%）。
        w = exposure.loc[qend, cols].fillna(0.0) / 100.0
        if w.sum() <= 0:
            continue
        seg = rets.loc[rets.index >= start]
        if seg.empty:
            continue
        # 该期暴露从披露日一直生效，直到被**更晚一期**覆盖（下方按期末顺序赋值，
        # 后者覆盖前者）—— 这正是"每半年按最新披露调仓"的行为。
        parts.append((seg.index, (seg * w).sum(axis=1)))
    if not parts:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(sorted(set().union(*[p[0] for p in parts])))
    out = pd.Series(0.0, index=idx)
    for i, v in parts:
        out.loc[i] = v
    return out.dropna().sort_index()


def compare_benchmarks(fund_nav: pd.Series, rbsa_syn_nav: pd.Series,
                       holding_ret: pd.Series) -> dict:
    """
    三条曲线的对比：基金实际 / RBSA 合成基准 / 持仓基准。

    都在**同一段共同区间**上比较，且持仓基准已按披露滞后起算，
    所以这是一个"可实现"的比较，不是事后最优的比较。
    """
    if holding_ret is None or holding_ret.empty:
        return {"ok": False, "reason": "持仓基准序列为空"}
    idx = holding_ret.index
    f = np.log(pd.Series(fund_nav).astype(float)).diff().reindex(idx).dropna()
    s = np.log(pd.Series(rbsa_syn_nav).astype(float)).diff().reindex(idx).dropna()
    common = f.index.intersection(s.index)
    if len(common) < 60:
        return {"ok": False,
                "reason": f"共同区间只有 {len(common)} 个交易日，不足以比较"}
    f, s, h = f.loc[common], s.loc[common], holding_ret.loc[common]

    def _stat(r, name):
        eq = float(np.exp(r.sum()))
        n = len(r)
        return {
            "基准": name,
            "区间收益": eq - 1,
            "年化收益": float(np.exp(r.mean() * 252) - 1),
            "年化波动": float(r.std(ddof=1) * np.sqrt(252)),
            "与基金相关": float(r.corr(f)),
            "相对基金IR": float(r.mean() / r.std(ddof=1) * np.sqrt(252))
            if r.std(ddof=1) > 0 else np.nan,
        }

    rows = [_stat(f, "基金实际"), _stat(s, "RBSA合成基准"), _stat(h, "持仓基准")]
    # 持仓基准 vs RBSA 基准：跟踪差异
    d = h - s
    rows.append({
        "基准": "持仓−RBSA（跟踪差）",
        "区间收益": float(np.exp(d.sum()) - 1),
        "年化收益": float(np.exp(d.mean() * 252) - 1),
        "年化波动": float(d.std(ddof=1) * np.sqrt(252)),
        "与基金相关": float(d.corr(f)),
        "相对基金IR": float(d.mean() / d.std(ddof=1) * np.sqrt(252))
        if d.std(ddof=1) > 0 else np.nan,
    })
    return {"ok": True, "stats": pd.DataFrame(rows), "n_days": int(len(common))}
