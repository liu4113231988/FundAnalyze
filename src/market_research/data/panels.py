from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import market_research.config as C
from market_research.data import storage
from market_research.data.cache import _cached, _cache_info_write, _with_info
from market_research.data.providers.funds import NAV_BASIS_CUM, NAV_BASIS_TOTAL, fetch_fund_nav, fetch_fund_total_return
from market_research.data.providers.indices import fetch_universe
from market_research.data.providers.valuation import fetch_pe_panel

def align_prices(px: pd.DataFrame, nav: pd.Series, max_ffill: int = 5,
                 min_coverage: float = 0.97) -> pd.DataFrame:
    """
    把基金净值与指数价格对齐到共同交易日。

    规则：
      1. 以基金净值的交易日为基准（基金开放日才是可交易日）；
      2. 指数缺失值前向填充，但连续填充不超过 max_ffill 天；
      3. 先剔除区间内覆盖率不足 min_coverage 的指数（如已停编的老指数），
         再掐头去尾保留全部序列均有有效值的区间。
    """
    df = px.reindex(nav.index).ffill(limit=max_ffill)
    df["__FUND__"] = nav.reindex(df.index)

    # 覆盖率过滤：优先剔除"中途停编"的指数，避免整段历史被截断
    cov = df.notna().mean()
    dropped = cov[(cov < min_coverage) & (df.columns != "__FUND__")].index.tolist()
    if dropped:
        print(f"[data] 覆盖率不足已剔除 {len(dropped)} 个：" +
              ", ".join(f"{c}({C.NAME_OF.get(c, c)})" for c in dropped))
    df = df.drop(columns=dropped)

    # 掐头去尾并剔除中间缺失日（连续缺失超过 max_ffill 的交易日直接放弃）
    return df.dropna()


def to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """价格 → 对数收益。"""
    r = np.log(df).diff().dropna()
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def load_pe_panel(verbose: bool = True) -> pd.DataFrame:
    from market_research.modeling.signals import PE_SYMBOL, PE_FALLBACK

    syms = sorted(set(PE_SYMBOL.values()) | {PE_FALLBACK})
    df = _cached("pe_panel", lambda: fetch_pe_panel(syms, verbose=verbose))
    # 缓存可能因当初部分标的抓取失败而缺列：增量补取后合并并回写
    missing = [s for s in syms if s not in df.columns]
    if missing:
        add = fetch_pe_panel(missing, verbose=verbose)
        if add.shape[1]:
            df = pd.concat([df, add], axis=1).sort_index()
            storage.write("pe_panel", df, _with_info({}, df))
    return df


def load_index_universe(start: Optional[str] = None,
                        end: Optional[str] = None,
                        refresh: bool = False,
                        verbose: bool = True) -> pd.DataFrame:
    """
    候选指数池的价格宽表（**全局缓存**，与基金无关）。

    ★ 为什么要单独缓存：指数价格对一只基金和对另一只基金是同一份数据。
    早期实现把它塞进 load_panel 的 per-fund 缓存里，于是每分析一只基金
    都要重新抓 47 个指数 —— 扩池到 50 只就是 2350 次网络请求。
    拆出来之后，无论分析多少只基金，指数只抓一次。

    返回列 = 候选指数代码（不含 __FUND__），**未做对齐** —— 对齐依赖具体
    基金的交易日轴，仍由 load_panel 做。

    ★ start/end 默认 None 而不是 `C.START_DATE`：Python 的默认参数在
      **函数定义时**求值，写成 `start=C.START_DATE` 会让运行时修改
      `C.START_DATE`（如 `--window-years 2`）完全无效。None 表示"用当前配置"。
    """
    start = start or C.START_DATE
    end = end if end is not None else C.END_DATE
    cache_name = f"index_universe_{start}" + (f"_to{end}" if end else "")
    meta = {"v": C.DATA_CACHE_VERSION, "n_candidates": len(C.NAME_OF),
            "start": start, "end": end}

    def build() -> pd.DataFrame:
        if verbose:
            print(f"[data] 抓取候选指数池 {len(C.NAME_OF)} 个 ...")
        return fetch_universe(C.NAME_OF.keys(), start, end, verbose=verbose)

    df = _cached(cache_name, build, meta=meta, refresh=refresh)
    # ★ 缺失资产必须留痕（N-07）：部分行情失败会静默得到一个"不完整指数池"，
    #   下游的覆盖率过滤会把它们当"中途停编"剔除，掩盖了真实原因是抓取失败。
    #   无条件写入（空列表 = 无缺失），否则"没这个字段"和"没缺失"分不清。
    missing = [c for c in C.NAME_OF if c not in df.columns]
    _cache_info_write(cache_name, {
        "missing_assets": missing,
        "n_missing": len(missing),
        "n_expected": len(C.NAME_OF)})
    if missing:
        print(f"[data] ⚠ 候选池缺 {len(missing)}/{len(C.NAME_OF)} 个指数"
              f"（{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}）"
              f"——重跑本命令可补齐，已成功的部分不会重复请求")
    return df


def load_fund_nav_cached(fund_code: str, start: Optional[str] = None,
                         refresh: bool = False,
                         basis: str = NAV_BASIS_TOTAL) -> pd.Series:
    """
    单只基金的收益序列（独立缓存，~100 KB，避免与指数抓取耦合）。

    basis = NAV_BASIS_TOTAL（默认，P0-01 修正口径）：日频总回报，含分红再投。
            由累计净值 + 单位净值 + 月频累计收益率校准得到。
    basis = NAV_BASIS_CUM：数据源的累计净值直算。**已实测系统性低估
            （14 只基金平均 −23.19%，最差 −76.12%）**，仅用于兼容与对照。

    缓存名带 basis 后缀，两种口径互不覆盖。
    """
    import market_research.tasks.sync as sync
    sync.register("fund", fund_code, basis)
    if not refresh:
        history = sync.cached("fund", fund_code, basis, start, None)
        if history is not None:
            result = history.iloc[:, 0].copy()
            result.name = f"fund_{fund_code}"
            return result
    suffix = "" if basis == NAV_BASIS_CUM else f"_{basis}"
    # None → 当前配置的起点（见 load_index_universe 的说明）
    start = start or C.START_DATE
    cache_name = f"fund_nav_{fund_code}_{start}{suffix}"
    meta = {"v": C.DATA_CACHE_VERSION, "code": fund_code, "start": start,
            "basis": basis}

    class _DegradedNav(Exception):
        """总回报口径不可用。绝不能把退化结果写进 total 缓存名下。"""

    def build() -> pd.DataFrame:
        if basis == NAV_BASIS_TOTAL:
            s = fetch_fund_total_return(fund_code, start)
            # ★ 复查发现的缓存污染：旧实现把退回的 cum 序列原样返回，
            #   _cached 会把它写进 `_total` 后缀的缓存名（meta 还标
            #   basis="total"），之后所有读取持续命中口径错误的缓存。
            #   这里显式拒绝：退回结果只本次内存使用，不落盘。
            if bool(getattr(s, "attrs", {}).get("degraded")):
                raise _DegradedNav(
                    f"{fund_code} 无法构造总回报口径（单位净值或累计收益率锚点缺失）")
            return s.to_frame()
        return fetch_fund_nav(fund_code, start).to_frame()

    try:
        df = _cached(cache_name, build, meta=meta, refresh=refresh)
    except _DegradedNav as e:
        print(f"[data] ⚠ {e}；本次退回累计净值口径，结果【未缓存】")
        return fetch_fund_nav(fund_code, start)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s.name = f"fund_{fund_code}"
    return s


def fetch_start_for(start: Optional[str] = None) -> str:
    """
    **抓取**起点：取 `start` 与 `C.FETCH_START_DATE` 的较早者。

    ★ 为什么分析与抓取必须用两个起点（2026-10-12 改）：
    缓存名里带 start（`fund_nav_{code}_{start}_total.csv`）。分析窗口改成滚动
    近 N 年之后，如果抓取也跟着滚动，那么每换一次窗口、甚至每天运行都会让
    全部缓存 miss —— 离线环境下等于拿不到任何数据；在线环境下则是每次重抓
    全部指数，白白放大几十倍请求量。

    所以抓取一律用固定长区间（缓存稳定 + 预热期充足），拿到之后再裁剪到
    分析窗口。若用户显式要求比 FETCH_START_DATE 更早的区间，才真正提前抓取。

    ISO 格式（YYYY-MM-DD）字符串可直接字典序比较，等价于日期比较。
    """
    s = str(start) if start else ""
    # 取**较早者**：分析起点晚于固定起点时用固定起点（白赚一段预热数据，
    # 且缓存必然命中）；只有用户显式要求更早的区间才真的提前抓取。
    if not s or s > C.FETCH_START_DATE:
        return C.FETCH_START_DATE
    return s


def load_panel(fund_code: str, start: Optional[str] = None,
               end: Optional[str] = None, refresh: bool = False,
               verbose: bool = True,
               nav_basis: str = NAV_BASIS_TOTAL) -> pd.DataFrame:
    """
    返回对齐后的价格面板：列 = 候选指数 + __FUND__，**已裁剪到 [start, end]**。

    存储分层（★ 面板本身**不再缓存**，即时重建）：

        指数池  index_universe_{fs}.csv         全局一份（与基金无关）
        净值    fund_nav_{code}_{fs}.csv         每只一份（~100 KB）
        面板    = 两者 → align_prices() 即时重建（纯内存，<100 ms）

    为什么不缓存面板：面板里 95% 以上的列是指数，逐基金缓存会让同样的指数
    数据存 N 份（1 MB × 基金数）。拆成两个原子数据源之后，存储与抓取量都从
    O(基金 × 指数) 降到 O(基金 + 指数)；对齐是毫秒级内存操作，重建不亏。

    ★ `start` 是**分析**起点，抓取用 fetch_start_for(start)（通常更早）。
    对齐按**完整抓取区间**做（覆盖率筛选需要全样本），之后再裁剪到分析窗口 ——
    顺序反过来会让覆盖率统计只看到窗口内的样本，把"曾经存在过的指数"误判为
    缺失而剔除。
    """
    start = start or C.START_DATE
    end = end if end is not None else C.END_DATE
    fs = fetch_start_for(start)
    nav = load_fund_nav_cached(fund_code, fs, refresh=refresh, basis=nav_basis)
    if verbose:
        tag = "总回报" if nav_basis == NAV_BASIS_TOTAL else "累计净值(旧口径)"
        print(f"[data] 基金 {fund_code} {tag} {len(nav)} 行  "
              f"{nav.index[0].date()} ~ {nav.index[-1].date()}")
    px = load_index_universe(fs, end, refresh=refresh, verbose=verbose)
    out = align_prices(px, nav)

    # 裁剪到分析窗口
    if start:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end)]
    if out.empty:
        raise ValueError(
            f"{fund_code} 在 {start or '全区间'} ~ {end or '最新'} 内没有可用数据"
            f"（抓取区间 {fs} ~ {end or '最新'}，请放宽分析窗口或检查代码）")
    if verbose:
        print(f"[data] 对齐后：{out.shape[0]} 交易日 × {out.shape[1]} 条序列 "
              f"({out.index[0].date()} ~ {out.index[-1].date()})"
              + (f"  [分析窗口 {start} 起]" if start else ""))
    return out
