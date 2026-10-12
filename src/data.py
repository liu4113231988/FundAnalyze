# -*- coding: utf-8 -*-
"""
数据层：基金复权净值 + 候选指数行情，统一对齐后落盘缓存。

数据源（均为免费公开接口）
    - 基金净值 : 天天基金 pingzhongdata（akshare: fund_open_fund_info_em，累计净值走势）
    - 宽基/风格: 新浪指数日线（akshare: stock_zh_index_daily）
    - 申万行业 : 申万宏源一级行业指数（akshare: index_hist_sw）

注意
    停牌/缺失按交易日历前向填充，但不允许跨超过 5 个交易日填充
    （防止长期停牌被伪装成 0 收益）。

★ 收益口径（P0-01，尚未核实）
    `fetch_fund_nav` 取的是数据源的「累计净值走势」。原模块注释称它
    "已含分红再投效果"——**这个等式没有经过真实样本核实**。累计净值通常是
    「单位净值 + 历史累计分红」，而累计分红是**存量**、不参与后续增长，
    所以直接取对数差分会把收益率系统性、且随时间加重地压低。
    在本仓库用真实分红样本核实之前，基于该口径的所有结论都必须标注
    "口径待修正"。正确口径见 `total_return_index()`（需单位净值 + 分红/拆分事件）。
    指数侧用的是**价格指数**（不含股息），与基金总回报之间还差一个股息率。
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _default_http_timeout() -> float:
    """抓取超时（秒）。FIL_HTTP_TIMEOUT=0 表示不限制。"""
    try:
        v = float(os.environ.get("FIL_HTTP_TIMEOUT", "30"))
    except (TypeError, ValueError):                        # noqa: BLE001
        return 30.0
    return v if v > 0 else 0.0


@contextlib.contextmanager
def http_timeout(seconds: Optional[float] = None):
    """
    只在抓取期间生效的 socket 默认超时（P1-07 / N-05）。

    ★ 为什么不能写在模块级 `socket.setdefaulttimeout(30)`：
      它是**进程全局**的，会影响本进程里之后新建的**所有** socket ——
      包括 Flask 服务器的监听 socket 与已 accept 的连接 socket。
      本意只是防止 AkShare 请求永久挂起，结果把 Web 服务也罩进同一个
      30 秒超时里（长连接、慢客户端都可能被误杀）。改成上下文管理器后，
      只有真正在抓数据的代码段受影响。
    """
    old = socket.getdefaulttimeout()
    sec = _default_http_timeout() if seconds is None else seconds
    socket.setdefaulttimeout(sec if sec and sec > 0 else None)
    try:
        yield
    finally:
        socket.setdefaulttimeout(old)

from . import config as C


# ------------------------------------------------------------------ 通用重试
def _retry(fn, tries: int = 4, wait: float = 1.2, **kw):
    last = None
    for i in range(tries):
        # ★ 超时只在抓取期间生效（N-05）：AkShare 的请求没有暴露 timeout 参数，
        #   但 socket 级的全局默认值会波及 Web 服务器，所以必须局部设置。
        with http_timeout():
            try:
                return fn(**kw)
            except Exception as e:      # noqa: BLE001
                last = e
        time.sleep(wait * (i + 1))
    raise last


def _clean_series(s: pd.Series, start: Optional[str] = None) -> pd.Series:
    """通用清洗：剔除非正值、按日期去重、截断起点。"""
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")]
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s


# ------------------------------------------------------------------ 基金净值
def fetch_fund_nav(code: str, start: Optional[str] = None) -> pd.Series:
    """
    返回基金**累计净值**序列（index=交易日）。

    ★ 口径警告（P0-01）：本序列是数据源的「累计净值走势」，**不是**经过核验的
    红利再投资复权净值。它与真实总回报的偏离程度取决于该基金的历史分红总额，
    且偏离随时间加重。需要正确口径时，请取得单位净值与分红/拆分事件后调用
    `total_return_index()`，或用独立来源交叉验证后再替换本函数。
    """
    import akshare as ak

    df = _retry(ak.fund_open_fund_info_em, symbol=code, indicator="累计净值走势")
    df.columns = ["date", "nav"]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["nav"].astype(float).sort_index()
    # 注意：必须按"索引（日期）"去重，不能按数值去重——
    # 累计净值存在大量数值相同的相邻记录，按数值去重会误删四成以上样本。
    s.name = f"fund_{code}"
    return _clean_series(s, start)


# ------------------------------------------------------------------ 净值口径
# ★ P0-01：口径必须显式可选、可核对，不能靠"模块说明里的一句话"默认成立。
NAV_BASIS_CUM = "cum"        # 数据源「累计净值走势」直接取对数差分（当前默认）
NAV_BASIS_TOTAL = "total"    # 由单位净值 + 分红/拆分事件构造的总回报（红利再投资）

NAV_BASIS_NOTE = {
    NAV_BASIS_CUM: (
        "累计净值直算。**尚未与真实分红/拆分样本核实**：累计净值通常是"
        "「单位净值 + 历史累计分红」，它不等于红利再投资净值。由于累计分红是"
        "存量、不参与后续增长，收益率会被系统性且**随时间加重**地压低"
        "（见 TODO P0-01）。在核实前，基于本口径的结论不得作为决策依据。"),
    NAV_BASIS_TOTAL: (
        "总回报（红利再投资）：由单位净值 + 分红/拆分事件逐日构造，"
        "是基金收益的正确口径。指数侧若用价格指数，则与之差一个股息率。"),
}


def total_return_index(unit_nav: pd.Series,
                       dividends: Optional[pd.DataFrame] = None,
                       splits: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    由**单位净值 + 分红/拆分事件**构造总回报指数（起点 1.0，红利再投资口径）。

        r_t = (NAV_t × 份额因子 + D_t − NAV_{t−1}) / NAV_{t−1}

    即除息日把现金分红加回收益：除权后净值下跌的那部分不是亏损。
    拆分（1 拆 k）把**拆分日之前**的净值与分红统一折算到当前份额口径，
    这样净值序列在事件日连续，事件本身不产生虚假收益。

    unit_nav   : 单位净值序列（index=交易日）
    dividends  : DataFrame(index=除息日, 列 per_unit=每份现金分红)，可 None
    splits     : DataFrame(index=拆分日, 列 ratio=拆分比例，2 表示 1 拆 2)，可 None

    返回与 unit_nav 同索引的总回报指数序列（首日 = 1.0）。
    """
    s = pd.Series(unit_nav).astype(float).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return s.iloc[0:0]

    fac = pd.Series(1.0, index=s.index, dtype=float)
    if splits is not None and len(splits):
        sp = splits.copy()
        sp.index = pd.to_datetime(sp.index)
        col = "ratio" if "ratio" in sp.columns else sp.columns[0]
        for d, ratio in zip(sp.index, sp[col]):
            try:
                rr = float(ratio)
            except (TypeError, ValueError):
                continue
            if np.isfinite(rr) and rr > 0:
                fac[s.index < d] /= rr

    adj = s * fac
    d = pd.Series(0.0, index=adj.index, dtype=float)
    if dividends is not None and len(dividends):
        dv = dividends.copy()
        dv.index = pd.to_datetime(dv.index)
        col = "per_unit" if "per_unit" in dv.columns else dv.columns[0]
        amt = dv[col].astype(float)
        amt = amt[amt.index.isin(d.index)]
        d.loc[amt.index] += amt.values * fac.reindex(amt.index).values

    prev = adj.shift(1)
    r = (adj + d - prev) / prev
    r = r.where(prev > 0)
    r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod()


# ------------------------------------------------------------------ 指数行情
def fetch_index_sina(code: str, start: Optional[str] = None) -> pd.Series:
    """新浪指数日线收盘价。code 形如 sh000300 / sz399006。"""
    import akshare as ak

    df = _retry(ak.stock_zh_index_daily, symbol=code)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["close"].astype(float).sort_index()
    s.name = code
    return _clean_series(s, start)


def fetch_index_sw(code: str, start: Optional[str] = None) -> pd.Series:
    """申万一级行业指数收盘价。code 形如 801010。"""
    import akshare as ak

    df = _retry(ak.index_hist_sw, symbol=code, period="day")
    dcol = "日期" if "日期" in df.columns else df.columns[1]
    ccol = "收盘" if "收盘" in df.columns else "close"
    df[dcol] = pd.to_datetime(df[dcol])
    s = df.set_index(dcol)[ccol].astype(float).sort_index()
    s.name = code
    return _clean_series(s, start)


# ------------------------------------------------------------------ 批量抓取
def _fetch_one(code: str, start: str, end: Optional[str]):
    if code in C.SW_INDUSTRY:
        s = fetch_index_sw(code, start)
    else:
        s = fetch_index_sina(code, start)
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    return code, s


def fetch_universe(codes: Iterable[str], start: str, end: Optional[str] = None,
                   verbose: bool = True, workers: int = 6) -> pd.DataFrame:
    """
    并发抓取整个候选池，返回 宽表 DataFrame(index=日期, columns=指数代码)。

    串行抓 44 个指数约需 45 秒，6 线程并发可压到 8 秒左右。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    codes = list(codes)
    series: Dict[str, pd.Series] = {}
    ok = fail = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, c, start, end): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                code, s = fut.result()
                if len(s) > 200:
                    series[code] = s
                    ok += 1
                else:
                    fail += 1
                    if verbose:
                        print(f"  [data] {code} {C.NAME_OF.get(code,''):8s} "
                              f"跳过（仅 {len(s)} 行）")
            except Exception as e:      # noqa: BLE001
                fail += 1
                if verbose:
                    print(f"  [data] {code} {C.NAME_OF.get(code,''):8s} "
                          f"失败: {type(e).__name__} {str(e)[:50]}")

    if verbose:
        print(f"  [data] 候选池抓取完成：成功 {ok} / 失败 {fail}")
    px = pd.DataFrame(series).sort_index()
    return px


def data_stamp() -> str:
    """
    底层数据的粗粒度指纹（P1-05）。

    结果缓存旧键里只有 (基金, 模型集, 参数, 精度)，**不含底层数据摘要**：
    行情追加或重抓之后，缓存键不变，页面会一直显示旧结果，看起来像"刷新无效"。
    这里用数据目录里 CSV 的「文件数 + 最大 mtime」做指纹 —— 任何一次写入都会
    改变它，于是缓存键随之改变，结果自然刷新。开销仅一次目录 stat。
    """
    try:
        fs = list(C.DATA_DIR.glob("*.csv"))
        if not fs:
            return "empty"
        mx = max(f.stat().st_mtime for f in fs)
        return f"{len(fs)}:{int(mx)}"
    except Exception:                                   # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------ 清洗对齐
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


# ------------------------------------------------------------------ 指数估值
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


def load_pe_panel(verbose: bool = True) -> pd.DataFrame:
    from .signal import PE_SYMBOL, PE_FALLBACK

    syms = sorted(set(PE_SYMBOL.values()) | {PE_FALLBACK})
    df = _cached("pe_panel", lambda: fetch_pe_panel(syms, verbose=verbose))
    # 缓存可能因当初部分标的抓取失败而缺列：增量补取后合并并回写
    missing = [s for s in syms if s not in df.columns]
    if missing:
        add = fetch_pe_panel(missing, verbose=verbose)
        if add.shape[1]:
            df = pd.concat([df, add], axis=1).sort_index()
            df.to_csv(C.DATA_DIR / "pe_panel.csv", encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ 缓存层
def _cache_meta_path(name: str) -> Path:
    return C.DATA_DIR / f"{name}.meta.json"


def _fingerprint(meta: dict) -> dict:
    """
    只取**参与有效性判等**的字段。

    ★ 必须把 "info" 排除掉（N-07）：info 里放的是抓取时间、数据末日、行数、
    缺失资产这类**描述性**信息，它们每次抓取都会变。若让它们参与相等比较，
    缓存会每次都判为失效 → 每次全量重建，等于没有缓存。
    """
    return {k: v for k, v in (meta or {}).items() if k != "info"}


def _cache_meta_valid(name: str, meta: dict) -> bool:
    """
    缓存是否仍有效。

    用 sidecar 元数据而不是把版本号塞进文件名，是为了**不作废已有的 14 份
    panel 缓存**——它们仍然有效，强行改名会让离线环境（无 akshare）直接跑不起来。
    没有元数据文件的旧缓存一律视为有效。
    """
    fp = _cache_meta_path(name)
    if not fp.exists():
        return True
    try:
        saved = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:               # noqa: BLE001
        return True
    return _fingerprint(saved) == _fingerprint(meta)


def _cache_meta_write(name: str, meta: dict):
    try:
        _cache_meta_path(name).write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except Exception:               # noqa: BLE001
        pass


def cache_info(name: str) -> dict:
    """
    读某个缓存的描述性信息（P1-05 / N-07）：抓取时间、数据末日、行数、缺失资产。

    用途：回答"这份缓存是什么时候抓的、抓到哪天、缺了哪些资产" ——
    旧缓存只有一个版本号，出问题时无法判断该重抓还是该补取。
    """
    fp = _cache_meta_path(name)
    if not fp.exists():
        return {}
    try:
        return (json.loads(fp.read_text(encoding="utf-8")) or {}).get("info", {})
    except Exception:                                    # noqa: BLE001
        return {}


def _cache_info_write(name: str, info: dict):
    """把描述性信息合并进现有 meta（不触碰参与判等的指纹字段）。"""
    fp = _cache_meta_path(name)
    try:
        cur = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    except Exception:                                    # noqa: BLE001
        cur = {}
    cur.setdefault("info", {}).update(info)
    _cache_meta_write(name, cur)


def _os_replace_retry(src: Path, dst: Path, retries: int = 6):
    """os.replace 带重试。Windows 下目标被另一线程短暂占用会抛 WinError 32。"""
    import os
    import time
    for i in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def _atomic_csv_write(df: pd.DataFrame, path: Path):
    """
    原子写 CSV：先写同目录临时文件再 os.replace。

    为什么需要：直接 to_csv 到目标路径，进程在写一半时被杀会留下**半截 CSV**，
    下次读取虽然能兜底重建，但会白白多一次网络请求。
    临时名必须每次唯一 —— Web 端是多线程，只加 pid 仍会撞名。
    """
    import os
    import threading
    import uuid
    tmp = path.with_name(f"{path.name}.{os.getpid()}."
                         f"{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        df.to_csv(tmp, encoding="utf-8-sig")
        _os_replace_retry(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _cached(name: str, builder, meta: Optional[dict] = None):
    fp = C.DATA_DIR / f"{name}.csv"
    stale = meta is not None and not _cache_meta_valid(name, meta)
    if stale:
        print(f"[data] 缓存 {name} 口径已变更，重新抓取")
    if fp.exists() and not stale:
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df):
                # ★ 命中旧缓存时也要补写元数据，否则版本守卫永远不会被"武装"：
                # 之后有人改 DATA_CACHE_VERSION 时，_cache_meta_valid 会因为
                # 找不到元数据而一律判为有效，缓存照样命中，守卫形同虚设。
                if meta is not None:
                    _cache_meta_write(name, _with_info(meta, df))
                return df
        except Exception as e:      # noqa: BLE001
            print(f"[data] 缓存 {name} 损坏（{type(e).__name__}），重新构建")
    df = builder()
    _atomic_csv_write(df, fp)
    if meta is not None:
        _cache_meta_write(name, _with_info(meta, df))
    return df


def _with_info(meta: dict, df) -> dict:
    """
    给 meta 补上描述性 info（N-07）：抓取时间、数据末日、行数。

    这些字段**不参与** _cache_meta_valid 的判等（见 _fingerprint），
    否则每次抓取时间不同都会让缓存失效。
    """
    meta = dict(meta or {})
    info = dict(meta.get("info") or {})
    info.setdefault("fetched_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(df, pd.DataFrame) and len(df.index):
        try:
            info.setdefault("data_end", str(pd.Timestamp(df.index[-1]).date()))
            info.setdefault("data_start", str(pd.Timestamp(df.index[0]).date()))
            info.setdefault("n_rows", int(len(df)))
            info.setdefault("n_cols", int(df.shape[1]))
        except (TypeError, ValueError, IndexError):        # noqa: BLE001
            pass
    meta["info"] = info
    return meta


# ------------------------------------------------------------------ 分层缓存
def load_index_universe(start: str = C.START_DATE,
                        end: Optional[str] = C.END_DATE,
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
    """
    cache_name = f"index_universe_{start}" + (f"_to{end}" if end else "")
    meta = {"v": C.DATA_CACHE_VERSION, "n_candidates": len(C.NAME_OF),
            "start": start, "end": end}
    if refresh:
        for p in (C.DATA_DIR / f"{cache_name}.csv",
                  C.DATA_DIR / f"{cache_name}.meta.json"):
            if p.exists():
                p.unlink()

    def build() -> pd.DataFrame:
        if verbose:
            print(f"[data] 抓取候选指数池 {len(C.NAME_OF)} 个 ...")
        return fetch_universe(C.NAME_OF.keys(), start, end, verbose=verbose)

    df = _cached(cache_name, build, meta=meta)
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


def load_fund_nav_cached(fund_code: str, start: str = C.START_DATE,
                         refresh: bool = False) -> pd.Series:
    """
    单只基金的累计净值（独立缓存，~100 KB，避免与指数抓取耦合）。

    注意：_cached 统一走 CSV，读回来是 DataFrame，这里取第一列还原成 Series。
    """
    cache_name = f"fund_nav_{fund_code}_{start}"
    meta = {"v": C.DATA_CACHE_VERSION, "code": fund_code, "start": start}
    if refresh:
        for p in (C.DATA_DIR / f"{cache_name}.csv",
                  C.DATA_DIR / f"{cache_name}.meta.json"):
            if p.exists():
                p.unlink()

    def build() -> pd.DataFrame:
        return fetch_fund_nav(fund_code, start).to_frame()

    df = _cached(cache_name, build, meta=meta)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s.name = f"fund_{fund_code}"
    return s


def load_panel(fund_code: str, start: str = C.START_DATE,
               end: Optional[str] = C.END_DATE, refresh: bool = False,
               verbose: bool = True) -> pd.DataFrame:
    """
    返回对齐后的价格面板：列 = 候选指数 + __FUND__。

    存储分层（★ 面板本身**不再缓存**，即时重建）：

        指数池  index_universe_{start}.csv     全局一份（与基金无关）
        净值    fund_nav_{code}_{start}.csv     每只一份（~100 KB）
        面板    = 两者 → align_prices() 即时重建（纯内存，<100 ms）

    为什么不缓存面板：面板里 95% 以上的列是指数，逐基金缓存会让同样的指数
    数据存 N 份（1 MB × 基金数）。拆成两个原子数据源之后，存储与抓取量都从
    O(基金 × 指数) 降到 O(基金 + 指数)；对齐是毫秒级内存操作，重建不亏。
    """
    nav = load_fund_nav_cached(fund_code, start, refresh=refresh)
    if verbose:
        print(f"[data] 基金 {fund_code} 净值 {len(nav)} 行  "
              f"{nav.index[0].date()} ~ {nav.index[-1].date()}")
    px = load_index_universe(start, end, refresh=refresh, verbose=verbose)
    out = align_prices(px, nav)
    if verbose:
        print(f"[data] 对齐后：{out.shape[0]} 交易日 × {out.shape[1]} 条序列 "
              f"({out.index[0].date()} ~ {out.index[-1].date()})")
    return out
