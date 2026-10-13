# -*- coding: utf-8 -*-
"""
基金池扩充：按 `docs/扩池预注册规则.md` 冻结的规则筛选样本。

**规则已在拉取任何数据之前冻结**（见该文档的提交时间）。本模块只执行规则，
不在这里新增或放宽任何条件；确需修改必须先改那份文档并记录时间与原因。

为什么需要预注册：当前 14 只是人工挑的便利样本，有三类偏差
（存续偏差 / 选择偏差 / 同源暴露）。直接扩数量不解决独立性问题 ——
200 只高度相关的基金，有效样本量可能仍是几个。所以规则针对"独立性"而非"数量"。

★ 两个**做不到**的规则项（免费数据源限制，必须在报告里如实写）：
  1. **存续偏差无法消除**：`ak.fund_name_em()` 是在场基金列表，
     **已清盘/已合并的基金不在其中**，没有免费的清盘基金历史清单。
     因此扩池只覆盖"活到今天的主动权益基金"。
  2. **规模口径降级**：规则要求"成立满 2 年时规模 ≥ 2 亿"，
     但接口只给**当前规模**（`fund_individual_basic_info_xq` 的"最新规模"）。
     改用当前规模近似，会把"后来做大的基金"纳入、"中途缩水的基金"误排除。

实现分两步：
  1. `candidates()`  名称/类型粗筛（零请求，快）→ 候选 + 排除原因
  2. `enrich()`      逐只取基本信息（成立/规模/管理人/策略关键词）→ 详情 + 排除原因
  3. `build_pool()`  按成立时间分层抽样，单管理人限 3 只
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------- 冻结的规则常量
# 类型：主动权益（股票型 + 偏股/灵活混合）。指数/债券/货币/FOF/QDII 不在此列。
ALLOWED_TYPES = ("股票型", "混合型-偏股", "混合型-灵活")

# 名称里出现这些词 → 剔除（主题/行业基金的"风格"由契约决定，不是 RBSA 要解释的对象；
# 指数/联接/FOF/QDII/债/货币同理）。词表刻意保守（宁多勿漏），每条排除都会留痕。
NAME_EXCLUDE_WORDS = (
    # 被动 / 联接 / 跨境
    "指数", "ETF", "联接", "FOF", "QDII", "LOF", "分级", "等权", "增强",
    # 固定收益 / 货币
    "债", "货币", "转债", "纯债", "定开", "理财",
    # 行业主题（申万一级 + 常见赛道）
    "行业", "消费", "医药", "医疗", "健康", "军工", "国防", "新能源", "汽车",
    "半导体", "芯片", "白酒", "科技", "计算机", "传媒", "通信", "电子",
    "金融", "银行", "证券", "保险", "地产", "房地产", "资源", "煤炭", "石油",
    "钢铁", "有色", "金属", "化工", "农业", "牧业", "食品", "饮料", "家电",
    "纺织", "服装", "轻工", "造纸", "建材", "建筑", "装饰", "电力", "公用",
    "交通", "运输", "物流", "航空", "航运", "零售", "商贸", "社会", "服务",
    "旅游", "酒店", "餐饮", "教育", "环保", "美丽", "美容", "护理", "机械",
    "设备", "制造", "能源", "材料", "石化", "演艺", "影视", "游戏", "体育",
    "央企", "国企", "一带一路", "京津冀", "长三角", "粤港澳", "科创", "创业",
)
# 份额类别：同一组合只保留一类。排除显式的非默认份额与非 A 类后缀。
SHARE_SUFFIX_EXCLUDE = ("C", "E", "H", "I", "O", "R", "Y", "B", "D")
MIN_SCALE_YI = 2.0            # 规模下限（亿元），口径为"当前规模"（见模块 docstring）
MAX_PER_MANAGER = 3           # 单家管理人纳入上限
DEFAULT_SEED = 20261013       # 抽样种子，保证可复现
# ★ 成立满 2 年。这不是新增规则，而是规则"取成立满 2 年时规模 ≥ 2 亿元"的
#   前提：成立不满 2 年就取不到那个时点的规模。同时近 5 年窗口 + 2 年下限
#   也要求基金至少有 2 年历史，否则样本不足以评估（曾出现 2026 年成立的基金
#   被选入，等于占名额而无数据）。
MIN_AGE_DAYS = 730


def _share_class(name: str) -> Optional[str]:
    """从简称推断份额类别；无法判断返回 None（视作默认份额）。"""
    if "(" in name or "（" in name:
        return "other"                     # 后端 / 场内 / 美元等，一律排除
    core = name.rstrip("A")
    if name.endswith("A") and len(core) >= 2:
        return "A"
    for s in SHARE_SUFFIX_EXCLUDE:
        if name.endswith(s) and len(name.rstrip(s)) >= 2:
            return s
    return None                            # 无后缀 = 默认份额


def candidates(refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    第一步：按类型 + 名称规则粗筛（不逐只请求，快）。

    返回 (候选 DataFrame, 排除记录 DataFrame)。
    排除记录列：基金代码 / 基金简称 / 排除原因 —— 规则要求逐条留痕。
    """
    import akshare as ak
    d = ak.fund_name_em()
    d = d[["基金代码", "基金简称", "基金类型"]].copy()
    d["基金代码"] = d["基金代码"].astype(str).str.zfill(6)

    keep_idx, excl = [], []
    for _, r in d.iterrows():
        name, ftype = str(r["基金简称"]), str(r["基金类型"])
        why = None
        if ftype not in ALLOWED_TYPES:
            why = f"类型不符（{ftype}）"
        else:
            hit = next((w for w in NAME_EXCLUDE_WORDS if w in name), None)
            if hit:
                why = f"名称含主题/被动词「{hit}」"
            else:
                sc = _share_class(name)
                if sc == "other":
                    why = "非默认份额（含括号后缀）"
                elif sc not in (None, "A"):
                    why = f"非 A 类份额（{sc} 类）"
        if why:
            excl.append({"基金代码": r["基金代码"], "基金简称": name,
                         "排除原因": why})
        else:
            keep_idx.append(r.name)
    return d.loc[keep_idx].reset_index(drop=True), pd.DataFrame(excl)


def enrich(codes: List[str], verbose: bool = True,
           sleep: float = 0.1) -> pd.DataFrame:
    """
    第二步：逐只取基本信息（成立时间 / 当前规模 / 管理人 / 投资策略关键词）。

    `投资策略` 用来二次确认"被动/指数"——名称没写但策略说明是被动管理的要剔除。
    """
    import akshare as ak
    rows = []
    t0 = time.time()
    for i, c in enumerate(codes, 1):
        try:
            d = ak.fund_individual_basic_info_xq(symbol=c)
            kv = dict(zip(d["item"].astype(str), d["value"].astype(str)))
            rows.append({
                "基金代码": c,
                "基金简称": kv.get("基金名称", c),
                "成立时间": kv.get("成立时间", ""),
                "当前规模_亿": _parse_scale(kv.get("最新规模", "")),
                "管理人": kv.get("基金公司", ""),
                "策略": kv.get("投资策略", "")[:200],
            })
        except Exception as e:                                # noqa: BLE001
            rows.append({"基金代码": c, "基金简称": c,
                         "成立时间": "", "当前规模_亿": np.nan,
                         "管理人": "", "策略": "",
                         "_err": f"{type(e).__name__}"})
        if verbose and i % 25 == 0:
            print(f"  [fundpool] {i}/{len(codes)} 只已取详情"
                  f"（{time.time() - t0:.0f}s）")
        time.sleep(sleep)
    return pd.DataFrame(rows)


def _parse_scale(s: str) -> float:
    """'197.40亿' → 197.4；'---' / 空 → NaN。"""
    s = str(s).strip()
    if not s or s in ("---", "—", "暂无", "0"):
        return np.nan
    try:
        if s.endswith("亿"):
            return float(s[:-1])
        if s.endswith("万"):
            return float(s[:-1]) / 1e4
        return float(s)
    except ValueError:
        return np.nan


def build_pool(target_per_layer: int = 20, seed: int = DEFAULT_SEED,
               verbose: bool = True, enrich_n: Optional[int] = None,
               pre_sample: int = 1200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按冻结规则构造基金池。

    分层（成立时间）：2015 前 / 2015–2019 / 2020 后 各约 1/3。
    每层内按代码排序后**固定种子**随机抽，保证可复现；单管理人上限 3 只。

    `enrich_n`   调试用：只对前 N 个候选取详情。
    `pre_sample` 预抽样：候选太多时先随机抽这一批再取详情。
                 默认 1200 —— 全量 3775 只逐只取详情要约 20 分钟，
                 而分层只需要每层 20 只，1200 已足够覆盖三层
                 （若某层不足会自动在报告中显示该层实际只数）。

    返回 (入选池 DataFrame, 排除记录 DataFrame)。
    """
    cand, excl = candidates()
    if verbose:
        print(f"[fundpool] 粗筛后候选 {len(cand)} 只，"
              f"名称/类型规则排除 {len(excl)} 只")
    if enrich_n:
        cand = cand.head(enrich_n).reset_index(drop=True)
    elif pre_sample and len(cand) > pre_sample:
        cand = cand.sample(pre_sample, random_state=seed).reset_index(drop=True)
        if verbose:
            print(f"[fundpool] 预抽样 {pre_sample} 只取详情"
                  f"（全量 {len(cand)} 只取详情要约 20 分钟，无必要）")
    det = enrich(cand["基金代码"].tolist(), verbose=verbose)

    # 合并详情与排除记录
    all_excl = [excl]
    det = det.copy()
    det["成立时间"] = pd.to_datetime(det["成立时间"], errors="coerce")
    det["层"] = _layer(det["成立时间"])
    if "_err" in det:
        bad = det[det["_err"].notna()]
        if len(bad):
            all_excl.append(pd.DataFrame({
                "基金代码": bad["基金代码"], "基金简称": bad["基金简称"],
                "排除原因": "详情获取失败（" + bad["_err"].astype(str) + "）"}))
            det = det[det["_err"].isna()]

    def _drop(mask, why):
        nonlocal det
        if mask.any():
            all_excl.append(pd.DataFrame({
                "基金代码": det.loc[mask, "基金代码"],
                "基金简称": det.loc[mask, "基金简称"],
                "排除原因": why}))
            det = det[~mask]

    # 策略二次确认：被动/指数化
    passive = det["策略"].str.contains("被动|指数化|复制|跟踪.*指数", regex=True,
                                       na=False)
    _drop(passive, "投资策略为被动指数化")
    _drop(det["当前规模_亿"].isna(), "规模未知")
    _drop(det["当前规模_亿"] < MIN_SCALE_YI,
          f"当前规模 < {MIN_SCALE_YI} 亿（口径降级：规则要求成立满2年时规模）")
    _drop(det["成立时间"].isna(), "成立时间未知")
    _drop(det["成立时间"] > pd.Timestamp.today() - pd.Timedelta(days=MIN_AGE_DAYS),
          f"成立未满 {MIN_AGE_DAYS} 天（规则要求成立满 2 年，"
          f"且样本不足以评估）")

    # 分层抽样 + 单管理人上限
    # ★ n_mgr 必须在**三层之外**计数：放在层内会每层重置，上限退化成
    #   "每层每家 3 只"（三层累计最多 9 只）—— 实测曾出现单家 5 只。
    rng = np.random.default_rng(seed)
    picked = []
    n_mgr: Dict[str, int] = {}
    for layer in ("2015前", "2015-2019", "2020后"):
        sub = det[det["层"] == layer].sort_values("基金代码")
        if sub.empty:
            continue
        order = rng.permutation(len(sub))
        for j in order:
            r = sub.iloc[j]
            m = str(r["管理人"])
            if n_mgr.get(m, 0) >= MAX_PER_MANAGER:
                continue
            n_mgr[m] = n_mgr.get(m, 0) + 1
            picked.append(r)
            if sum(1 for p in picked if p["层"] == layer) >= target_per_layer:
                break
    pool = pd.DataFrame(picked).reset_index(drop=True)
    # 未被选中的也要留痕（原因 = 未抽中 / 管理人超限）
    chosen = set(pool["基金代码"])
    rest = det[~det["基金代码"].isin(chosen)]
    if len(rest):
        all_excl.append(pd.DataFrame({
            "基金代码": rest["基金代码"], "基金简称": rest["基金简称"],
            "排除原因": "通过规则但未抽中（或所属管理人已达 3 只上限）"}))

    ex = pd.concat([e for e in all_excl if len(e)], ignore_index=True)
    if verbose:
        print(f"[fundpool] 入选 {len(pool)} 只，排除合计 {len(ex)} 条")
    return pool, ex


def _layer(s: pd.Series) -> pd.Series:
    """成立时间 → 分层标签。"""
    return pd.cut(
        s, bins=[pd.Timestamp("1900-01-01"), pd.Timestamp("2014-12-31"),
                 pd.Timestamp("2019-12-31"), pd.Timestamp("2100-01-01")],
        labels=["2015前", "2015-2019", "2020后"]).astype(object)


def load_pool() -> pd.DataFrame:
    """
    读入入选清单（`output/data/fund_pool.csv`），供其他脚本复用。

    返回空 DataFrame 表示还没扩池 —— 调用方应回退到 C.EXTENDED_FUNDS。
    """
    fp = C.DATA_DIR / "fund_pool.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(fp, dtype={"基金代码": str})
        return d
    except Exception:                                         # noqa: BLE001
        return pd.DataFrame()


def pool_codes() -> List[str]:
    """入选基金代码列表（未扩池时回退到 C.EXTENDED_FUNDS）。"""
    d = load_pool()
    if d.empty or "基金代码" not in d.columns:
        return list(C.EXTENDED_FUNDS.keys())
    return [str(c).zfill(6) for c in d["基金代码"].tolist()]


def pool_names() -> Dict[str, str]:
    """入选基金的 代码 → 简称。"""
    d = load_pool()
    if d.empty or "基金代码" not in d.columns:
        return dict(C.EXTENDED_FUNDS)
    return {str(r["基金代码"]).zfill(6): str(r["基金简称"])
            for _, r in d.iterrows()}
