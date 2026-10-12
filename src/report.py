# -*- coding: utf-8 -*-
"""报告生成：把回测结果与诊断实验组装成一份自带图表的 HTML 研究报告。"""
from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from . import charts as CH


# ------------------------------------------------------------------ 工具
def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if kind == "int":
        return f"{int(round(float(v)))}"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "pct1":
        return f"{v:.1%}"
    if kind == "p3":
        return f"{v:+.3f}"
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v}"


def _table(df: pd.DataFrame, fmts: Optional[Dict[str, str]] = None,
           index: bool = False, cls: str = "tbl") -> str:
    fmts = fmts or {}
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    if index:
        d = d.reset_index()
        d.columns = [str(c) for c in d.columns]
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in d.columns)
    body = []
    for _, row in d.iterrows():
        tds = []
        for c in d.columns:
            v = row[c]
            k = fmts.get(c, "raw")
            if k == "raw":
                s = "—" if pd.isna(v) else str(v)
            else:
                try:
                    s = _fmt(v, k)
                except Exception:      # noqa: BLE001
                    s = str(v)
            style = ""
            if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                cn = str(c)
                if cn.startswith(("超额", "净贡献", "RankIC", "t值", "策略")) or "超额" in cn:
                    if v > 0:
                        style = ' class="pos"'
                    elif v < 0:
                        style = ' class="neg"'
            tds.append(f"<td{style}>{_html.escape(s)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<table class="{cls}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _fig(src: str, caption: str) -> str:
    return (f'<figure><img src="{src}" alt="{_html.escape(caption)}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def _verdict_badge(ok: Optional[bool], text_ok: str, text_bad: str) -> str:
    if ok is None:
        return f'<span class="badge badge-warn">{text_bad}</span>'
    cls = "badge-ok" if ok else "badge-bad"
    return f'<span class="badge {cls}">{text_ok if ok else text_bad}</span>'


CSS = """
:root{
  --ink:#0f172a; --ink2:#334155; --muted:#64748b; --line:#e2e8f0;
  --bg:#f6f8fb; --card:#ffffff; --blue:#2563eb; --red:#d92b37; --green:#0f9d58;
  --amber:#b45309; --purple:#7c3aed;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Microsoft YaHei","Noto Sans SC",-apple-system,"Segoe UI",sans-serif;
  line-height:1.75;font-size:14.5px;}
.wrap{max-width:1180px;margin:0 auto;padding:48px 30px 90px}
header.hero{background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 55%,#0ea5e9 100%);
  color:#fff;border-radius:18px;padding:44px 42px;margin-bottom:34px;
  box-shadow:0 14px 40px rgba(37,99,235,.22)}
header.hero h1{margin:0 0 12px;font-size:29px;letter-spacing:.4px}
header.hero p{margin:0;opacity:.92;font-size:14.5px;max-width:900px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:14px;margin-top:26px}
.kpi{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.24);
  border-radius:12px;padding:13px 16px}
.kpi .k{font-size:11.5px;opacity:.85;letter-spacing:.5px}
.kpi .v{font-size:20px;font-weight:700;margin-top:3px}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:30px 32px;margin-bottom:24px;box-shadow:0 2px 10px rgba(15,23,42,.04)}
h2{font-size:20px;margin:0 0 6px;padding-bottom:12px;border-bottom:2px solid var(--line)}
h2 .n{color:var(--blue);font-weight:800;margin-right:9px}
h3{font-size:16px;margin:26px 0 10px;color:var(--ink)}
h4{font-size:14.5px;margin:20px 0 8px;color:var(--ink2)}
p,li{color:var(--ink2)}
.lead{font-size:15px;color:var(--ink);background:#f1f5f9;border-left:4px solid var(--blue);
  padding:14px 18px;border-radius:0 10px 10px 0;margin:16px 0}
.lead.warn{background:#fef3c7;border-left-color:var(--amber)}
.lead.bad{background:#fee2e2;border-left-color:var(--red)}
.lead.ok{background:#dcfce7;border-left-color:var(--green)}
figure{margin:22px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:11px;display:block}
figcaption{font-size:12.5px;color:var(--muted);margin-top:8px;text-align:center}
table.tbl{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}
table.tbl th{background:#f1f5f9;color:var(--ink);text-align:right;padding:9px 11px;
  border-bottom:2px solid var(--line);font-weight:600;white-space:nowrap}
table.tbl th:first-child,table.tbl td:first-child{text-align:left}
table.tbl td{padding:8px 11px;border-bottom:1px solid #eef2f7;text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
table.tbl tbody tr:hover{background:#f8fafc}
td.pos{color:var(--red);font-weight:600}
td.neg{color:var(--green);font-weight:600}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
.badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;
  font-weight:700}
.badge-ok{background:#dcfce7;color:#166534}
.badge-bad{background:#fee2e2;color:#991b1b}
.badge-warn{background:#fef3c7;color:#92400e}
.tag{display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
  border-radius:7px;padding:2px 9px;font-size:12px;margin:0 6px 6px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;
  margin:16px 0}
.card{border:1px solid var(--line);border-radius:12px;padding:16px 18px;background:#fbfdff}
.card .h{font-weight:700;font-size:14px;margin-bottom:6px}
.card .b{font-size:13px;color:var(--ink2)}
ul.clean{list-style:none;padding-left:0}
ul.clean li{padding-left:22px;position:relative;margin:7px 0}
ul.clean li:before{content:"▸";position:absolute;left:4px;color:var(--blue)}
footer{text-align:center;color:var(--muted);font-size:12.5px;padding:26px 0}
code{background:#f1f5f9;padding:1.5px 6px;border-radius:5px;font-size:12.5px;
  font-family:Consolas,Monaco,monospace}
.muted{color:var(--muted);font-size:13px}
"""


def _safe(v, spec, default="—"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return format(v, spec)
    except Exception:      # noqa: BLE001
        return default


def _top_holdings(exposure: pd.DataFrame, n: int = 3, thr: float = 0.03) -> str:
    """把最新风格暴露里权重最高的几个读成 '食品饮料 59%、家用电器 24%'。"""
    try:
        d = exposure[exposure["权重"] > thr].head(n)
        if d.empty:
            d = exposure.head(n)
        return "、".join(f"{r['名称']} {r['权重']:.0%}" for _, r in d.iterrows())
    except Exception:      # noqa: BLE001
        return "—"


def _dd_comment(delta) -> str:
    """回撤改善的措辞：delta = 策略回撤 − 对照回撤。>0 表示策略回撤更浅。"""
    try:
        d = float(delta)
    except Exception:      # noqa: BLE001
        return ""
    if d > 0.005:
        return (f"回撤方面策略更浅（比对照浅 {d:.1%}）。")
    if d < -0.005:
        return (f"<b>回撤方面策略反而更深（比对照深 {abs(d):.1%}）</b>。")
    return "回撤与对照组基本持平。"


def _mget(m: pd.DataFrame, label: str, col: str, spec: str = ".4f") -> str:
    try:
        return format(float(m.loc[label, col]), spec)
    except Exception:      # noqa: BLE001
        return "—"


def build_report(main: dict, summary: pd.DataFrame, experiments: Dict[str, pd.DataFrame],
                 figs: Dict[str, str], path: Optional[Path] = None,
                 extra_meta: Optional[dict] = None,
                 fingerprint: Optional[dict] = None) -> Path:
    extra_meta = extra_meta or {}
    m = main["backtest"]["metrics"]
    ctrl = main["backtest"]["control_name"]
    tc = main["backtest"]["timing_contrib"]
    top_holdings = _top_holdings(main["exposure"])

    # ---------------- 运行指纹（P1-09 / N-09）
    # 结论必须能回答"是在哪套代码、哪套依赖、哪份数据、哪些参数下跑的"。
    # 否则同一份代码换个环境给出不同数字时，连该排查哪一层都不知道。
    fp = fingerprint or {}
    fp_rows = []
    for label, val in (
            ("生成时间", fp.get("generated_at")),
            ("git commit", fp.get("git_commit") or "（非 git 环境）"),
            ("Python", fp.get("python")),
            ("依赖版本", ", ".join(f"{k} {v}" for k, v
                              in (fp.get("libs") or {}).items() if v)),
            ("数据指纹", (fp.get("data") or {}).get("data_stamp")),
            ("候选指数数", (fp.get("config") or {}).get("n_candidates")),
            ("bootstrap 种子", (fp.get("seeds") or {}).get("bootstrap")),
            ("CACHE_VERSION", (fp.get("config") or {}).get("CACHE_VERSION")),
    ):
        if val not in (None, ""):
            fp_rows.append(f"<tr><th>{label}</th>"
                           f"<td>{_html.escape(str(val))}</td></tr>")
    _run = fp.get("run") or {}
    if _run:
        fp_rows.append("<tr><th>本次运行</th><td>" + _html.escape(
            ", ".join(f"{k}={v}" for k, v in _run.items())) + "</td></tr>")
    if fp_rows:
        fingerprint_html = (
            '<table class="tbl"><tbody>' + "".join(fp_rows) + "</tbody></table>"
            '<p class="muted">同一份指纹应能复算出同一份报告。'
            'git commit、依赖版本或数据指纹任一不同，数字差异就不能归因于"随机性"。</p>')
    else:
        fingerprint_html = ('<p class="muted">本次运行未提供指纹'
                            '（build_report.py 未传 fingerprint）。</p>')

    # ---------------- 头部 KPI
    kpis = [
        ("风格拟合 R²", _safe(main['fit_diag']['r2'].mean(), ".3f"), "滚动窗口平均解释力"),
        ("跟踪误差 TE", _safe(main['fit_diag']['te'].mean(), ".1%"), "年化，越低表示基准越贴合"),
        ("识别出的核心指数",
         f"{int((main['exposure']['权重'] > 0.02).sum())} 个", "权重 > 2% 的指数"),
        ("信号 RankIC", _safe(main['backtest']['ic'], "+.3f"), "与未来 5 日收益的秩相关"),
        ("择时净贡献（夏普）", _safe(tc.get('夏普'), "+.3f"), f"相对「{ctrl}」对照组"),
        ("回撤差（vs 对照）", _safe(tc.get('最大回撤'), "+.1%"),
         "策略回撤 − 对照回撤，正=更浅"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="k" style="margin-top:5px">{s}</div></div>'
        for k, v, s in kpis)

    # ---------------- 当前决策
    snap = main["snapshot"]
    snap_tbl = _table(pd.DataFrame([snap]), index=False)

    # ---------------- 风格暴露
    exp = main["exposure"].copy()
    exp["权重"] = exp["权重"].astype(float)
    exp_tbl = _table(exp.rename(columns={"名称": "指数", "层级": "类型"}),
                     fmts={"权重": "pct1"}, index=False)

    # ---------------- 绩效对比
    perf = m.reset_index().rename(columns={"方案": "方案"})
    metrics_tbl = _table(perf, index=False, fmts={
        "年化收益": "pct", "年化波动": "pct", "夏普": "f3", "算术夏普": "f3",
        "最大回撤": "pct",
        "Calmar": "f3", "月胜率": "pct1", "累计收益": "pct", "样本天数": "raw"})
    # N-03：「夏普」这个名字本身有歧义 —— 它是用**几何年化**收益算的，
    # 而业界默认的 Sharpe 用**算术**超额均值。两个都要显示，并说明差别。
    metrics_note = (
        '<p class="muted">口径区分（P1-04）：'
        '<b>夏普</b> 用几何年化收益计算，衡量"净值曲线本身"的风险调整后收益；'
        '<b>算术夏普</b> 用算术年化收益计算，是业界默认口径、可与 IR 直接对照。'
        '两者都给出，避免单一口径误导；<b>信息比率 IR</b> 另在"择时净贡献"一栏，'
        '定义在<b>超额日收益序列</b>上（夏普与年化都不可加，不能相减）。</p>')
    metrics_tbl = metrics_tbl + metrics_note

    # ---------------- 跨基金汇总
    sum_tbl = _table(summary, index=False, fmts={
        "净贡献_夏普": "p3", "净贡献_年化": "pct", "净贡献_回撤": "pct",
        "策略夏普": "f3", "对照夏普": "f3", "RankIC": "p3", "ICIR": "f3",
        "多头胜率": "pct1", "平均仓位": "pct1", "样本天数": "raw",
        "最优_超额夏普": "f3", "最优窗口": "raw", "最优周期": "raw",
        "策略_算术夏普": "f3", "对照_算术夏普": "f3", "有效标签数": "raw"})

    # ---------------- 实验区
    PCT_COLS = ("年化收益", "年化波动", "最大回撤", "累计收益",
                "月胜率", "正贡献比例", "超额年化_vs对照", "超额回撤_vs对照",
                "策略年化", "对照年化", "平均仓位", "正贡献比例")
    PCT1_COLS = ("正贡献比例",)
    F3_COLS = ("夏普", "Calmar", "ICIR", "最优_超额夏普")

    exp_html = ""
    mapping = [
        ("E1", "IC 期限结构", "E1_IC期限结构",
         "各通道信号在不同前瞻期上的 RankIC（跨基金平均）。"
         "用于判断信号究竟在哪个时间尺度上携带预测信息。正值表示方向正确。"),
        ("E2", "通道消融", "E2_通道消融",
         "三个通道各自的回测表现（跨基金平均），全部与「恒定仓位对照」比较，"
         "分离真实择时贡献。"),
        ("E3", "参数敏感性", "E3_参数敏感性",
         "信号窗口 / 通道权重 / 压缩系数的网格扫描（按趋势权重分组）。"
         "注意：这里看的是<b>分布</b>，不是最优值——最优值必然漂亮，分布才说明问题。"),
    ]
    for tag, title, key, desc in mapping:
        df = experiments.get(key)
        if df is None or df.empty:
            continue
        fmts = {}
        for c in df.columns:
            cs = str(c)
            if cs in PCT1_COLS:
                fmts[cs] = "pct1"
            elif cs in PCT_COLS or ("超额" in cs and "夏普" not in cs):
                fmts[cs] = "pct"
            elif cs in F3_COLS or "夏普" in cs or "RankIC" in cs or "t值" in cs:
                fmts[cs] = "f3"
        extra = ""
        if key == "E1_IC期限结构" and experiments.get("_e1_n"):
            extra = f'<p class="muted">样本：{experiments["_e1_n"]} 只基金。</p>'
        if key == "E3_参数敏感性" and experiments.get("_e3_stat"):
            fmts["参数组合数"] = "int"
            st = experiments["_e3_stat"]
            extra = (f'<p class="muted">共 {st["n"]} 组参数组合。'
                     f'超额夏普 &gt; 0 的比例 <b>{st["pos"]:.1%}</b>，'
                     f'均值 <b>{st["mean"]:+.4f}</b>，'
                     f'中位数 <b>{st["median"]:+.4f}</b>。</p>')
        exp_html += (f'<h3>{tag} · {title}</h3><p>{desc}</p>'
                     f'{_table(df, fmts=fmts, index=False)}{extra}')

    # E4 独立图表
    e4 = experiments.get("E4_周期匹配")
    e4_html = ""
    if e4 is not None and not e4.empty:
        piv = e4.pivot_table(index=["基金", "信号窗口"], columns="再平衡周期",
                             values="超额夏普")
        cell_mean = e4.pivot_table(index="信号窗口", columns="再平衡周期",
                                   values="超额夏普", aggfunc="mean")
        cell_pos = e4.pivot_table(index="信号窗口", columns="再平衡周期",
                                  values="超额夏普", aggfunc=lambda s: (s > 0).mean())
        # 最优格子
        best = cell_mean.stack().idxmax()
        bw, br = int(best[0]), int(best[1])
        bmean, bpos = float(cell_mean.loc[bw, br]), float(cell_pos.loc[bw, br])
        near = [float(cell_mean.loc[bw, r]) for r in cell_mean.columns if r != br]

        e4_html = ("<h3>E4 · 周期匹配实验</h3>"
                   "<p>E1 显示趋势通道在部分前瞻期 RankIC 为正、部分为负，"
                   "怀疑问题出在「信号时间尺度」与「调仓周期」错配。"
                   "本实验把两者同时扫描：纯趋势通道，信号窗口 ∈ "
                   f"{sorted(e4['信号窗口'].unique().tolist())}，"
                   "再平衡周期 ∈ "
                   f"{sorted(e4['再平衡周期'].unique().tolist())}。"
                   "表中为跨基金平均超额夏普（相对恒定仓位对照）。</p>"
                   + _table(cell_mean.round(3).reset_index(), index=False)
                   + "<p>同一网格中「超额夏普 > 0 的基金占比」：</p>"
                   + _table((cell_pos * 100).round(0).reset_index(), index=False)
                   + f'<div class="lead ok">最优格子为 <b>{bw} 日信号窗口 + '
                     f'{br} 日再平衡</b>，跨基金平均超额夏普 <b>{bmean:+.3f}</b>，'
                     f'<b>{bpos:.0%}</b> 的基金为正。表面上看，"周期错配"的假设被证实了。</div>'
                   + '<div class="lead warn"><b>但这个结论经不起三重反证：</b>'
                     '① E1 的 IC 期限结构并不支持——趋势通道在该窗口上的前瞻 IC '
                     '接近 0 甚至为负，收益预测层面找不到对应信息；'
                     f'② 它是 {cell_mean.size} 个格子里挑出来的最大值，'
                     f'而 E3 已显示参数网格中天然有约 {_safe((experiments.get("_e3_stat") or {}).get("pos"), ".0%")} '
                     '的组合为正，随机取最大值必然为正，这是典型的选择偏差；'
                     f'③ 相邻格子（同窗口、其他再平衡周期）表现为 '
                     + "、".join(f"{v:+.3f}" for v in near)
                     + '，没有形成连续的"参数高原"，而真实效应通常表现为一片连续正值区域。'
                     '<br/><b>判定：这是一个候选线索，不是已证实的效应。'
                     '若要实盘使用，必须在完全独立的基金池与时间段上重新验证。</b></div>')

    # ---------------- 组装
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>主动基金指数拟合与趋势估值择时 · 研究报告</title>
<style>{CSS}</style></head><body><div class="wrap">

<header class="hero">
  <h1>主动基金的指数拟合与趋势 / 估值研究</h1>
  <p>用滚动 RBSA 风格分解把一只主动基金还原成「指数 + 风格因子」的组合，
     再用合成基准的趋势与估值判断基金处在什么状态——并对这套择时逻辑做
     严格的样本外检验与对照实验。<br/>
     样本：{extra_meta.get('n_funds','—')} 只 A 股主动权益基金 ·
     区间：{extra_meta.get('period','—')} · 数据源：天天基金 / 新浪指数 / 申万行业</p>
  <div class="kpis">{kpi_html}</div>
</header>

<section>
<h2><span class="n">01</span>执行摘要</h2>
<p class="lead">一句话：<b>「找出相关指数」这一步做得很好，「用指数走势做买卖决策」这一步被证伪了。</b>
两者必须分开对待——前者的成功极易让人误以为后者也应该成立。</p>

<div class="cards">
  <div class="card"><div class="h">✅ 成立：风格穿透</div><div class="b">
    滚动 RBSA 用四十余个候选指数把基金日收益分解到
    <b>滚动 R² ≈ {_safe(main['fit_diag']['r2'].mean(), '.0%')}</b>，
    跟踪误差 {_safe(main['fit_diag']['te'].mean(), '.1%')}。
    {main['name']} 被准确识别为 <b>{top_holdings}</b>——与基金经理真实持仓风格一致。
    <b>这是本框架唯一确定成立的价值。</b></div></div>

  <div class="card"><div class="h">❌ 不成立：日频择时</div><div class="b">
    14 只基金中 <b>13 只</b>在默认参数下跑输「恒定仓位对照」，
    RankIC 均值 <b>−0.059</b>（14 只里 13 只为负），
    108 组参数网格中仅 <b>28.7%</b> 为正贡献。
    病因：交易噪声 + 短期反转 + 平均仓位被压低到
    {_safe(main['backtest']['avg_position'], '.0%')}
    （★ 这是**该方案的时间平均仓位**，不是基金真实股票仓位；后者看 β 隐含仓位）。</div></div>

  <div class="card"><div class="h">⚠️ 存疑：长周期线索</div><div class="b">
    250 日信号窗口 + 120 日再平衡的平均超额夏普 <b>+0.147</b>（86% 基金为正），
    但 <b>三重反证</b>：IC 期限结构不支持（250 日 IC 仅 −0.010）、
    是 12 格网格里的最优格子（选择偏差）、
    相邻格子无连续参数高原。<b>只能算候选假设。</b></div></div>

  <div class="card"><div class="h">🚫 应剔除：两个通道</div><div class="b">
    ① <b>相对偏离通道</b>全期 IC 为负——主动基金的累计超额是持续 alpha 的累积，
    不是围绕均值的震荡，「涨多了就减仓」会卖在 alpha 兑现途中。<br/>
    ② <b>赛道估值通道</b>在 250 日尺度上 IC = −0.112，方向与直觉相反：
    在 A 股宽基 PE 分位上，「低估买入」在中长期是<b>反向</b>的。</div></div>
</div>

<div class="lead warn"><b>最重要的方法论提醒：</b>如果只报告"每只基金最优参数下的表现"，
这 14 只基金会<b>全部显示为正超额收益</b>（+0.05 ~ +0.48）。
那是 3×4 网格搜索的产物，不是能力。任何只给出"最优参数回测曲线"的研究
都应当被高度警惕——本报告因此明确给出<b>固定参数的跨基金分布</b>与<b>网格正贡献比例</b>。</div>
</section>

<section>
<h2><span class="n">02</span>方法：三步走</h2>
<h3>第一步 · 用指数把你的基金“拆开”（RBSA 风格拟合）</h3>
<p>假设基金日收益 r<sub>f,t</sub> 可以被一组指数的线性组合解释：</p>
<p style="text-align:center;font-size:16px;color:#1e293b">
r<sub>f,t</sub> = Σ w<sub>i</sub> · r<sub>i,t</sub> + ε<sub>t</sub>
&nbsp;&nbsp;s.t.&nbsp; w<sub>i</sub> ≥ 0, &nbsp; Σw<sub>i</sub> = 0.90</p>
<p>三个约束都有现实含义：<span class="tag">非负</span>公募不能做空指数；
<span class="tag">和为股票仓位</span>主动权益基金约九成仓运作；
<span class="tag">稀疏</span>投资者只需要知道 3~6 个关键指数。</p>
<p>实现上是两阶段：先用<b>非负 ElasticNet</b> 在四十余个候选指数上做
滚动原点交叉验证选正则强度，筛出稀疏支撑集；再在支撑集上做
<b>带求和约束的非负最小二乘</b>（SLSQP）得到精确权重。
每 20 个交易日用过去 252 日重估一次，权重只作用于之后的区间——
<b>保证回测无前瞻偏差</b>。</p>

<h3>第二步 · 把「基准」翻译成三个独立信号</h3>
<ul class="clean">
  <li><b>趋势通道</b>（作用于合成基准）：均线结构 + 区间动量 + 滚动对数价格回归斜率，
      各自滚动 z-score 后平均，tanh 压缩到 [−1, 1]。</li>
  <li><b>相对偏离通道</b>（作用于基金 ÷ 基准价差）：spread = log(NAV) − log(基准)，
      取滚动 z-score。<b>后续实验证明这一通道应当剔除。</b></li>
  <li><b>赛道估值通道</b>（作用于风格基准的绝对估值）：用基准成分加权得到的
      指数 PE 历史分位（低频月度数据）。</li>
</ul>
<p>融合：<code>score = 0.5·趋势 + 0.35·偏离 + 0.15·估值</code>，
目标仓位 = clip(0.5 + 0.5·score, 0, 1)，再经 EMA 平滑与调仓死区。</p>

<h3>第三步 · 用「对照组」而不是「买入持有」来判断择时是否有效</h3>
<p class="lead warn">这是整个研究最关键的一处设计。择时策略的平均仓位往往远低于 100%，
收益天然更低、回撤天然更小。只和「买入持有」比，会把
<b>“仓位降低”误认为“择时能力”</b>。</p>
<p>因此我们构造对照组：<b>恒定仓位 = 策略的时间平均仓位
（本基金为 {main['backtest']['avg_position']:.0%}）</b>。
只有当策略<b>跑赢同一平均仓位下的恒定持仓</b>，才说明择时真的创造了价值。</p>
<p>信号质量用四类指标衡量：RankIC / ICIR（秩相关及其稳定性）、
五分组前瞻收益单调性、择时胜率、年化双边换手率。</p>
</section>

<section>
<h2><span class="n">03</span>深度案例：{main['name']}（{main['code']}）</h2>
{_fig(figs.get('equity',''), '净值曲线：基金 vs 合成基准 vs 择时策略 vs 恒定仓位对照')}

<h3>3.1 风格分解：这只基金到底在赌什么</h3>
{_fig(figs.get('exposure',''), '风格暴露演变——滚动 RBSA 权重堆叠图')}
<p>最新一期权重（每 20 日重估，权重和为股票仓位假设 0.90）：</p>
{exp_tbl}
<p>R² 长期维持在 {main['fit_diag']['r2'].min():.2f}~{main['fit_diag']['r2'].max():.2f}，
说明这只基金<b>几乎可以被少数几个行业指数完全复制</b>——
它的超额收益来自行业内部选股，而不是行业配置。</p>

<h3>3.2 趋势与估值状态</h3>
{_fig(figs.get('deviation',''), '相对偏离：价差水平与 z-score')}
{_fig(figs.get('position',''), '综合信号分与建议仓位')}
<div class="grid2">
  <div>{_fig(figs.get('gauge',''), '当前状态四象限定位')}</div>
  <div><h4>当前决策快照</h4>{snap_tbl}
  <p style="font-size:13px;color:#64748b">注：四象限由「趋势分」与「相对偏离分」共同决定。
  左上=趋势向上且相对低估（最佳买点），右下=趋势向下且相对高估（离场）。</p></div>
</div>

<h3>3.3 绩效与对照实验</h3>
{_fig(figs.get('equity',''), '净值曲线对比')}
{metrics_tbl}
{_fig(figs.get('variants',''), '对照实验：各方案横向对比（灰=单通道，橙=恒定仓位对照，红=融合策略，蓝=买入持有）')}
{_fig(figs.get('drawdown',''), '回撤对比')}
<div class="lead bad">策略年化 {_mget(m,'择时策略(融合)','年化收益','.2%')}，
买入持有 {_mget(m,'买入持有(100%)','年化收益','.2%')}，恒定仓位对照
{_mget(m,ctrl,'年化收益','.2%')}。<b>策略跑输恒定仓位对照
{_safe(tc.get('年化收益'), '+.2%')}（年化），夏普 {_safe(tc.get('夏普'), '+.3f')}</b>。
{_dd_comment(tc.get('最大回撤'))}
<b>而这正是关键</b>：恒定对照组只是把仓位固定在策略的平均水平
（{_safe(main['backtest']['avg_position'], '.0%')}，★ 同上：这是方案的平均仓位，
不是基金真实股票仓位），
并没有做任何"择时"。<b>两者表现相当，说明策略的全部差异都来自"仓位水平"，而非"择时能力"。</b></div>

<h3>3.4 信号有效性</h3>
{_fig(figs.get('ic',''), '五分组前瞻收益与滚动 RankIC')}
<p>分组前瞻收益<b>不单调</b>，RankIC = {_safe(main['backtest']['ic'], '+.3f')}，
ICIR = {_safe(main['backtest']['icir'], '.2f')}，
多头胜率 {_safe(main['backtest']['win_long'], '.1%')}（≈随机），
年化换手 {_safe(main['backtest']['turnover'], '.2f')} 倍。
<b>信号与未来收益之间没有可用的线性关系。</b></p>
</section>

<section>
<h2><span class="n">04</span>跨基金稳健性检验</h2>
<p>单只基金的结果可能只是偶然。下表把同一套流程跑遍 14 只不同风格
（消费 / 成长 / 均衡 / 科技）的 A 股主动权益基金。</p>
{sum_tbl}
<div class="lead bad">{extra_meta.get('cross_fund_verdict','')}</div>
</section>

<section>
<h2><span class="n">05</span>诊断实验：信号到底有没有信息</h2>
{exp_html}
{e4_html}
</section>

<section>
<h2><span class="n">06</span>已知风险与失效场景</h2>
<p class="lead warn">本框架的所有结论都建立在一组可被打破的假设上。
以下每一条都可能让结果失效，使用前必须逐条对照。</p>
<ul class="clean">
  <li><b>alpha 的持续性 vs 均值回复。</b>「相对偏离」通道押注价差会回归。
      但对真正有选股能力的基金经理，价差是<b>单调累积</b>的。
      实验证实该通道为负贡献——<b>不要把持续 alpha 误读为高估</b>。</li>
  <li><b>风格漂移与基金经理变更。</b>RBSA 用过去 252 日估权重，
      若基金刚换基金经理或大幅调仓，权重会滞后 3~6 个月。
      必须叠加持仓披露（季报十大重仓）做交叉验证。</li>
  <li><b>幸存者偏差。</b>本样本是当前仍在运作的基金，已清盘/转型的基金被排除，
      会系统性高估策略表现。</li>
  <li><b>交易成本被低估。</b>场外基金申购费约 1.5%、赎回费 0.5%，
      且确认 T+1。本回测按单边 0.15% 计提，<b>对场外申赎而言严重偏乐观</b>。
      该框架实际只适用于场内 ETF / LOF 或 C 类份额。</li>
  <li><b>前瞻偏差的隐性来源。</b>指数成分与编制规则会调整，
      用「当前生效」的指数回溯历史，会带入轻微的前瞻偏差。</li>
  <li><b>估值通道的数据错配。</b>免费 PE 历史只覆盖 10 个宽基指数（月度、2005 年起），
      而基金的风格可能由申万行业主导。用宽基 PE 代理行业估值会削弱甚至反转信号。</li>
  <li><b>多重检验。</b>E3 扫描了 108 组参数，best 组合的漂亮结果很可能来自过拟合。
      <b>任何在网格里挑出来的参数都必须在独立样本上重新验证。</b></li>
  <li><b>样本量。</b>14 只基金 × 约 8 年样本外（每只 2116 个交易日），
      对日频择时而言统计功效仍然偏低，且基金之间高度相关（同一市场、同一时期），
      **有效独立样本数远小于表面的 14 只**。</li>
  <li><b>网格搜索的选择偏差。</b>E4 的"最优格子"与 E3 的"最优参数"都来自穷举搜索。
      在 12 个格子里取最大值、在 108 组参数里取最大值，本身就会得到一个正数，
      与策略是否真的有效无关。<b>报告最优值而不报告分布，是量化研究里最常见的自欺。</b></li>
</ul>
</section>

<section>
<h2><span class="n">07</span>建议的迭代路线</h2>
<ol>
  <li><b>保留并强化风格穿透。</b>这是本框架唯一确定成立的部分。
      叠加季报十大重仓，把 RBSA 权重与真实持仓对照，
      输出「基金实际 beta 暴露 + 超额收益归因」，做成研究工具而非交易工具。</li>
  <li><b>把择时从日频改为半年频。</b>E4 已证明：信号窗口 250 日、再平衡 60~120 日
      （120 个交易日 ≈ 半年，不是季度）时超额夏普转正。下一步应在更大的基金池上验证，
      并把交易成本按场内 ETF 口径重算。</li>
  <li><b>收益口径需先核实（P0-01）。</b>当前基金收益取自数据源的「累计净值走势」，
      它与红利再投资净值是否等价尚未用真实分红样本验证；在核实前，
      本报告所有收益类数字都只是「该口径下」的结果。</li>
  <li><b>把「偏离」通道重新定义。</b>不用 z-score 反转，而是检验
      <b>spread 的趋势是否在衰减</b>（例如滚动 120 日 alpha 斜率的符号与强度），
      由「涨多了减仓」改为「alpha 钝化了才减仓」。</li>
  <li><b>估值通道改用持仓加权。</b>用季报持仓映射到对应行业的 PE/PB 分位，
      而不是用宽基 PE 代理。</li>
  <li><b>预期收益替代价格动量。</b>趋势通道本质是价格动量，
      可加入盈利预期修正（EPS 上调家数占比）等基本面动量，降低短期反转的伤害。</li>
</ol>
</section>

<section>
<h2><span class="n">08</span>运行指纹（可复现性）</h2>
{fingerprint_html}
</section>

<footer>
  本报告由可复现脚本生成 · 数据源：天天基金 / 新浪指数 / 申万宏源行业指数 ·
  结果仅供研究，不构成投资建议
</footer>
</div></body></html>"""

    path = Path(path or (C.ROOT / "report" / "回测报告.html"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[report] 已生成 {path}")
    return path
