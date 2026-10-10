/* Fund Lens · 前端逻辑 */
'use strict';

// ==================================================================== 全局状态
const S = {
  cat: null,          // 目录（模型清单、已知基金）
  result: null,       // 当前分析结果
  sel: null,          // 选中的模型 key
  best: null,         // 当前"最佳"模型 key
  params: {},         // 每个模型的超参覆盖
  rollWin: 120,
  charts: {},         // echarts 实例缓存
};

const $ = (id) => document.getElementById(id);
const fmtPct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v * 100).toFixed(d) + '%';
const fmtNum = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toFixed(d);
const fmtSign = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(d);
const fmtPctSign = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';

// A股习惯：涨红跌绿
const UP = '#ef4a4a', DOWN = '#12b886', FLAT = '#8b98ab';
const colOf = (v) => v > 0 ? UP : v < 0 ? DOWN : FLAT;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok && r.status !== 202) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch (e) { }
    throw new Error(msg);
  }
  return r.json();
}
function showAlert(msg) {
  const a = $('alert');
  if (!msg) { a.hidden = true; return; }
  a.textContent = msg; a.hidden = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function busy(on, txt) {
  $('loading').hidden = !on;
  if (txt) $('loadingTxt').textContent = txt;
}

// ==================================================================== 初始化
(async function init() {
  try {
    S.cat = await api('/api/catalog');
  } catch (e) {
    showAlert('无法连接后端：' + e.message); return;
  }
  // 示例基金
  const sel = $('presetSel');
  S.cat.known_funds.forEach(f => {
    const o = document.createElement('option');
    o.value = f.code; o.textContent = `${f.code} ${f.name}`;
    sel.appendChild(o);
  });
  const chips = $('emptyChips');
  S.cat.known_funds.slice(0, 10).forEach(f => {
    const c = document.createElement('div');
    c.className = 'chip'; c.textContent = f.code;
    c.onclick = () => { $('codeInput').value = f.code; run(f.code); };
    chips.appendChild(c);
  });
  renderModelPicker();
  $('rebalSel').onchange = updateRunHint;
  $('autoEquity').onchange = updateRunHint;

  $('searchForm').onsubmit = e => {
    e.preventDefault();
    const c = $('codeInput').value.trim();
    if (!/^\d{6}$/.test(c)) { showAlert('请输入 6 位数字基金代码，例如 110022'); return; }
    run(c);
  };
  sel.onchange = e => { if (e.target.value) run(e.target.value); };

  $('btnRefresh').onclick = () => S.result && run(S.result.code, { refresh: true });
  $('btnReLearn').onclick = () => S.result && run(S.result.code, { force: true });
  $('btnSaveSel').onclick = () => {
    if (!S.sel) return;
    S.best = S.sel;
    renderModelStrip(); renderRank();
    showAlert(`已把「${S.result.models[S.sel].name}」设为最佳模型基准。`);
  };
  $('rollWin').onchange = e => {
    S.rollWin = +e.target.value || 120;
    if (S.result) renderCorr();
  };
  $('btnApplyParams').onclick = applyParams;
  $('btnResetParams').onclick = () => {
    if (S.sel) delete S.params[S.sel];
    renderTuner();
  };

  initBacktestPicker();
  $('btRun').onclick = runBacktest;
})();

// 模型勾选：预设按钮批量勾选，也可逐个自定义
function renderModelPicker() {
  const list = $('mpList');
  $('mpQuick').textContent = `快速 ${S.cat.quick_models.length} 个`;
  $('mpFull').textContent = `全部 ${S.cat.full_models.length} 个`;
  S.cat.models.forEach(m => {
    const lab = document.createElement('label');
    lab.className = 'mp-item';
    lab.title = m.desc;
    lab.innerHTML = `<input type="checkbox" data-k="${m.key}"> ${m.name}`;
    list.appendChild(lab);
  });
  $('mpQuick').onclick = () => applyPreset(S.cat.quick_models);
  $('mpFull').onclick = () => applyPreset(S.cat.full_models);
  list.querySelectorAll('input').forEach(cb => { cb.onchange = updateRunHint; });
  applyPreset(S.cat.quick_models);
}
function applyPreset(keys) {
  $('mpList').querySelectorAll('input').forEach(cb => {
    cb.checked = keys.includes(cb.dataset.k);
  });
  updateRunHint();
}
function selectedModels() {
  return [...$('mpList').querySelectorAll('input:checked')].map(cb => cb.dataset.k);
}
function selectedAuto() {
  return $('autoEquity').checked;
}
function selectedRebal() {
  const v = +$('rebalSel').value;
  return v === 0 ? 0 : v;      // 0 表示不抽稀
}
function updateRunHint() {
  const n = selectedModels().length;
  const rb = $('rebalSel').selectedOptions[0].textContent.split('·')[1].trim();
  $('runHint').textContent = `本次将跑 ${n} 个模型 · 重估点${rb}` +
    (selectedAuto() ? ' · 仓位自适应' : '');
}

// ==================================================================== 运行分析
async function run(code, opts = {}) {
  opts = opts || {};
  showAlert(null);
  const models = selectedModels();
  if (!models.length) { showAlert('请至少勾选一个模型'); return; }
  const params = JSON.parse(JSON.stringify(S.params));
  $('runBtn').disabled = true;
  $('progressWrap').hidden = false;
  setProgress('提交任务…', 0.01, '');

  try {
    const body = {
      code, models,
      params,
      max_rebal: selectedRebal(),
      auto_equity: selectedAuto(),
      refresh: !!opts.refresh,
    };
    // 强制重学：绕过结果缓存
    if (opts.force) body.refresh = true;
    const r = await api('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (r.cached && r.result) {
      applyResult(r.result);
      $('progressWrap').hidden = true;
      return;
    }
    await poll(r.job_id);
  } catch (e) {
    showAlert('分析失败：' + e.message);
    $('progressWrap').hidden = true;
  } finally {
    $('runBtn').disabled = false;
  }
}

function setProgress(stage, pct, detail) {
  $('pStage').textContent = stage;
  $('pPct').textContent = Math.round((pct || 0) * 100) + '%';
  $('pFill').style.width = Math.max(2, (pct || 0) * 100) + '%';
  $('pDetail').textContent = detail || '';
}

async function poll(jid) {
  for (;;) {
    await new Promise(r => setTimeout(r, 900));
    let j;
    try { j = await api('/api/job/' + jid); }
    catch (e) { showAlert('任务查询失败：' + e.message); $('progressWrap').hidden = true; return; }
    setProgress(j.stage, j.progress, j.detail);
    if (j.status === 'done') {
      const r = await api('/api/result/' + jid);
      $('progressWrap').hidden = true;
      applyResult(r.result);
      return;
    }
    if (j.status === 'error') {
      $('progressWrap').hidden = true;
      showAlert('分析失败：' + j.error);
      return;
    }
  }
}

function applyResult(res) {
  S.result = res;
  S.best = res.meta.best_model || Object.keys(res.models)[0];
  S.sel = S.best;
  $('empty').hidden = true;
  $('main').hidden = false;
  renderHead();
  renderInsights();
  renderModelStrip();
  renderRank();
  renderSelected();
  renderNav();
  renderCorr();
  syncBacktestDefaults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==================================================================== 头部
function renderHead() {
  const r = S.result, st = r.nav.stats;
  $('fundName').textContent = `${r.name}（${r.code}）`;
  const m = r.meta;
  $('fundMeta').textContent =
    `${st.start} ~ ${st.end} · ${st.n_days} 交易日 · 候选指数 ${m.n_assets} 个 · ` +
    `窗口 ${m.window} 日 / 重估 ${m.refit} 日 · 用时 ${m.elapsed}s` +
    (m.from_cache ? ' · 缓存' : '');

  const bm = r.models[S.best];
  const kpi = [
    { lb: '最新累计净值', vl: fmtNum(st.latest, 4), sb: `累计 ${fmtPctSign(st.total_return)}` },
    { lb: '年化收益', vl: fmtPctSign(st.ann_return, 1), sb: `年化波动 ${fmtPct(st.ann_vol, 1)}` },
    { lb: '最大回撤', vl: fmtPct(st.max_drawdown, 1), sb: `当前 60日波动 ${fmtPct(st.current_vol60, 1)}`, cls: 'bad' },
    { lb: '拟合模型', vl: bm ? bm.name.slice(0, 8) : '—', sb: bm ? `样本外R² ${fmtNum(bm.metrics.oos_r2, 3)}` : '', cls: 'good' },
    { lb: '跟踪误差', vl: bm ? fmtPct(bm.metrics.oos_te, 1) : '—', sb: bm ? `方向一致 ${fmtPct(bm.metrics.win, 1)}` : '' },
  ];
  $('kpis').innerHTML = kpi.map(k => `
    <div class="kpi ${k.cls || ''}">
      <div class="lb">${k.lb}</div><div class="vl">${k.vl}</div>
      <div class="sb">${k.sb || ''}</div>
    </div>`).join('');
}

function renderInsights() {
  const ins = S.result.insights || {};
  const order = [
    ['penetration', 'good', '✓ 风格穿透成立'],
    ['exposure_gap', 'warn', '⚠ 仓位假设检验'],
    ['overfit', 'warn', '⚠ 过拟合检查'],
    ['drift', 'warn', '◎ 风格漂移'],
    ['exposure_fix', 'info', '≡ 如何解读低 R²'],
    ['te_spread', 'info', '≡ 模型差异'],
    ['coverage', 'info', '≡ 候选池'],
    ['trend', 'info', '≡ 近期动量'],
  ];
  $('insights').innerHTML = order.filter(([k]) => ins[k]).map(([k, cls, hd]) => {
    const it = ins[k];
    const realCls = it.warn ? (k === 'overfit' || k === 'drift' ||
                                k === 'exposure_gap' ? 'warn' : 'info') : cls;
    return `<div class="card ${realCls}">
      <div class="hd">${hd}</div><div class="bd">${it.text}</div></div>`;
  }).join('');
}

// ==================================================================== 模型条
function renderModelStrip() {
  const r = S.result;
  const keys = r.meta.models;
  $('modelStrip').innerHTML = keys.map(k => {
    const m = r.models[k]; if (!m) return '';
    const err = m.metrics.oos_r2 === null;
    return `<div class="mcard ${k === S.sel ? 'sel' : ''}" data-k="${k}">
      ${k === S.best ? '<span class="best">最佳</span>' : ''}
      <div class="nm">${m.name}</div>
      <div class="r2" style="color:${err ? 'var(--bad)' : 'var(--fg)'}">
        ${err ? '失败' : fmtNum(m.metrics.oos_r2, 3)}</div>
      <div class="sub">TE ${err ? '—' : fmtPct(m.metrics.oos_te, 1)} · ${
        m.interpretable ? '可解释' : '黑箱'}</div>
    </div>`;
  }).join('');
  [...$('modelStrip').children].forEach(c => {
    c.onclick = () => { S.sel = c.dataset.k; renderModelStrip(); renderRank(); renderSelected(); };
  });
}

// ==================================================================== 排名表
function renderRank() {
  const r = S.result;
  const rows = r.rank;
  const maxR2 = Math.max(...rows.map(x => x['样本外R²'] ?? 0), 0.0001);
  const head = ['模型', '流派', '可解释', '样本外 R²', '年化 TE', '窗口内 R²',
    '训练/样本外差', '方向一致', '隐含仓位', '耗时(s)'];
  const html = `<thead><tr>${head.map((h, i) =>
    `<th class="${i < 3 ? 'l' : ''}">${h}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(x => {
      const k = x.key, m = r.models[k] || {};
      const r2 = x['样本外R²'], gap = (x['窗口内R²'] ?? 0) - (x['样本外R²'] ?? 0);
      const bt = x['隐含仓位'];
      const w = Math.max(0, (r2 ?? 0)) / maxR2 * 100;
      const eq = S.cat?.rbsa?.equity_ratio ?? 0.9;
      const bLow = bt !== null && bt !== undefined && Math.abs(bt) < eq * 0.7;
      return `<tr class="clickable ${k === S.sel ? 'sel' : ''}" data-k="${k}">
        <td class="l">${x['模型']} ${k === S.best ? '<span class="tag">最佳</span>' : ''}</td>
        <td class="l">${x['流派']}</td>
        <td class="l">${x['可解释']}</td>
        <td class="bar-cell">${r2 === null ? '—' :
          `<div class="bg" style="right:14px;width:${w}%"></div><span>${fmtNum(r2, 4)}</span>`}</td>
        <td>${fmtPct(x['样本外TE'], 1)}</td>
        <td>${fmtNum(x['窗口内R²'], 4)}</td>
        <td style="color:${gap > 0.08 ? 'var(--warn)' : 'var(--fg3)'}">${fmtSign(gap, 3)}</td>
        <td>${fmtPct(x['方向一致率'], 1)}</td>
        <td style="color:${bLow ? 'var(--warn)' : 'var(--fg2)'}">${fmtPct(bt, 0)}</td>
        <td>${fmtNum(x['耗时'], 1)}</td>
      </tr>`;
    }).join('') + '</tbody>';
  $('rankTable').innerHTML = html;
  [...$('rankTable').querySelectorAll('tr.clickable')].forEach(tr => {
    tr.onclick = () => { S.sel = tr.dataset.k; renderModelStrip(); renderRank(); renderSelected(); };
  });

  // 排序解读
  const best = rows[0];
  const worst = rows[rows.length - 1];
  const gap = (best['样本外R²'] ?? 0) - (worst['样本外R²'] ?? 0);
  $('rankNote').innerHTML =
    `排名按<b>样本外 R²</b> 降序——即"用上一窗口拟合好的模型去预测下一段"的表现。
     最佳：<b style="color:var(--fg)">${best['模型']}（${fmtNum(best['样本外R²'], 3)}）</b>，
     最弱：${worst['模型']}（${fmtNum(worst['样本外R²'], 3)}），跨度 ${fmtNum(gap, 3)}。
     注意"窗口内 R²"高只代表训练集拟合得好，
     <b>任何模型若训练/样本外差 &gt; 0.08，都说明它只是在背历史，不能用来外推</b>。`;
}

// ==================================================================== 选中模型详情
function renderSelected() {
  const k = S.sel; if (!k) return;
  const m = S.result.models[k]; if (!m) return;
  $('selName').textContent = m.name;
  $('selFamily').textContent = m.family;
  $('selInterp').textContent = m.interpretable ? '权重可解释' : '权重=重要性(不可当持仓)';
  $('selInterp').style.color = m.interpretable ? 'var(--ok)' : 'var(--warn)';
  $('selTime').textContent = `用时 ${fmtNum(m.metrics.elapsed, 1)}s`;
  $('selDesc').textContent = m.desc;

  const gap = (m.metrics.fit_r2_mean ?? 0) - (m.metrics.oos_r2 ?? 0);
  const beta = m.metrics.beta, vr = m.metrics.vol_ratio;
  const eq = S.cat?.rbsa?.equity_ratio ?? 0.9;
  const betaLow = beta !== null && beta !== undefined &&
    Math.abs(beta) < eq * 0.7;
  const kpi = [
    { lb: '样本外 R²', vl: fmtNum(m.metrics.oos_r2, 4), sb: '排序依据', cls: 'good' },
    { lb: '年化跟踪误差', vl: fmtPct(m.metrics.oos_te, 1), sb: '越低越贴合' },
    { lb: '窗口内 R²', vl: fmtNum(m.metrics.fit_r2_mean, 4), sb: '训练集拟合度' },
    { lb: '训练−样本外', vl: fmtSign(gap, 3), sb: gap > 0.08 ? '过拟合明显' : '差距可控',
      cls: gap > 0.08 ? 'warn' : '' },
    { lb: '隐含股票仓位', vl: fmtPct(beta, 0), sb: `基准波动/基金 ${fmtNum(vr, 1)}×`,
      cls: betaLow ? 'warn' : '' },
    { lb: '方向一致率', vl: fmtPct(m.metrics.win, 1), sb: '涨跌方向命中' },
    { lb: '有效权重数', vl: m.metrics.n_weights, sb: `更新于 ${m.weights_last_date || '—'}` },
  ];
  $('selKpis').innerHTML = kpi.map(x => `
    <div class="kpi ${x.cls || ''}"><div class="lb">${x.lb}</div>
    <div class="vl">${x.vl}</div><div class="sb">${x.sb}</div></div>`).join('');

  // 仓位失配时给一句提示，避免用户误以为模型失效
  const noteEl = $('selNote');
  if (betaLow) {
    noteEl.innerHTML = `<div class="lead warn" style="margin:0 0 14px">
      <b>提示：</b>该基金隐含股票仓位仅 ${fmtPct(beta, 0)}，而合成基准按
      ${fmtPct(eq, 0)} 满仓构建（基准波动是基金的 ${fmtNum(vr, 1)} 倍）。
      这会系统性压低 R²——<b>低 R² 反映的是仓位差异，不是风格识别失败</b>。
      参见上方「仓位假设检验」。</div>`;
    noteEl.hidden = false;
  } else { noteEl.hidden = true; }

  drawCurve(m); drawResid(m); drawWHist(m); drawWeights(m);
  renderTuner();
}

function chart(id, opt) {
  let c = S.charts[id];
  if (c && !c.isDisposed()) { c.setOption(opt, true); return c; }
  const el = $(id); if (!el) return null;
  c = echarts.init(el, null, { renderer: 'canvas' });
  c.setOption(opt);
  S.charts[id] = c;
  return c;
}
const AX = {
  axisLine: { lineStyle: { color: '#2f3b4d' } },
  axisLabel: { color: '#6f7f96', fontSize: 10.5 },
  splitLine: { lineStyle: { color: 'rgba(35,45,59,.55)' } },
};
const PAL = ['#4a9eff', '#7c5cff', '#12b886', '#f0a020', '#ef4a4a',
  '#00c2d1', '#e857a6', '#8bc34a'];

function drawCurve(m) {
  const d = m.curve;
  chart('chartCurve', {
    grid: { left: 52, right: 18, top: 18, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 } },
    legend: { data: ['基金净值', '合成基准'], textStyle: { color: '#a9b6c8', fontSize: 11 },
      top: 0, itemWidth: 16, itemHeight: 8 },
    xAxis: { type: 'category', data: d.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...AX, axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: [
      { name: '基金净值', type: 'line', data: d.fund, showSymbol: false,
        lineStyle: { width: 1.9, color: PAL[0] } },
      { name: '合成基准', type: 'line', data: d.synth, showSymbol: false,
        lineStyle: { width: 1.4, color: PAL[1], type: 'dashed' } },
    ]
  });
}

function drawResid(m) {
  const d = m.residual;
  const vals = (d.value || []).filter(v => v !== null);
  const mx = vals.length ? Math.max(...vals.map(Math.abs)) : 1;
  chart('chartResid', {
    grid: { left: 52, right: 18, top: 18, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : Number(v).toFixed(4)) },
    xAxis: { type: 'category', data: d.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(124,92,255,.12)',
      handleStyle: { color: '#7c5cff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: [{
      type: 'bar', data: (d.value || []).map(v =>
        v === null ? null : { value: v, itemStyle: { color: colOf(v) } }),
      barMaxWidth: 2.2,
    }]
  });
}

function drawWHist(m) {
  const h = m.weights_hist;
  if (!h || !h.assets || !h.assets.length) {
    chart('chartWHist', { title: { text: '无权重历史', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  chart('chartWHist', {
    grid: { left: 52, right: 18, top: 34, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : (v * 100).toFixed(1) + '%') },
    legend: { data: h.assets, textStyle: { color: '#a9b6c8', fontSize: 10.5 },
      top: 0, itemWidth: 13, itemHeight: 7, type: 'scroll' },
    xAxis: { type: 'category', data: h.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: h.assets.map((a, i) => ({
      name: a, type: 'line', stack: 'w', areaStyle: { opacity: 0.68 },
      showSymbol: false, lineStyle: { width: 0.6 },
      itemStyle: { color: PAL[i % PAL.length] },
      data: h.series[i]
    }))
  });
}

function drawWeights(m) {
  const items = (m.latest_weights && m.latest_weights.items) || [];
  const arr = items.slice().reverse();
  if (!arr.length) {
    chart('chartWeights', { title: { text: '无权重数据', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  chart('chartWeights', {
    grid: { left: 92, right: 62, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: '#1b2431', borderColor: '#2f3b4d', textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : (v * 100).toFixed(2) + '%') },
    xAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    yAxis: { type: 'category', data: arr.map(x => x.name), ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 11 } },
    series: [{
      type: 'bar', barMaxWidth: 15,
      data: arr.map((x, i) => ({ value: x.weight,
        itemStyle: { color: PAL[(arr.length - 1 - i) % PAL.length] } })),
      label: { show: true, position: 'right', color: '#a9b6c8', fontSize: 10.5,
        formatter: p => (p.value * 100).toFixed(1) + '%' }
    }]
  });
}

// ==================================================================== 超参表单
function renderTuner() {
  const k = S.sel; if (!k) return;
  const spec = S.cat.models.find(m => m.key === k);
  if (!spec) return;
  const cur = Object.assign({}, spec.params, S.params[k] || {});
  const HINTS = {
    alpha_scale: '正则强度缩放，越大越稀疏',
    l1_ratio: 'L1 占比，越大越稀疏',
    max_assets: '最多保留几个指数',
    min_weight: '低于此权重视为噪声',
    top_k: '按相关性取前 K 个',
    n_components: '主成分个数',
    learning_rate: ' boosting 学习率',
    max_iter: '提升轮数 / 迭代次数',
    max_depth: '树最大深度',
    n_estimators: '树棵数',
    hidden: '隐藏层，如 32,16',
  };
  $('tunerBody').innerHTML = Object.entries(spec.params).map(([pk, dflt]) => {
    const v = cur[pk];
    let inp;
    if (pk === 'hidden') {
      inp = `<input type="text" data-pk="${pk}" value="${(v || []).join(',')}">`;
    } else if (typeof dflt === 'boolean') {
      inp = `<input type="checkbox" data-pk="${pk}" ${v ? 'checked' : ''}>`;
    } else if (typeof dflt === 'number') {
      const step = dflt < 0.01 ? 0.01 : dflt < 1 ? 0.01 : 1;
      inp = `<input type="number" data-pk="${pk}" value="${v}" step="${step}">`;
    } else {
      inp = `<input type="text" data-pk="${pk}" value="${v}">`;
    }
    return `<div class="pfield"><label>${pk}</label>${inp}
      <span class="hint">默认 ${JSON.stringify(dflt)} · ${HINTS[pk] || ''}</span></div>`;
  }).join('');
}

async function applyParams() {
  const k = S.sel; if (!k) return;
  const spec = S.cat.models.find(m => m.key === k);
  const o = {};
  $('tunerBody').querySelectorAll('[data-pk]').forEach(el => {
    const pk = el.dataset.pk;
    if (el.type === 'checkbox') { o[pk] = el.checked; return; }
    const dflt = spec.params[pk];
    let v = el.value;
    if (pk === 'hidden') v = v.split(/[,，]/).map(x => +x).filter(x => !Number.isNaN(x));
    else if (typeof dflt === 'number') v = Number(v);
    o[pk] = v;
  });
  S.params[k] = o;
  $('progressWrap').hidden = false;
  setProgress('重新学习 ' + spec.name, 0.02, '');
  try {
    const r = await api('/api/refit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: S.result.code, model: k, params: o,
                              max_rebal: selectedRebal(),
                              auto_equity: selectedAuto() })
    });
    await pollRefit(r.job_id, k);
  } catch (e) {
    $('progressWrap').hidden = true;
    showAlert('重新学习失败：' + e.message);
  }
}

async function pollRefit(jid, k) {
  for (;;) {
    await new Promise(r => setTimeout(r, 800));
    const j = await api('/api/job/' + jid);
    setProgress(j.stage, j.progress, j.detail);
    if (j.status === 'done') {
      const r = await api('/api/result/' + jid);
      $('progressWrap').hidden = true;
      S.result.models[k] = r.result;          // 就地替换
      // 重算排名。键名必须加引号：'样本外R²' 含上标字符，不是合法标识符。
      S.result.rank = Object.values(S.result.models).map(m => ({
        key: m.key, '模型': m.name, '流派': m.family,
        '可解释': m.interpretable ? '是' : '否',
        '样本外R²': m.metrics.oos_r2, '样本外TE': m.metrics.oos_te,
        '窗口内R²': m.metrics.fit_r2_mean, '窗口内TE': m.metrics.fit_te_mean,
        '方向一致率': m.metrics.win, '耗时': m.metrics.elapsed,
      })).sort((a, b) => (b['样本外R²'] ?? -9) - (a['样本外R²'] ?? -9));
      const stillBest = Object.keys(S.result.models).reduce((a, x) =>
        (S.result.models[x].metrics.oos_r2 ?? -9) > (S.result.models[a].metrics.oos_r2 ?? -9) ? x : a,
        Object.keys(S.result.models)[0]);
      S.best = stillBest;
      renderModelStrip(); renderRank(); renderSelected(); renderHead();
      showAlert(`「${r.result.name}」已用新参数重新学习，样本外 R² = ${
        fmtNum(r.result.metrics.oos_r2, 4)}。当前最佳仍是「${S.result.models[stillBest].name}」。`);
      return;
    }
    if (j.status === 'error') {
      $('progressWrap').hidden = true;
      showAlert('重新学习失败：' + j.error); return;
    }
  }
}

// ==================================================================== 净值
function renderNav() {
  const n = S.result.nav;
  chart('chartNav', {
    grid: { left: 56, right: 18, top: 14, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : Number(v).toFixed(4)) },
    xAxis: { type: 'category', data: n.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...AX, axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: [{
      type: 'line', data: n.nav, showSymbol: false,
      lineStyle: { width: 1.9, color: PAL[0] },
      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [{ offset: 0, color: 'rgba(74,158,255,.28)' },
                     { offset: 1, color: 'rgba(74,158,255,0)' }] } },
      markLine: { silent: true, symbol: 'none',
        lineStyle: { color: '#6f7f96', type: 'dashed', width: 1 },
        data: [{ yAxis: n.nav[0] }] }
    }]
  });

  chart('chartDD', {
    grid: { left: 56, right: 18, top: 14, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 }, valueFormatter: v => v == null ? '—' : (v * 100).toFixed(2) + '%' },
    xAxis: { type: 'category', data: n.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    series: [{ type: 'line', data: n.drawdown, showSymbol: false,
      lineStyle: { width: 1.2, color: DOWN },
      areaStyle: { color: 'rgba(18,184,134,.22)' } }]
  });

  chart('chartVol', {
    grid: { left: 56, right: 18, top: 14, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 }, valueFormatter: v => v == null ? '—' : (v * 100).toFixed(1) + '%' },
    xAxis: { type: 'category', data: n.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    series: [{ type: 'line', data: n.vol60, showSymbol: false,
      lineStyle: { width: 1.3, color: PAL[3] } }]
  });

  drawMonthly(n.monthly, n.years);
}

function drawMonthly(monthly, years) {
  if (!monthly || !Object.keys(monthly).length) {
    chart('chartMonthly', { title: { text: '数据不足', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  const yearsR = years.slice().reverse();
  const data = [];
  yearsR.forEach((y, yi) => {
    for (let mth = 1; mth <= 12; mth++) {
      const v = (monthly[y] || {})[String(mth)];
      data.push([mth - 1, yi,
        v === undefined || v === null ? '-' : +(v * 100).toFixed(2)]);
    }
  });
  chart('chartMonthly', {
    grid: { left: 52, right: 18, top: 12, bottom: 46 },
    tooltip: { position: 'top', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => p.value[0] === '-' ? '' : `${yearsR[p.value[1]]}-${String(p.value[0] + 1).padStart(2, '0')}<br/><b style="color:${colOf(p.value[2])}">${p.value[2]}%</b>` },
    xAxis: { type: 'category', data: Array.from({ length: 12 }, (_, i) => i + 1 + '月'),
      ...AX, splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,.015)', 'transparent'] } } },
    yAxis: { type: 'category', data: yearsR, ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 10.5 } },
    visualMap: { min: -25, max: 25, calculable: false, orient: 'horizontal',
      left: 'center', bottom: 4, itemHeight: 90, itemWidth: 12,
      textStyle: { color: '#6f7f96', fontSize: 10 },
      inRange: { color: ['#12b886', '#0f8f68', '#1b2431', '#c0392b', '#ef4a4a'] } },
    series: [{ type: 'heatmap', data,
      label: { show: true, fontSize: 9.5, color: '#dbe4f0',
        formatter: p => p.value[0] === '-' ? '' : p.value[2].toFixed(1) },
      itemStyle: { borderColor: '#0d1117', borderWidth: 1.5 } }]
  });
}

// ==================================================================== 相关性
function renderCorr() {
  const c = S.result.correlation;

  const top = c.top.slice().reverse();
  chart('chartCorrBar', {
    grid: { left: 96, right: 58, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: '#1b2431', borderColor: '#2f3b4d', textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => {
        const r = top[p[0].dataIndex];
        return `<b>${r.name}</b><br/>${r.layer}<br/>相关系数 ${fmtSign(r.corr, 4)}`;
      } },
    xAxis: { type: 'value', min: -1, max: 1, ...AX },
    yAxis: { type: 'category', data: top.map(x => x.name), ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 10.5 } },
    series: [{
      type: 'bar', barMaxWidth: 14,
      data: top.map(x => ({ value: x.corr, itemStyle: { color: colOf(x.corr) } })),
      label: { show: true, position: 'right', fontSize: 10.5, color: '#a9b6c8',
        formatter: p => fmtSign(p.value, 2) }
    }]
  });

  const L = c.layers.slice();
  chart('chartLayer', {
    grid: { left: 96, right: 58, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => { const r = L[p[0].dataIndex];
        return `<b>${r.layer}</b><br/>包含 ${r.n} 个指数<br/>平均相关 ${fmtSign(r.avg_corr, 3)}`; } },
    xAxis: { type: 'value', min: -1, max: 1, ...AX },
    yAxis: { type: 'category', data: L.map(x => x.layer), ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 10.5 } },
    series: [{
      type: 'bar', barMaxWidth: 20,
      data: L.map(x => ({ value: x.avg_corr, itemStyle: { color: colOf(x.avg_corr) } })),
      label: { show: true, position: 'right', fontSize: 11, color: '#a9b6c8',
        formatter: p => `${fmtSign(p.value, 3)} (${L[p.dataIndex].n})` }
    }]
  });

  drawMatrix(c.matrix);
  drawRollCorr(c.rolling);

  $('driftNote').innerHTML = (c.drift || []).map(d => {
    const big = Math.abs(d.delta || 0) > 0.10;
    return `<div class="card ${big ? 'warn' : 'info'}" style="margin-bottom:10px">
      <div class="hd">${d.name} · 相关性 ${fmtSign(d.delta, 3)}</div>
      <div class="bd">前段 ${fmtSign(d.early, 3)} → 后段 ${fmtSign(d.late, 3)}。
      ${big ? '<b>变化超过 0.10，风格发生实质漂移。</b>' : '变化在 0.10 以内，暴露相对稳定。'}
      </div></div>`;
  }).join('');
}

function drawMatrix(mx) {
  const data = [];
  mx.rows.forEach((r, i) => mx.cols.forEach((c, j) => {
    const v = mx.values[i][j];
    if (v !== null) data.push([j, i, v]);
  }));
  chart('chartMatrix', {
    grid: { left: 118, right: 18, top: 8, bottom: 92 },
    tooltip: { position: 'top', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => `${mx.rows[p.value[1]]} × ${mx.cols[p.value[0]]}<br/><b>${fmtSign(p.value[2], 3)}</b>` },
    xAxis: { type: 'category', data: mx.cols, ...AX,
      axisLabel: { ...AX.axisLabel, rotate: 52, fontSize: 9.5, interval: 0 } },
    yAxis: { type: 'category', data: mx.rows, ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 9.5, interval: 0 } },
    visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal',
      left: 'center', bottom: 4, itemWidth: 11, itemHeight: 110,
      textStyle: { color: '#6f7f96', fontSize: 10 },
      inRange: { color: ['#12b886', '#1b2431', '#ef4a4a'] } },
    series: [{ type: 'heatmap', data,
      label: { show: true, fontSize: 8.5, color: '#c8d3e2',
        formatter: p => p.value[2].toFixed(2) },
      itemStyle: { borderColor: '#0d1117', borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: '#e6edf6', borderWidth: 1.5 } } }]
  });
}

function drawRollCorr(roll) {
  const keys = Object.keys(roll || {});
  if (!keys.length) {
    chart('chartRollCorr', { title: { text: '数据不足', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  chart('chartRollCorr', {
    grid: { left: 56, right: 18, top: 34, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 } },
    legend: { data: keys, textStyle: { color: '#a9b6c8', fontSize: 10.5 },
      top: 0, itemWidth: 14, itemHeight: 7, type: 'scroll' },
    xAxis: { type: 'category', data: roll[keys[0]].dates, ...AX,
      axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', min: 0, max: 1, ...AX },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: keys.map((k, i) => ({
      name: k, type: 'line', showSymbol: false,
      lineStyle: { width: 1.4 },
      itemStyle: { color: PAL[i % PAL.length] },
      data: roll[k].values
    }))
  });
}

// ==================================================================== 买卖决策回测
function initBacktestPicker() {
  S.cat.models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.key; o.textContent = m.name;
    $('btModel').appendChild(o);
  });
}

function syncBacktestDefaults() {
  const st = S.result.nav.stats;
  // 仅在用户尚未选择时预填，避免分析刷新后覆盖已选区间
  if (!$('btStart').value && st.start) $('btStart').value = st.start;
  if (!$('btEnd').value && st.end) $('btEnd').value = st.end;
  if (S.sel && S.cat.models.some(m => m.key === S.sel)) $('btModel').value = S.sel;
  $('btAuto').checked = $('autoEquity').checked;
}

async function runBacktest() {
  if (!S.result) { showAlert('请先分析一只基金'); return; }
  const start = $('btStart').value, end = $('btEnd').value;
  if (start && end && start >= end) { showAlert('起始日需早于结束日'); return; }

  $('btRun').disabled = true;
  $('progressWrap').hidden = false;
  setProgress('提交回测任务…', 0.02, '');
  try {
    const r = await api('/api/backtest', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: S.result.code,
        model: $('btModel').value,
        auto_equity: $('btAuto').checked,
        start: start || null,
        end: end || null,
      })
    });
    await pollBacktest(r.job_id);
  } catch (e) {
    $('progressWrap').hidden = true;
    showAlert('回测失败：' + e.message);
  } finally {
    $('btRun').disabled = false;
  }
}

async function pollBacktest(jid) {
  for (;;) {
    await new Promise(r => setTimeout(r, 800));
    const j = await api('/api/job/' + jid);
    setProgress(j.stage, j.progress, j.detail);
    if (j.status === 'done') {
      const r = await api('/api/result/' + jid);
      $('progressWrap').hidden = true;
      renderBacktest(r.result);
      return;
    }
    if (j.status === 'error') {
      $('progressWrap').hidden = true;
      showAlert('回测失败：' + j.error);
      return;
    }
  }
}

function renderBacktest(p) {
  $('btOut').hidden = false;
  drawBtEquity(p);
  drawBtHealth(p);
  renderBtKpis(p);
  renderBtMetrics(p);
  renderBtTrades(p);
  $('btOut').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function btLineStyle(k) {
  if (k.includes('健康')) return { width: 2.2, color: PAL[0] };
  if (k.includes('趋势')) return { width: 1.4, color: PAL[3] };
  if (k.includes('恒定')) return { width: 1.2, color: FLAT, type: 'dashed' };
  return { width: 1.3, color: PAL[1], type: 'dashed' };   // 买入持有
}

function drawBtEquity(p) {
  const keys = Object.keys(p.equity.series);
  chart('btChartEquity', {
    grid: { left: 52, right: 18, top: 32, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : Number(v).toFixed(4)) },
    legend: { data: keys, textStyle: { color: '#a9b6c8', fontSize: 10.5 },
      top: 0, itemWidth: 15, itemHeight: 8, type: 'scroll' },
    xAxis: { type: 'category', data: p.equity.dates, ...AX,
      axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...AX,
      axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: keys.map(k => ({
      name: k, type: 'line', data: p.equity.series[k], showSymbol: false,
      lineStyle: btLineStyle(k),
    }))
  });
}

function drawBtHealth(p) {
  const h = p.health, th = p.meta.thresholds;
  chart('btChartHealth', {
    grid: { left: 52, right: 46, top: 32, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 } },
    legend: { data: ['健康分', '持仓'], textStyle: { color: '#a9b6c8', fontSize: 11 },
      top: 0, itemWidth: 15, itemHeight: 8 },
    xAxis: { type: 'category', data: h.dates, ...AX,
      axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: [
      { type: 'value', min: -1, max: 1, ...AX,
        name: '健康分', nameTextStyle: { color: '#6f7f96', fontSize: 10 } },
      { type: 'value', min: 0, max: 1, splitLine: { show: false },
        axisLine: { lineStyle: { color: '#2f3b4d' } },
        axisLabel: { color: '#6f7f96', fontSize: 10 } },
    ],
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: '健康分', type: 'line', data: h.score, showSymbol: false,
        lineStyle: { width: 1.5, color: PAL[0] },
        markLine: {
          symbol: 'none', silent: true,
          data: [
            { yAxis: th.exit, lineStyle: { color: DOWN, type: 'dashed', width: 1 },
              label: { color: DOWN, fontSize: 9.5, position: 'insideStartTop',
                formatter: '卖出线 ' + th.exit } },
            { yAxis: th.enter, lineStyle: { color: UP, type: 'dashed', width: 1 },
              label: { color: UP, fontSize: 9.5, position: 'insideStartBottom',
                formatter: '买入线 ' + th.enter } },
          ],
        } },
      { name: '持仓', type: 'line', step: 'end', data: h.position, yAxisIndex: 1,
        showSymbol: false, lineStyle: { width: 1.4, color: PAL[3] },
        areaStyle: { color: 'rgba(240,160,32,.12)' } },
    ]
  });
}

function renderBtKpis(p) {
  const main = p.metrics.find(x => x['方案'] && x['方案'].includes('健康')) || p.metrics[0];
  const tc = p.summary.timing_contrib || {};
  const cards = [
    { lb: '年化收益', vl: fmtPctSign(main['年化收益'], 1),
      sb: `择时贡献 ${fmtPctSign(tc['年化收益'])}`,
      cls: main['年化收益'] > 0 ? 'good' : 'bad' },
    { lb: '夏普比率', vl: fmtNum(main['夏普'], 2),
      sb: `择时贡献 ${fmtSign(tc['夏普'], 2)}` },
    { lb: '最大回撤', vl: fmtPct(main['最大回撤'], 1),
      sb: `择时改善 ${fmtPctSign(tc['最大回撤'])}`, cls: 'bad' },
    { lb: 'Calmar', vl: fmtNum(main['Calmar'], 2),
      sb: `择时贡献 ${fmtSign(tc['Calmar'], 2)}` },
    { lb: '平均仓位', vl: fmtPct(p.summary.avg_position, 0),
      sb: `对照 ${p.summary.control_name}` },
    { lb: '年化换手', vl: fmtNum(p.summary.turnover, 2), sb: '单边换手倍数' },
    { lb: 'RankIC', vl: fmtNum(p.summary.ic, 3),
      sb: `ICIR ${fmtNum(p.summary.icir, 2)}` },
  ];
  $('btKpis').innerHTML = cards.map(x => `
    <div class="kpi ${x.cls || ''}"><div class="lb">${x.lb}</div>
    <div class="vl">${x.vl}</div><div class="sb">${x.sb}</div></div>`).join('');
}

function renderBtMetrics(p) {
  const cols = ['年化收益', '年化波动', '夏普', '最大回撤', 'Calmar', '月胜率', '累计收益'];
  const html = `<thead><tr><th class="l">方案</th>${cols.map(c => `<th>${c}</th>`).join('')}
    </tr></thead><tbody>` +
    p.metrics.map(x => {
      const main = x['方案'] && x['方案'].includes('健康');
      return `<tr ${main ? 'class="sel"' : ''}>
        <td class="l">${x['方案']}</td>
        <td style="color:${colOf(x['年化收益'])}">${fmtPctSign(x['年化收益'], 1)}</td>
        <td>${fmtPct(x['年化波动'], 1)}</td>
        <td>${fmtNum(x['夏普'], 2)}</td>
        <td style="color:${DOWN}">${fmtPct(x['最大回撤'], 1)}</td>
        <td>${fmtNum(x['Calmar'], 2)}</td>
        <td>${fmtPct(x['月胜率'], 0)}</td>
        <td style="color:${colOf(x['累计收益'])}">${fmtPctSign(x['累计收益'], 1)}</td>
      </tr>`;
    }).join('') + '</tbody>';
  $('btMetrics').innerHTML = html;
}

function renderBtTrades(p) {
  let body;
  if (!p.trades.length) {
    // 零交易必须区分「信号判断为一直持有」和「阈值根本够不到」——
    // 后者意味着该方案 ≡ 买入持有，它与对照的差额只是仓位差。
    const hs = p.meta.health_stats || {};
    let why;
    if (hs.unreliable_ratio > 0.5) {
      why = `拟合不可信天数占 <b>${fmtPct(hs.unreliable_ratio, 0)}</b>`
        + `（R² 中位 ${fmtNum(hs.r2_median, 2)} &lt; 0.5，合成基准解释不了这只基金），`
        + '<b>门控已停用</b>';
    } else if (hs.gap_to_exit != null) {
      why = `健康分历史最低 <b>${fmtNum(hs.min, 2)}</b>，离场线 ${fmtNum(hs.exit, 2)}，`
        + `还差 <b>${fmtNum(hs.gap_to_exit, 2)}</b>，未触及阈值`;
    } else {
      why = '健康分未触及离场阈值';
    }
    body = `<tr><td colspan="5" style="color:var(--fg3);text-align:left">
      该区间内<b>零交易</b>：${why}，全程持有。<br>
      <span style="color:var(--fg2)">因此本方案 ≡ 买入持有，
      绩效差异只来自仓位水平，<b>不构成择时能力的证据</b>。</span></td></tr>`;
  } else {
    body = p.trades.map(t => {
      const buy = t.action === '买入';
      return `<tr>
        <td class="l">${t.date}</td>
        <td class="l">${t.exec_date}</td>
        <td style="color:${buy ? UP : DOWN};font-weight:600">${buy ? '买入 ▲' : '卖出 ▼'}</td>
        <td>${fmtNum(t.health, 2)}</td>
        <td class="l" style="color:var(--fg2)">${t.reason || '—'}</td>
      </tr>`;
    }).join('');
  }
  $('btTrades').innerHTML =
    `<thead><tr><th class="l">决策日</th><th class="l">执行日(次日)</th>
      <th>动作</th><th>健康分</th><th class="l">触发原因</th></tr></thead>
     <tbody>${body}</tbody>`;
}

// ==================================================================== 自适应
let rzT = null;
window.addEventListener('resize', () => {
  clearTimeout(rzT);
  rzT = setTimeout(() => {
    Object.values(S.charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
  }, 140);
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) Object.values(S.charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
});