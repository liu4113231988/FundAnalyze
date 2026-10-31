/* 股票与指数工作区：入口切换、请求状态和市场图表。 */
'use strict';
window.MarketApp = (() => {
  const el = id => document.getElementById(id);
  const views = { stock: {}, index: {}, fund: {} };
  const charts = new Map();
  let kind = 'stock', catalog, fund, requestData, alert, generation = 0, result = null, chartType = 'candlestick';
  const pct = value => value == null ? '—' : (value * 100).toFixed(2) + '%';
  const num = value => value == null ? '—' : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  const date = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  function status(message, error = false) {
    el('marketStatus').hidden = !message;
    el('marketStatus').textContent = message || '';
    el('marketStatus').classList.toggle('is-error', error);
  }
  function switchKind(next) {
    if (!views[next] || (catalog && next === kind && el('marketStart').value)) return;
    views[kind] = { code: el('codeInput').value, result,
      start: el('marketStart').value, end: el('marketEnd').value,
      adjust: el('marketAdjust').value, benchmark: el('marketBenchmark').value };
    generation++;
    if (fund.invalidate) fund.invalidate();
    kind = next;
    result = views[kind].result || null;
    charts.forEach(c => c.dispose()); charts.clear();
    document.querySelectorAll('[data-asset]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.asset === kind)));
    el('marketView').hidden = kind === 'fund';
    el('fundView').hidden = kind !== 'fund';
    el('main').hidden = kind !== 'fund' || !fund.hasResult();
    el('empty').hidden = kind !== 'fund' || fund.hasResult();
    el('progressWrap').hidden = true;
    el('runBtn').disabled = false;
    el('marketRecalculate').disabled = el('marketRefresh').disabled = false;
    el('fundApply').disabled = false;
    el('runBtn').removeAttribute('aria-busy');
    alert(null); status(null);
    el('codeInput').value = views[kind].code || '';
    el('codeInput').placeholder = kind === 'fund' ? '基金代码，如 110022' : kind === 'stock' ? '股票代码，如 600519' : '指数代码，如 sh000300';
    const items = kind === 'fund' ? catalog.known_funds : kind === 'stock' ? catalog.assets.stocks : catalog.assets.indices;
    el('presetSel').replaceChildren(new Option('— 示例 —', ''), ...items.map(x => new Option(`${x.code} ${x.name}`, x.code)));
    el('presetSel').setAttribute('aria-label', kind === 'fund' ? '示例基金' : kind === 'stock' ? '示例股票' : '示例指数');
    el('presetSel').title = kind === 'fund' ? '示例基金' : kind === 'stock' ? '示例股票' : '示例指数';
    if (kind === 'fund') {
      if (fund.hasResult()) fund.show();
      return;
    }
    el('marketStart').value = views[kind].start || date(new Date(new Date().setFullYear(new Date().getFullYear() - 1)));
    el('marketEnd').value = views[kind].end || date(new Date());
    el('marketAdjust').value = views[kind].adjust ?? 'qfq';
    el('marketBenchmark').value = views[kind].benchmark || 'sh000300';
    const stock = kind === 'stock';
    el('adjustField').hidden = !stock;
    el('marketTitle').textContent = stock ? '股票分析' : '指数观察';
    el('marketEyebrow').textContent = stock ? 'EQUITY RESEARCH' : 'INDEX RESEARCH';
    el('marketSubtitle').textContent = stock ? '从价格、成交与风险，观察一只股票。' : '从市场基准出发，观察长期走势与风险。';
    el('marketEmptyTitle').textContent = stock ? '选择一只股票，展开分析' : '选择一个指数，观察市场';
    el('marketExamples').replaceChildren(...items.slice(0, 4).map(item => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'market-example';
      const name = document.createElement('strong'); name.textContent = item.name;
      const code = document.createElement('span'); code.textContent = item.code;
      button.append(name, code); button.onclick = () => { el('codeInput').value = item.code; submit(item.code); };
      return button;
    }));
    el('marketEmpty').hidden = !!result; el('marketResult').hidden = !result;
    if (result) render(result);
  }
  async function submit(code, refresh = false) {
    code = code.trim();
    if (!/^(?:(?:sh|sz|bj))?\d{6}$/i.test(code)) { status('请输入正确的证券代码，例如 600519 或 sh000300。', true); return; }
    const form = el('marketFilters');
    if (!form.reportValidity()) return;
    if (el('marketStart').value >= el('marketEnd').value) { status('开始日期需早于结束日期。', true); return; }
    if (Date.parse(el('marketEnd').value) - Date.parse(el('marketStart').value) < 30 * 86400000) { status('分析区间最短需 30 个自然日，请扩大区间。', true); return; }
    const token = ++generation;
    const body = {kind, code, start: el('marketStart').value, end: el('marketEnd').value,
      adjust: el('marketAdjust').value, benchmark: el('marketBenchmark').value, refresh};
    el('runBtn').disabled = true; el('runBtn').setAttribute('aria-busy', 'true');
    el('marketRecalculate').disabled = el('marketRefresh').disabled = true;
    status(`正在读取 ${code} 的日线行情…`);
    try {
      const request = await requestData('/api/market/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      while (token === generation) {
        if (token !== generation) return;
        await new Promise(resolve => setTimeout(resolve, 700));
        if (token !== generation) return;
        const job = await requestData('/api/job/' + request.job_id);
        if (token !== generation) return;
        if (job.status === 'error') throw new Error(job.error);
        status(`${code} · ${job.stage} · ${Math.round((job.progress || 0) * 100)}%`);
        if (job.status === 'done') {
          const response = await requestData('/api/result/' + request.job_id);
          if (token !== generation) return;
          result = response.result; render(result);
          const changed = body.start !== el('marketStart').value || body.end !== el('marketEnd').value
            || body.adjust !== el('marketAdjust').value || body.benchmark !== el('marketBenchmark').value;
          status(changed ? '设置已更改，当前图表对应提交时的设置；点击「应用区间」更新结果。' : null); return;
        }
      }
    } catch (error) {
      if (token === generation) status('分析失败：' + error.message, true);
    } finally {
      if (token === generation) { el('runBtn').disabled = false; el('runBtn').removeAttribute('aria-busy'); el('marketRecalculate').disabled = el('marketRefresh').disabled = false; }
    }
  }
  function plot(id, dates, series, percent = false, extra = {}) {
    if (!charts.has(id)) charts.set(id, echarts.init(el(id)));
    charts.get(id).setOption({ animation: false, color: ['#71afff', '#d5b76d', '#a1b0c6', '#55bfa4'],
      backgroundColor: 'transparent', tooltip: {trigger: 'axis'}, legend: {textStyle: {color: '#a9b6c8'}, top: 0},
      grid: {left: 65, right: 22, top: 38, bottom: 60},
      xAxis: {type: 'category', data: dates, boundaryGap: true, axisLabel: {color: '#8796ab'}, axisLine: {lineStyle: {color: '#2f3b4d'}}},
      yAxis: {type: 'value', scale: true, axisLabel: {color: '#8796ab', formatter: percent ? v => (v * 100).toFixed(0) + '%' : undefined}, splitLine: {lineStyle: {color: '#232d3b'}}},
      dataZoom: [{type: 'inside'}, {type: 'slider', height: 16, bottom: 6, borderColor: '#2f3b4d'}], series, ...extra }, true);
  }
  const line = (name, data, extra = {}) => ({name, type: 'line', data, symbol: 'none', connectNulls: false, ...extra});
  function price() {
    const s = result.series;
    document.querySelectorAll('[data-chart]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.chart === chartType)));
    const technical = result.technical;
    const markers = technical ? technical.signals.map(signal => ({name: signal.type === 'buy' ? '关注' : '警戒', coord:[signal.date,signal.close], value:signal.type === 'buy' ? '关' : '警', itemStyle:{color:signal.type === 'buy' ? '#c88869' : '#558d7e'}})) : [];
    const base = chartType === 'line' ? line('收盘', s.close) : {name: '日线', type: 'candlestick', data: s.candles, itemStyle: {color: '#ef4a4a', color0: '#12b886', borderColor: '#ef4a4a', borderColor0: '#12b886'}};
    base.markPoint = {symbol:'pin',symbolSize:30,data:markers,label:{fontSize:10}};
    const bands = [];
    if (technical) {
      const byDate = new Map(technical.series.dates.map((d,i)=>[d,i]));
      for (const [key,name] of [['boll_upper','布林上轨'],['boll_lower','布林下轨']]) bands.push(line(name,s.dates.map(d=>byDate.has(d)?technical.series[key][byDate.get(d)]:null),{lineStyle:{type:'dashed',width:1,opacity:.65}}));
    }
    plot('marketPrice', s.dates, [base, ...Object.entries(s.ma).map(([n, data]) => line('MA' + n, data)), ...bands]);
  }
  function renderTechnical(data) {
    const t = data.technical;
    el('stockTechnical').hidden = !t; el('stockIndicatorCharts').hidden = !t;
    if (!t) return;
    const part = (tag,text,cls) => {const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;};
    el('technicalAsOf').textContent = `信号日 ${t.as_of} · 收盘后`;
    el('technicalTrend').textContent = t.trend;
    el('technicalAction').textContent = t.action;
    el('technicalScore').textContent = t.ready ? `规则趋势分 ${t.score} / 100 · 分数不是上涨概率` : '趋势合成至少需要 65 根日线';
    el('technicalReasons').replaceChildren(...t.reasons.map(text=>part('li',text)));
    el('technicalConflict').hidden = !t.conflict;
    el('technicalConflict').textContent = t.conflict || '';
    const l=t.latest;
    el('technicalMetrics').replaceChildren(...[
      ['RSI14',num(l.rsi),'30 / 70 为观察阈值'],['MACD 柱',num(l.macd),'2 × (DIF − DEA)'],
      ['ATR14',num(l.atr),`约为收盘价 ${pct(t.levels.atr_distance)}`],['成交量比',l.volume_ratio==null?'—':num(l.volume_ratio)+' 倍','对比此前 20 日均量'],
    ].map(([label,value,note])=>{const box=part('div','');box.append(part('span',label),part('strong',value),part('small',note));return box;}));
    const context=t.context, values=context?.latest || {};
    el('technicalContext').replaceChildren(...(context ? [
      ['基准环境',context.environment,context.benchmark],
      ['相对基准 · 20日',pct(values.relative20),`60日 ${pct(values.relative60)}`],
      ['ADX14',num(values.adx),'趋势强度，不表示涨跌方向'],
      ['ATR 历史分位',pct(values.atr_percentile),`布林带宽分位 ${pct(values.bandwidth_percentile)}`],
      ['成交确认',values.volume_ratio==null?'—':num(values.volume_ratio)+' 倍',`成交额比 ${num(values.amount_ratio)} · 换手率比 ${num(values.turnover_ratio)}`],
    ].map(([label,value,note])=>{const box=part('div','');box.append(part('span',label),part('strong',value),part('small',note));return box;}) : []));
    el('technicalContextNote').textContent = context ? context.note + ' 缺失数据以 — 显示；相对比较本身时强弱为零。' : '';
    el('technicalValidation').replaceChildren(...t.forecasts.map(f=>{
      const box=part('article','','forecast-block'), c=f.comparison;
      box.append(part('h5',`${f.horizon}日 · 原模型 / 候选模型`));
      if(!c?.n){box.append(part('p','共同成熟样本不足，暂不比较。'));return box;}
      box.append(part('p',`${c.n} 笔 · ${c.start} — ${c.end}`),
        part('p',`方向准确率 ${pct(c.base.accuracy)} / ${pct(c.candidate.accuracy)}`),
        part('p',`Brier ${c.base.brier.toFixed(4)} / ${c.candidate.brier.toFixed(4)} · 改善 ${c.brier_improvement.toFixed(4)}`),
        part('p',c.ok?'达到展示样本门槛，仍需独立测试验证。':'少于20笔，不判断候选优劣。'));
      box.append(part('p',`候选当前上涨概率 ${pct(f.candidate?.probability)} · 仅作对照`));
      const regimeNames={strong:'基准偏强',weak:'基准偏弱',mixed:'分歧 / 震荡',unknown:'环境未知'};
      for(const row of c.regimes || []) {
        box.append(part('p',`${regimeNames[row.regime]} · ${row.n}笔` + (row.n ? ` · Brier ${row.base.brier.toFixed(4)} / ${row.candidate.brier.toFixed(4)}` : '') + (row.ok ? '' : ' · 样本不足20笔')));
      }
      for(const key of ['base','candidate']) {
        box.append(part('strong',key==='base'?'原模型概率分组':'候选模型概率分组'));
        for(const bin of c[key].calibration) box.append(part('p',`${pct(bin.lower)}～${pct(bin.upper)} · ${bin.n}笔 · 平均预测 ${pct(bin.probability)} / 实际上涨 ${pct(bin.frequency)}`));
      }
      return box;
    }));
    const basis=kind==='index'?'指数点位':data.meta.adjust==='hfq'?'后复权':data.meta.adjust==='qfq'?'前复权':'不复权';
    el('technicalLevels').replaceChildren(...[
      ['前 20 日低点',t.levels.support],['前 20 日高点',t.levels.resistance],['风控观察线 · 收盘 − 2ATR',t.levels.atr_stop_reference],
    ].map(([label,value])=>{const box=part('p','');box.append(part('span',label),part('strong',num(value)));return box;}),part('small',`${basis}坐标；支撑/阻力可能失效，观察线不保证成交，也不是基于持仓成本的止损。`));
    el('stockForecasts').replaceChildren(...t.forecasts.map(f=>{
      const box=part('article','','forecast-block');box.append(part('h5',`${f.horizon} 个交易日后 · 方向估计`));
      if(!f.ok){box.append(part('strong','样本不足'),part('p',f.reason+' 可等待数据中心补齐历史后重试。'));return box;}
      const e=f.evaluation;
      box.append(part('strong',pct(f.probability)),part('p','模型估计上涨概率 · 当前收盘至未来收盘'));
      box.append(part('span',!e.ok?'验证样本不足':e.advantage?'历史 Brier 优于频率基准':'历史 Brier 未优于频率基准','forecast-verdict'));
      if(e.n){box.append(part('p',`历史 ${e.n} 笔非重叠评估 · 方向准确率 ${pct(e.accuracy)} / 基准 ${pct(e.baseline_accuracy)}`),part('p',`Brier ${e.brier.toFixed(3)} / 基准 ${e.baseline_brier.toFixed(3)} · 越小越好`),part('small',`${e.start} — ${e.end}`));}
      else box.append(part('p','所选区间尚无成熟评估标签。'));
      box.append(part('small',`训练 ${f.training_rows} 个已成熟样本；概率未单独校准。`)); return box;
    }));
    el('technicalRules').textContent=t.rules;
    el('technicalNotices').replaceChildren(...t.notices.map(text=>part('li',text)));
    const series=t.series;
    plot('stockMacd',series.dates,[line('DIF',series.dif),line('DEA',series.dea),{name:'MACD',type:'bar',data:series.macd.map(v=>({value:v,itemStyle:{color:v>=0?'#c88869':'#558d7e'}}))}]);
    plot('stockRsi',series.dates,[line('RSI14',series.rsi,{markLine:{symbol:'none',silent:true,label:{color:'#aabbbf'},lineStyle:{color:'#7d8f95',type:'dashed'},data:[{yAxis:30},{yAxis:70}]}})],false,{yAxis:{type:'value',min:0,max:100,axisLabel:{color:'#8796ab'},splitLine:{lineStyle:{color:'#25343c'}}}});
    el('stockSignalHistory').replaceChildren(...(t.signals.length?t.signals.map(signal=>{const row=part('div','','signal-row');row.append(part('time',signal.date),part('span',signal.label),part('span',`收盘 ${num(signal.close)}`),part('span',`趋势分 ${signal.score}`));return row;}):[part('p','当前区间没有首次满足的规则条件。','note')]));
  }
  function render(data) {
    el('marketEmpty').hidden = true; el('marketResult').hidden = false;
    const s = data.stats, m = data.meta;
    const basis = m.adjust === 'qfq' ? '前复权' : m.adjust === 'hfq' ? '后复权' : '不复权';
    el('marketName').textContent = `${data.name} · ${data.code.toUpperCase()}`;
    el('marketMeta').textContent = `${m.start} — ${m.end} · ${m.n_days} 个交易日 · ${kind === 'stock' ? basis : '价格指数'} · ${m.source || '历史日线'}`;
    if (data.quality) {
      const q = data.quality;
      el('marketMeta').textContent += ` · 请求 ${q.requested_start} — ${q.requested_end}`;
      if (q.fetched_at) el('marketMeta').textContent += ` · 抓取于 ${q.fetched_at}`;
    }
    el('marketKpis').replaceChildren(...[
      [kind === 'stock' ? '最新收盘价' : '最新收盘点位', num(s.latest), `最近交易日 ${pct(s.change)}`],
      ['区间收益', pct(s.total_return), `年化 ${pct(s.ann_return)}`], ['最大回撤', pct(s.max_drawdown), '区间峰值至谷值'],
      ['年化波动', pct(s.volatility), '按 252 个交易日'], ['距 20 日均线', pct(s.ma20_gap), `距 60 日均线 ${pct(s.ma60_gap)}`],
    ].map(([label, value, note]) => {
      const box = document.createElement('div'); box.className = 'kpi';
      for (const [cls, text] of [['lb', label], ['vl', value], ['sb', note]]) { const part = document.createElement('div'); part.className = cls; part.textContent = text; box.appendChild(part); }
      return box;
    }));
    price();
    renderTechnical(data);
    el('marketPriceNote').textContent = 'MA20 / MA60 / MA120 为简单均线。叠加布林带（20 日均值 ± 2 倍总体标准差）及最近条件标记；样本不足时留空。';
    const series = data.series;
    el('marketVolumeUnit').textContent = `（${m.volume_unit}）`;
    el('marketVolumeNote').textContent = series.volume ? `最近交易日成交量 ${num(s.latest_volume)} ${m.volume_unit}` : '数据源未提供成交量。';
    plot('marketVolume', series.dates, series.volume ? [{name: '成交量', type: 'bar', data: series.volume, itemStyle: {color: '#637d9f'}}] : []);
    const c = data.comparison;
    plot('marketCompare', c.ok ? c.dates : [], c.ok ? [line(data.name, c.asset), line(c.name, c.benchmark)] : []);
    el('marketCompareNote').textContent = c.ok ? `${c.name} · ${c.n_days} 个共同交易日；仅在共同起点归一化。` : '基准数据不足，价格与风险分析仍可查看。';
    el('marketRelative').replaceChildren(...(c.ok ? [
      `相对 ${c.name} 区间收益差 ${pct(c.excess_return)}`, `Beta ${num(c.beta)}`, `日收益相关系数 ${num(c.correlation)}`,
    ] : ['暂无可用的基准比较']).map(text => {const p = document.createElement('p'); p.textContent = text; return p;}));
    plot('marketDrawdown', series.dates, [line('回撤', series.drawdown, {areaStyle: {opacity: 0.12}})], true);
    plot('marketVolatility', series.dates, [line('20 日年化波动', series.volatility)], true);
    el('marketNotes').replaceChildren(...data.warnings.map(text => {const li = document.createElement('li'); li.textContent = text; return li;}));
  }
  function init(cat, services) {
    catalog = cat; fund = services.fund; requestData = services.request; alert = services.alert;
    document.querySelectorAll('[data-asset]').forEach(button => {button.onclick = () => switchKind(button.dataset.asset);});
    document.querySelectorAll('[data-chart]').forEach(button => {button.onclick = () => {chartType = button.dataset.chart; if (result) price();};});
    el('marketFilters').onsubmit = event => {event.preventDefault(); submit(el('codeInput').value);};
    el('marketRefresh').onclick = () => result && submit(result.code, true);
    el('marketFilters').onchange = () => {if (result) status('区间或口径已更改，点击「应用区间」更新结果。');};
    window.addEventListener('resize', () => charts.forEach(c => c.resize()));
    switchKind('stock');
  }
  return {init, submit, get kind() {return kind;}};
})();
