/* Persistent asset synchronization, independent of the research workspace. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const dialog = $('dataCenter');
  const states = {pending: '等待同步', running: '同步中', success: '已同步', error: '同步失败'};
  const kinds = {stock: '股票', fund: '基金', index: '指数'};
  let snapshot, busy = false;
  const names = new Map();
  function node(tag, text, cls) {const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e;}
  function feedback(message, error = false) {$('syncFeedback').textContent = message; $('syncFeedback').hidden = !message; $('syncFeedback').classList.toggle('is-error', error);}
  async function api(url, body) {
    const response = await fetch(url, body ? {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)} : undefined);
    const data = await response.json();
    if (!response.ok) throw Error(data.error || '请求失败，请稍后重试');
    return data;
  }
  function basis(a) {return a.kind === 'fund' ? (a.basis === 'total' ? '总回报' : '累计净值') : a.kind === 'index' ? '价格指数' : ({qfq:'前复权', hfq:'后复权'}[a.basis] || '不复权');}
  function timestamp(value) {return value ? new Date(value).toLocaleString('zh-CN', {timeZone:'Asia/Shanghai', hour12:false}) : '尚未完成';}
  function renderRows() {
    if (!snapshot) return;
    const query = $('assetSearch').value.trim().toLowerCase(), state = $('assetState').value;
    const filtered = snapshot.assets.filter(a => (!state || a.state === state) && `${a.code} ${names.get(a.code) || ''} ${kinds[a.kind]}`.toLowerCase().includes(query));
    const order = {error:0, running:1, pending:2, success:3};
    filtered.sort((a,b) => order[a.state] - order[b.state] || a.code.localeCompare(b.code));
    const rows = filtered.map(a => {
      const tr = node('tr');
      const asset = node('td'); asset.append(node('strong', names.get(a.code) || a.code), node('span', `${a.code.toUpperCase()} · ${kinds[a.kind]} · ${basis(a)}`, 'asset-detail'));
      const status = node('td'); status.append(node('span', states[a.state] || '等待同步', `state-badge state-${a.state}`));
      if (a.error) status.append(node('p', a.error, 'asset-error'));
      const time = node('td', timestamp(a.last_success), 'asset-time');
      const count = node('td'); count.append(node('strong', Number(a.rows || 0).toLocaleString()), node('span', a.mode === 'full' ? '全历史同步' : a.mode === 'incremental' ? '增量更新' : '首次补齐', 'asset-detail'));
      const actions = node('td'), button = node('button', a.state === 'error' ? '重试' : a.state === 'running' ? '同步中' : a.state === 'pending' ? '已排队' : '重新同步', 'btn ghost sync-row-action');
      button.type = 'button'; button.disabled = ['running','pending'].includes(a.state);
      button.setAttribute('aria-label', `${button.textContent} ${names.get(a.code) || a.code} ${basis(a)}`);
      button.onclick = async () => {
        button.disabled = true;
        try {const response = await api('/api/sync/retry', {asset_key:a.asset_key}); feedback(response.queued ? `${a.code} 已排队，将在当前同步批次后处理。` : '该资产正在同步。'); await refresh();}
        catch(error) {feedback(error.message, true); button.disabled = false;}
      };
      actions.append(button); tr.append(asset,status,time,count,actions); return tr;
    });
    $('syncAssetRows').replaceChildren(...rows);
    $('syncEmpty').hidden = filtered.length > 0;
    $('syncEmpty').textContent = snapshot.assets.length ? '没有符合条件的资产，试试其他代码或状态。' : '尚未添加资产，从上方添加一只证券开始。';
  }
  function render() {
    const list = snapshot.assets;
    const counts = Object.fromEntries(Object.keys(kinds).map(kind => [kind, new Set(list.filter(a => a.kind === kind).map(a => a.code)).size]));
    const failed = list.filter(a => a.state === 'error').length;
    const waiting = list.filter(a => ['running','pending'].includes(a.state)).length;
    $('syncIndicator').textContent = !list.length ? '尚未添加资产 · 数据中心' : failed ? `${failed} 项同步需处理 · 数据中心` : waiting ? `${waiting} 项等待或同步中 · 数据中心` : '数据同步正常 · 数据中心';
    $('openDataCenter').classList.toggle('has-error', failed > 0);
    $('syncSchedule').textContent = `每日 ${snapshot.time} · 北京时间${snapshot.enabled ? '' : ' · 服务内调度已停用'}`;
    $('syncOverview').replaceChildren(...[
      ['股票',counts.stock], ['基金',counts.fund], ['指数',counts.index], ['待同步 / 进行中',waiting], ['失败口径',failed],
    ].map(([label,value]) => {const p=node('div');p.append(node('span',label),node('strong',String(value)));return p;}));
    $('universeSummary').replaceChildren(node('span','你的研究范围','universe-caption'), ...Object.keys(kinds).map(kind=>{const p=node('span');p.append(node('b',String(counts[kind])),document.createTextNode(' '+kinds[kind]));return p;}));
    $('syncUpdated').textContent = `状态更新于 ${new Date().toLocaleTimeString('zh-CN')} · 列表按同步口径显示，同一证券可能有多条`;
    renderRows();
  }
  async function refresh() {
    if (busy) return;
    busy = true; $('refreshSync').disabled = true; $('refreshSync').textContent = '读取中…';
    try {snapshot = await api('/api/sync'); render();}
    catch(error) {$('syncIndicator').textContent = '同步状态不可用 · 数据中心'; feedback('无法读取同步状态：' + error.message, true); if (!snapshot) {$('syncEmpty').hidden = false; $('syncEmpty').textContent = '同步状态读取失败，请点击刷新状态重试。';}}
    finally {busy = false; $('refreshSync').disabled = false; $('refreshSync').textContent = '刷新状态';}
  }
  function changeKind() {
    const kind = $('addAssetKind').value;
    $('addBasisField').hidden = kind === 'index';
    const options = kind === 'fund' ? [['total','总回报'],['cum','累计净值']] : [['qfq','前复权'],['hfq','后复权'],['','不复权']];
    $('addAssetBasis').replaceChildren(...options.map(([value,text]) => new Option(text,value)));
    $('addAssetCode').placeholder = kind === 'fund' ? '如 110022' : kind === 'index' ? '如 sh000300' : '如 600519';
  }
  $('openDataCenter').onclick = () => {dialog.showModal(); refresh();};
  $('closeDataCenter').onclick = () => dialog.close();
  dialog.addEventListener('close', () => $('openDataCenter').focus());
  dialog.addEventListener('click', event => {if(event.target === dialog) {const r=dialog.getBoundingClientRect(); if(event.clientX<r.left || event.clientX>r.right || event.clientY<r.top || event.clientY>r.bottom) dialog.close();}});
  $('refreshSync').onclick = () => {feedback(''); refresh();};
  $('assetSearch').oninput = renderRows; $('assetState').onchange = renderRows; $('addAssetKind').onchange = changeKind;
  $('addAssetForm').onsubmit = async event => {
    event.preventDefault(); const button = $('addAssetButton'); button.disabled = true;
    try {const code=$('addAssetCode').value.trim(); await api('/api/sync/assets',{kind:$('addAssetKind').value,code,basis:$('addAssetBasis').value}); feedback(`${code} 已加入研究范围；新增口径会自动补齐历史。`); $('assetSearch').value=''; $('assetState').value=''; await refresh();}
    catch(error) {feedback(error.message,true);} finally {button.disabled=false;}
  };
  $('workspaceDate').textContent = new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric',weekday:'long',timeZone:'Asia/Shanghai'});
  api('/api/catalog').then(cat => {for (const item of [...cat.known_funds,...cat.assets.stocks,...cat.assets.indices]) names.set(item.code,item.name); if(snapshot) renderRows();}).catch(()=>{});
  refresh();
  setInterval(() => {if (!document.hidden) refresh();}, 20000);
})();
