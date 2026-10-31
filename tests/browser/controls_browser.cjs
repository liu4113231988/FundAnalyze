/* 使用已启动的隔离 Edge CDP 实例；Node 22+，无需新增测试依赖。 */
const fs = require('node:fs');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:8877';
const output = 'output/ui-evidence';
fs.mkdirSync(output, {recursive: true});
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const pages = await (await fetch('http://127.0.0.1:9229/json')).json();
  const target = pages.find(p => p.type === 'page');
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(resolve => socket.addEventListener('open', resolve, {once: true}));
  let seq = 0;
  const pending = new Map(), errors = [];
  socket.addEventListener('message', async event => {
    const message = JSON.parse(event.data);
    if (message.id) {const request = pending.get(message.id); if (!request) return; pending.delete(message.id); message.error ? request.reject(message.error) : request.resolve(message.result);}
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
  });
  function command(method, params = {}) {return new Promise((resolve, reject) => {const id = ++seq; pending.set(id, {resolve, reject}); socket.send(JSON.stringify({id, method, params}));});}
  async function evaluate(expression) {
    const response = await command('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (response.exceptionDetails) throw Error(response.exceptionDetails.text);
    return response.result.value;
  }
  async function until(expression, seconds = 30) {
    for (let i = 0; i < seconds * 4; i++) {if (await evaluate(expression)) return; await delay(250);}
    throw Error('Timed out: ' + expression);
  }
  async function capture(name) {const data = await command('Page.captureScreenshot', {format: 'png'}); fs.writeFileSync(`${output}/${name}.png`, Buffer.from(data.data, 'base64'));}
  const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const set = (id, value) => evaluate(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(value)}`);
  async function key(key,code=key){await command('Input.dispatchKeyEvent',{type:'keyDown',key,code,windowsVirtualKeyCode:{Enter:13,Escape:27,ArrowDown:40,ArrowUp:38,ArrowRight:39}[key]||0});await command('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:{Enter:13,Escape:27,ArrowDown:40,ArrowUp:38,ArrowRight:39}[key]||0});}
  try {
    await command('Runtime.enable');await command('Page.enable');
    await command('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
    await command('Page.navigate',{url:base});
    await until("window.MarketApp?.kind==='stock' && document.querySelector('.control-calendar-button') && document.querySelector('#marketStart').value");
    assert.equal(await evaluate("document.querySelectorAll('select').length===document.querySelectorAll('select.control-native').length"),true);
    await set('marketStart','2024-02-20');
    await click('.control-date:has(#marketStart) .control-calendar-button');
    await until("document.querySelector('.calendar-popover:popover-open')");await capture('controls-desktop-calendar');
    await click('.calendar-popover [aria-label="选择年份"]');await capture('controls-desktop-years');
    await click('.calendar-period.is-selected');
    assert.equal(await evaluate("document.querySelectorAll('.calendar-period').length"),12);
    await evaluate("[...document.querySelectorAll('.calendar-period')].find(b=>b.textContent==='2月').click()");
    await click('[data-date="2024-02-29"]');
    assert.equal(await evaluate("document.querySelector('#marketStart').value"),'2024-02-29');
    assert.equal(await evaluate("document.querySelector('.control-date:has(#marketStart) .control-date-text').value"),'2024-02-29');
    // Manual input rejects impossible dates instead of normalizing to March.
    await evaluate("const field=document.querySelector('.control-date:has(#marketStart) .control-date-text');field.value='2025-02-29';field.dispatchEvent(new Event('change',{bubbles:true}))");
    assert.equal(await evaluate("document.querySelector('#marketStart').value"),'');
    assert.equal(await evaluate("document.querySelector('.control-date:has(#marketStart) .control-date-text').validity.valid"),false);
    await capture('controls-invalid-date');
    await evaluate("{const n=document.querySelector('#marketStart');n.min='2024-02-10';n.max='2024-02-25';n.value='2024-02-20'}");
    await click('.control-date:has(#marketStart) .control-calendar-button');
    assert.equal(await evaluate("document.querySelector('[data-date=\"2024-02-29\"]').disabled"),true);
    await key('Escape');
    await evaluate("{const n=document.querySelector('#marketStart');n.removeAttribute('min');n.removeAttribute('max');n.required=false}");await delay(50);
    await click('.control-date:has(#marketStart) .control-calendar-button');
    await evaluate("[...document.querySelectorAll('.calendar-footer button')].find(b=>b.textContent==='清空').click()");
    assert.equal(await evaluate("document.querySelector('#marketStart').value"),'');
    await evaluate("{const n=document.querySelector('#marketStart');n.required=true;const f=document.querySelector('.control-date:has(#marketStart) .control-date-text');f.value='20250101';f.dispatchEvent(new Event('change',{bubbles:true}))}");
    assert.equal(await evaluate("document.querySelector('#marketStart').value"),'2025-01-01');
    await set('marketStart','2025-01-01');
    await click('.control-select:has(#marketBenchmark) .control-trigger');
    await until("document.querySelector('.select-popover:popover-open')");await capture('controls-desktop-select');
    await key('ArrowDown');await key('Enter');
    assert.equal(await evaluate("document.querySelector('#marketBenchmark').value"),'sh000905');
    assert.equal(await evaluate("document.querySelector('.control-select:has(#marketBenchmark) .control-value').textContent"),'中证500');
    // Native programmatic updates and dynamic options stay synchronized.
    await set('marketAdjust','hfq');
    assert.equal(await evaluate("document.querySelector('.control-select:has(#marketAdjust) .control-value').textContent"),'后复权');
    await evaluate("document.querySelector('#marketAdjust').disabled=true");await delay(100);
    assert.equal(await evaluate("document.querySelector('.control-select:has(#marketAdjust) .control-trigger').disabled"),true);
    await evaluate("document.querySelector('#marketAdjust').disabled=false");
    await click('[data-asset="index"]');await delay(100);
    assert.match(await evaluate("document.querySelector('.control-select:has(#presetSel) .control-value').textContent"),/示例/);
    await click('#openDataCenter');await until("document.querySelector('#dataCenter').open");
    await click('.control-select:has(#addAssetKind) .control-trigger');
    await until("document.querySelector('.select-popover:popover-open')");await capture('controls-dialog-select');
    await evaluate("[...document.querySelectorAll('.select-option')].find(b=>b.textContent.includes('基金')).click()");await delay(100);
    assert.equal(await evaluate("document.querySelector('#addAssetKind').value"),'fund');
    assert.equal(await evaluate("document.querySelector('.control-select:has(#addAssetBasis) .control-value').textContent"),'总回报');
    await click('.control-select:has(#addAssetBasis) .control-trigger');await key('Escape');
    assert.equal(await evaluate("document.querySelector('#dataCenter').open"),true);
    assert.equal(await evaluate("document.activeElement===document.querySelector('.control-select:has(#addAssetBasis) .control-trigger')"),true);
    await evaluate("document.querySelector('#addAssetBasis').replaceChildren()");await delay(50);
    await click('.control-select:has(#addAssetBasis) .control-trigger');
    assert.match(await evaluate("document.querySelector('.select-popover').textContent"),/暂无可选项/);
    await capture('controls-empty-options');await key('Escape');
    await evaluate("document.querySelector('#addAssetKind').dispatchEvent(new Event('change',{bubbles:true}))");await delay(50);
    await click('#closeDataCenter');
    await until("!document.querySelector('#dataCenter').open && document.activeElement.id==='openDataCenter'");
    await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await evaluate("window.dispatchEvent(new Event('resize'));document.querySelector('#marketFilters').scrollIntoView({block:'start'})");
    await click('.control-date:has(#marketStart) .control-calendar-button');await delay(250);await capture('controls-mobile-calendar');
    assert.equal(await evaluate("(()=>{const r=document.querySelector('.calendar-popover').getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&r.bottom<=innerHeight})()"),true);
    await key('ArrowRight');await key('Enter');
    await click('.control-select:has(#marketBenchmark) .control-trigger');await delay(150);await capture('controls-mobile-select');
    assert.equal(await evaluate("document.querySelector('.select-popover').getBoundingClientRect().right<=innerWidth"),true);
    await key('Escape');
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
    assert.equal(errors.length,0,errors.join(';'));
    console.log('Controls: all selects enhanced, calendar/leap/manual validation, selection and events, dynamic/disabled state, modal layering, Escape/focus, mobile bounds: OK');
  }finally{socket.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
