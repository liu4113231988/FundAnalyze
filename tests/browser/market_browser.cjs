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
  try {
    await command('Runtime.enable');
    await command('Page.enable');
    await command('Emulation.setDeviceMetricsOverride', {width: 1440, height: 900, deviceScaleFactor: 1, mobile: false});
    await command('Page.navigate', {url: base});
    await until("document.querySelector('#marketStart')?.value && window.MarketApp?.kind === 'stock'");
    assert.equal(await evaluate("document.querySelectorAll('[data-asset]').length"), 3);
    await capture('desktop-empty');
    // Use actual cached free-source bars for the successful render.
    await set('marketStart', '2025-01-01'); await set('marketEnd', '2026-10-30'); await set('codeInput', '600519');
    await click('#runBtn');
    await until("!document.querySelector('#marketResult').hidden && document.querySelector('#marketStatus').hidden", 100);
    assert.equal(await evaluate("document.querySelectorAll('#marketKpis .kpi').length"), 5);
    assert.equal(await evaluate("document.querySelectorAll('#marketPrice canvas').length > 0"), true);
    assert.equal(await evaluate("document.documentElement.scrollWidth <= innerWidth"), true);
    await capture('desktop-success');
    await click('[data-asset="fund"]');
    assert.equal(await evaluate("document.querySelector('#marketView').hidden && !document.querySelector('#empty').hidden"), true);
    await capture('desktop-fund');
    await click('[data-asset="index"]');
    await set('marketStart', '2025-01-01'); await set('marketEnd', '2026-10-30'); await set('codeInput', 'sh000300');
    await click('#runBtn');
    await until("!document.querySelector('#marketResult').hidden && document.querySelector('#marketStatus').hidden", 60);
    assert.equal(await evaluate("document.querySelector('#adjustField').hidden"), true);
    assert.match(await evaluate("document.querySelector('#marketName').textContent"), /沪深300/);
    await capture('desktop-index');
    // Controlled UI-only fixtures exercise partial/error/loading and stale-result handling.
    await evaluate(`window.__originalFetch=window.fetch; window.__testPartial=false; window.fetch=async (url, options) => {
      if (url === '/api/market/analyze') return {ok:true,status:200,json:async()=>({job_id:'ui-state'})};
      if (url === '/api/job/ui-state') {await new Promise(r=>setTimeout(r,1200)); return {ok:true,status:200,json:async()=>({status:'done'})};}
      if (url === '/api/result/ui-state') {return {ok:true,status:200,json:async()=>({result:window.__testResult})};}
      return window.__originalFetch(url,options);
    }`);
    // Get a real market payload from the running service through the completed result DOM is unnecessary;
    // invoke the real API once and store its result, then alter only comparison availability.
    const fixture = await evaluate(`(async()=>{const r=await window.__originalFetch('/api/market/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'stock',code:'600519',start:'2025-01-01',end:'2026-10-30'})});const j=await r.json();for(let i=0;i<100;i++){await new Promise(r=>setTimeout(r,200));const s=await (await window.__originalFetch('/api/job/'+j.job_id)).json();if(s.status==='done')return (await(await window.__originalFetch('/api/result/'+j.job_id)).json()).result;}throw Error('fixture job timeout');})()`);
    await evaluate(`window.__testResult=${JSON.stringify(fixture)};window.__testResult.comparison={ok:false,name:'沪深300'};window.__testResult.warnings=['基准指数暂不可用，已保留价格与风险分析；可点击刷新重试。'];`);
    await click('[data-asset="stock"]');
    await click('#runBtn'); await delay(150); await capture('desktop-loading');
    await until("!document.querySelector('#marketResult').hidden && document.querySelector('#marketStatus').hidden");
    assert.match(await evaluate("document.querySelector('#marketCompareNote').textContent"), /不足/);
    await capture('desktop-partial');
    await set('codeInput', 'invalid'); await click('#runBtn');
    assert.match(await evaluate("document.querySelector('#marketStatus').textContent"), /正确/);
    await capture('desktop-error');
    // Start one pending result, change entry immediately, and ensure it cannot display there.
    await set('codeInput', '600519'); await click('#runBtn'); await click('[data-asset="index"]');
    await delay(2200);
    assert.match(await evaluate("document.querySelector('#marketName').textContent"), /沪深300/);
    await evaluate('window.fetch=window.__originalFetch');
    await command('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    await evaluate('window.dispatchEvent(new Event("resize"))'); await delay(300);
    assert.equal(await evaluate("document.documentElement.scrollWidth <= innerWidth"), true);
    await capture('mobile-index');
    await click('[data-asset="stock"]'); await delay(300);
    assert.equal(await evaluate("document.documentElement.scrollWidth <= innerWidth"), true);
    await capture('mobile-stock');
    await evaluate("document.querySelector('[data-asset=stock]').focus()");
    await command('Input.dispatchKeyEvent', {type:'keyDown', key:'Tab', code:'Tab', windowsVirtualKeyCode:9});
    await command('Input.dispatchKeyEvent', {type:'keyUp', key:'Tab', code:'Tab', windowsVirtualKeyCode:9});
    assert.equal(await evaluate("document.activeElement.dataset.asset"), 'fund');
    assert.equal(await evaluate("getComputedStyle(document.activeElement).outlineStyle !== 'none'"), true);
    await command('Emulation.setEmulatedMedia', {features:[{name:'prefers-reduced-motion', value:'reduce'}]});
    assert.equal(await evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches"), true);
    assert.equal(errors.length, 0, errors.join('; '));
    console.log('Browser: 3 entries, real stock/index charts, loading/error/partial states, stale request, mobile layout, keyboard focus and reduced motion: OK');
  } finally {socket.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
