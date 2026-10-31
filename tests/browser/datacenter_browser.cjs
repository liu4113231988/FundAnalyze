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
    await command('Runtime.enable'); await command('Page.enable');
    await command('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
    await command('Page.navigate',{url:base});
    await until("document.querySelector('#universeSummary b') && window.MarketApp?.kind === 'stock'");
    await capture('polish-desktop-home');
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'),true);
    await click('#openDataCenter');
    await until("document.querySelectorAll('#syncAssetRows tr').length > 0 && !document.querySelector('#refreshSync').disabled");
    assert.equal(await evaluate("document.querySelector('#dataCenter').open"),true);
    await capture('polish-desktop-data');
    await set('assetSearch','600519');
    await evaluate("document.querySelector('#assetSearch').dispatchEvent(new Event('input'))");
    assert.ok(await evaluate("document.querySelectorAll('#syncAssetRows tr').length > 0"));
    assert.match(await evaluate("document.querySelector('#syncAssetRows').textContent"),/600519/);
    await set('assetSearch','nothingmatches'); await evaluate("document.querySelector('#assetSearch').dispatchEvent(new Event('input'))");
    assert.equal(await evaluate("document.querySelector('#syncEmpty').hidden"),false);
    await capture('polish-empty-search');
    await set('assetSearch',''); await evaluate("document.querySelector('#assetSearch').dispatchEvent(new Event('input'))");
    // UI-only states and mutation interception, never add test symbols to production.
    await evaluate(`(async()=>{window.__dcFetch=window.fetch;window.__dcSnapshot=await(await fetch('/api/sync')).json();window.__dcMode='partial';window.__dcRequests=[];
      window.fetch=async(url,options)=>{
        if(url==='/api/sync'){
          await new Promise(r=>setTimeout(r,500));
          if(window.__dcMode==='error')return {ok:false,json:async()=>({error:'测试：行情服务暂不可用'})};
          const s=structuredClone(window.__dcSnapshot);
          if(window.__dcMode==='empty')s.assets=[];
          else{s.assets=s.assets.slice(0,4);s.assets[0].state='error';s.assets[0].error='测试：上游连接超时，上一份行情已保留。';s.assets[1].state='running';}
          return {ok:true,json:async()=>s};
        }
        if(url==='/api/sync/retry'||url==='/api/sync/assets'){window.__dcRequests.push({url,body:JSON.parse(options.body)});return {ok:true,json:async()=>({ok:true,queued:true})};}
        return window.__dcFetch(url,options);
      };})()`);
    await click('#refreshSync'); await capture('polish-loading');
    await until("!document.querySelector('#refreshSync').disabled");
    await capture('polish-partial');
    await click('#syncAssetRows .sync-row-action');
    await until("window.__dcRequests.length===1 && !document.querySelector('#refreshSync').disabled");
    assert.equal(await evaluate("window.__dcRequests[0].url"),'/api/sync/retry');
    await set('addAssetCode','600519'); await click('#addAssetButton');
    await until("window.__dcRequests.length===2 && !document.querySelector('#addAssetButton').disabled");
    assert.equal(await evaluate("window.__dcRequests[1].body.basis"),'qfq');
    await evaluate("window.__dcMode='error'"); await click('#refreshSync');
    await until("document.querySelector('#syncFeedback').classList.contains('is-error')");
    await capture('polish-error');
    await evaluate("window.__dcMode='empty'"); await click('#refreshSync');
    await until("!document.querySelector('#refreshSync').disabled");
    assert.equal(await evaluate("document.querySelector('#syncEmpty').hidden"),false);
    assert.equal(await evaluate("document.querySelector('#syncFeedback').hidden"),true);
    await capture('polish-empty-data');
    // A mutation during an older status request must trigger a trailing refresh.
    await evaluate(`window.__dcMode='partial';window.__dcReads=0;window.__dcWrappedFetch=window.fetch;
      window.fetch=async(url,options)=>{if(url==='/api/sync')window.__dcReads++;return window.__dcWrappedFetch(url,options);}`);
    await click('#refreshSync');
    await set('addAssetCode','600519'); await click('#addAssetButton');
    await until("window.__dcReads>=2 && !document.querySelector('#refreshSync').disabled");
    await evaluate("window.fetch=window.__dcFetch;document.querySelector('#assetKind').value='';document.querySelector('#assetKind').dispatchEvent(new Event('change',{bubbles:true}))"); await click('#refreshSync');
    await until("document.querySelectorAll('#syncAssetRows tr').length>4 && !document.querySelector('#refreshSync').disabled");
    await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await evaluate("window.dispatchEvent(new Event('resize'))"); await delay(250);
    assert.equal(await evaluate("document.querySelector('#dataCenter').scrollWidth<=document.querySelector('#dataCenter').clientWidth"),true);
    await capture('polish-mobile-data');
    // Native dialog focus cannot tab outside; Escape returns to opener.
    await evaluate("document.querySelector('#closeDataCenter').focus()");
    await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
    await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Tab',code:'Tab',windowsVirtualKeyCode:9});
    assert.equal(await evaluate("document.querySelector('#dataCenter').contains(document.activeElement)"),true);
    assert.equal(await evaluate("getComputedStyle(document.activeElement).outlineStyle !== 'none'"),true);
    await command('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    await command('Input.dispatchKeyEvent',{type:'keyUp',key:'Escape',code:'Escape',windowsVirtualKeyCode:27});
    await until("!document.querySelector('#dataCenter').open");
    await until("document.activeElement.id === 'openDataCenter'");
    await capture('polish-mobile-home');
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
    await command('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
    assert.equal(await evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches"),true);
    assert.equal(errors.length,0,errors.join(';'));
    console.log('Data center: real counts, search, empty/loading/error/partial, add/retry, native dialog keyboard, mobile: OK');
  } finally {socket.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
