/* UI-only checks using an existing fund result, without submitting real jobs.
 * node tests/browser/fund_browser.cjs [base URL] [cached-result JSON path]
 * Requires an isolated Edge CDP instance on port 9229.
 */
const fs = require('node:fs');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:8877';
const fixture = JSON.parse(fs.readFileSync(process.argv[3] || 'output/ui-evidence/fund-style-fixture.json', 'utf8'));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const pages = await (await fetch('http://127.0.0.1:9229/json')).json();
  const socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl);
  await new Promise(resolve => socket.addEventListener('open', resolve, {once:true}));
  let serial = 0;
  const pending = new Map(), errors = [];
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const request = pending.get(message.id); if (!request) return;
      pending.delete(message.id);
      message.error ? request.reject(message.error) : request.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
  });
  const command = (method, params={}) => new Promise((resolve,reject) => {
    const id = ++serial; pending.set(id,{resolve,reject}); socket.send(JSON.stringify({id,method,params}));
  });
  async function evaluate(expression) {
    const response = await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if (response.exceptionDetails) throw Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
  async function until(expression) {
    for (let i=0;i<120;i++) {if (await evaluate(expression)) return; await pause(100);}
    throw Error('Timed out: '+expression);
  }
  const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  async function capture(name) {
    const shot = await command('Page.captureScreenshot',{format:'png'});
    fs.mkdirSync('output/ui-evidence',{recursive:true});
    fs.writeFileSync(`output/ui-evidence/${name}.png`,Buffer.from(shot.data,'base64'));
  }
  try {
    await command('Runtime.enable'); await command('Page.enable');
    await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await command('Page.navigate',{url:base});
    await until("window.MarketApp?.kind==='stock' && document.querySelector('#fundStart').value");
    await click('[data-asset="fund"]');
    assert.equal(await evaluate("!document.querySelector('#fundView').hidden && !document.querySelector('#empty').hidden"),true);
    assert.equal(await evaluate("document.querySelectorAll('#emptyChips button').length > 0"),true);
    assert.equal(await evaluate("getComputedStyle(document.querySelector('#fundFilters')).backgroundColor === getComputedStyle(document.querySelector('#marketFilters')).backgroundColor"),true);
    assert.equal(await evaluate("windowStart(3,'2024-02-29')"),'2021-02-28');
    await capture('fund-desktop-empty');
    await evaluate(`window.__fundFixture=${JSON.stringify(fixture)};window.__fundRequests=[];window.__fundFetch=window.fetch;
      window.fetch=async(url,options)=>{if(url==='/api/analyze'){window.__fundRequests.push(JSON.parse(options.body));return {ok:true,json:async()=>({cached:true,result:structuredClone(window.__fundFixture)})};}return window.__fundFetch(url,options);};`);
    await evaluate("document.querySelector('#codeInput').value=window.__fundFixture.code;document.querySelector('#fundStart').value='2025-01-01';document.querySelector('#fundEnd').value='2026-01-01';document.querySelector('#fundStart').dispatchEvent(new Event('change',{bubbles:true}))");
    await click('#fundApply'); await until("!document.querySelector('#alert').hidden");
    assert.equal(await evaluate('window.__fundRequests.length'),0);
    assert.match(await evaluate("document.querySelector('#alert').textContent"),/2 年/);
    await evaluate("applyPreset(Object.keys(window.__fundFixture.models));document.querySelector('#codeInput').value=window.__fundFixture.code;document.querySelector('#fundStart').value=window.__fundFixture.meta.window_requested||window.__fundFixture.meta.start;document.querySelector('#fundEnd').value=window.__fundFixture.meta.window_requested_end||fundDate(new Date())");
    await click('#fundApply');
    await until("!document.querySelector('#main').hidden && document.querySelector('#chartNav canvas')");
    assert.equal(await evaluate('window.__fundRequests.length'),1);
    assert.equal(await evaluate("window.__fundRequests[0].end===document.querySelector('#fundEnd').value"),true);
    assert.equal(await evaluate("document.querySelector('#chartNav').compareDocumentPosition(document.querySelector('#modelStrip')) & Node.DOCUMENT_POSITION_FOLLOWING"),4);
    assert.equal(await evaluate("document.querySelectorAll('#kpis .kpi').length"),5);
    await capture('fund-desktop-result');
    await evaluate("document.querySelector('#fundStart').value='2020-01-01';document.querySelector('#fundStart').dispatchEvent(new Event('change',{bubbles:true}))");
    assert.equal(await evaluate("!document.querySelector('#fundStatus').hidden"),true);
    await click('[data-asset="stock"]'); assert.equal(await evaluate("document.querySelector('#fundView').hidden"),true);
    await click('[data-asset="fund"]'); assert.equal(await evaluate("!document.querySelector('#fundStatus').hidden && !document.querySelector('#main').hidden"),true);
    await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await evaluate("window.dispatchEvent(new Event('resize'));window.scrollTo(0,0)"); await pause(350);
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'),true);
    await capture('fund-mobile-result');
    await click('.fund-settings > summary');
    assert.equal(await evaluate("document.querySelector('.fund-settings').open"),true);
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'),true);
    assert.equal(errors.length,0,errors.join(';'));
    console.log('Fund UI: shared style, visible settings, date validation/request, real cached charts, asset switching, pending settings and mobile bounds: OK');
  } finally {await evaluate('if(window.__fundFetch)window.fetch=window.__fundFetch').catch(()=>{}); socket.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
