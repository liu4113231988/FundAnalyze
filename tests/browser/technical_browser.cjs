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
    await until("window.MarketApp?.kind==='stock' && document.querySelector('#marketStart')?.value");
    await set('marketStart','2025-01-01'); await set('marketEnd','2026-10-30'); await set('codeInput','600519');
    await click('#runBtn');
    await until("!document.querySelector('#stockTechnical').hidden && document.querySelector('#marketStatus').hidden",100);
    assert.equal(await evaluate("document.querySelectorAll('#stockForecasts article').length"),2);
    assert.equal(await evaluate("document.querySelectorAll('#technicalMetrics>div').length"),4);
    assert.match(await evaluate("document.querySelector('#stockForecasts').textContent"),/Brier/);
    assert.equal(await evaluate("document.querySelectorAll('#stockMacd canvas').length>0 && document.querySelectorAll('#stockRsi canvas').length>0"),true);
    assert.ok(await evaluate("document.querySelectorAll('#stockSignalHistory .signal-row').length>0"));
    await evaluate("document.querySelector('#stockTechnical').scrollIntoView({block:'start'})");await delay(250);await capture('technical-desktop-overview');
    await evaluate("document.querySelector('.forecast-heading').scrollIntoView({block:'start'})");await delay(250);await capture('technical-desktop-forecast');
    await evaluate("document.querySelector('#stockIndicatorCharts').scrollIntoView({block:'start'})");await delay(250);await capture('technical-desktop-indicators');
    await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await evaluate("window.dispatchEvent(new Event('resize'));document.querySelector('#stockTechnical').scrollIntoView({block:'start'})");await delay(300);await capture('technical-mobile-overview');
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
    await evaluate("document.querySelector('.forecast-heading').scrollIntoView({block:'start'})");await delay(250);await capture('technical-mobile-forecast');
    await evaluate("document.querySelector('#stockIndicatorCharts').scrollIntoView({block:'start'})");await delay(250);await capture('technical-mobile-indicators');
    // Keep actual bars, replace only technical block with actual short-history output.
    await evaluate(`window.__techFetch=fetch;window.__techResult=null;`);
    const payload=await evaluate(`(async()=>{const j=await(await fetch('/api/market/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:'stock',code:'600519',start:'2025-01-01',end:'2026-10-30'})})).json();for(let n=0;n<100;n++){await new Promise(r=>setTimeout(r,250));const s=await(await fetch('/api/job/'+j.job_id)).json();if(s.status==='done')return(await(await fetch('/api/result/'+j.job_id)).json()).result;}throw Error('No fixture');})()`);
    const short=JSON.parse(fs.readFileSync('output/ui-evidence/technical-short.json','utf8'));
    payload.technical=short;
    await evaluate(`window.__techResult=${JSON.stringify(payload)};window.fetch=async(url,options)=>{
      if(url==='/api/market/analyze')return{ok:true,json:async()=>({job_id:'technical-short'})};
      if(url==='/api/job/technical-short')return{ok:true,json:async()=>({status:'done'})};
      if(url==='/api/result/technical-short')return{ok:true,json:async()=>({result:window.__techResult})};
      return window.__techFetch(url,options);
    }`);
    await click('#runBtn'); await until("document.querySelector('#technicalAction').textContent==='等待样本'");
    assert.match(await evaluate("document.querySelector('#stockForecasts').textContent"),/样本不足/);
    await evaluate("document.querySelector('#stockTechnical').scrollIntoView({block:'start'})");await delay(250);await capture('technical-short-history');
    await evaluate('window.fetch=window.__techFetch');
    await click('[data-asset="index"]');await set('codeInput','sh000300');await set('marketStart','2025-01-01');await set('marketEnd','2026-10-30');await click('#runBtn');
    await until("!document.querySelector('#marketResult').hidden && document.querySelector('#marketStatus').hidden",60);
    assert.equal(await evaluate("document.querySelector('#stockTechnical').hidden && document.querySelector('#stockIndicatorCharts').hidden"),true);
    assert.equal(errors.length,0,errors.join(';'));
    console.log('Technical UI: real indicators, forecasts and validation, signal events, short-history degradation, stock/index isolation, desktop/mobile: OK');
  } finally {socket.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
