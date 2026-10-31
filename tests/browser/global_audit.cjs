/* Offline page-contract checks; dedicated Edge CDP 9229 and local API 8877. */
const fs = require('node:fs');
const assert = require('node:assert/strict');
const delay = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const pages = await (await fetch('http://127.0.0.1:9229/json')).json();
  const socket = new WebSocket(pages.find(p => p.type === 'page').webSocketDebuggerUrl);
  await new Promise(r => socket.addEventListener('open', r, {once:true}));
  let serial = 0;
  const pending = new Map(), errors = [];
  socket.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (m.id) {const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(m.error) : p.resolve(m.result);}
    if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.text);
  });
  const cmd = (method, params={}) => new Promise((resolve,reject) => {
    const id = ++serial; pending.set(id,{resolve,reject}); socket.send(JSON.stringify({id,method,params}));
  });
  const evaluate = async expression => {
    const r = await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const until = async expression => {
    for (let i=0;i<100;i++) {if(await evaluate(expression))return; await delay(100);}
    throw Error('Timed out: '+expression);
  };
  try {
    await cmd('Runtime.enable'); await cmd('Page.enable');
    await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await cmd('Page.navigate',{url:process.argv[2] || 'http://127.0.0.1:8877'});
    await until("window.MarketApp && document.querySelector('#marketStart').value");
    const fund = JSON.parse(fs.readFileSync('output/ui-evidence/fund-style-fixture.json','utf8'));
    const stock = JSON.parse(fs.readFileSync(process.argv[3] || 'output/ui-evidence/stock-fixture.json','utf8'));
    await evaluate(`window.__fund=${JSON.stringify(fund)};window.__stock=${JSON.stringify(stock)};
      window.__originalFetch=fetch;window.__calls=[];window.__holdBacktest=false;
      window.fetch=async(url,options)=>{
        const reply=data=>({ok:true,status:200,json:async()=>data});
        if(url.startsWith('/api/name/'))return reply({found:true});
        if(url==='/api/market/analyze'){window.__calls.push(JSON.parse(options.body));return reply({job_id:'market-audit'});}
        if(url==='/api/job/market-audit')return reply({status:'done'});
        if(url==='/api/result/market-audit')return reply({result:structuredClone(window.__stock)});
        if(url==='/api/analyze')return reply({cached:true,result:structuredClone(window.__fund)});
        if(url==='/api/backtest'){window.__btBody=JSON.parse(options.body);return reply({job_id:'backtest-audit'});}
        if(url==='/api/job/backtest-audit')return new Promise(resolve=>window.__releaseBacktest=()=>resolve(reply({status:'done'})));
        if(url==='/api/result/backtest-audit'){window.__staleBacktestApplied=true;throw Error('Stale result must not be read');}
        return window.__originalFetch(url,options);
      }`);
    // Market date minimum is enforced before submission; chart switch survives repeated active navigation.
    await evaluate("document.querySelector('#codeInput').value='600519';document.querySelector('#marketStart').value='2025-01-01';document.querySelector('#marketEnd').value='2025-01-10'");
    await click('#runBtn');
    assert.equal(await evaluate('__calls.length'),0);
    assert.match(await evaluate("document.querySelector('#marketStatus').textContent"),/30/);
    await evaluate("document.querySelector('#marketEnd').value='2026-01-01'");
    await click('#runBtn');
    assert.equal(await evaluate("document.querySelector('#marketRecalculate').disabled"),true);
    await click('[data-asset="stock"]');
    await until("!document.querySelector('#marketResult').hidden && !document.querySelector('#runBtn').disabled");
    if (stock.technical?.context) {
      assert.equal(await evaluate("document.querySelectorAll('#technicalContext>div').length"),5);
      assert.match(await evaluate("document.querySelector('#technicalValidation').textContent"),/平均预测/);
      if(stock.technical.forecasts[0].comparison.regimes) assert.match(await evaluate("document.querySelector('#technicalValidation').textContent"),/基准偏强/);
    }
    await click('[data-chart="line"]');
    assert.equal(await evaluate("document.querySelector('[data-chart=line]').getAttribute('aria-pressed')"),'true');
    await click('[data-asset="index"]');
    assert.equal(await evaluate("document.querySelector('#presetSel').title"),'示例指数');
    assert.equal(await evaluate("document.querySelector('#adjustField').hidden"),true);
    await evaluate("window.__stock.kind='index';window.__stock.code='sh000300';window.__stock.name='沪深300';document.querySelector('#codeInput').value='sh000300'");
    await click('#runBtn'); await until("document.querySelector('#marketName').textContent.includes('沪深300')");
    assert.equal(await evaluate("document.querySelector('#stockTechnical').hidden"),!stock.technical);
    // Manual best/selection survive round-trip; header follows manual best immediately.
    await click('[data-asset="fund"]');
    await evaluate("applyResult(structuredClone(__fund));S.sel=Object.keys(S.result.models).at(-1);renderModelStrip();renderSelected()");
    await click('#btnSaveSel');
    const selection = await evaluate('S.sel');
    await click('[data-asset="stock"]'); await click('[data-asset="fund"]');
    assert.equal(await evaluate('S.sel'), selection);
    assert.equal(await evaluate('S.best'), selection);
    // New analysis clears previous backtest and resets rolling-window control to the returned data.
    await evaluate("document.querySelector('#btOut').hidden=false;S.rollWin=60;document.querySelector('#rollWin').value='60';applyResult(structuredClone(__fund))");
    assert.equal(await evaluate("document.querySelector('#btOut').hidden"),true);
    assert.equal(await evaluate("+document.querySelector('#rollWin').value"),fund.correlation.roll_window || 120);
    // Request captures current fitted configuration, not un-applied controls.
    await evaluate("document.querySelector('#autoEquity').checked=!S.result.meta.auto_equity;syncBacktestDefaults()");
    assert.equal(await evaluate("document.querySelector('#btAuto').checked"),fund.meta.auto_equity);
    await click('#btRun'); await until('!!window.__releaseBacktest');
    assert.equal(await evaluate('__btBody.window_start'),fund.meta.window_requested || fund.meta.start);
    assert.equal(await evaluate('__btBody.max_rebal'),fund.meta.max_rebal);
    await click('[data-asset="index"]');
    await evaluate('__releaseBacktest()'); await delay(150);
    assert.equal(await evaluate('!!window.__staleBacktestApplied'),false);
    assert.equal(await evaluate("document.querySelector('#progressWrap').hidden"),true);
    await click('[data-asset="fund"]');
    assert.equal(await evaluate("document.querySelector('#btRun').disabled"),false);
    // All three workspaces fit a narrow phone viewport.
    await cmd('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    for (const kind of ['stock','index','fund']) {
      await click(`[data-asset="${kind}"]`); await delay(200);
      if (stock.technical?.context && kind==='stock') {
        await evaluate("document.querySelector('#technicalValidation').parentElement.open=true;document.querySelector('#technicalContext').scrollIntoView({block:'start'})");
        const contextShot = await cmd('Page.captureScreenshot',{format:'png'});
        fs.writeFileSync('output/ui-evidence/context-mobile.png',Buffer.from(contextShot.data,'base64'));
      }
      assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,kind);
    }
    const shot = await cmd('Page.captureScreenshot',{format:'png'});
    fs.writeFileSync('output/ui-evidence/global-audit-mobile.png',Buffer.from(shot.data,'base64'));
    assert.deepEqual(errors,[]);
    console.log('Global UI: market minimum/active tab/loading, stock/index charts, model selection, stale backtest, fitted settings, correlation reset, mobile bounds: OK');
  } finally {await evaluate('if(window.__originalFetch)window.fetch=window.__originalFetch').catch(()=>{});socket.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
