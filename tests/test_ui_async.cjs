const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('webapp/static/app.js', 'utf8');
const nodes = new Map();
const pending = [];
const context = {URLSearchParams, setTimeout: fn => fn(),
  S: {result: {code:'110022', meta:{window_requested:'2021-01-01', window_requested_end:'2025-01-01'}}},
  $: id => {if (!nodes.has(id)) nodes.set(id, {}); return nodes.get(id);},
  api: url => new Promise(resolve => pending.push({url, resolve})),
  renderCorr() {}, showAlert(message) {throw Error(message);},
  applyResult() {throw Error('Stale fund result applied');},
  setProgress() {throw Error('Stale fund progress applied');},
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('let fundGeneration ='), source.indexOf('// ==================================================================== 运行分析')), context);
vm.runInContext(source.slice(source.indexOf('async function poll(jid'), source.indexOf('function applyResult(res)')), context);
(async () => {
  const first = context.updateCorrelation(60);
  const second = context.updateCorrelation(120);
  const query = new URL('http://local' + pending[0].url).searchParams;
  assert.equal(query.get('start'), '2021-01-01');
  assert.equal(query.get('end'), '2025-01-01');
  pending[1].resolve({correlation:'new'}); await second;
  pending[0].resolve({correlation:'old'}); await first;
  assert.equal(context.S.result.correlation, 'new');
  const third = context.updateCorrelation(90);
  context.S.result = {correlation:'other fund'};
  pending[2].resolve({correlation:'obsolete'}); await third;
  assert.equal(context.S.result.correlation, 'other fund');
  const poll = context.poll('old-job');
  await Promise.resolve();
  vm.runInContext('fundGeneration++', context);
  pending[3].resolve({status:'done'}); await poll;
  assert.equal(pending.length, 4);
  console.log('Correlation window, response ordering, replaced fund and stale polling: OK');
})().catch(error => {console.error(error); process.exitCode=1;});
