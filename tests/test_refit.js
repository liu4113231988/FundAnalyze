// 重算必须绑定已分析窗口，并刷新完整诊断结果。
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('webapp/static/app.js', 'utf8');
const functionText = source.slice(source.indexOf('async function applyParams()'), source.indexOf('// ==================================================================== 净值', source.indexOf('async function applyParams()')));
let submitted;
let applied;
const context = {
  S: {sel: 'ridge', params: {}, cat: {models: [{key: 'ridge', name: 'Ridge', params: {alpha_scale: 1}}]},
      result: {code: '110022', meta: {models: ['ridge', 'pcr'], params: {pcr: {n_components: 2}}, max_rebal: 30, auto_equity: false, window_requested: '2021-01-01', end: '2026-01-01'}}},
  $: id => id === 'tunerBody' ? {querySelectorAll: () => [{dataset: {pk: 'alpha_scale'}, type: 'number', value: '2'}]} : {},
  setProgress() {}, showAlert(message) { if (message) throw Error(message); },
  api: async (url, options) => { submitted = JSON.parse(options.body); return {cached: true, result: {insights: 'updated', style_drift: 'updated'}}; },
  applyResult(result) { applied = result; },
};
vm.createContext(context);
vm.runInContext(functionText, context);
(async () => {
  await context.applyParams();
  assert.equal(submitted.start, '2021-01-01');
  assert.equal(submitted.end, '2026-01-01');
  assert.equal(submitted.max_rebal, 30);
  assert.equal(submitted.auto_equity, false);
  assert.deepEqual(submitted.models, ['ridge', 'pcr']);
  assert.deepEqual(submitted.params, {pcr: {n_components: 2}, ridge: {alpha_scale: 2}});
  assert.equal(applied.style_drift, 'updated');
  console.log('Refit configuration and full-result refresh: OK');
})().catch(e => { console.error(e); process.exitCode = 1; });
