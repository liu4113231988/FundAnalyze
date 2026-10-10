/**
 * 前端渲染验证：用 jsdom 加载真实页面 + 真实 API 数据，
 * 捕获所有运行时错误与未捕获 Promise，并断言关键 DOM 是否被正确填充。
 *
 * 用法： node tests/dom_check.js http://127.0.0.1:8760
 */
const { JSDOM, VirtualConsole } = require('jsdom');

const BASE = process.argv[2] || 'http://127.0.0.1:8760';
const CODE = process.argv[3] || '110022';

const errors = [];
const warns = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => {
  // 过滤掉 echarts / canvas 相关的 jsdom 限制，这些在真浏览器里没问题
  const m = String(e.message || e);
  if (/Not implemented|getContext|Could not parse CSS|HTMLCanvasElement/i.test(m)) {
    warns.push('[jsdom限制] ' + m.slice(0, 80));
  } else {
    errors.push('[jsdomError] ' + (e.stack || m).slice(0, 400));
  }
});
vc.on('error', (...a) => errors.push('[console.error] ' + a.join(' ').slice(0, 300)));
vc.on('warn', (...a) => warns.push('[console.warn] ' + a.join(' ').slice(0, 200)));

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  console.log('▶ 加载页面:', BASE);
  const html = await (await fetch(BASE + '/')).text();

  const dom = new JSDOM(html, {
    url: BASE + '/', runScripts: 'dangerously',
    resources: undefined,          // 不用 'usable'，脚本一律手动注入
    pretendToBeVisual: true, virtualConsole: vc,
  });
  const { window } = dom;
  const doc = window.document;

  // jsdom 没有内置 fetch / Headers —— 注入 Node 18+ 的原生 fetch，
  // 否则 app.js 里的 api() 会直接抛 ReferenceError，测试就失去意义。
  if (!window.fetch) {
    window.fetch = (input, init) => {
      const url = typeof input === 'string' && input.startsWith('/')
        ? BASE + input : input;
      return fetch(url, init);
    };
  }
  if (!window.Headers) window.Headers = Headers;

  // echarts 需要 canvas，jsdom 没有 → 打桩，只记录调用不报错。
  // 脚本不自动加载，所以这个桩不会被真 echarts 覆盖。
  const charts = [];
  window.echarts = {
    init(el) {
      const inst = {
        __el: el && el.id, __opt: null, disposed: false,
        setOption(o) { this.__opt = o; charts.push(this); return this; },
        resize() { }, isDisposed() { return this.disposed; }, dispose() { },
      };
      return inst;
    },
  };

  // 手动注入脚本（jsdom 的 resources:'usable' 对本地 http 有时不可靠）
  for (const src of ['/static/app.js']) {
    const code = await (await fetch(BASE + src)).text();
    try {
      window.eval(code);
    } catch (e) {
      errors.push('[eval ' + src + '] ' + (e.stack || e).slice(0, 400));
    }
  }
  await sleep(1500);

  const $ = id => doc.getElementById(id);
  const txt = id => ($(id) ? $(id).textContent.trim() : '<missing>');

  let fails = 0;
  const chk = (cond, msg) => { console.log((cond ? '  OK   ' : '  FAIL ') + msg); if (!cond) fails++; return cond; };

  console.log('\n[1] 初始化');
  chk($('empty') !== null, '空状态节点存在');
  chk($('empty').hidden === false, '初始显示空状态');
  chk($('presetSel').options.length > 5, `示例下拉 ${$('presetSel').options.length} 项`);
  chk($('emptyChips').children.length > 3, `示例 chip ${$('emptyChips').children.length} 个`);
  chk(txt('runHint').includes('模型'), '运行提示已生成: ' + txt('runHint'));

  console.log('\n[2] 提交分析任务');
  $('codeInput').value = CODE;
  const form = $('searchForm');
  const ev = new window.Event('submit', { bubbles: true, cancelable: true });
  form.dispatchEvent(ev);
  await sleep(300);
  // 进度条可能一闪而过（缓存命中时秒回），因此只要"曾经显示过"或"已完成"即算通过
  const sawProgress = $('progressWrap').hidden === false ||
                      $('pStage').textContent !== '准备中…';
  chk(sawProgress, '任务已提交（进度条出现或直接完成）');
  console.log('    阶段:', txt('pStage'), '|', txt('pPct'));

  // 轮询等待
  let done = false;
  for (let i = 0; i < 120; i++) {
    await sleep(1000);
    if ($('progressWrap').hidden && $('main').hidden === false) { done = true; break; }
    if (i % 8 === 0) console.log('    ', txt('pStage'), txt('pPct'));
  }
  chk(done, '分析完成并渲染主区域');
  if (!done) { report(); process.exit(1); }

  console.log('\n[3] 头部与 KPI');
  console.log('    基金:', txt('fundName'));
  console.log('    meta:', txt('fundMeta').slice(0, 110));
  chk(txt('fundName').includes(CODE), '基金名含代码');
  chk($('kpis').children.length >= 5, `KPI 卡 ${$('kpis').children.length} 个`);
  const kpiTxt = $('kpis').textContent;
  chk(/R²|净值|回撤/.test(kpiTxt), 'KPI 含关键指标');
  chk(!/NaN|undefined/.test(kpiTxt), 'KPI 无 NaN/undefined');

  console.log('\n[4] 结论卡');
  chk($('insights').children.length >= 3, `结论卡 ${$('insights').children.length} 张`);
  const insTxt = $('insights').textContent;
  chk(insTxt.includes('风格穿透'), '含风格穿透结论');
  chk(insTxt.includes('过拟合'), '含过拟合检查');
  chk(!/NaN|undefined/.test(insTxt), '结论无 NaN/undefined');

  console.log('\n[5] 模型条与排名表');
  const cards = $('modelStrip').children;
  chk(cards.length >= 3, `模型卡 ${cards.length} 个`);
  chk($('rankTable').querySelectorAll('tbody tr').length >= 3,
    `排名表 ${$('rankTable').querySelectorAll('tbody tr').length} 行`);
  const rankTxt = $('rankTable').textContent;
  chk(!/NaN|undefined/.test(rankTxt), '排名表无 NaN');
  chk(/0\.\d{3}/.test(rankTxt), '排名表含 R² 数值');
  console.log('    ' + rankTxt.replace(/\s+/g, ' ').slice(0, 240));

  console.log('\n[6] 模型详情');
  console.log('    选中:', txt('selName'), '|', txt('selFamily'), '|', txt('selInterp'));
  chk(txt('selName') !== '—', '选中模型名已填充');
  chk($('selKpis').children.length >= 5, `详情 KPI ${$('selKpis').children.length} 个`);
  chk(txt('selDesc').length > 15, '模型描述已填充');

  console.log('\n[7] 超参表单');
  const fields = $('tunerBody').querySelectorAll('[data-pk]');
  chk(fields.length >= 2, `超参输入 ${fields.length} 个`);
  const tunerTxt = $('tunerBody').textContent;
  chk(/默认/.test(tunerTxt), '超参显示默认值');

  console.log('\n[8] 图表实例');
  console.log('    已创建图表:', charts.length);
  chk(charts.length >= 10, `图表 ${charts.length} 个（应≥10）`);
  const ids = charts.map(c => c.__el);
  console.log('    ', ids.join(', '));
  ['chartCurve', 'chartResid', 'chartWHist', 'chartWeights',
   'chartNav', 'chartDD', 'chartVol', 'chartMonthly',
   'chartCorrBar', 'chartLayer', 'chartMatrix', 'chartRollCorr'].forEach(id => {
    chk(ids.includes(id), `图表 ${id} 已渲染`);
  });
  // 抽查一个 option 的内容
  const cv = charts.find(c => c.__el === 'chartCurve');
  if (cv && cv.__opt) {
    const s = cv.__opt.series || [];
    console.log('    chartCurve series:', s.map(x => `${x.name}(${(x.data || []).length}点)`).join(', '));
    chk(s.length === 2 && s[0].data.length > 100, '拟合曲线有 2 条序列且数据充足');
  }

  console.log('\n[9] 相关性区块');
  const corrIds = ids.filter(i => /chart(Corr|Layer|Matrix|RollCorr)/.test(i));
  chk(corrIds.length >= 4, `相关性图表 ${corrIds.length} 个`);
  chk($('driftNote').children.length >= 1, `漂移提示 ${$('driftNote').children.length} 条`);

  console.log('\n[10] 切换模型（点击第 2 张卡）');
  const before = txt('selName');
  if (cards[1]) {
    cards[1].dispatchEvent(new window.Event('click', { bubbles: true }));
    await sleep(400);
    const after = txt('selName');
    chk(after !== before, `切换后详情更新: ${before} → ${after}`);
  } else { chk(false, '没有可点击的第二张卡'); }

  console.log('\n[11] 点击排名表行');
  const row = $('rankTable').querySelector('tbody tr');
  if (row) {
    row.dispatchEvent(new window.Event('click', { bubbles: true }));
    await sleep(300);
    chk(txt('selName') !== '—', '表格点击生效: ' + txt('selName'));
  } else chk(false, '排名表无行');

  console.log('\n[12] 设为最佳模型');
  const bestBefore = ($('rankTable').textContent.match(/最佳/) || []).length;
  $('btnSaveSel').dispatchEvent(new window.Event('click', { bubbles: true }));
  await sleep(400);
  const bestAfter = ($('rankTable').textContent.match(/最佳/) || []).length;
  chk(bestAfter >= 1, `最佳模型标记存在（之前 ${bestBefore} 处）`);
  chk(!$('alert').hidden, '显示切换提示');

  console.log('\n[13] 改超参并重新学习（RF 换成更浅的树）');
  // 切到 RF，确保是可调超参的黑箱模型
  const rfCard = [...$('modelStrip').children].find(c =>
    /随机森林/.test(c.textContent));
  if (rfCard) {
    rfCard.dispatchEvent(new window.Event('click', { bubbles: true }));
    await sleep(400);
    chk(/随机森林/.test(txt('selName')), '已切到随机森林');
    chk($('selInterp').textContent.includes('重要性'), 'RF 标记为不可解释');

    const before = [...$('rankTable').querySelectorAll('tbody tr')]
      .find(tr => /随机森林/.test(tr.textContent));
    const r2Before = before ? before.textContent.match(/0\.\d{4}/g) : null;

    // 改 max_depth=2（更浅 → 更欠拟合），验证重学习确实改变结果
    const depthInput = $('tunerBody').querySelector('[data-pk="max_depth"]');
    chk(!!depthInput, '找到 max_depth 输入框');
    if (depthInput) {
      depthInput.value = '2';
      $('btnApplyParams').dispatchEvent(new window.Event('click', { bubbles: true }));
      let ok = false;
      for (let i = 0; i < 60; i++) {
        await sleep(1000);
        if ($('progressWrap').hidden) { ok = true; break; }
      }
      chk(ok, '重新学习完成');
      const after = [...$('rankTable').querySelectorAll('tbody tr')]
        .find(tr => /随机森林/.test(tr.textContent));
      const r2After = after ? after.textContent.match(/0\.\d{4}/g) : null;
      console.log('    RF 重新学习前:', r2Before && r2Before[1],
                  '→ 后:', r2After && r2After[1]);
      chk(JSON.stringify(r2Before) !== JSON.stringify(r2After),
          '超参改变后 R² 发生变化');
      chk($('alert').textContent.includes('重新学习'), '提示已更新: ' +
          $('alert').textContent.slice(0, 90));
    }
  } else chk(false, '未找到随机森林模型卡');

  console.log('\n[14] 恢复默认超参');
  $('btnResetParams').dispatchEvent(new window.Event('click', { bubbles: true }));
  await sleep(300);
  const dv = $('tunerBody').querySelector('[data-pk="max_depth"]');
  chk(dv && dv.value === '6', 'max_depth 已恢复默认 6');

  console.log('\n' + '='.repeat(60));
  console.log(`失败项：${fails}`);
  if (errors.length) {
    console.log('\n运行时错误:');
    [...new Set(errors)].slice(0, 12).forEach(e => console.log('  ' + e));
  } else console.log('无运行时错误');
  if (warns.length) {
    console.log(`\njsdom 限制提示 ${warns.length} 条（真浏览器无此问题）`);
  }
  console.log('='.repeat(60));
  process.exit(fails || errors.length ? 1 : 0);

  function report() {
    console.log('\n运行时错误:');
    [...new Set(errors)].slice(0, 10).forEach(e => console.log('  ' + e));
    process.exit(1);
  }
})().catch(e => { console.error('测试脚本异常:', e); process.exit(1); });