/**
 * 前端渲染验证：用 jsdom 加载真实页面 + 真实 API 数据，
 * 捕获所有运行时错误与未捕获 Promise，并断言关键 DOM 是否被正确填充。
 *
 * 用法： node tests/dom_check.js http://127.0.0.1:8760
 * 依赖： npm install --no-save jsdom   （仓库没有 package.json，按需临时安装即可）
 *        node_modules/ 已在 .gitignore 中，不会污染仓库。
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
      // ★ app.js 首行是 'use strict'，严格模式下**间接 eval 的 const 不会
      // 泄漏到全局作用域**，所以测试里 `window.eval('METRICS')` 会 ReferenceError。
      // 真浏览器里 <script> 是全局脚本、不存在这个问题。
      // 这里追加一行把词典显式导出，供测试做静态一致性检查。
      window.eval(code + '\n;window.__METRICS = METRICS;window.__S = S;');
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

  console.log('\n[15] 指标提示（tooltip）');
  chk($('mtip') !== null, '浮层节点 #mtip 已注入');
  const mhs = [...doc.querySelectorAll('.mh')];
  chk(mhs.length >= 25, `带解释的指标标签 ${mhs.length} 个（应≥25）`);

  // ★ 静态一致性：每个 data-m 都必须在 METRICS 里有定义。
  // 否则页面上会出现"有下划线但悬停无反应"的死标签 ——
  // 这类问题不会报错，只会静默地少给用户一段解释。
  const dict = window.__METRICS || {};
  const keys = Object.keys(dict);
  chk(keys.length >= 35, `METRICS 词典 ${keys.length} 条`);
  const dead = [...new Set(mhs.map(e => e.dataset.m))].filter(k => !keys.includes(k));
  chk(dead.length === 0, dead.length
    ? `悬空提示（METRICS 缺定义）：${dead.join(', ')}`
    : '无悬空提示，所有 data-m 均有词典条目');

  // 三段结构完整性
  const full = Object.values(dict)
    .filter(v => v && v.what && v.how && v.read).length;
  chk(full === keys.length, `每条均含 是什么/怎么算/怎么看 三段（${full}/${keys.length}）`);
  const thin = Object.entries(dict)
    .filter(([, v]) => typeof v.read === 'string' && v.read.length < 40)
    .map(([k]) => k);
  chk(thin.length === 0, thin.length
    ? `「怎么看」过短（<40 字，可能没写判据）：${thin.join(', ')}`
    : '「怎么看」均给出了可操作判据，无敷衍条目');

  // 触发行为：mouseover → 显示且有内容；移开 → 隐藏
  const probe = mhs.find(e => e.dataset.m === 'sharpe') || mhs[0];
  probe.dispatchEvent(new window.MouseEvent('mouseover', { bubbles: true }));
  await sleep(60);
  const tip = $('mtip');
  chk(tip.classList.contains('on'), `悬停 ${probe.dataset.m} → 浮层显示`);
  chk(/是什么/.test(tip.textContent) && /怎么看/.test(tip.textContent),
    '浮层含「是什么 / 怎么看」两段');
  console.log('    ' + tip.textContent.replace(/\s+/g, ' ').slice(0, 150));

  const other = mhs.find(e => e.dataset.m !== probe.dataset.m) || mhs[1];
  other.dispatchEvent(new window.MouseEvent('mouseover', { bubbles: true }));
  await sleep(60);
  chk(tip.textContent !== '' && tip.classList.contains('on'),
    `移到另一个标签后内容已切换（${other.dataset.m}）`);

  doc.body.dispatchEvent(new window.MouseEvent('mouseover', { bubbles: true }));
  await sleep(60);
  chk(!tip.classList.contains('on'), '移开 → 浮层隐藏');

  // 回测区（跑完才有数据）的指标标签也应带提示
  $('btRun').dispatchEvent(new window.Event('click', { bubbles: true }));
  let btDone = false;
  for (let i = 0; i < 60; i++) {
    await sleep(1000);
    if (!$('btOut').hidden) { btDone = true; break; }
  }
  chk(btDone, '回测运行完成');
  if (btDone) {
    const btMh = [...$('btOut').querySelectorAll('.mh')].map(e => e.dataset.m);
    chk(btMh.length >= 8, `回测区指标标签 ${btMh.length} 个`);
    chk($('btKpis').querySelectorAll('.mh').length >= 7,
      `回测 KPI 卡提示 ${$('btKpis').querySelectorAll('.mh').length} 个`);
    const btDead = [...new Set(btMh)].filter(k => !keys.includes(k));
    chk(btDead.length === 0, btDead.length
      ? `回测区悬空提示：${btDead.join(', ')}` : '回测区无悬空提示');
    // 健康分卡上的"择时贡献/对照"必须来自该行自身（不是主策略的）
    const kt = $('btKpis').textContent;
    chk(/对照/.test(kt) && /择时贡献/.test(kt), '回测 KPI 含对照与择时贡献');
    chk(!/NaN|undefined/.test(kt), '回测 KPI 无 NaN/undefined');
  }

  // 覆盖度：词典不应有从未被引用的死条目。
  // ★ 必须放在回测之后 —— 回测表的表头在跑完之前根本不存在，
  //   放在前面会把 monthly_win / cumulative 误判成"死条目"。
  const allMh = [...doc.querySelectorAll('.mh')];
  chk(allMh.length >= 35, `全页指标标签 ${allMh.length} 个（含回测区）`);
  const unused = keys.filter(k => !allMh.some(e => e.dataset.m === k));
  chk(unused.length === 0, unused.length
    ? `词典中未被任何标签引用的死条目：${unused.join(', ')}`
    : '词典条目全部被页面引用，无死条目');

  console.log('\n[16] 趋势状态（描述性）');
  const seg = $('trendSeg');
  chk(seg !== null && seg.children.length === 6,
    `回看窗口选择器 ${seg ? seg.children.length : 0} 档（应为 1/2/3/5/10/20）`);
  const onBtn = seg ? seg.querySelector('button.on') : null;
  chk(onBtn && onBtn.textContent === '10', `默认档位 = ${onBtn ? onBtn.textContent : '—'}`);

  // 后端必须在**日频**序列上判定，再按绘图下标采样下发。
  // 若长度对不上，说明判定是在降采样序列上做的 —— "回看 10 日"会变成
  // "回看 10 个采样点 ≈ 40 个交易日"，语义失真且不会报错。
  const S = window.__S;
  chk(!!S && !!S.result, '应用状态 S 可访问（测试注入）');
  let trendOk = true, trendMsg = '';
  for (const k of Object.keys((S && S.result && S.result.models) || {})) {
    const mm = S.result.models[k];
    const tr = (mm.curve || {}).trend;
    if (!tr || !tr.codes || !tr.stats) { trendOk = false; trendMsg = `${k} 缺 trend`; break; }
    for (const n of tr.levels) {
      const c = tr.codes[String(n)];
      if (!c || c.length !== mm.curve.dates.length) {
        trendOk = false;
        trendMsg = `${k} N=${n} codes ${c ? c.length : 0} ≠ dates ${mm.curve.dates.length}`;
      }
    }
    if (!trendOk) break;
  }
  chk(trendOk, trendOk ? '每个模型的 trend.codes 长度 = 绘图点数（判定在日频上做）'
    : 'codes 与绘图点数不一致：' + trendMsg);

  const card = $('trendCard');
  chk(card && card.textContent.length > 40, '状态卡已渲染');
  const cardTxt = card ? card.textContent : '';
  chk(/上升趋势|下降趋势|震荡|拐头/.test(cardTxt), '状态卡含状态标签: ' +
    (card ? (card.querySelector('.tc-state') || {}).textContent : ''));
  chk(card.querySelectorAll('.tc-grid > div').length === 5,
    `判据明细 ${card.querySelectorAll('.tc-grid > div').length} 项（应为 5）`);
  chk(/体制态 \d+ 段/.test(cardTxt) && /拐头 \d+ 次/.test(cardTxt),
    '状态卡含历史可靠性统计（体制态段数 + 拐头次数）');
  chk(!/NaN|undefined|Infinity/.test(cardTxt), '状态卡无 NaN/undefined');

  // 色带（体制态）与拐头点（瞬时标记）必须都画进图表
  const cvT = charts.filter(c => c.__el === 'chartCurve').pop();
  const synthSeries = cvT && cvT.__opt
    ? (cv.__opt.series || []).find(s => s.name === '合成基准') : null;
  chk(!!synthSeries, '合成基准序列存在');
  if (synthSeries) {
    const areas = ((synthSeries.markArea || {}).data || []).length;
    const pts = ((synthSeries.markPoint || {}).data || []).length;
    chk(areas > 0, `历史状态色带 ${areas} 段`);
    chk(pts > 0, `拐头标记 ${pts} 个`);
  }

  // 切换回看窗口：状态卡与色带都要跟着变
  const beforeTxt = card.querySelector('.tc-state').textContent;
  const lastBtn = seg.children[seg.children.length - 1];
  lastBtn.dispatchEvent(new window.Event('click', { bubbles: true }));
  await sleep(400);
  chk(/回看 20 个交易日/.test($('trendCard').textContent),
    '切到 N=20 后状态卡窗口已更新');
  chk(seg.querySelector('button.on').textContent === '20', '选择器高亮已跟随');
  const cv2 = charts.filter(c => c.__el === 'chartCurve').pop();
  const s2 = cv2 && cv2.__opt
    ? (cv2.__opt.series || []).find(s => s.name === '合成基准') : null;
  const areas2 = s2 ? ((s2.markArea || {}).data || []).length : 0;
  chk(areas2 > 0, `切换后色带已重画（${areas2} 段，状态 ${beforeTxt}）`);

  // 定位声明：必须写明"不是买卖建议"
  const noteTxt = $('main').textContent;
  chk(noteTxt.includes('不是买卖建议'), '页面含「不是买卖建议」的定位声明');
  chk(/RankIC[\s\S]{0,40}−0\.073|RankIC[\s\S]{0,40}-0\.073/.test(noteTxt),
    '声明中给出了日频择时证伪的依据（RankIC −0.073）');

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