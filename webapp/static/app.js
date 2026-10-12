/* Fund Lens · 前端逻辑 */
'use strict';

// ==================================================================== 全局状态
const S = {
  cat: null,          // 目录（模型清单、已知基金）
  result: null,       // 当前分析结果
  sel: null,          // 选中的模型 key
  best: null,         // 当前"最佳"模型 key
  params: {},         // 每个模型的超参覆盖
  rollWin: 120,
  lookback: 10,       // 趋势状态的回看窗口（交易日，1~20）
  charts: {},         // echarts 实例缓存
};

const $ = (id) => document.getElementById(id);
const fmtPct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v * 100).toFixed(d) + '%';
const fmtNum = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toFixed(d);
const fmtSign = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(d);
const fmtPctSign = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';

// A股习惯：涨红跌绿
const UP = '#ef4a4a', DOWN = '#12b886', FLAT = '#8b98ab';
const colOf = (v) => v > 0 ? UP : v < 0 ? DOWN : FLAT;

// ============================================================ 指标词典
// 每个指标给三段：是什么 / 怎么算 / 怎么看。
// 写作原则：用大白话，不拿术语解释术语；「怎么看」必须给出可操作的判据
//           （多大算大、方向如何、什么情况下会误导）。
const METRICS = {
  // ---------------- 净值与风险 ----------------
  latest_nav: {
    what: '复权后的累计净值，把历次分红再投资也算进去了。',
    how: '用基金公布的<b>单位净值</b>与<b>累计净值</b>还原分红，再累乘得到；'
      + '这样不同基金之间才可比。',
    read: '只比较绝对高低没有意义（不同基金成立时点不同），'
      + '要看它的<b>走势斜率</b>和<b>回撤幅度</b>。',
  },
  total_return: {
    what: '整个统计区间内一共涨了多少（含分红再投资）。',
    how: '<code>末值 / 初值 − 1</code>。',
    read: '区间越长越容易“看起来很大”。2016 年至今的 10 年区间里，'
      + '翻倍并不罕见，别把它当成“年化 100%”。',
  },
  ann_return: {
    what: '把区间总收益折算成“平均每年赚多少”，方便跨基金、跨区间比较。',
    how: '<code>(1 + 累计收益)^(252 / 交易日数) − 1</code>，'
      + '按每年 252 个交易日折算。',
    read: '这是<b>历史回溯</b>，不是预期收益。A 股主动权益基金'
      + '在 2016 年以来的区间里，年化 10%~25% 都常见；'
      + '但同一只基金换个起点就能差很多，<b>务必结合最大回撤一起看</b>。',
  },
  ann_vol: {
    what: '收益的波动程度，也就是“净值上下颠簸得有多厉害”。',
    how: '日收益的<b>标准差 × √252</b>（年化）。',
    read: 'A 股主动权益基金通常 18%~28%。'
      + '它衡量的是<b>颠簸</b>不是<b>亏损</b>，但颠簸大的基金在下跌段通常也更狠。',
  },
  vol60: {
    what: '最近 60 个交易日（约 3 个月）的年化波动。',
    how: '最近 60 日日收益标准差 × √252。',
    read: '用来判断<b>当下</b>的风险水平是否比长期平均更高。'
      + '明显高于历史年化波动时，说明市场正在剧烈变化，'
      + '此时任何模型（包括本页的拟合）都会更不可靠。',
  },
  max_dd: {
    what: '从历史最高点跌到最低点的最大幅度，衡量“最难受的时候有多难受”。',
    how: '逐日计算 <code>当前净值 / 历史最高净值 − 1</code>，取最小值。',
    read: '<b>这是持有体验的核心指标</b>，比年化收益更重要。'
      + '主动权益基金 40%~60% 的回撤并不罕见。'
      + '如果一次 50% 的回撤就会让你中途赎回，那这只基金就不适合你，'
      + '无论它的年化多好看。',
  },
  monthly_win: {
    what: '月度收益为正的月份占比。',
    how: '统计所有完整自然月，算涨的月份比例。',
    read: '长期看 55%~65% 属于正常。它和夏普相关但不等价：'
      + '夏普还考虑了“赢的时候赢多少”。',
  },
  cumulative: {
    what: '某个方案在回测区间内累计的收益（含交易成本）。',
    how: '把每日净收益连乘起来。',
    read: '和年化收益是同一件事的两种表达。'
      + '看方案对比时，<b>年化更直观、累计更适合看绝对金额</b>。',
  },

  // ---------------- 拟合质量 ----------------
  oos_r2: {
    what: '样本外 R²：用“上一段时间学到的指数组合”去解释“接下来一段”的基金收益，'
      + '能解释掉多少比例的波动。',
    how: '滚动窗口 252 日拟合一次，权重冻结；'
      + '下一段用这组权重合成基准，算 <code>1 − 残差方差 / 基金方差</code>。',
    read: '<b>这是本页所有排名的唯一依据。</b>'
      + '0.9 以上说明风格清晰且稳定；'
      + '0.6~0.8 属于风格不鲜明或正在变化；'
      + '低于 0.5 通常意味着<b>仓位假设失配</b>（见「隐含股票仓位」）'
      + '而不是"没有风格"，应先看那条提示再下结论。',
  },
  fit_r2: {
    what: '窗口内 R²：模型在它“自己见过”的那些数据上的拟合度。',
    how: '每个滚动窗口内直接算 <code>1 − 残差方差 / 基金方差</code>。',
    read: '<b>高不代表能预测</b>。树模型、核方法、神经网络这类黑箱'
      + '天然能把这个数做到 1.00，那只是“背下了历史”。'
      + '所以它不能用来排名，只能用来和样本外 R² 对比。',
  },
  gap: {
    what: '过拟合程度：窗口内 R² 比样本外 R² 高多少。',
    how: '<code>窗口内 R² − 样本外 R²</code>。',
    read: '经验判据：<b>&gt; 0.08 就算过拟合明显</b>，'
      + '说明模型学到的那些细节换到新数据就不成立了。'
      + '黑箱模型这个差值通常远大于线性约束类，'
      + '这也是本页优先推荐可解释模型的理由。',
  },
  oos_te: {
    what: '年化跟踪误差：基金实际收益与合成基准收益之间偏离的幅度。',
    how: '<code>日残差的标准差 × √252</code>（样本外口径）。',
    read: '和 R² 是同一件事的两面：<b>TE 越小 = 基金越像那个指数组合</b>。'
      + '主动基金一般 4%~10%；'
      + '超过 12% 说明它要么风格很不稳定，要么仓位假设不对。',
  },
  win: {
    what: '方向一致率：基金和合成基准当天“同涨同跌”的比例。',
    how: '统计 <code>sign(基金日收益) == sign(基准日收益)</code> 的天数占比。',
    read: '这是比 R² 更“粗糙”但更直观的指标，正常在 60%~75%。'
      + '它高而 R² 低，通常说明<b>方向对但幅度不对</b>（仓位差）；'
      + '两者都低才是真的没拟合上。',
  },
  beta: {
    what: '隐含股票仓位：模型反推出的“这只基金有多少钱真的买了股票”。',
    how: '不做固定 90% 的假设，而是让模型自己去估计'
      + '“基金总波动 / 基准波动”这个比例（即隐含 β）。'
      + '只在【仓位约束自适应】开启时生效。',
    read: '<b>这是全页最容易误判的一项。</b>'
      + '如果它显著低于 90%（例如 30%），'
      + '说明合成基准按满仓构建、而基金其实大半是债或现金，'
      + '于是<b>低 R² 反映的是仓位差异，不是风格识别失败</b>。'
      + '遇到这种情况请把「仓位约束自适应」打开重跑。',
  },
  vol_ratio: {
    what: '合成基准的波动是基金自身波动的几倍。',
    how: '<code>基准年化波动 / 基金年化波动</code>。',
    read: '正常接近 1。远大于 1（比如 2 倍以上）'
      + '说明基准“太猛”了，通常就是仓位假设过高造成的，'
      + '此时 R² 会被系统性压低。',
  },
  n_weights: {
    what: '有效权重数：这只基金实际被归因到了几个指数上。',
    how: '统计权重超过阈值（如 1%）的指数个数。',
    read: '3~6 个最可读；'
      + '几十个说明权重被摊薄成了噪声，'
      + '这时“食品饮料 59% + 家用电器 24%”那种清晰结论就不成立了。',
  },
  elapsed: {
    what: '该模型完成全部滚动拟合所用的时间（秒）。',
    how: '实测墙钟时间。',
    read: '只影响使用体验。'
      + '但它常常和复杂度相关：<b>耗时高、样本外 R² 却不高的模型，'
      + '性价比就是负的</b>。',
  },
  interpretable: {
    what: '该模型的权重能不能当成“真实可买的持仓组合”来读。',
    how: '线性约束类（弹性网 / Lasso / Ridge / TopK / PCR）可解释；'
      + '树 / 核 / 神经网络类不可解释。',
    read: '<b>这是选择模型时最该优先看的一列。</b>'
      + '黑箱模型输出的数字只是“重要性排序”，'
      + '不能理解成“这只基金买了 30% 白酒”，照着它调仓会出错。',
  },

  // ---------------- 回测评价 ----------------
  sharpe: {
    what: '夏普比率：每承担一单位波动能换来多少超额收益。',
    how: '<code>(几何年化收益 − 无风险利率) / 年化波动</code>。',
    read: '长期 0.5 以上算不错，1.0 以上很好。'
      + '但它<b>对“低仓位”有天然偏爱</b>——仓位低、波动小，夏普就好看。'
      + '所以本页必须配「恒定仓位对照」一起看，'
      + '单看夏普会被仓位差异骗到。'
      + '<br><b>注意口径</b>：这里用的是<b>几何</b>年化收益。'
      + '业界默认的 Sharpe 用<b>算术</b>年化收益，见「算术夏普」。',
  },
  sharpe_arith: {
    what: '算术夏普：用<b>算术</b>年化收益算的夏普，业界默认口径。',
    how: '<code>(算术年化收益 − 无风险利率) / 年化波动</code>，'
      + '即 <code>mean(日超额) × 252 / (std × √252)</code>。',
    read: '与「信息比率 IR」可直接对照（IR 的分母是<b>超额</b>波动）。'
      + '波动越大，它与几何夏普差得越多；两者接近说明路径平滑。',
  },
  win_rate: {
    what: '多头胜率：给出做多信号的那些天里，未来真的上涨的比例。',
    how: '分母只计<b>有完整前瞻收益</b>的天数。'
      + '窗口尾部没有未来数据的天数被剔除，不会算成"猜错"。',
    read: '<b>必须连"有效标签数"一起看。</b>若有效标签远少于样本天数，'
      + '这个胜率的置信度很低，不能当作"预测准确率"。',
  },
  calmar: {
    what: 'Calmar 比率：年化收益 ÷ 最大回撤。',
    how: '<code>年化收益 / |最大回撤|</code>。',
    read: '它比夏普更贴近持有体验，因为它惩罚的是<b>最难受的那一段</b>'
      + '而不是平均颠簸。0.5 以上算健康。',
  },
  avg_pos: {
    what: '该方案在回测区间内的平均持仓比例（时间加权）。',
    how: '对每日仓位取时间平均，不是“买卖次数平均”。',
    read: '<b>读所有超额指标前必须先看这个数。</b>'
      + '平均仓位 40% 的策略天然比 100% 的少赚、少跌，'
      + '直接比收益毫无意义。'
      + '这正是本页为每个方案单独配一个「恒定仓位对照」的原因。',
  },
  turnover: {
    what: '年化换手：平均每年要买卖相当于整个持仓多少倍的量。',
    how: '把每日仓位变动绝对值累加，再除以年数。'
      + '买一次卖一次（0%→100%→0%）计为 2。',
    read: '<b>换手是交易成本敏感度的直接放大器</b>'
      + '（成本影响 ≈ 换手 × 单边费率）。'
      + '年换手 0.3 意味着三年才动一次，几乎不受费率影响；'
      + '年换手 1.4 意味着成本从 0.15% 抬到 1.5% 时'
      + '净贡献会多掉约 1.9 个百分点。'
      + '场外基金真实申赎费率是场内 ETF 的十倍，'
      + '所以高换手策略基本只能在场内执行。',
  },
  rank_ic: {
    what: 'RankIC：当日信号排名与未来收益排名的相关性。',
    how: '每日算信号值与未来 5 日收益的<b>秩相关</b>，再取时间平均。',
    read: '正数 = 信号方向对（信号高→未来涨得多）。'
      + '日频择时里 |RankIC| 能稳定到 0.03 已属不易。'
      + '<b>本项目 14 只基金实测均值为负</b>，'
      + '所以这个数在这里主要起“证伪”作用。',
  },
  icir: {
    what: 'ICIR：RankIC 的稳定性，即“信号方向对得有多一贯”。',
    how: '<code>RankIC 均值 / RankIC 标准差</code>。',
    read: '比单看 RankIC 更严格：均值高但不稳定（ICIR 低）'
      + '意味着信号只在少数时段有效，'
      + '<b>这种策略在实盘上会赚赚赔赔然后被手续费磨平</b>。',
  },
  timing_contrib: {
    what: '择时净贡献：该方案相对它<b>自己那份</b>恒定仓位对照的超额。',
    how: '<code>方案指标 − 同平均仓位恒定仓位方案的指标</code>。',
    read: '<b>这是本项目唯一认可的“择时能力”度量。</b>'
      + '为什么必须各自配对：平均仓位 100% 的方案减去平均 43% 的对照，'
      + '会凭空得到 +12% 的“超额年化”，那纯粹是仓位差。'
      + '只有平均仓位一致时，差值才代表择时。'
      + '接近 0 = 择时没有价值；为正才说明真的择对了。',
  },
  control: {
    what: '恒定仓位对照：一个“死拿着不动”的基准方案。',
    how: '仓位固定等于被测方案的时间平均仓位，'
      + '只在建仓时交易一次，之后永不调仓。',
    read: '它的作用是回答：<b>“如果我只是保持同样的平均仓位、'
      + '完全不做择时，结果会怎样？”</b>'
      + '被测方案跑不赢它，就说明所有交易都是负价值。',
  },
  health_score: {
    what: '健康分：把四个拟合诊断压缩成的一个 −1 ~ +1 的分数，'
      + '用来判断“这只基金的风格模型还准不准”。',
    how: '四个诊断分项（R² 相对下降、TE 扩张、alpha 的 t 值、权重漂移速度）'
      + '各映射到 [−1, +1]，再按 <code>最差项 × 0.6 + 整体均值 × 0.4</code> 合成。',
    read: '<b>接近 +1 = 一切正常，无需动作。</b>'
      + '之所以要“最差项占大头”，是因为四个分量描述的是'
      + '互相独立的失效模式，任何一个严重恶化都值得警惕；'
      + '而“整体均值”那一项是为了防止被最吵的单个分量牵着走——'
      + '<b>单项抖动不足以动手，需要整体也在恶化</b>。',
  },
  exit_line: {
    what: '离场线：健康分跌破它并<b>持续 10 个交易日</b>才真正卖出。',
    how: '默认 −0.5，事先固定，不依据回测结果调整。',
    read: '要求“连续 10 日”是为了滤掉单日尖峰（例如某天 TE 异常跳升）。'
      + '这个阈值不是调优出来的，而是按<b>各分量自身的噪声幅度</b>标定的，'
      + '所以不应再拿回测表现去回调它。',
  },
  enter_line: {
    what: '买入线：健康分回升到它之上并持续，才重新买入。',
    how: '默认 −0.2。',
    read: '它与离场线之间的 −0.5 ~ −0.2 叫<b>迟滞带</b>，'
      + '作用是防止健康分在阈值附近来回穿梭时'
      + '把仓位反复买卖、被手续费磨死。',
  },
  c_r2: {
    what: '拟合优度的恶化程度：当前的 R² 比它自己两年来的常态低多少。',
    how: '<code>(当前R² − 两年中位R²) / 20pp</code>，截断到 [−1, +1]。',
    read: '用<b>相对自身</b>而不是绝对水位，'
      + '是因为不同基金的 R² 天生差很多'
      + '（消费类 0.95、轮动型 0.6）。'
      + '用绝对阈值判“不健康”，等于永久判低 R² 的基金出局。',
  },
  c_te: {
    what: '跟踪误差的扩张倍数：现在的 TE 是常态的几倍。',
    how: '<code>(1.6 − 当前TE/两年中位TE) / 0.6</code>，截断到 [−1, +1]。',
    read: 'TE 扩张说明<b>基金的风险结构变了</b>'
      + '（可能是风格切换、也可能是基金经理在赌方向）。'
      + '扩张 60% 归零、扩张 120% 触底。',
  },
  c_alpha: {
    what: '实现 alpha 的显著性：基金相对<b>自己的复制组合</b>跑赢还是跑输，'
      + '这个偏离在统计上有多可信。',
    how: '先算超额收益的 t 值（2 年滚动窗口），'
      + '再 <code>t / 3</code> 截断到 [−1, +1]。',
    read: '这是四个分量里<b>经济含义最清晰</b>的一个：'
      + '判断的是“基金经理的选股能力是否真的在衰减”。'
      + '之所以用 t 值而不是原始 alpha，'
      + '是因为原始 alpha 的噪声比它本身还大得多。'
      + '<code>t ≤ −3</code>（约年化跑输 13%）才算严重钝化。',
  },
  c_drift: {
    what: '风格漂移速度：模型权重相对上一次重估变了多少。',
    how: '<code>Σ|Δw|</code> 再取 3 日平滑，'
      + '与自身两年常态比较，换算到 [−1, +1]。',
    read: '漂移剧烈说明<b>风格不稳定</b>——'
      + '今天消费明天科技，那么“风格穿透”的结论本身就靠不住了。'
      + '注意 20 日重估的基金天生漂移更大，所以同样必须用相对口径。',
  },

  // ---------------- 趋势状态（描述性） ----------------
  trend_state: {
    what: '按「高低点结构」给合成基准贴的形态标签，'
      + '只有五档：上升、下降、震荡、拐头向上、拐头向下。',
    how: '取最近 N 个交易日为一组、紧邻的再前 N 日为一组，比较两组的极值：'
      + '<b>高点抬高且低点抬高 → 上升</b>；<b>高点降低且低点降低 → 下降</b>；'
      + '其余 → 震荡。拐头另需四个条件同时成立（见「回看窗口」）。',
    read: '<b>这是形态描述，不是买卖建议、不含任何持仓含义。</b>'
      + '本项目已用 14 只基金做样本外检验证明：日频方向信号无法超越恒定仓位对照'
      + '（RankIC 均值 −0.073，13/14 只为负）。'
      + '它唯一的用途是让你"用一句话说出这段走势长什么样"。'
      + '实测该判据并非纯噪声：真实序列的方向性标签占比比打乱时间顺序后高 '
      + '<b>25~28 个百分点</b>，说明它确实抓到了短期结构。',
  },
  trend_lookback: {
    what: '回看窗口 N：判定时用最近多少个交易日的数据。'
      + '本页可选 1 / 2 / 3 / 5 / 10 / 20 个交易日。',
    how: 'N 决定两组窗口各取多少天。N=1 时只比较今昨两日收盘；'
      + 'N=20 时比较最近 20 日与之前 20 日的高低点。',
    read: '<b>N 越大，状态越稳定，但也越迟钝。</b>'
      + '实测（4 只基金合成基准，2368 个交易日）体制态的平均段长：'
      + '<b>N=5 → 3.4 天，N=10 → 5.3 天，N=20 → 7.4 天</b>。'
      + 'N=1 会退化成"单日涨跌"——判不出震荡，也判不出拐头。'
      + '默认 10 日（约两个交易周），是能读出形态与灵敏度之间的折中。'
      + '状态卡里会实时显示当前 N 下的历史段长，可直接核对它是否可读。',
  },

  // ---------------- 稳健性统计 ----------------
  n_eff: {
    what: '有效独立样本量：N 只基金的“超额收益”在统计上相当于几个互不相关的观测。',
    how: '<code>(Σλ)² / Σλ²</code>，λ 是横截面相关矩阵的特征值。',
    read: '<b>本项目最核心的一条方法论。</b>'
      + '14 只基金如果超额高度同步（都靠同一个市场信号），'
      + 'n_eff 可能只有 2~3，那“14 只里 12 只为正”≈“3 只里 2.5 只为正”，'
      + '证据强度被严重高估。'
      + '实测：长周期趋势 n_eff=2.30（几乎无用），'
      + '健康门控 n_eff=8.80（真的接近 14 份证据）。'
      + '<b>看到任何“N 只为正”的结论，先问 n_eff 是多少。</b>',
  },
  boot_p: {
    what: '移动块 bootstrap 的 p 值：'
      + '“如果策略完全没有预测能力，出现当前这种结果的可能性有多大”。',
    how: '先把每只基金的超额序列减去均值（构造“无能力”的零假设），'
      + '再用 60 日移动块重采样 2000 次。'
      + '<b>所有基金共用同一组重采样日期</b>，'
      + '否则等于假设 14 个观测互不相关，恰好抹掉要检验的东西。',
    read: '<b>小于 0.05 才算显著。</b>'
      + '本项目实测：长周期趋势 0.147、波动率目标 0.432（都不显著），'
      + '健康门控 0.029（显著）。'
      + '注意 p 值只回答“是不是运气”，不回答“能赚多少钱”——'
      + '仍要结合效应量一起看。',
  },
  pc1: {
    what: '第一主成分解释方差：14 只基金的超额收益里，'
      + '有多少比例可以被<b>同一个共同因子</b>解释。',
    how: '对超额收益矩阵做 PCA，取最大特征值占总特征值的比例。',
    read: '越低越好。'
      + '超过 50% 说明这些基金的“超额”其实大部分是同一个信号，'
      + '“跨基金一致”就是假的一致。'
      + '实测：长周期趋势 63%（几乎全同步）、健康门控 16.5%（各自独立）。',
  },
  nw_t: {
    what: 'Newey-West t 值：单只基金的超额收益在统计上是否显著（已修正自相关）。',
    how: '对超额序列做 NW 自相关调整后算 t 值，|t| > 1.96 即 5% 水平显著。',
    read: '这是<b>最严格的逐基金检验</b>。'
      + '本项目三个策略都是 0/14 显著 —— '
      + '说明没有任何单只基金的超额站得住，'
      + '显著性（如果有）完全来自横截面聚合。'
      + '这一点必须如实报告，不能只报 p 值。',
  },
};

// ------------------------------------------------ 提示触发器
// 用法：mh('oos_r2', '样本外 R²') → 带下划线与 ? 图标、可悬停解释的标签
function mh(key, text) {
  if (!METRICS[key]) return text;
  return `<span class="mh" data-m="${key}">${text}<i class="mi">?</i></span>`;
}

// 全局浮层：hover 触发（桌面），click 触发（触屏）
(function initMetricTip() {
  const tip = document.createElement('div');
  tip.id = 'mtip';
  document.body.appendChild(tip);
  let cur = null;

  // 放置策略：优先正下方 → 正上方 → 右侧 → 左侧 → 强制钳位。
  // 之所以要「左右」这一档：解释文字较长（300~400px 高），
  // 在 600px 以下的窗口里上下都放不下，只钳到角落会正好盖住被悬停的标签，
  // 用户一边看解释一边看不到自己在问哪个指标。
  function place(el) {
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const vw = window.innerWidth, vh = window.innerHeight;
    const M = 8;
    const clampX = (x) => Math.max(M, Math.min(x, vw - tw - M));
    const clampY = (y) => Math.max(M, Math.min(y, vh - th - M));

    // 1) 正下方   2) 正上方（水平居中于标签）
    if (r.bottom + 9 + th <= vh - M) {
      tip.style.left = clampX(r.left + r.width / 2 - tw / 2) + 'px';
      tip.style.top = (r.bottom + 9) + 'px';
      return;
    }
    if (r.top - th - 9 >= M) {
      tip.style.left = clampX(r.left + r.width / 2 - tw / 2) + 'px';
      tip.style.top = (r.top - th - 9) + 'px';
      return;
    }
    // 3) 右侧   4) 左侧（垂直居中于标签）
    const yc = clampY(r.top + r.height / 2 - th / 2);
    if (r.right + 9 + tw <= vw - M) {
      tip.style.left = (r.right + 9) + 'px';
      tip.style.top = yc + 'px';
      return;
    }
    if (r.left - tw - 9 >= M) {
      tip.style.left = (r.left - tw - 9) + 'px';
      tip.style.top = yc + 'px';
      return;
    }
    // 5) 都放不下（窗口极小）：钳位兜底
    tip.style.left = clampX(r.left + r.width / 2 - tw / 2) + 'px';
    tip.style.top = clampY(r.bottom + 9) + 'px';
  }

  function show(el) {
    const m = METRICS[el.dataset.m];
    if (!m) return;
    // ★ 小标题必须用独立的类（.t-k），不能用 <b>：
    //   CSS 里若写成 `.t-row > b{display:block}`，会把正文里所有作强调用的
    //   <b> 一并变成块级元素，句子会被拆成一行一个词。
    tip.innerHTML = `<div class="t-hd">${m.title || el.textContent.trim()}</div>`
      + `<div class="t-row"><span class="t-k">是什么</span>${m.what}</div>`
      + (m.how ? `<div class="t-row"><span class="t-k">怎么算</span>${m.how}</div>` : '')
      + (m.read ? `<div class="t-row"><span class="t-k">怎么看</span>${m.read}</div>` : '');
    tip.classList.add('on');
    cur = el;
    place(el);
  }
  function hide() { tip.classList.remove('on'); cur = null; }

  // 单一 mouseover 处理器：进入 .mh 就显示，离开到别处就隐藏。
  // 比 mouseover + mouseout 配对更稳（子元素 i.mi 会造成边界抖动）。
  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest ? e.target.closest('.mh') : null;
    if (!el) { if (cur) hide(); return; }
    if (el !== cur) show(el);
  });
  // 触屏：点一下显示，点别处收起；滚轮/滚动时收起
  document.addEventListener('click', (e) => {
    const el = e.target.closest ? e.target.closest('.mh') : null;
    if (el) { el === cur ? hide() : show(el); }
    else hide();
  });
  window.addEventListener('scroll', hide, true);
  window.addEventListener('resize', hide);
})();

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok && r.status !== 202) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error || msg; } catch (e) { }
    throw new Error(msg);
  }
  return r.json();
}
function showAlert(msg) {
  const a = $('alert');
  if (!msg) { a.hidden = true; return; }
  a.textContent = msg; a.hidden = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
// （已删除从未被调用的 busy() 与对应的 #loading 遮罩）

// 代码合法但名录里查不到时给个提示。查不到也可能只是名录接口不可用，
// 所以只在「明确查到名录且未命中」时才提示，且不阻断分析。
async function checkName(code) {
  try {
    const r = await api(`/api/name/${code}`);
    if (r && r.found === false && !r.error) {
      showAlert(`未在基金名录中查到 ${code}，请确认代码是否正确。`);
    }
  } catch (e) { /* 名录查询失败不阻断主流程 */ }
}

// ==================================================================== 初始化
(async function init() {
  try {
    S.cat = await api('/api/catalog');
  } catch (e) {
    showAlert('无法连接后端：' + e.message); return;
  }
  // 示例基金
  const sel = $('presetSel');
  S.cat.known_funds.forEach(f => {
    const o = document.createElement('option');
    o.value = f.code; o.textContent = `${f.code} ${f.name}`;
    sel.appendChild(o);
  });
  const chips = $('emptyChips');
  S.cat.known_funds.slice(0, 10).forEach(f => {
    const c = document.createElement('div');
    c.className = 'chip'; c.textContent = f.code;
    c.onclick = () => { $('codeInput').value = f.code; run(f.code); };
    chips.appendChild(c);
  });
  renderModelPicker();
  $('rebalSel').onchange = updateRunHint;
  $('autoEquity').onchange = updateRunHint;

  $('searchForm').onsubmit = e => {
    e.preventDefault();
    const c = $('codeInput').value.trim();
    if (!/^\d{6}$/.test(c)) { showAlert('请输入 6 位数字基金代码，例如 110022'); return; }
    checkName(c);      // 代码存在性提示（异步，不阻塞）
    run(c);
  };
  sel.onchange = e => { if (e.target.value) run(e.target.value); };

  $('btnRefresh').onclick = () => S.result && run(S.result.code, { refresh: true });
  $('btnReLearn').onclick = () => S.result && run(S.result.code, { force: true });
  $('btnSaveSel').onclick = () => {
    if (!S.sel) return;
    S.best = S.sel;
    renderModelStrip(); renderRank();
    showAlert(`已把「${S.result.models[S.sel].name}」设为最佳模型基准。`);
  };
  // 滚动窗口改动 → 只重算相关性区块（/api/corr），不必重跑拟合。
  // 改之前它只是改了个本地变量、本地重绘，后端根本不知道，是个死控件。
  $('rollWin').onchange = async e => {
    S.rollWin = +e.target.value || 120;
    if (!S.result) return;
    try {
      const r = await api(`/api/corr?code=${S.result.code}&win=${S.rollWin}`);
      S.result.correlation = r.correlation;
      renderCorr();
    } catch (err) {
      showAlert('滚动相关重算失败：' + err.message);
    }
  };
  $('btnApplyParams').onclick = applyParams;
  $('btnResetParams').onclick = () => {
    if (S.sel) delete S.params[S.sel];
    renderTuner();
  };

  initBacktestPicker();
  $('btRun').onclick = runBacktest;
})();

// 模型勾选：预设按钮批量勾选，也可逐个自定义
function renderModelPicker() {
  const list = $('mpList');
  $('mpQuick').textContent = `快速 ${S.cat.quick_models.length} 个`;
  $('mpFull').textContent = `全部 ${S.cat.full_models.length} 个`;
  S.cat.models.forEach(m => {
    const lab = document.createElement('label');
    lab.className = 'mp-item';
    lab.title = m.desc;
    lab.innerHTML = `<input type="checkbox" data-k="${m.key}"> ${m.name}`;
    list.appendChild(lab);
  });
  $('mpQuick').onclick = () => applyPreset(S.cat.quick_models);
  $('mpFull').onclick = () => applyPreset(S.cat.full_models);
  list.querySelectorAll('input').forEach(cb => { cb.onchange = updateRunHint; });
  applyPreset(S.cat.quick_models);
}
function applyPreset(keys) {
  $('mpList').querySelectorAll('input').forEach(cb => {
    cb.checked = keys.includes(cb.dataset.k);
  });
  updateRunHint();
}
function selectedModels() {
  return [...$('mpList').querySelectorAll('input:checked')].map(cb => cb.dataset.k);
}
function selectedAuto() {
  return $('autoEquity').checked;
}
function selectedRebal() {
  const v = +$('rebalSel').value;
  return v === 0 ? 0 : v;      // 0 表示不抽稀
}
function updateRunHint() {
  const n = selectedModels().length;
  const rb = $('rebalSel').selectedOptions[0].textContent.split('·')[1].trim();
  $('runHint').textContent = `本次将跑 ${n} 个模型 · 重估点${rb}` +
    (selectedAuto() ? ' · 仓位自适应' : '');
}

// ==================================================================== 运行分析
async function run(code, opts = {}) {
  opts = opts || {};
  showAlert(null);
  const models = selectedModels();
  if (!models.length) { showAlert('请至少勾选一个模型'); return; }
  const params = JSON.parse(JSON.stringify(S.params));
  $('runBtn').disabled = true;
  $('progressWrap').hidden = false;
  setProgress('提交任务…', 0.01, '');

  try {
    const body = {
      code, models,
      params,
      max_rebal: selectedRebal(),
      auto_equity: selectedAuto(),
      refresh: !!opts.refresh,
    };
    // 强制重学：绕过结果缓存
    if (opts.force) body.refresh = true;
    const r = await api('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (r.cached && r.result) {
      applyResult(r.result);
      $('progressWrap').hidden = true;
      return;
    }
    await poll(r.job_id);
  } catch (e) {
    showAlert('分析失败：' + e.message);
    $('progressWrap').hidden = true;
  } finally {
    $('runBtn').disabled = false;
  }
}

function setProgress(stage, pct, detail) {
  $('pStage').textContent = stage;
  $('pPct').textContent = Math.round((pct || 0) * 100) + '%';
  $('pFill').style.width = Math.max(2, (pct || 0) * 100) + '%';
  $('pDetail').textContent = detail || '';
}

async function poll(jid) {
  for (;;) {
    await new Promise(r => setTimeout(r, 900));
    let j;
    try { j = await api('/api/job/' + jid); }
    catch (e) { showAlert('任务查询失败：' + e.message); $('progressWrap').hidden = true; return; }
    setProgress(j.stage, j.progress, j.detail);
    if (j.status === 'done') {
      const r = await api('/api/result/' + jid);
      $('progressWrap').hidden = true;
      applyResult(r.result);
      return;
    }
    if (j.status === 'error') {
      $('progressWrap').hidden = true;
      showAlert('分析失败：' + j.error);
      return;
    }
  }
}

function applyResult(res) {
  S.result = res;
  S.best = res.meta.best_model || Object.keys(res.models)[0];
  S.sel = S.best;
  $('empty').hidden = true;
  $('main').hidden = false;
  renderHead();
  renderInsights();
  renderModelStrip();
  renderRank();
  renderSelected();
  renderNav();
  renderCorr();
  syncBacktestDefaults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==================================================================== 头部
function renderHead() {
  const r = S.result, st = r.nav.stats;
  $('fundName').textContent = `${r.name}（${r.code}）`;
  const m = r.meta;
  $('fundMeta').textContent =
    `${st.start} ~ ${st.end} · ${st.n_days} 交易日 · 候选指数 ${m.n_assets} 个 · ` +
    `窗口 ${m.window} 日 / 重估 ${m.refit} 日 · 用时 ${m.elapsed}s` +
    (m.from_cache ? ' · 缓存' : '');

  const bm = r.models[S.best];
  const kpi = [
    { lb: mh('latest_nav', '最新累计净值'), vl: fmtNum(st.latest, 4),
      sb: `${mh('total_return', '累计')} ${fmtPctSign(st.total_return)}` },
    { lb: mh('ann_return', '年化收益'), vl: fmtPctSign(st.ann_return, 1),
      sb: `${mh('ann_vol', '年化波动')} ${fmtPct(st.ann_vol, 1)}` },
    { lb: mh('max_dd', '最大回撤'), vl: fmtPct(st.max_drawdown, 1),
      sb: `${mh('vol60', '当前 60日波动')} ${fmtPct(st.current_vol60, 1)}`, cls: 'bad' },
    { lb: '拟合模型', vl: bm ? bm.name.slice(0, 8) : '—',
      sb: bm ? `${mh('oos_r2', '样本外R²')} ${fmtNum(bm.metrics.oos_r2, 3)}` : '', cls: 'good' },
    { lb: mh('oos_te', '跟踪误差'), vl: bm ? fmtPct(bm.metrics.oos_te, 1) : '—',
      sb: bm ? `${mh('win', '方向一致')} ${fmtPct(bm.metrics.win, 1)}` : '' },
  ];
  $('kpis').innerHTML = kpi.map(k => `
    <div class="kpi ${k.cls || ''}">
      <div class="lb">${k.lb}</div><div class="vl">${k.vl}</div>
      <div class="sb">${k.sb || ''}</div>
    </div>`).join('');
}

function renderInsights() {
  const ins = S.result.insights || {};
  const order = [
    ['penetration', 'good', '✓ 风格穿透成立'],
    ['exposure_gap', 'warn', '⚠ 仓位假设检验'],
    ['overfit', 'warn', '⚠ 过拟合检查'],
    ['drift', 'warn', '◎ 风格漂移'],
    ['exposure_fix', 'info', '≡ 如何解读低 R²'],
    ['te_spread', 'info', '≡ 模型差异'],
    ['coverage', 'info', '≡ 候选池'],
    ['trend', 'info', '≡ 近期动量'],
  ];
  $('insights').innerHTML = order.filter(([k]) => ins[k]).map(([k, cls, hd]) => {
    const it = ins[k];
    const realCls = it.warn ? (k === 'overfit' || k === 'drift' ||
                                k === 'exposure_gap' ? 'warn' : 'info') : cls;
    return `<div class="card ${realCls}">
      <div class="hd">${hd}</div><div class="bd">${it.text}</div></div>`;
  }).join('');
}

// ==================================================================== 模型条
function renderModelStrip() {
  const r = S.result;
  const keys = r.meta.models;
  $('modelStrip').innerHTML = keys.map(k => {
    const m = r.models[k]; if (!m) return '';
    const err = m.metrics.oos_r2 === null;
    return `<div class="mcard ${k === S.sel ? 'sel' : ''}" data-k="${k}">
      ${k === S.best ? '<span class="best">最佳</span>' : ''}
      <div class="nm">${m.name}</div>
      <div class="r2" style="color:${err ? 'var(--bad)' : 'var(--fg)'}">
        ${err ? '失败' : fmtNum(m.metrics.oos_r2, 3)}</div>
      <div class="sub">TE ${err ? '—' : fmtPct(m.metrics.oos_te, 1)} · ${
        m.interpretable ? '可解释' : '黑箱'}</div>
    </div>`;
  }).join('');
  [...$('modelStrip').children].forEach(c => {
    c.onclick = () => { S.sel = c.dataset.k; renderModelStrip(); renderRank(); renderSelected(); };
  });
}

// ==================================================================== 排名表
function renderRank() {
  const r = S.result;
  const rows = r.rank;
  // ★ 负 R² 必须进入归一化基准（P0-05 / N-02）：
  //   旧写法 Math.max(0, r2) 让负值的条形宽度归零，视觉上与"恰好为 0"无法区分，
  //   等于把彻底失败的模型显示成"平庸但可用"的模型。
  const r2vals = rows.map(x => x['样本外R²'])
    .filter(v => v !== null && v !== undefined && !isNaN(v));
  const maxAbsR2 = Math.max(...r2vals.map(v => Math.abs(v)), 0.0001);
  const head = ['模型', '流派', mh('interpretable', '可解释'),
    mh('oos_r2', '样本外 R²'), mh('oos_te', '年化 TE'), mh('fit_r2', '窗口内 R²'),
    mh('gap', '训练/样本外差'), mh('win', '方向一致'), mh('beta', '隐含仓位'),
    mh('elapsed', '耗时(s)')];
  const html = `<thead><tr>${head.map((h, i) =>
    `<th class="${i < 3 ? 'l' : ''}">${h}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(x => {
      const k = x.key, m = r.models[k] || {};
      const r2 = x['样本外R²'];
      const noR2 = (r2 === null || r2 === undefined || isNaN(r2));
      const neg = !noR2 && r2 < 0;
      // NaN / 缺失不得当 0：旧写法 `?? 0` 会把"没有数据"显示成"差距可控"
      const hasGap = !noR2 && x['窗口内R²'] !== null && x['窗口内R²'] !== undefined
                     && !isNaN(x['窗口内R²']);
      const gap = hasGap ? (x['窗口内R²'] - r2) : null;
      const bt = x['隐含仓位'];
      const w = noR2 ? 0 : Math.abs(r2) / maxAbsR2 * 100;
      const eq = S.cat?.rbsa?.equity_ratio ?? 0.9;
      // ★ 去掉 Math.abs(beta)：负 beta 意味着合成基准与基金方向相反，
      //   比"仓位偏低"严重得多；旧写法把它当成"接近 0"而不告警。
      const bLow = bt !== null && bt !== undefined && !isNaN(bt)
                   && (bt < eq * 0.7 || bt < 0);
      const r2col = noR2 ? 'var(--fg3)' : (neg ? 'var(--bad)' : 'var(--fg)');
      return `<tr class="clickable ${k === S.sel ? 'sel' : ''}" data-k="${k}">
        <td class="l">${x['模型']} ${k === S.best ? '<span class="tag">最佳</span>' : ''}</td>
        <td class="l">${x['流派']}</td>
        <td class="l">${x['可解释']}</td>
        <td class="bar-cell" ${noR2 ? 'title="R² 未定义（窗口内目标无变异）"' :
          neg ? 'title="负 R²：模型连常数基准都不如"' : ''}>${noR2 ? '—' :
          `<div class="bg" style="right:14px;width:${w}%;` +
          `background:${neg ? 'var(--bad)' : 'var(--ok)'};` +
          `opacity:${neg ? 0.5 : 1}"></div>` +
          `<span style="color:${r2col}">${fmtNum(r2, 4)}</span>`}</td>
        <td>${fmtPct(x['样本外TE'], 1)}</td>
        <td>${fmtNum(x['窗口内R²'], 4)}</td>
        <td style="color:${gap === null ? 'var(--fg3)' : (gap > 0.08 ? 'var(--warn)' : 'var(--fg3)')}">${gap === null ? '—' : fmtSign(gap, 3)}</td>
        <td>${fmtPct(x['方向一致率'], 1)}</td>
        <td style="color:${bLow ? 'var(--warn)' : 'var(--fg2)'}" ${bt !== null && bt < 0 ? 'title="负 beta：合成基准与基金方向相反"' : ''}>${fmtPct(bt, 0)}</td>
        <td>${fmtNum(x['耗时'], 1)}</td>
      </tr>`;
    }).join('') + '</tbody>';
  $('rankTable').innerHTML = html;
  [...$('rankTable').querySelectorAll('tr.clickable')].forEach(tr => {
    tr.onclick = () => { S.sel = tr.dataset.k; renderModelStrip(); renderRank(); renderSelected(); };
  });

  // 排序解读
  const best = rows[0];
  const worst = rows[rows.length - 1];
  const gap = (best['样本外R²'] ?? 0) - (worst['样本外R²'] ?? 0);
  $('rankNote').innerHTML =
    `排名按<b>样本外 R²</b> 降序——即"用上一窗口拟合好的模型去预测下一段"的表现。
     最佳：<b style="color:var(--fg)">${best['模型']}（${fmtNum(best['样本外R²'], 3)}）</b>，
     最弱：${worst['模型']}（${fmtNum(worst['样本外R²'], 3)}），跨度 ${fmtNum(gap, 3)}。
     注意"窗口内 R²"高只代表训练集拟合得好，
     <b>任何模型若训练/样本外差 &gt; 0.08，都说明它只是在背历史，不能用来外推</b>。`;
}

// ==================================================================== 选中模型详情
function renderSelected() {
  const k = S.sel; if (!k) return;
  const m = S.result.models[k]; if (!m) return;
  $('selName').textContent = m.name;
  $('selFamily').textContent = m.family;
  $('selInterp').textContent = m.interpretable ? '权重可解释' : '权重=重要性(不可当持仓)';
  $('selInterp').style.color = m.interpretable ? 'var(--ok)' : 'var(--warn)';
  $('selTime').textContent = `用时 ${fmtNum(m.metrics.elapsed, 1)}s`;
  $('selDesc').textContent = m.desc;

  const gap = (m.metrics.fit_r2_mean ?? 0) - (m.metrics.oos_r2 ?? 0);
  const beta = m.metrics.beta, vr = m.metrics.vol_ratio;
  const eq = S.cat?.rbsa?.equity_ratio ?? 0.9;
  const betaLow = beta !== null && beta !== undefined &&
    Math.abs(beta) < eq * 0.7;
  const kpi = [
    { lb: mh('oos_r2', '样本外 R²'), vl: fmtNum(m.metrics.oos_r2, 4),
      sb: '排序依据', cls: 'good' },
    { lb: mh('oos_te', '年化跟踪误差'), vl: fmtPct(m.metrics.oos_te, 1),
      sb: '越低越贴合' },
    { lb: mh('fit_r2', '窗口内 R²'), vl: fmtNum(m.metrics.fit_r2_mean, 4),
      sb: '训练集拟合度' },
    { lb: mh('gap', '训练−样本外'), vl: fmtSign(gap, 3),
      sb: gap > 0.08 ? '过拟合明显' : '差距可控',
      cls: gap > 0.08 ? 'warn' : '' },
    { lb: mh('beta', '隐含股票仓位'), vl: fmtPct(beta, 0),
      sb: `${mh('vol_ratio', '基准波动/基金')} ${fmtNum(vr, 1)}×`,
      cls: betaLow ? 'warn' : '' },
    { lb: mh('win', '方向一致率'), vl: fmtPct(m.metrics.win, 1), sb: '涨跌方向命中' },
    { lb: mh('n_weights', '有效权重数'), vl: m.metrics.n_weights,
      sb: `更新于 ${m.weights_last_date || '—'}` },
  ];
  $('selKpis').innerHTML = kpi.map(x => `
    <div class="kpi ${x.cls || ''}"><div class="lb">${x.lb}</div>
    <div class="vl">${x.vl}</div><div class="sb">${x.sb}</div></div>`).join('');

  // 仓位失配时给一句提示，避免用户误以为模型失效
  const noteEl = $('selNote');
  if (betaLow) {
    noteEl.innerHTML = `<div class="lead warn" style="margin:0 0 14px">
      <b>提示：</b>该基金隐含股票仓位仅 ${fmtPct(beta, 0)}，而合成基准按
      ${fmtPct(eq, 0)} 满仓构建（基准波动是基金的 ${fmtNum(vr, 1)} 倍）。
      这会系统性压低 R²——<b>低 R² 反映的是仓位差异，不是风格识别失败</b>。
      参见上方「仓位假设检验」。</div>`;
    noteEl.hidden = false;
  } else { noteEl.hidden = true; }

  renderTrendPanel();
  drawCurve(m); drawResid(m); drawWHist(m); drawWeights(m);
  renderTuner();
}

function chart(id, opt) {
  let c = S.charts[id];
  if (c && !c.isDisposed()) { c.setOption(opt, true); return c; }
  const el = $(id); if (!el) return null;
  c = echarts.init(el, null, { renderer: 'canvas' });
  c.setOption(opt);
  S.charts[id] = c;
  return c;
}
const AX = {
  axisLine: { lineStyle: { color: '#2f3b4d' } },
  axisLabel: { color: '#6f7f96', fontSize: 10.5 },
  splitLine: { lineStyle: { color: 'rgba(35,45,59,.55)' } },
};
const PAL = ['#4a9eff', '#7c5cff', '#12b886', '#f0a020', '#ef4a4a',
  '#00c2d1', '#e857a6', '#8bc34a'];

// ============================================================ 趋势状态（描述性）
// ★ 定位：这是**形态描述**，不是买卖建议、不产出任何持仓含义。
//   本项目已用 14 只基金做样本外检验证明：日频方向信号无法超越「恒定仓位对照」
//   （RankIC 均值 −0.073，13/14 只为负，见 docs/研究方案.md E0~E4）。
//   这里只回答"合成基准现在长什么样"，不回答"该不该买"。
//
// ── 判定逻辑在 src/trend_state.py，本文件只负责画 ────────────────────
//   为什么不放在前端算：后端下发的曲线是**降采样**的（step = len//400，
//   实测约 4 天一个点）。在前端算"最近 10 日"实际是"10 个采样点 ≈ 40 个
//   交易日"，语义完全失真；而要前端拿到日频序列，curve 会从 67KB 涨到
//   ~335KB。改由后端在日频序列上算好、把状态压成每回看窗口一条字符串下发，
//   6 档 × 474 点 ≈ 2.8KB/模型 —— 便宜约 20 倍，且只有一份实现。
//   判据细节（高低点结构 + 拐头的四个附加条件）见该模块的 docstring。
//
// 编码：U 上升 / D 下降 / R 震荡 / u 拐头向上 / d 拐头向下 / . 样本不足
const TREND = {
  u: { label: '拐头向上', color: '#ff9a3c' },
  U: { label: '上升趋势', color: '#ef4a4a' },
  R: { label: '震荡',     color: '#8b98ab' },
  D: { label: '下降趋势', color: '#12b886' },
  d: { label: '拐头向下', color: '#2f9e8f' },
};
// 可选回看窗口。1~5 是"几天涨跌"，10/20 才勉强能读出短期趋势形态。
// 实测（4 只基金合成基准，2368 日）体制态（上升/下降/震荡）的平均段长：
//   N=5 → 3.4 天，N=10 → 5.3 天，N=20 → 7.4 天（p90 分别 8 / 13 / 19 天）
// 默认 10：均值约 5 个交易日，是一个交易周的量级。
const TREND_LEVELS = [1, 2, 3, 5, 10, 20];
const TREND_DEFAULT_N = 10;
const TURN_CODES = ['u', 'd'];

/**
 * 把后端下发的状态串压成连续段（画色带用）。
 * 编码：U 上升 / D 下降 / R 震荡 / u 拐头向上 / d 拐头向下 / . 样本不足
 */
function trendRuns(dates, codes) {
  const runs = [];
  let s = -1, cur = null;
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i];
    if (!c || c === '.') {
      if (s >= 0) runs.push([s, i - 1, cur]);
      s = -1; cur = null; continue;
    }
    if (s < 0) { s = i; cur = c; }
    else if (c !== cur) { runs.push([s, i - 1, cur]); s = i; cur = c; }
  }
  if (s >= 0) runs.push([s, codes.length - 1, cur]);
  return runs;
}

/** 回看窗口选择器 + 当前状态卡（挂在拟合曲线面板上）。 */
function renderTrendPanel() {
  const m = S.result && S.result.models[S.sel];
  if (!m || !m.curve) return;

  // ---- 选择器
  const seg = $('trendSeg');
  if (seg) {
    seg.innerHTML = TREND_LEVELS.map(n =>
      `<button type="button" data-n="${n}" class="${n === S.lookback ? 'on' : ''}">${n}</button>`
    ).join('');
    [...seg.children].forEach(b => {
      b.onclick = () => {
        const v = Number(b.dataset.n);
        if (v === S.lookback) return;
        S.lookback = v;
        renderTrendPanel();
        drawCurve(S.result.models[S.sel]);   // 色带要跟着重画
      };
    });
  }

  // ---- 状态（数据来自后端 src/trend_state.py，在日频序列上算好）
  const d = m.curve;
  const N = S.lookback;
  const tr = d.trend || {};
  const at = (tr.last || {})[String(N)];
  const hist = (tr.stats || {})[String(N)];
  const card = $('trendCard');
  if (!card) return;
  if (!at || !hist) {
    card.innerHTML = `<div class="tc-empty">样本不足 ${2 * N} 个交易日，无法判定。</div>`;
    return;
  }
  const st = TREND[at.state] || TREND.R;
  const posTxt = `${Math.round(at.pos * 100)}%`;
  const thin = N <= 2;
  card.innerHTML = `
    <div class="tc-head">
      <span class="tc-state" style="color:${st.color}">
        <i style="background:${st.color}"></i>${st.label}</span>
      <span class="tc-sub">合成基准 · 回看 ${N} 个交易日 ·
        本组 ${at.win_start} ~ ${at.win_end}，前一组 ${at.prior_start} ~ ${at.prior_end}</span>
    </div>
    <div class="tc-grid">
      <div><i>本组高点</i><b>${at.hr.toFixed(4)}</b>
        <em>出现在第 ${at.arg_h}/${N} 日</em></div>
      <div><i>本组低点</i><b>${at.lr.toFixed(4)}</b>
        <em>出现在第 ${at.arg_l}/${N} 日</em></div>
      <div><i>前一组高点</i><b>${at.hp.toFixed(4)}</b></div>
      <div><i>前一组低点</i><b>${at.lp.toFixed(4)}</b></div>
      <div><i>当前在区间位置</i><b>${posTxt}</b><em>0% = 区间最低</em></div>
    </div>
    <div class="tc-hist">
      <b>该回看窗口的历史可靠性（按日频全样本统计）：</b>
      体制态 ${hist.regime_runs} 段（中位 ${hist.regime_median} 天 /
      均值 ${hist.regime_mean} 天 / p90 ${hist.regime_p90} 天）
      · 拐头 ${hist.turn_runs} 次
      ${hist.turn_every ? `（平均每 ${hist.turn_every} 个交易日一次）` : ''}
      ${hist.regime_mean < 3 ? '　⚠ 段长过短：这个窗口下状态接近每天翻转，读不出趋势'
        : ''}
    </div>
    <div class="tc-legend">
      ${['U', 'D', 'R'].map(k =>
        `<span><i class="bar" style="background:${TREND[k].color}"></i>${TREND[k].label}</span>`
      ).join('')}
      ${TURN_CODES.map(k =>
        `<span><i class="dot" style="background:${TREND[k].color}"></i>${TREND[k].label}</span>`
      ).join('')}
      <span class="tc-lb">色带 = 状态区间　圆点 = 拐头（瞬时标记，仅 1 天）　
        拖动图表下方滑块放大后色带才可读（全区间看是一片细条纹）</span>
    </div>
    ${thin ? `<div class="tc-warn">回看窗口 ${N} 日只能比较单日/两日的涨跌：
      出不了「震荡」，也出不了「拐头」。要读形态请选 5 日以上。</div>` : ''}`;
}

function drawCurve(m) {
  const d = m.curve;
  const codes = ((d.trend || {}).codes || {})[String(S.lookback)] || '';
  // 体制态（上升/下降/震荡）→ 竖向色带；拐头是 1 天的瞬时标记，
  // 画成色带宽度不足 1 像素、根本看不见，所以改为在曲线上打点。
  const runs = trendRuns(d.dates, codes)
    .filter(([, , c]) => TURN_CODES.indexOf(c) < 0)
    .map(([a, b, c]) => ([
      { xAxis: d.dates[a], itemStyle: { color: TREND[c].color } },
      { xAxis: d.dates[b] },
    ]));
  const turns = [];
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i];
    if (TURN_CODES.indexOf(c) >= 0) {
      turns.push({
        name: TREND[c].label,
        coord: [d.dates[i], d.synth[i]],
        value: TREND[c].label,
        itemStyle: { color: TREND[c].color },
      });
    }
  }
  chart('chartCurve', {
    grid: { left: 52, right: 18, top: 18, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 } },
    legend: { data: ['基金净值', '合成基准'], textStyle: { color: '#a9b6c8', fontSize: 11 },
      top: 0, itemWidth: 16, itemHeight: 8 },
    xAxis: { type: 'category', data: d.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...AX, axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: [
      { name: '基金净值', type: 'line', data: d.fund, showSymbol: false,
        lineStyle: { width: 1.9, color: PAL[0] } },
      { name: '合成基准', type: 'line', data: d.synth, showSymbol: false,
        lineStyle: { width: 1.4, color: PAL[1], type: 'dashed' },
        markArea: {
          silent: true, z: 0,
          // 色带密度很高（状态平均每 5 天切换一次，而绘图只有约 474 个点），
          // 全区间看就是一片细条纹，所以压到 0.13 —— 让价格曲线保持主体，
          // 放大到一两年区间后色带才真正可读。
          itemStyle: { opacity: 0.13, borderWidth: 0 },
          data: runs,
        },
        markPoint: {
          symbol: 'circle', symbolSize: 6,
          data: turns,
          label: { show: false },
          tooltip: { formatter: (p) => p.data.value },
        } },
    ]
  });
}

function drawResid(m) {
  const d = m.residual;
  const vals = (d.value || []).filter(v => v !== null);
  const mx = vals.length ? Math.max(...vals.map(Math.abs)) : 1;
  chart('chartResid', {
    grid: { left: 52, right: 18, top: 18, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : Number(v).toFixed(4)) },
    xAxis: { type: 'category', data: d.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(124,92,255,.12)',
      handleStyle: { color: '#7c5cff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: [{
      type: 'bar', data: (d.value || []).map(v =>
        v === null ? null : { value: v, itemStyle: { color: colOf(v) } }),
      barMaxWidth: 2.2,
    }]
  });
}

function drawWHist(m) {
  const h = m.weights_hist;
  if (!h || !h.assets || !h.assets.length) {
    chart('chartWHist', { title: { text: '无权重历史', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  chart('chartWHist', {
    grid: { left: 52, right: 18, top: 34, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : (v * 100).toFixed(1) + '%') },
    legend: { data: h.assets, textStyle: { color: '#a9b6c8', fontSize: 10.5 },
      top: 0, itemWidth: 13, itemHeight: 7, type: 'scroll' },
    xAxis: { type: 'category', data: h.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: h.assets.map((a, i) => ({
      name: a, type: 'line', stack: 'w', areaStyle: { opacity: 0.68 },
      showSymbol: false, lineStyle: { width: 0.6 },
      itemStyle: { color: PAL[i % PAL.length] },
      data: h.series[i]
    }))
  });
}

function drawWeights(m) {
  const items = (m.latest_weights && m.latest_weights.items) || [];
  const arr = items.slice().reverse();
  if (!arr.length) {
    chart('chartWeights', { title: { text: '无权重数据', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  chart('chartWeights', {
    grid: { left: 92, right: 62, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: '#1b2431', borderColor: '#2f3b4d', textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : (v * 100).toFixed(2) + '%') },
    xAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    yAxis: { type: 'category', data: arr.map(x => x.name), ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 11 } },
    series: [{
      type: 'bar', barMaxWidth: 15,
      data: arr.map((x, i) => ({ value: x.weight,
        itemStyle: { color: PAL[(arr.length - 1 - i) % PAL.length] } })),
      label: { show: true, position: 'right', color: '#a9b6c8', fontSize: 10.5,
        formatter: p => (p.value * 100).toFixed(1) + '%' }
    }]
  });
}

// ==================================================================== 超参表单
function renderTuner() {
  const k = S.sel; if (!k) return;
  const spec = S.cat.models.find(m => m.key === k);
  if (!spec) return;
  const cur = Object.assign({}, spec.params, S.params[k] || {});
  const HINTS = {
    alpha_scale: '正则强度缩放，越大越稀疏',
    l1_ratio: 'L1 占比，越大越稀疏',
    max_assets: '最多保留几个指数',
    min_weight: '低于此权重视为噪声',
    top_k: '按相关性取前 K 个',
    n_components: '主成分个数',
    learning_rate: ' boosting 学习率',
    max_iter: '提升轮数 / 迭代次数',
    max_depth: '树最大深度',
    n_estimators: '树棵数',
    hidden: '隐藏层，如 32,16',
  };
  $('tunerBody').innerHTML = Object.entries(spec.params).map(([pk, dflt]) => {
    const v = cur[pk];
    let inp;
    if (pk === 'hidden') {
      inp = `<input type="text" data-pk="${pk}" value="${(v || []).join(',')}">`;
    } else if (typeof dflt === 'boolean') {
      inp = `<input type="checkbox" data-pk="${pk}" ${v ? 'checked' : ''}>`;
    } else if (typeof dflt === 'number') {
      const step = dflt < 0.01 ? 0.01 : dflt < 1 ? 0.01 : 1;
      inp = `<input type="number" data-pk="${pk}" value="${v}" step="${step}">`;
    } else {
      inp = `<input type="text" data-pk="${pk}" value="${v}">`;
    }
    return `<div class="pfield"><label>${pk}</label>${inp}
      <span class="hint">默认 ${JSON.stringify(dflt)} · ${HINTS[pk] || ''}</span></div>`;
  }).join('');
}

async function applyParams() {
  const k = S.sel; if (!k) return;
  const spec = S.cat.models.find(m => m.key === k);
  const o = {};
  $('tunerBody').querySelectorAll('[data-pk]').forEach(el => {
    const pk = el.dataset.pk;
    if (el.type === 'checkbox') { o[pk] = el.checked; return; }
    const dflt = spec.params[pk];
    let v = el.value;
    if (pk === 'hidden') v = v.split(/[,，]/).map(x => +x).filter(x => !Number.isNaN(x));
    else if (typeof dflt === 'number') v = Number(v);
    o[pk] = v;
  });
  S.params[k] = o;
  $('progressWrap').hidden = false;
  setProgress('重新学习 ' + spec.name, 0.02, '');
  try {
    const r = await api('/api/refit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: S.result.code, model: k, params: o,
                              max_rebal: selectedRebal(),
                              auto_equity: selectedAuto() })
    });
    await pollRefit(r.job_id, k);
  } catch (e) {
    $('progressWrap').hidden = true;
    showAlert('重新学习失败：' + e.message);
  }
}

async function pollRefit(jid, k) {
  for (;;) {
    await new Promise(r => setTimeout(r, 800));
    const j = await api('/api/job/' + jid);
    setProgress(j.stage, j.progress, j.detail);
    if (j.status === 'done') {
      const r = await api('/api/result/' + jid);
      $('progressWrap').hidden = true;
      S.result.models[k] = r.result;          // 就地替换
      // 重算排名。键名必须加引号：'样本外R²' 含上标字符，不是合法标识符。
      S.result.rank = Object.values(S.result.models).map(m => ({
        key: m.key, '模型': m.name, '流派': m.family,
        '可解释': m.interpretable ? '是' : '否',
        '样本外R²': m.metrics.oos_r2, '样本外TE': m.metrics.oos_te,
        '窗口内R²': m.metrics.fit_r2_mean, '窗口内TE': m.metrics.fit_te_mean,
        '方向一致率': m.metrics.win, '耗时': m.metrics.elapsed,
      })).sort((a, b) => (b['样本外R²'] ?? -9) - (a['样本外R²'] ?? -9));
      const stillBest = Object.keys(S.result.models).reduce((a, x) =>
        (S.result.models[x].metrics.oos_r2 ?? -9) > (S.result.models[a].metrics.oos_r2 ?? -9) ? x : a,
        Object.keys(S.result.models)[0]);
      S.best = stillBest;
      renderModelStrip(); renderRank(); renderSelected(); renderHead();
      showAlert(`「${r.result.name}」已用新参数重新学习，样本外 R² = ${
        fmtNum(r.result.metrics.oos_r2, 4)}。当前最佳仍是「${S.result.models[stillBest].name}」。`);
      return;
    }
    if (j.status === 'error') {
      $('progressWrap').hidden = true;
      showAlert('重新学习失败：' + j.error); return;
    }
  }
}

// ==================================================================== 净值
function renderNav() {
  const n = S.result.nav;
  chart('chartNav', {
    grid: { left: 56, right: 18, top: 14, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : Number(v).toFixed(4)) },
    xAxis: { type: 'category', data: n.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...AX, axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: [{
      type: 'line', data: n.nav, showSymbol: false,
      lineStyle: { width: 1.9, color: PAL[0] },
      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [{ offset: 0, color: 'rgba(74,158,255,.28)' },
                     { offset: 1, color: 'rgba(74,158,255,0)' }] } },
      markLine: { silent: true, symbol: 'none',
        lineStyle: { color: '#6f7f96', type: 'dashed', width: 1 },
        data: [{ yAxis: n.nav[0] }] }
    }]
  });

  chart('chartDD', {
    grid: { left: 56, right: 18, top: 14, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 }, valueFormatter: v => v == null ? '—' : (v * 100).toFixed(2) + '%' },
    xAxis: { type: 'category', data: n.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    series: [{ type: 'line', data: n.drawdown, showSymbol: false,
      lineStyle: { width: 1.2, color: DOWN },
      areaStyle: { color: 'rgba(18,184,134,.22)' } }]
  });

  chart('chartVol', {
    grid: { left: 56, right: 18, top: 14, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 }, valueFormatter: v => v == null ? '—' : (v * 100).toFixed(1) + '%' },
    xAxis: { type: 'category', data: n.dates, ...AX, axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: v => (v * 100).toFixed(0) + '%' } },
    series: [{ type: 'line', data: n.vol60, showSymbol: false,
      lineStyle: { width: 1.3, color: PAL[3] } }]
  });

  drawMonthly(n.monthly, n.years);
}

function drawMonthly(monthly, years) {
  if (!monthly || !Object.keys(monthly).length) {
    chart('chartMonthly', { title: { text: '数据不足', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  const yearsR = years.slice().reverse();
  const data = [];
  yearsR.forEach((y, yi) => {
    for (let mth = 1; mth <= 12; mth++) {
      const v = (monthly[y] || {})[String(mth)];
      data.push([mth - 1, yi,
        v === undefined || v === null ? '-' : +(v * 100).toFixed(2)]);
    }
  });
  chart('chartMonthly', {
    grid: { left: 52, right: 18, top: 12, bottom: 46 },
    tooltip: { position: 'top', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => p.value[0] === '-' ? '' : `${yearsR[p.value[1]]}-${String(p.value[0] + 1).padStart(2, '0')}<br/><b style="color:${colOf(p.value[2])}">${p.value[2]}%</b>` },
    xAxis: { type: 'category', data: Array.from({ length: 12 }, (_, i) => i + 1 + '月'),
      ...AX, splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,.015)', 'transparent'] } } },
    yAxis: { type: 'category', data: yearsR, ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 10.5 } },
    visualMap: { min: -25, max: 25, calculable: false, orient: 'horizontal',
      left: 'center', bottom: 4, itemHeight: 90, itemWidth: 12,
      textStyle: { color: '#6f7f96', fontSize: 10 },
      inRange: { color: ['#12b886', '#0f8f68', '#1b2431', '#c0392b', '#ef4a4a'] } },
    series: [{ type: 'heatmap', data,
      label: { show: true, fontSize: 9.5, color: '#dbe4f0',
        formatter: p => p.value[0] === '-' ? '' : p.value[2].toFixed(1) },
      itemStyle: { borderColor: '#0d1117', borderWidth: 1.5 } }]
  });
}

// ==================================================================== 相关性
function renderCorr() {
  const c = S.result.correlation;

  const top = c.top.slice().reverse();
  chart('chartCorrBar', {
    grid: { left: 96, right: 58, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: '#1b2431', borderColor: '#2f3b4d', textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => {
        const r = top[p[0].dataIndex];
        return `<b>${r.name}</b><br/>${r.layer}<br/>相关系数 ${fmtSign(r.corr, 4)}`;
      } },
    xAxis: { type: 'value', min: -1, max: 1, ...AX },
    yAxis: { type: 'category', data: top.map(x => x.name), ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 10.5 } },
    series: [{
      type: 'bar', barMaxWidth: 14,
      data: top.map(x => ({ value: x.corr, itemStyle: { color: colOf(x.corr) } })),
      label: { show: true, position: 'right', fontSize: 10.5, color: '#a9b6c8',
        formatter: p => fmtSign(p.value, 2) }
    }]
  });

  const L = c.layers.slice();
  chart('chartLayer', {
    grid: { left: 96, right: 58, top: 10, bottom: 24 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => { const r = L[p[0].dataIndex];
        return `<b>${r.layer}</b><br/>包含 ${r.n} 个指数<br/>平均相关 ${fmtSign(r.avg_corr, 3)}`; } },
    xAxis: { type: 'value', min: -1, max: 1, ...AX },
    yAxis: { type: 'category', data: L.map(x => x.layer), ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 10.5 } },
    series: [{
      type: 'bar', barMaxWidth: 20,
      data: L.map(x => ({ value: x.avg_corr, itemStyle: { color: colOf(x.avg_corr) } })),
      label: { show: true, position: 'right', fontSize: 11, color: '#a9b6c8',
        formatter: p => `${fmtSign(p.value, 3)} (${L[p.dataIndex].n})` }
    }]
  });

  drawMatrix(c.matrix);
  drawRollCorr(c.rolling);

  $('driftNote').innerHTML = (c.drift || []).map(d => {
    const big = Math.abs(d.delta || 0) > 0.10;
    return `<div class="card ${big ? 'warn' : 'info'}" style="margin-bottom:10px">
      <div class="hd">${d.name} · 相关性 ${fmtSign(d.delta, 3)}</div>
      <div class="bd">前段 ${fmtSign(d.early, 3)} → 后段 ${fmtSign(d.late, 3)}。
      ${big ? '<b>变化超过 0.10，风格发生实质漂移。</b>' : '变化在 0.10 以内，暴露相对稳定。'}
      </div></div>`;
  }).join('');
}

function drawMatrix(mx) {
  const data = [];
  mx.rows.forEach((r, i) => mx.cols.forEach((c, j) => {
    const v = mx.values[i][j];
    if (v !== null) data.push([j, i, v]);
  }));
  chart('chartMatrix', {
    grid: { left: 118, right: 18, top: 8, bottom: 92 },
    tooltip: { position: 'top', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      formatter: p => `${mx.rows[p.value[1]]} × ${mx.cols[p.value[0]]}<br/><b>${fmtSign(p.value[2], 3)}</b>` },
    xAxis: { type: 'category', data: mx.cols, ...AX,
      axisLabel: { ...AX.axisLabel, rotate: 52, fontSize: 9.5, interval: 0 } },
    yAxis: { type: 'category', data: mx.rows, ...AX,
      axisLabel: { ...AX.axisLabel, fontSize: 9.5, interval: 0 } },
    visualMap: { min: -1, max: 1, calculable: true, orient: 'horizontal',
      left: 'center', bottom: 4, itemWidth: 11, itemHeight: 110,
      textStyle: { color: '#6f7f96', fontSize: 10 },
      inRange: { color: ['#12b886', '#1b2431', '#ef4a4a'] } },
    series: [{ type: 'heatmap', data,
      label: { show: true, fontSize: 8.5, color: '#c8d3e2',
        formatter: p => p.value[2].toFixed(2) },
      itemStyle: { borderColor: '#0d1117', borderWidth: 1 },
      emphasis: { itemStyle: { borderColor: '#e6edf6', borderWidth: 1.5 } } }]
  });
}

function drawRollCorr(roll) {
  const keys = Object.keys(roll || {});
  if (!keys.length) {
    chart('chartRollCorr', { title: { text: '数据不足', left: 'center',
      textStyle: { color: '#6f7f96', fontSize: 12, fontWeight: 'normal' } } });
    return;
  }
  chart('chartRollCorr', {
    grid: { left: 56, right: 18, top: 34, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 } },
    legend: { data: keys, textStyle: { color: '#a9b6c8', fontSize: 10.5 },
      top: 0, itemWidth: 14, itemHeight: 7, type: 'scroll' },
    xAxis: { type: 'category', data: roll[keys[0]].dates, ...AX,
      axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', min: 0, max: 1, ...AX },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: keys.map((k, i) => ({
      name: k, type: 'line', showSymbol: false,
      lineStyle: { width: 1.4 },
      itemStyle: { color: PAL[i % PAL.length] },
      data: roll[k].values
    }))
  });
}

// ==================================================================== 买卖决策回测
function initBacktestPicker() {
  S.cat.models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.key; o.textContent = m.name;
    $('btModel').appendChild(o);
  });
}

function syncBacktestDefaults() {
  const st = S.result.nav.stats;
  // 仅在用户尚未选择时预填，避免分析刷新后覆盖已选区间
  if (!$('btStart').value && st.start) $('btStart').value = st.start;
  if (!$('btEnd').value && st.end) $('btEnd').value = st.end;
  if (S.sel && S.cat.models.some(m => m.key === S.sel)) $('btModel').value = S.sel;
  $('btAuto').checked = $('autoEquity').checked;
}

async function runBacktest() {
  if (!S.result) { showAlert('请先分析一只基金'); return; }
  const start = $('btStart').value, end = $('btEnd').value;
  if (start && end && start >= end) { showAlert('起始日需早于结束日'); return; }

  $('btRun').disabled = true;
  $('progressWrap').hidden = false;
  setProgress('提交回测任务…', 0.02, '');
  try {
    const r = await api('/api/backtest', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: S.result.code,
        model: $('btModel').value,
        auto_equity: $('btAuto').checked,
        start: start || null,
        end: end || null,
      })
    });
    await pollBacktest(r.job_id);
  } catch (e) {
    $('progressWrap').hidden = true;
    showAlert('回测失败：' + e.message);
  } finally {
    $('btRun').disabled = false;
  }
}

async function pollBacktest(jid) {
  for (;;) {
    await new Promise(r => setTimeout(r, 800));
    const j = await api('/api/job/' + jid);
    setProgress(j.stage, j.progress, j.detail);
    if (j.status === 'done') {
      const r = await api('/api/result/' + jid);
      $('progressWrap').hidden = true;
      renderBacktest(r.result);
      return;
    }
    if (j.status === 'error') {
      $('progressWrap').hidden = true;
      showAlert('回测失败：' + j.error);
      return;
    }
  }
}

function renderBacktest(p) {
  $('btOut').hidden = false;
  drawBtEquity(p);
  drawBtHealth(p);
  renderBtKpis(p);
  renderBtMetrics(p);
  renderBtTrades(p);
  $('btOut').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function btLineStyle(k) {
  if (k.includes('健康')) return { width: 2.2, color: PAL[0] };
  if (k.includes('趋势')) return { width: 1.4, color: PAL[3] };
  if (k.includes('恒定')) return { width: 1.2, color: FLAT, type: 'dashed' };
  return { width: 1.3, color: PAL[1], type: 'dashed' };   // 买入持有
}

function drawBtEquity(p) {
  const keys = Object.keys(p.equity.series);
  chart('btChartEquity', {
    grid: { left: 52, right: 18, top: 32, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 },
      valueFormatter: v => (v === null ? '—' : Number(v).toFixed(4)) },
    legend: { data: keys, textStyle: { color: '#a9b6c8', fontSize: 10.5 },
      top: 0, itemWidth: 15, itemHeight: 8, type: 'scroll' },
    xAxis: { type: 'category', data: p.equity.dates, ...AX,
      axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', scale: true, ...AX,
      axisLabel: { ...AX.axisLabel, formatter: v => v.toFixed(2) } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 4,
      borderColor: '#2f3b4d', fillerColor: 'rgba(74,158,255,.12)',
      handleStyle: { color: '#4a9eff' }, textStyle: { color: '#6f7f96', fontSize: 9 } }],
    series: keys.map(k => ({
      name: k, type: 'line', data: p.equity.series[k], showSymbol: false,
      lineStyle: btLineStyle(k),
    }))
  });
}

function drawBtHealth(p) {
  const h = p.health, th = p.meta.thresholds;
  chart('btChartHealth', {
    grid: { left: 52, right: 46, top: 32, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: '#1b2431', borderColor: '#2f3b4d',
      textStyle: { color: '#e6edf6', fontSize: 12 } },
    legend: { data: ['健康分', '持仓'], textStyle: { color: '#a9b6c8', fontSize: 11 },
      top: 0, itemWidth: 15, itemHeight: 8 },
    xAxis: { type: 'category', data: h.dates, ...AX,
      axisLabel: { ...AX.axisLabel, hideOverlap: true } },
    yAxis: [
      { type: 'value', min: -1, max: 1, ...AX,
        name: '健康分', nameTextStyle: { color: '#6f7f96', fontSize: 10 } },
      { type: 'value', min: 0, max: 1, splitLine: { show: false },
        axisLine: { lineStyle: { color: '#2f3b4d' } },
        axisLabel: { color: '#6f7f96', fontSize: 10 } },
    ],
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: '健康分', type: 'line', data: h.score, showSymbol: false,
        lineStyle: { width: 1.5, color: PAL[0] },
        markLine: {
          symbol: 'none', silent: true,
          data: [
            { yAxis: th.exit, lineStyle: { color: DOWN, type: 'dashed', width: 1 },
              label: { color: DOWN, fontSize: 9.5, position: 'insideStartTop',
                formatter: '卖出线 ' + th.exit } },
            { yAxis: th.enter, lineStyle: { color: UP, type: 'dashed', width: 1 },
              label: { color: UP, fontSize: 9.5, position: 'insideStartBottom',
                formatter: '买入线 ' + th.enter } },
          ],
        } },
      { name: '持仓', type: 'line', step: 'end', data: h.position, yAxisIndex: 1,
        showSymbol: false, lineStyle: { width: 1.4, color: PAL[3] },
        areaStyle: { color: 'rgba(240,160,32,.12)' } },
    ]
  });
}

function renderBtKpis(p) {
  const main = p.metrics.find(x => x['方案'] && x['方案'].includes('健康')) || p.metrics[0];
  const nm = main['方案'];
  // ★ 净贡献 / 平均仓位 / 对照必须取**这一行自己的**值，不能拿主策略的。
  // 早先的实现：指标取「拟合健康决策」行，但 tc 与对照取的是
  // "择时策略(融合)"的 —— 两个不同方案的数被拼在同一张卡上。
  // 平均仓位 100% 的门控配 45% 的对照，会出现"择时贡献"看起来很大的假象。
  const tc = (p.summary.timing_contrib_all || {})[nm]
    || p.summary.timing_contrib || {};
  const posAll = p.summary.avg_position_all || {};
  const ctrlAll = p.summary.controls || {};
  const avgPos = (nm in posAll) ? posAll[nm] : p.summary.avg_position;
  const ctrlName = ctrlAll[nm] || p.summary.control_name;
  const cards = [
    { lb: mh('ann_return', '年化收益'), vl: fmtPctSign(main['年化收益'], 1),
      sb: `${mh('timing_contrib', '择时贡献')} ${fmtPctSign(tc['年化收益'])}`,
      cls: main['年化收益'] > 0 ? 'good' : 'bad' },
    { lb: mh('sharpe', '夏普比率'), vl: fmtNum(main['夏普'], 2),
      // N-03：几何夏普与算术夏普分开显示。旧页面只给一个"夏普"，
      // 而它是用几何年化收益算的，与业界默认口径不可直接对照。
      sb: `${mh('sharpe_arith', '算术夏普')} ` +
          `${p.summary.sharpe_arithmetic ?? main['算术夏普'] ?? '—'}`
          + `（几何 ${fmtNum(main['夏普'], 2)}）` },
    { lb: mh('max_dd', '最大回撤'), vl: fmtPct(main['最大回撤'], 1),
      sb: `${mh('timing_contrib', '择时改善')} ${fmtPctSign(tc['最大回撤'])}`, cls: 'bad' },
    { lb: mh('calmar', 'Calmar'), vl: fmtNum(main['Calmar'], 2),
      sb: `${mh('timing_contrib', '择时贡献')} ${fmtSign(tc['Calmar'], 2)}` },
    { lb: mh('avg_pos', '平均仓位'), vl: fmtPct(avgPos, 0),
      sb: `${mh('control', '对照')} ${ctrlName}` },
    { lb: mh('turnover', '年化换手'), vl: fmtNum(p.summary.turnover, 2),
      sb: '单边换手倍数' },
    { lb: mh('rank_ic', 'RankIC'), vl: fmtNum(p.summary.ic, 3),
      sb: `${mh('icir', 'ICIR')} ${fmtNum(p.summary.icir, 2)}` },
    // N-02：胜率的分母口径必须一起给。旧页面只显示"99.17%"，
    // 用户无从知道其中有多少是根本没有前瞻标签的天数。
    { lb: mh('win_rate', '多头胜率'), vl: fmtPct(p.summary.win_long, 1),
      sb: `有效标签 ${p.summary.n_valid_label ?? '—'} 天`
          + `（已剔除尾部无前瞻收益的天数）` },
  ];
  $('btKpis').innerHTML = cards.map(x => `
    <div class="kpi ${x.cls || ''}"><div class="lb">${x.lb}</div>
    <div class="vl">${x.vl}</div><div class="sb">${x.sb}</div></div>`).join('');
}

function renderBtMetrics(p) {
  const cols = [mh('ann_return', '年化收益'), mh('ann_vol', '年化波动'),
    mh('sharpe', '夏普'), mh('sharpe_arith', '算术夏普'),
    mh('max_dd', '最大回撤'), mh('calmar', 'Calmar'),
    mh('monthly_win', '月胜率'), mh('cumulative', '累计收益')];
  const html = `<thead><tr><th class="l">方案</th>${cols.map(c => `<th>${c}</th>`).join('')}
    </tr></thead><tbody>` +
    p.metrics.map(x => {
      const main = x['方案'] && x['方案'].includes('健康');
      return `<tr ${main ? 'class="sel"' : ''}>
        <td class="l">${x['方案']}</td>
        <td style="color:${colOf(x['年化收益'])}">${fmtPctSign(x['年化收益'], 1)}</td>
        <td>${fmtPct(x['年化波动'], 1)}</td>
        <td>${fmtNum(x['夏普'], 2)}</td>
        <td>${fmtNum(x['算术夏普'], 2)}</td>
        <td style="color:${DOWN}">${fmtPct(x['最大回撤'], 1)}</td>
        <td>${fmtNum(x['Calmar'], 2)}</td>
        <td>${fmtPct(x['月胜率'], 0)}</td>
        <td style="color:${colOf(x['累计收益'])}">${fmtPctSign(x['累计收益'], 1)}</td>
      </tr>`;
    }).join('') + '</tbody>';
  $('btMetrics').innerHTML = html;
}

function renderBtTrades(p) {
  let body;
  if (!p.trades.length) {
    // 零交易必须区分「信号判断为一直持有」和「阈值根本够不到」——
    // 后者意味着该方案 ≡ 买入持有，它与对照的差额只是仓位差。
    const hs = p.meta.health_stats || {};
    let why;
    if (hs.unreliable_ratio > 0.5) {
      why = `拟合不可信天数占 <b>${fmtPct(hs.unreliable_ratio, 0)}</b>`
        + `（R² 中位 ${fmtNum(hs.r2_median, 2)} &lt; 0.5，合成基准解释不了这只基金），`
        + '<b>门控已停用</b>';
    } else if (hs.gap_to_exit != null) {
      why = `健康分历史最低 <b>${fmtNum(hs.min, 2)}</b>，离场线 ${fmtNum(hs.exit, 2)}，`
        + `还差 <b>${fmtNum(hs.gap_to_exit, 2)}</b>，未触及阈值`;
    } else {
      why = '健康分未触及离场阈值';
    }
    body = `<tr><td colspan="5" style="color:var(--fg3);text-align:left">
      该区间内<b>零交易</b>：${why}，全程持有。<br>
      <span style="color:var(--fg2)">因此本方案 ≡ 买入持有，
      绩效差异只来自仓位水平，<b>不构成择时能力的证据</b>。</span></td></tr>`;
  } else {
    body = p.trades.map(t => {
      const buy = t.action === '买入';
      return `<tr>
        <td class="l">${t.date}</td>
        <td class="l">${t.exec_date}</td>
        <td style="color:${buy ? UP : DOWN};font-weight:600">${buy ? '买入 ▲' : '卖出 ▼'}</td>
        <td>${fmtNum(t.health, 2)}</td>
        <td class="l" style="color:var(--fg2)">${t.reason || '—'}</td>
      </tr>`;
    }).join('');
  }
  $('btTrades').innerHTML =
    `<thead><tr><th class="l">决策日</th><th class="l">执行日(次日)</th>
      <th>动作</th><th>健康分</th><th class="l">触发原因</th></tr></thead>
     <tbody>${body}</tbody>`;
}

// ==================================================================== 自适应
let rzT = null;
window.addEventListener('resize', () => {
  clearTimeout(rzT);
  rzT = setTimeout(() => {
    Object.values(S.charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
  }, 140);
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) Object.values(S.charts).forEach(c => { if (c && !c.isDisposed()) c.resize(); });
});