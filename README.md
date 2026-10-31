# Market Research · 股票、基金、指数研究

本地研究工作台，包含股票技术分析与方向估计、基金滚动 RBSA 与多模型比较、指数走势与风险分析。后端使用 FastAPI、独立计算进程和 DuckDB，环境与依赖由 uv 管理。

文档核对日期：2026-10-31。本次更新核对了源码、配置和已有验证记录，没有重新抓取全部行情或重算研究结论。

## 安装与启动

需要 uv；项目使用 Python 3.13，验证版本由 `.python-version` 固定为 3.13.14，依赖由 `uv.lock` 锁定。在项目根目录运行：

```powershell
uv sync --locked
uv run --locked market-research --port 8877
```

打开 [研究工作台](http://127.0.0.1:8877/)。接口文档位于 `/docs`，协议位于 `/openapi.json`，健康检查为 `/api/health`。未传 `--port` 时默认端口为 8760；8877 是上述示例及浏览器测试使用的端口。

Web 启动时默认拉起单实例 worker。分析任务在独立子进程计算；API 关闭后 worker 继续处理已提交任务与同步调度。停止 worker：

```powershell
uv run market-research-worker --stop
```

需要自行管理进程时，在两个终端分别启动：

```powershell
$env:FIL_AUTO_SERVICES = '0'
uv run --locked market-research-worker
# 另一个终端，同样设置 FIL_AUTO_SERVICES=0
uv run --locked market-research --port 8877
```

多个 API 进程必须使用同一运行数据目录；当前支持单机本地文件系统，不支持多台机器通过共享网络盘组成队列。

## Docker 运行

根目录提供 `Dockerfile` 和 `.dockerignore`。使用 Linux 容器，在项目根目录执行：

```powershell
docker build -t market-research:local .
docker run -d --init --name market-research -p 8877:8877 -v market-research-data:/data market-research:local
```

访问 `http://127.0.0.1:8877/`。镜像使用 Python 3.13，通过 `uv.lock` 安装运行依赖和非 editable 项目包；最终镜像只复制安装环境，包含前端和 SQL 包资源，以 UID/GID 10001 运行。`.dockerignore` 采用允许列表，本机 `.venv`、行情、结果、Git、Node 依赖及测试产物不进入构建上下文。

命名卷 `market-research-data` 挂载到 `/data`，行情数据库、队列和日志位于 `/data/output/data`，研究输出也保存在该卷中。容器首次运行创建独立数据，**不会自动导入本机已有行情**。若使用宿主目录代替命名卷，应确保该目录可由容器 UID/GID 10001 写入。停止/删除容器不会删除命名卷；备份或迁移 DuckDB 前先停止容器内所有写入进程。

API 默认启动同容器 worker 和已添加资产同步调度，每日北京时间 18:30 同步；容器需保持运行，无需在容器中安装 Windows 定时任务。可用 `-e FIL_SYNC_TIME=19:00` 调整时间，或 `-e FIL_SYNC_ENABLED=0` 停用自动同步。容器停止会结束其所有进程，持久队列在重启后按租约规则恢复中断任务。

```powershell
docker logs -f market-research
docker exec market-research market-research-cli --help
docker stop market-research
```

HTTP 健康检查使用 `/api/health`；后台计算和同步日志保留在数据卷中。多阶段安装与非 editable 环境复制方式参考 [uv 官方 Docker 指南](https://docs.astral.sh/uv/guides/integration/docker/)。当前维护环境未安装 Docker，本次仅完成文件及配置核对，镜像构建和容器运行需在具备 Docker 的环境验证。

## 页面功能

| 入口 | 已实现功能 | 范围 |
| --- | --- | --- |
| 股票 | 日线 K 线/收盘线、均线、成交量、收益风险、基准比较、技术规则、5/20 交易日方向估计 | 沪深京 A 股；支持前复权、不复权和后复权 |
| 基金 | 净值、相关性、11 种拟合器比较、风格权重与漂移、超参重新学习、诊断回测 | 默认近 5 年，分析窗口下限 2 年 |
| 指数 | 走势、均线、收益风险、基准比较、技术规则及方向估计 | 页面支持 `sh000xxx`、`sz399xxx`；基金研究另使用申万行业指数 |
| 数据中心 | 添加证券、按类型/代码/名称/状态组合筛选、分页查看同步进度与错误、单口径重试 | 默认每页 20 条，可选 10/20/50/100 条；只同步已添加资产 |

股票示例 `600519`、`sz000001`；指数建议使用 `sh000300` 等完整代码。分析结果同时显示请求与实际行情区间，并提示相对基准缺少的日线；尚未核对交易日历，不据此直接判为过期或停牌。股票/指数区间至少 30 个自然日，基础统计至少需要 21 根日线；更长预热要求由技术分析模块单独提示。

日期与单选下拉由 `api/static/controls.js` 统一增强，支持键盘、直接输入日期、年月跳转和动态选项；不支持 Popover 的浏览器保留原生控件。

### 股票技术分析

股票和指数已共用技术分析模块。新增市场环境（基准相对 MA60 与20日动量）、20/60日相对强弱、ADX、ATR/布林带宽历史分位，以及成交量、成交额、换手率相对此前20日均值。缺少基准或成交字段时保留基础分析，缺失值不以零替代。指标按日线收盘计算；指数条件仍需在实际ETF交易上另行验证。

原方向模型保持为基线；固定候选额外使用上述环境与量价特征。页面按相同成熟日期、同样的非重叠抽样比较两者准确率、Brier和概率分组实际频率，不据展示区间排名自动替换原模型。分组是校准诊断，不是已经完成概率校准；独立未来验证、样本扩展和扣费交易验证仍在 TODO。

页面已新增信号日基准环境分组。离线验证可使用固定证券清单，对波动、量价、相对强弱及全部特征分别消融，开发期滚动训练、留出期冻结训练截止日，保存全部比较及数据指纹。运行方式、缺失处理、统计近似和当前证据范围见[股票指数验证流程](docs/股票指数验证流程.md)；扣费交易验证和独立未来数据验证仍待完成。

指标包括 MA20/60、MACD(12,26,9)、RSI14、布林带(20,2)、ATR14、量比与此前 20 日高低点。RSI/ATR 使用 Wilder 平滑。趋势规则为等权计分，首次满足条件才记录事件；规则分不等于上涨概率。

方向估计使用固定参数 `StandardScaler + LogisticRegression(C=0.1)`，预测未来 5/20 个交易日收盘是否上涨。每 21 个交易日重估，训练使用 252～756 个标签已成熟样本，缩放器只拟合训练数据。默认从同步历史中读取至多 1500 根日线预热，展示区间与训练区间分开；历史不足或复权历史不一致时降级处理。

验证展示成熟预测的准确率、Brier 分数及当时训练集上涨频率基准，按预测周期间隔取样。至少 20 笔才比较 Brier 优劣；该比较不是显著性检验，模型概率未单独校准。规则和模型方向冲突会提示，不输出精确目标股价，也不发送交易订单。

### 基金研究

候选池定义 47 个指数，实际使用数量由覆盖率过滤决定。所有拟合器共用滚动切分；黑箱模型使用真实样本外预测，其重要性不应解释为真实持仓。可以调参重新学习、手动选择模型、查看风格漂移及诊断回测。

基金页沿用股票页的标题、筛选条、示例卡、概览和图表样式。顶部支持近 2/3/5/10 年预设及自定义起止日期，分析区间至少两年；模型与仓位设置在首次分析前即可展开调整。修改设置后点击「应用区间」更新结果，净值与风险图表优先展示，随后是模型对比、调参和诊断回测。

同一评估区间内的模型排名存在选择偏差，不代表独立验证后的未来表现。分析窗口与回测窗口分开：`--window-*` 控制拟合样本，`analyze` 的 `--start/--end` 控制已拟合样本中的回测区间。

跨运行的长期模型档案、算法记录库和历史对比界面尚未实现，见[待评估方案](docs/模型档案与算法记录_重构方案.md)。

## 统一研究命令

根目录只保留 `run.py` 作为研究入口，业务实现位于 `market_research` 包。14 个命令共用注册表，支持帮助与显式参数转发：

```powershell
uv run python run.py --help
uv run python run.py analyze 110022 --window-years 5
uv run python run.py experiments --window-years 2
uv run python run.py holdings --help
uv run python run.py report --help
uv run python run.py cache-warm 110022
uv run python run.py cache-migrate
```

也可使用 `uv run market-research-cli ...` 或 `uv run python -m market_research.cli ...`。原 `uv run python run.py 110022` 仍默认基金分析；不带参数仍分析全部演示基金。查看帮助不会执行研究计算。

原来 10 个 `run_*` 文件归档到 `compat/entrypoints/`；原 `python run_experiments.py` 改为 `uv run python run.py experiments`。根目录另保留报告和缓存维护的 3 个兼容脚本。完整命令映射见[模块与入口](docs/模块与入口.md)。

## 数据存储与每日同步

行情直接写入 `FIL_DATA_DIR/market.duckdb`，没有 Parquet 过渡层。DuckDB 同时保存数据集、日线值、同步状态与分析任务。写入使用事务、短连接和跨进程文件锁。

旧行情 CSV 可通过 `cache-migrate` 导入，源文件保留；研究导出、持仓明细与样本池仍可使用 CSV，基金计算结果仍有本地 pickle 缓存。行情库中的任务结果以 JSON 持久化，和结果缓存是不同用途。

默认北京时间每天 **18:30** 同步已添加资产，包括实际用到的候选指数；演示目录不是同步名单。worker 在空闲时约每 30 秒检查待处理资产，批次运行期间等待当前批次结束。

| 类型 | 首次同步 | 后续同步 |
| --- | --- | --- |
| 不复权股票、指数 | 从 1990-01-01 请求数据源可提供的历史 | 回补最近 10 个自然日并合并 |
| 前/后复权股票 | 全历史 | 每次重新获取完整历史并替换，处理分红、拆分和数据源修订 |
| 基金 | 全历史 | 每次重建对应净值口径，处理分红及月度校准修订 |

全历史不代表上市前存在数据。空数据、历史倒退或复权历史被截断时保留旧数据并记录错误，服务运行期间按 30 分钟间隔重试。早期前复权价可能非正，分析时需改用后复权或缩短区间。当前没有完整可靠的公司行动源，因此复权历史采用整段刷新。

独立调度与单次检查：

```powershell
uv run market-research-sync
uv run market-research-sync --once
```

`--once` 仅检查当前到期/待处理资产，不会强制刷新所有已成功同步的资产。手动重试通过数据中心或 `/api/sync/retry` 提交。调度器通过同一同步锁互斥。

Windows 每日任务安装或更新：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_sync_task.ps1 -Time 18:30
```

任务名 `MarketResearch-DailySync`，优先使用项目 `.venv/Scripts/pythonw.exe` 和北京时间触发边界，支持错过时间后补跑；电脑需开机、当前用户需登录。重新安装时同步设置服务的 `FIL_SYNC_TIME`。`FIL_SYNC_ENABLED=0` 只关闭 worker 管理的同步，Windows 任务和独立 sync 进程需分别停用。

## 配置、接口和日志

保留 `FIL_*` 环境变量是为了兼容既有配置，不代表项目包名。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| FIL_HOME | 源码根目录；安装包通常为当前目录 | 运行产物根目录 |
| FIL_DATA_DIR | FIL_HOME/output/data | DuckDB、任务、缓存及日志目录 |
| FIL_HOST / FIL_PORT | 127.0.0.1 / 8760 | HTTP 监听地址，命令参数可覆盖 |
| FIL_AUTO_SERVICES | 1 | API 自动启动 worker |
| FIL_MAX_JOBS / FIL_QUEUE_LIMIT | 2 / 100 | 分析进程并发数；排队和运行任务总上限 |
| FIL_JOB_TIMEOUT / FIL_JOB_ATTEMPTS | 900 / 3 | 单次计算超时秒数；进程中断最大尝试次数 |
| FIL_SYNC_ENABLED / FIL_SYNC_TIME | 1 / 18:30 | worker 同步开关及北京时间 |
| FIL_CORS_ORIGINS | 空 | 逗号分隔跨域来源；同源页面无需配置 |
| FIL_START_DATE | 未设置 | 固定基金分析起点，覆盖滚动默认值，用于研究复现 |

任务通过 `POST /api/analyze`、`/api/refit`、`/api/backtest` 或 `/api/market/analyze` 提交，再查询 `/api/job/{jid}` 与 `/api/result/{jid}`。完成的基金计算可能直接命中结果缓存；以实际返回字段为准。取消为 `POST /api/job/{jid}/cancel`，进入错误终态并由 worker 终止对应进程；切换页面入口不会自动取消后台任务。

参数错误为 400，容量不足为 429，未完成结果为 202，不存在/过期任务为 404。队列及结果在 API 重启后可继续查询，终态满 24 小时后在新任务提交时清理。持久任务不等于长期研究档案。

同步接口为 `GET /api/sync`、`POST /api/sync/assets`（`kind/code/basis`）、`POST /api/sync/retry`（`asset_key`）。股票 basis 为 `qfq`、空字符串或 `hfq`，指数为空，基金为 `total` 或 `cum`。

API 使用 Uvicorn 日志；worker、单任务与同步日志位于 `FIL_DATA_DIR` 下的 `worker.log`、`job-{id}.log`、`sync-process.log` 和 `sync.log`。

## 代码与文档导航

```text
src/market_research/
  api/          FastAPI、路由、协议与 static/ 前端资源
  services/     API 和 worker 共用的分析编排
  market/       股票、指数和技术计算
  funds/        基金分析、持仓与样本池
  modeling/     拟合器、滚动评估、信号与回测
  data/         数据源、缓存、面板、DuckDB 与 migrations/
  tasks/        持久队列、worker 与已添加资产同步
  research/     实验、统计检验、报告及 commands/
  validation/   窗口与模型参数规则
  cli/          命令注册、分发及基础命令
compat/         旧导入转发与 entrypoints/ 归档脚本
tests/          按业务与测试类型分组
scripts/        Windows 同步任务安装脚本
```

| 文档 | 用途 |
| --- | --- |
| [模块与入口](docs/模块与入口.md) | 当前目录、模块职责、14 个命令与旧入口映射 |
| [架构设计](docs/架构设计.md) | 当前数据流、进程与恢复规则、部署约束 |
| [功能完整性检查](docs/功能完整性检查.md) | 当前已实现能力、缺口与验证边界 |
| [研究方案](docs/研究方案.md) | 基金研究方法、带样本/窗口口径的历史实验 |
| [结论快照](docs/结论快照_口径修正前.md) | 修正前基线及后续重算对照，不能当当前绩效 |
| [扩池预注册规则](docs/扩池预注册规则.md) | 冻结规则、历史执行状态与未满足条件 |
| [合成基准与净值差距](docs/合成基准与基金净值的差距.md) | 历史诊断、股息口径和残差解释限制 |
| [模型档案方案](docs/模型档案与算法记录_重构方案.md) | 尚未实现的产品提案及当前架构修订 |
| [TODO](TODO.md) | 当前待办摘要和有日期的历史审查记录 |
| [测试说明](tests/README.md) / [兼容说明](compat/README.md) | 验证方式及旧入口边界 |

## 验证

```powershell
uv run python -m unittest discover -s tests -t . -p 'test_*.py'
uv run python tests/regression/test_p0_fixes.py
uv run python tests/regression/test_regressions.py
node tests/frontend/test_refit.js
node tests/frontend/test_ui_async.cjs
```

现有验证记录为 68 项 unittest 通过，14 个统一子命令帮助检查通过，并完成 wheel 独立安装入口检查。文档更新没有重新执行完整数值研究或浏览器回归。测试前提、浏览器启动与实际基金计算见[测试说明](tests/README.md)。

依赖以 `pyproject.toml` 和 `uv.lock` 为准。`requirements.txt` 引用运行依赖兼容导出，更新命令为：

```powershell
uv export --locked --no-dev --no-emit-project --no-hashes -o requirements.lock.txt
```

## 研究解释与限制

基金历史研究支持将 RBSA 作为风格解释工具；权重不等于真实持仓，较高 R² 也不能证明预测能力。已有样本未建立稳定择时优势，时间向前验证、窗口变化与扩池曾改变结论。数值应连同样本、窗口、收益口径和产物日期阅读，不能把不同轮次结果混成一组。

基金总回报采用事后月度锚点校准，未来锚点可能修订历史；股票复权历史也会随公司行动修订。当前滚动验证不是严格历史时点可得的可执行回测。交易诊断未完整模拟申赎确认、持有期费用、滑点、停牌与涨跌停，不能由简化成本模型推导任何份额类别可直接交易。

本项目提供研究与人工判断辅助，不承诺未来收益，不自动交易。
