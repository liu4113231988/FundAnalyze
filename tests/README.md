# 测试说明

核对日期：2026-10-31。先在项目根目录运行 `uv sync --locked`。下列命令使用可安装的 `market_research` 包及当前测试目录；无需修改 `PYTHONPATH`。

| 目录 | 内容 |
| --- | --- |
| api | HTTP 校验、响应、缓存与任务提交 |
| data | DuckDB 事务、迁移及进程间访问 |
| funds | 基金池与持仓检查 |
| market | 股票/指数行情和纯技术计算 |
| tasks | 同步、内存隔离测试、持久队列及 worker |
| architecture | 模块边界、兼容导入、包资源与统一 CLI |
| frontend | 参数绑定及异步状态回归 |
| browser | Edge CDP 与 jsdom 检查 |
| regression | 通过独立 main 运行的数值回归 |
| integration | 使用实际行情或模型的计算验证 |
| tools | 数据源探针、计时和诊断工具 |

## 常规回归

```powershell
uv run python -m unittest discover -s tests -t . -p 'test_*.py'
uv run python tests/regression/test_p0_fixes.py
uv run python tests/regression/test_regressions.py
node tests/frontend/test_refit.js
node tests/frontend/test_ui_async.cjs
```

unittest 使用 `-t .` 按 `tests.*` 加载，避免分组目录遮蔽第三方包。自动发现不会调用研究脚本的 main，也不等于执行全部真实行情/完整模型验证。Windows ASGI 测试需要允许本机 socketpair；受限沙箱可能阻止该操作。

## 浏览器检查

使用 Node.js 22+、本地 8877 服务和专用 Edge CDP 9229。先启动服务，再用独立测试用户目录启动 Edge，例如：

```powershell
uv run market-research --port 8877
# 另一个终端；按实际安装位置调用 Edge
& "${env:ProgramFiles(x86)}/Microsoft/Edge/Application/msedge.exe" --remote-debugging-port=9229 "--user-data-dir=$PWD/output/edge-test-profile" http://127.0.0.1:8877/
```

保持测试浏览器打开，在项目根目录执行：

```powershell
node tests/browser/market_browser.cjs
node tests/browser/technical_browser.cjs
node tests/browser/datacenter_browser.cjs
node tests/browser/controls_browser.cjs
```

测试会操作 CDP 实例中的页面，截图/状态产物在 `output/ui-evidence/`。不要连接日常使用的浏览器配置。`tests/browser/dom_check.js` 是另一项 jsdom 检查，需额外具备它导入的 Node 依赖；不是上述无额外依赖 CDP 套件的前提。

## 实际基金计算与打包

```powershell
uv run python tests/integration/test_webapi.py 110022 3
uv run python run.py --help
uv run python -m market_research.cli experiments --help
uv build
```

`test_webapi.py` 直接调用基金分析函数，不经过 HTTP；3 是最大重估点数，不是模型数或运行次数。缓存缺失时可能联网、计算较久，不能把它当路由层测试。轮询/HTTP 测试由 api 组另行覆盖。

wheel 检查应在独立虚拟环境安装构建产物，在源码目录之外验证控制台命令、静态资源和 SQL 可读取；源码兼容脚本不是安装包用户的调用方式。

## 已有验证记录

最近代码调整记录为 68 项 unittest 通过、两套独立数值回归通过、两套前端回归通过、4 套 Edge 页面检查通过；统一 CLI 另验证 14 个子命令帮助和 wheel 独立安装。证据位于本地 output 日志和界面产物，通常不随 Git 分发。本次文档更新未重新执行完整研究和浏览器回归，记录不承诺其他机器、数据版本或网络下相同结果。
