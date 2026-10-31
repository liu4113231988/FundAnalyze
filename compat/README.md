# 旧导入与入口兼容层

核对日期：2026-10-31。新业务代码统一导入 `market_research`，当前对应关系见[模块与入口](../docs/模块与入口.md)。

- `src/`、`webapp/` 在本目录下是旧 Python 包名，文件仅转发到同一规范模块对象，保留历史导入、单例及 pickle 类路径。
- 源码工作区根部的 `src/__init__.py`、`webapp/__init__.py` 只桥接包搜索路径，不修改全局 `sys.path`。wheel 直接把本目录包安装为旧命名。
- `entrypoints/` 保存原来的 10 个 `run_*` 脚本，仅用于源码工作区兼容；根目录 `run.py` 及 3 个报告/缓存维护脚本同样只负责转发。
- 前端静态资源属于 `src/market_research/api/static/`，SQL 属于 `src/market_research/data/migrations/`，均随规范包发布，不再放在兼容层。

推荐使用 `uv run python run.py experiments --help` 或 `uv run market-research-cli experiments --help`。确需归档脚本时，可从已执行 `uv sync --locked` 的源码根目录运行 `uv run python compat/entrypoints/run_experiments.py --help`；wheel 用户应使用控制台/模块命令，不能依赖源码脚本路径。

兼容层不是另一套实现。新增模块不依赖旧命名，新增研究命令登记 `market_research.cli.registry`，不重新增加根目录 `run_*` 文件。
