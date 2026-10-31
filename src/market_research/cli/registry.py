"""Command metadata shared by the console, root script and legacy adapters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    module: str
    summary: str
    legacy_script: str


COMMAND_SPECS = (
    Command('analyze', 'cli.analyze', '基金分析与回测', 'run.py'),
    Command('cache-warm', 'cli.cache_warm', '行情缓存预热', 'warm_cache.py'),
    Command('cache-migrate', 'cli.cache_migrate', '旧行情缓存导入', 'migrate_cache.py'),
    Command('report', 'research.commands.report', '生成研究报告', 'build_report.py'),
    Command('expand-pool', 'research.commands.expand_pool', '构建基金研究样本池', 'run_expand_pool.py'),
    Command('experiments', 'research.commands.experiments', '基础实验与通道诊断', 'run_experiments.py'),
    Command('extended', 'research.commands.extended', '扩展样本与周期实验', 'run_extended.py'),
    Command('holdings', 'research.commands.holdings', '真实持仓验证', 'run_holdings.py'),
    Command('increment', 'research.commands.increment', '风格拟合增量价值', 'run_increment.py'),
    Command('power', 'research.commands.power', '统计功效分析', 'run_power.py'),
    Command('robustness', 'research.commands.robustness', '稳健性检验', 'run_robustness.py'),
    Command('selection', 'research.commands.selection', '选择偏差检验', 'run_selection.py'),
    Command('verify-basis', 'research.commands.verify_basis', '核对基金收益口径', 'run_verify_basis.py'),
    Command('walkforward', 'research.commands.walkforward', '时间向前验证', 'run_walkforward.py'),
)

COMMANDS = {command.name: command.module for command in COMMAND_SPECS}
LEGACY_COMMANDS = {command.legacy_script: command.name for command in COMMAND_SPECS}
