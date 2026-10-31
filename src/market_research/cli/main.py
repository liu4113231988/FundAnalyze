"""统一研究与数据维护入口；命令延迟加载并显式接收参数。"""
import argparse
from importlib import import_module
from pathlib import Path
import sys

from .registry import COMMANDS, COMMAND_SPECS, LEGACY_COMMANDS


def execute(command: str, arguments: list[str], *, prog: str):
    entry = import_module('market_research.' + COMMANDS[command]).main
    return entry(list(arguments), prog=prog)


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    subcommands = parser.add_subparsers(dest='command', required=True)
    for command in COMMAND_SPECS:
        subcommands.add_parser(command.name, add_help=False, help=command.summary)
    args, remaining = parser.parse_known_args(argv)
    return execute(args.command, remaining, prog=f'{parser.prog} {args.command}')


def root_main(argv=None):
    """One root script, retaining implicit legacy fund-analysis arguments."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    program = Path(sys.argv[0]).name
    if arguments and (arguments[0] in COMMANDS or arguments[0] in ('-h', '--help')):
        return main(arguments, prog=program)
    if not arguments or arguments[0].startswith('-') or (len(arguments[0]) == 6 and arguments[0].isdigit()):
        return execute('analyze', arguments, prog=program)
    return main(arguments, prog=program)


def legacy_main(filename, argv=None):
    """Single adapter for archived scripts and maintenance wrappers."""
    name = Path(filename).name
    try:
        command = LEGACY_COMMANDS[name]
    except KeyError:
        raise ValueError(f'未注册的旧入口：{name}') from None
    return execute(command, list(sys.argv[1:] if argv is None else argv), prog=name)


if __name__ == '__main__':
    raise SystemExit(main())
