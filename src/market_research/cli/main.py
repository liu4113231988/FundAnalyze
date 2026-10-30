"""Unified research and data-maintenance commands."""
import argparse
from importlib import import_module
import sys

COMMANDS = {'analyze': 'cli.analyze', 'cache-warm': 'cli.cache_warm', 'cache-migrate': 'cli.cache_migrate', 'report': 'research.commands.report', 'expand-pool': 'research.commands.expand_pool', 'experiments': 'research.commands.experiments', 'extended': 'research.commands.extended', 'holdings': 'research.commands.holdings', 'increment': 'research.commands.increment', 'power': 'research.commands.power', 'robustness': 'research.commands.robustness', 'selection': 'research.commands.selection', 'verify-basis': 'research.commands.verify_basis', 'walkforward': 'research.commands.walkforward'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in COMMANDS:
        sub.add_parser(name,add_help=False,help=name.replace('-',' '))
    args,remaining=parser.parse_known_args()
    original = sys.argv
    sys.argv=[f'market-research-cli {args.command}',*remaining]
    try:
        return import_module('market_research.'+COMMANDS[args.command]).main()
    finally:
        sys.argv=original

if __name__=='__main__':
    raise SystemExit(main())
