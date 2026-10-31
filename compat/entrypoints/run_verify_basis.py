"""Archived entry; prefer run.py subcommands."""
from market_research.cli.main import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main(__file__))
