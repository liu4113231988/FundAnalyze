"""Unified root entry; use --help to list commands."""
from market_research.cli.main import root_main as main

if __name__ == "__main__":
    raise SystemExit(main())
