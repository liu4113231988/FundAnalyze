"""Compatibility entry; implementation lives in market_research."""
from market_research.cli.cache_warm import main

if __name__ == "__main__":
    raise SystemExit(main())
