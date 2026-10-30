"""Compatibility entry; implementation lives in market_research."""
from market_research.research.commands.holdings import main

if __name__ == "__main__":
    raise SystemExit(main())
