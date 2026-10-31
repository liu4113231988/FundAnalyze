"""Legacy module alias; new code imports market_research.funds.pipeline."""
import sys
from importlib import import_module

_module = import_module("market_research.funds.pipeline")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
