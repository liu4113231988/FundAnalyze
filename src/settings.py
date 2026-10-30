"""Legacy module alias; new code imports market_research.settings."""
import sys
from importlib import import_module

_module = import_module("market_research.settings")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
