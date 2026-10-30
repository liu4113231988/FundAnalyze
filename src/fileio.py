"""Legacy module alias; new code imports market_research.data.files."""
import sys
from importlib import import_module

_module = import_module("market_research.data.files")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
