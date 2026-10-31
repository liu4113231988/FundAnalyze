"""Source-checkout bridge to the isolated compatibility package."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parent.parent / "compat" / "src")]
