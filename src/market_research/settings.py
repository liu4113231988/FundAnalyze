"""Runtime configuration shared by API, workers and scheduled sync."""
from dataclasses import dataclass
import os
from pathlib import Path


def home_dir():
    default = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()), Path.cwd())
    return Path(os.environ.get("FIL_HOME", default)).resolve()


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8760
    max_jobs: int = 2
    queue_limit: int = 100
    job_timeout: int = 900
    max_attempts: int = 3
    auto_services: bool = True

    @classmethod
    def load(cls):
        return cls(host=os.environ.get("FIL_HOST", "127.0.0.1"),
                   port=int(os.environ.get("FIL_PORT", "8760")),
                   max_jobs=max(1, int(os.environ.get("FIL_MAX_JOBS", "2"))),
                   queue_limit=max(1, int(os.environ.get("FIL_QUEUE_LIMIT", "100"))),
                   job_timeout=max(1, int(os.environ.get("FIL_JOB_TIMEOUT", "900"))),
                   max_attempts=max(1, int(os.environ.get("FIL_JOB_ATTEMPTS", "3"))),
                   auto_services=os.environ.get("FIL_AUTO_SERVICES", "1") != "0")
