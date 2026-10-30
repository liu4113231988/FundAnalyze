"""导出文件和派生结果的原子写入；行情持久化由 storage 负责。"""
import pickle
from pathlib import Path
import pandas as pd


def _os_replace_retry(src: Path, dst: Path, retries: int = 6):
    """os.replace 带重试。Windows 下目标被另一线程短暂占用会抛 WinError 32。"""
    import os
    import time
    for i in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def _atomic_csv_write(df: pd.DataFrame, path: Path):
    """唯一临时文件防止并发覆盖；替换失败仍清理临时文件。"""
    import os
    import threading
    import uuid
    tmp = path.with_name(f"{path.name}.{os.getpid()}."
                         f"{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        df.to_csv(tmp, encoding="utf-8-sig")
        _os_replace_retry(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _atomic_pickle_dump(obj, path: Path, retries: int = 6):
    """唯一临时文件防止并发覆盖；替换失败仍清理临时文件。"""
    import os
    import threading
    import uuid

    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump(obj, f)
        _os_replace_retry(tmp, path, retries)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
