from __future__ import annotations
import os, socket, time, contextlib
from typing import Optional
import pandas as pd

def _default_http_timeout() -> float:
    """抓取超时（秒）。FIL_HTTP_TIMEOUT=0 表示不限制。"""
    try:
        v = float(os.environ.get("FIL_HTTP_TIMEOUT", "30"))
    except (TypeError, ValueError):                        # noqa: BLE001
        return 30.0
    return v if v > 0 else 0.0


@contextlib.contextmanager
def http_timeout(seconds: Optional[float] = None):
    """
    只在抓取期间生效的 socket 默认超时（P1-07 / N-05）。

    ★ 为什么不能写在模块级 `socket.setdefaulttimeout(30)`：
      它是**进程全局**的，会影响本进程里之后新建的**所有** socket ——
      包括 Flask 服务器的监听 socket 与已 accept 的连接 socket。
      本意只是防止 AkShare 请求永久挂起，结果把 Web 服务也罩进同一个
      30 秒超时里（长连接、慢客户端都可能被误杀）。改成上下文管理器后，
      只有真正在抓数据的代码段受影响。
    """
    old = socket.getdefaulttimeout()
    sec = _default_http_timeout() if seconds is None else seconds
    socket.setdefaulttimeout(sec if sec and sec > 0 else None)
    try:
        yield
    finally:
        socket.setdefaulttimeout(old)


def _retry(fn, tries: int = 4, wait: float = 1.2, **kw):
    last = None
    for i in range(tries):
        # ★ 超时只在抓取期间生效（N-05）：AkShare 的请求没有暴露 timeout 参数，
        #   但 socket 级的全局默认值会波及 Web 服务器，所以必须局部设置。
        with http_timeout():
            try:
                return fn(**kw)
            except Exception as e:      # noqa: BLE001
                last = e
        time.sleep(wait * (i + 1))
    raise last


def _clean_series(s: pd.Series, start: Optional[str] = None) -> pd.Series:
    """通用清洗：剔除非正值、按日期去重、截断起点。"""
    s = s[s > 0]
    s = s[~s.index.duplicated(keep="last")]
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s
