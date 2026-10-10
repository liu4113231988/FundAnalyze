# -*- coding: utf-8 -*-
"""
启动 / 重启 Fund Lens Web 服务。

为什么需要这个脚本：Windows 上 `pkill -f` 不生效，反复重启会在端口上
堆积多个旧进程，curl 请求被分流到旧代码，出现"改了代码但没生效"的假象。
本脚本会先按端口清理，再启动，并用 /api/health 确认新进程已就绪。

用法：
    python webapp/serve.py                     # 默认 127.0.0.1:8760
    python webapp/serve.py --port 9000
    python webapp/serve.py --restart           # 先清理再启动
    python webapp/serve.py --host0.0.0.0      # 局域网访问
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def port_busy(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host, port)) == 0


def kill_port(port: int) -> int:
    """杀掉占用端口的所有进程，返回杀掉的进程数。"""
    killed = 0
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen "
             f"-ErrorAction SilentlyContinue | Select-Object -ExpandProperty "
             f"OwningProcess -Unique; "
             f"if ($c) {{ $c | ForEach-Object {{ "
             f"try {{ Stop-Process -Id $_ -Force -ErrorAction Stop; "
             f"Write-Output $_ }} catch {{ }} }} }}"],
            capture_output=True, text=True, timeout=40)
        killed = len([x for x in out.stdout.split() if x.strip()])
    except Exception as e:                              # noqa: BLE001
        print(f"  [serve] 端口清理异常（可忽略）: {e}")
    return killed


def wait_healthy(url: str, timeout: float = 40.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:                               # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--restart", action="store_true",
                    help="先杀掉占用端口的旧进程再启动")
    a = ap.parse_args()

    host = "127.0.0.1" if a.host in ("0.0.0.0", "host0.0.0.0") else a.host
    url = f"http://{host}:{a.port}"

    if a.restart or port_busy(host, a.port):
        print(f"[serve] 清理端口 {a.port} ...")
        n = kill_port(a.port)
        for _ in range(20):
            if not port_busy(host, a.port):
                break
            time.sleep(0.5)
        if port_busy(host, a.port):
            print(f"[serve] 警告：端口 {a.port} 仍被占用，可能启动失败")
        else:
            print(f"[serve] 已清理（{n} 个进程）")

    cmd = [sys.executable, "-u", str(ROOT / "webapp" / "app.py"),
           "--host", a.host, "--port", str(a.port)]
    print(f"[serve] 启动: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=str(ROOT))

    if wait_healthy(url + "/api/health"):
        print(f"\n  ✓ Fund Lens 已就绪 → {url}\n")
        print(f"    在浏览器打开该地址，输入 6 位基金代码（如 110022）即可。")
        print(f"    停止服务：Ctrl+C 或运行  python webapp/serve.py --restart\n")
    else:
        print(f"[serve] 未在 40s 内就绪，请检查输出。")
    try:
        p.wait()
    except KeyboardInterrupt:
        print("\n[serve] 已停止")


if __name__ == "__main__":
    main()