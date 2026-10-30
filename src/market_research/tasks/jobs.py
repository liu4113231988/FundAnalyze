"""单进程后台任务状态与限流；计算函数通过回调提交。"""
from __future__ import annotations
import uuid
import threading
from threading import Thread
import time
import traceback
from typing import Dict, List, Optional
from market_research.errors import JobsBusy

JOB_TTL_SEC = 30 * 60


class JobStore:
    """内存任务表；结果持久化由调用方负责，超时不会强杀 Python 线程。"""

    JOB_TIMEOUT_SEC = 15 * 60

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._cancel: set = set()

    def _prune_locked(self):
        """惰性清理：done/error 超过 TTL 的任务整条删除，避免结果常驻内存。"""
        now = time.time()
        expired = [jid for jid, j in self._jobs.items()
                   if j["status"] in ("done", "error")
                   and now - j["created"] >= JOB_TTL_SEC]
        for jid in expired:
            del self._jobs[jid]
            self._cancel.discard(jid)

    def _sweep_timeout_locked(self):
        """只标记超时；实际计算退出后由 runner 释放并发名额。"""
        now = time.time()
        for jid, j in self._jobs.items():
            if j["status"] == "running" and now - j["created"] > self.JOB_TIMEOUT_SEC:
                j["status"] = "error"
                j["error"] = (f"任务超时（>{self.JOB_TIMEOUT_SEC // 60} 分钟），"
                              "后台计算退出后释放名额；请检查网络")
                j["stage"] = "超时"
                self._cancel.add(jid)

    # ---- 相同任务合并（N-08）
    def find_running(self, key: str) -> Optional[str]:
        """同参数请求复用正在运行的任务。"""
        with self._lock:
            self._sweep_timeout_locked()
            for jid, j in self._jobs.items():
                if j["status"] == "running" and j.get("key") == key:
                    return jid
        return None

    def is_cancelled(self, jid: str) -> bool:
        with self._lock:
            return jid in self._cancel

    def cancel(self, jid: str) -> bool:
        """请求取消。worker 应在每个阶段检查 is_cancelled 并及时退出。"""
        with self._lock:
            if jid not in self._jobs or self._jobs[jid]["status"] != "running":
                return False
            self._cancel.add(jid)
            self._jobs[jid].update(cancel_requested=True, status="error",
                                   stage="已取消", error="任务已取消；后台计算退出后释放名额")
        return True

    def create(self, code: str, models: List[str], key: Optional[str] = None) -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._prune_locked()
            self._sweep_timeout_locked()
            self._jobs[jid] = {
                "id": jid, "code": code, "models": models,
                "key": key or f"{code}|{','.join(sorted(models))}",
                "status": "running", "stage": "准备数据", "progress": 0.0,
                "detail": [], "result": None, "error": None,
                "cancel_requested": False,
                "created": time.time(),
            }
        return jid

    def update(self, jid: str, **kw):
        with self._lock:
            if (jid in self._jobs and self._jobs[jid]["status"] == "running"
                    and jid not in self._cancel):
                self._jobs[jid].update(kw)

    def get(self, jid: str) -> Optional[dict]:
        with self._lock:
            self._sweep_timeout_locked()
            job = self._jobs.get(jid)
            return dict(job) if job else None

    def finish(self, jid: str, result: dict):
        self.update(jid, status="done", progress=1.0, stage="完成", result=result)

    def fail(self, jid: str, err: str):
        self.update(jid, status="error", stage="失败", error=err)

    def public(self, jid: str) -> Optional[dict]:
        j = self.get(jid)
        if not j:
            return None
        return {k: v for k, v in j.items() if k != "result"}


class JobRunner:
    """所有计算入口共用创建、去重、线程启动与名额释放。"""

    def __init__(self, store, max_jobs=2):
        self.store = store
        self.max_jobs = max(1, max_jobs)
        self._slots = threading.BoundedSemaphore(self.max_jobs)
        self._lock = threading.Lock()

    def submit(self, code, models, work, key=None):
        with self._lock:
            existing = self.store.find_running(key) if key else None
            if existing:
                return existing, True
            if not self._slots.acquire(blocking=False):
                raise JobsBusy(f"同时运行的分析任务已达上限 {self.max_jobs}，请稍后重试")
            jid = None
            try:
                jid = self.store.create(code, models, key=key)

                def worker():
                    try:
                        self.store.finish(jid, work(jid))
                    except Exception as error:
                        traceback.print_exc()
                        self.store.fail(jid, f"{type(error).__name__}: {str(error)[:300]}")
                    finally:
                        self._slots.release()

                Thread(target=worker, name=f"job-{jid}", daemon=True).start()
            except Exception:
                self._slots.release()
                if jid is not None:
                    self.store.fail(jid, "无法启动后台任务")
                raise
            return jid, False


# Production uses the cross-process store. The in-memory implementation above
# remains available for isolated algorithm tests and local callback consumers.
from market_research.tasks.store import DurableJobStore
JOBS = DurableJobStore()
