"""Process locks and hidden child launch shared by service entry points."""
from contextlib import contextmanager
import os
import subprocess
import sys
import market_research.config as C

_JOB_HANDLE = None


def contain_children():
    """On Windows, kill model grandchildren when their owning job exits.

    The non-inheritable handle is owned only by this calculation process. Kernel
    cleanup covers native crashes and forced termination, not just Python finally.
    """
    global _JOB_HANDLE
    if os.name != 'nt' or _JOB_HANDLE is not None:
        return
    import ctypes
    from ctypes import wintypes as W
    class Basic(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',W.DWORD),
                  ('min_memory',ctypes.c_size_t),('max_memory',ctypes.c_size_t),('process_limit',W.DWORD),
                  ('affinity',ctypes.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
    class Io(ctypes.Structure):
        _fields_=[(name,ctypes.c_uint64) for name in ('reads','writes','other','read_bytes','write_bytes','other_bytes')]
    class Limits(ctypes.Structure):
        _fields_=[('basic',Basic),('io',Io),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                  ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,W.LPCWSTR];kernel.CreateJobObjectW.restype=W.HANDLE
    kernel.SetInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD];kernel.SetInformationJobObject.restype=W.BOOL
    kernel.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];kernel.AssignProcessToJobObject.restype=W.BOOL
    kernel.GetCurrentProcess.restype=W.HANDLE
    kernel.CloseHandle.argtypes=[W.HANDLE]
    handle=kernel.CreateJobObjectW(None,None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits=Limits();limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle,kernel.GetCurrentProcess()):
        error=ctypes.get_last_error();kernel.CloseHandle(handle);raise ctypes.WinError(error)
    _JOB_HANDLE=handle


def terminate_tree(process):
    if os.name=='nt':
        # Calculation children use contain_children(); killing the owner closes
        # the job handle, which also terminates model pool descendants.
        if process.poll() is None:
            process.kill()
    else:
        import signal
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def exclusive(name):
    C.DATA_DIR.mkdir(parents=True,exist_ok=True)
    with (C.DATA_DIR/name).open('a+b') as file:
        file.seek(0,2)
        if not file.tell():
            file.write(b'0');file.flush()
        file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            if os.name=='nt':
                file.seek(0);msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(file.fileno(),fcntl.LOCK_UN)


def launch(module,*arguments,log_name='worker.log'):
    C.DATA_DIR.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'FIL_HOME':str(C.ROOT),'FIL_DATA_DIR':str(C.DATA_DIR),'PYTHONUNBUFFERED':'1','PYTHONUTF8':'1'}
    with (C.DATA_DIR/log_name).open('ab') as log:
        return subprocess.Popen([sys.executable,'-m',module,*arguments],cwd=C.ROOT,env=environment,
                                stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,
                                start_new_session=os.name!='nt')
