"""Queue durability, fencing, concurrency and isolated worker integration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
import market_research.config as C
import market_research.data.storage as storage
import market_research.tasks.sync as sync
from market_research.tasks.store import DurableJobStore, bind_attempt, LEASE_SECONDS
from market_research.errors import JobsBusy


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=C.ROOT/'output')
        self.config=patch.object(C,'DATA_DIR',Path(self.temp.name));self.config.start()
        self.store=DurableJobStore()

    def tearDown(self):
        self.config.stop();self.temp.cleanup()

    def enqueue(self,key='same'):
        return self.store.enqueue('sh000300',['index'],'market',{'kind':'index','code':'sh000300','start':'2024-01-01','end':'2024-06-01','adjust':'','benchmark':'sh000300','refresh':False},key=key)

    def test_cross_instance_dedup_and_queue_limit(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            results=list(pool.map(lambda _:self.enqueue(),range(6)))
        self.assertEqual(len({jid for jid,_ in results}),1)
        self.assertEqual(sum(not merged for _,merged in results),1)
        with patch.dict(os.environ,{'FIL_QUEUE_LIMIT':'1'}):
            with self.assertRaises(JobsBusy):self.enqueue('different')
        other=DurableJobStore()
        self.assertEqual(other.public(results[0][0])['status'],'queued')
        self.assertNotIn('result',other.public(results[0][0]))

    def test_lease_recovery_fences_late_progress_and_result(self):
        jid,_=self.enqueue()
        first=self.store.claim()
        self.assertTrue(self.store.heartbeat(jid,first['token']))
        with patch('market_research.tasks.store.time.time',return_value=time.time()+LEASE_SECONDS+1):
            second=self.store.claim()
        self.assertEqual(second['id'],jid)
        self.assertNotEqual(first['token'],second['token'])
        with bind_attempt(first['token']):
            self.store.update(jid,stage='stale')
            self.store.finish(jid,{'stale':True})
        self.assertEqual(self.store.get(jid)['status'],'running')
        with bind_attempt(second['token']):
            self.store.update(jid,stage='current',progress=.8)
            self.store.finish(jid,{'answer':42})
        self.assertEqual(self.store.get(jid)['result'],{'answer':42})
        self.store.fail(jid,'late failure',token=first['token'])
        self.assertEqual(self.store.get(jid)['status'],'done')

    def test_crash_retry_bound_and_cancel(self):
        jid,_=self.enqueue()
        with patch.dict(os.environ,{'FIL_JOB_ATTEMPTS':'2'}):
            for _ in range(2):
                job=self.store.claim();self.store.retry_crashed(jid,job['token'],'crash')
        self.assertEqual(self.store.get(jid)['status'],'error')
        second,_=self.enqueue('cancel')
        self.assertTrue(self.store.cancel(second))
        self.assertIsNone(self.store.claim())

    def test_worker_process_runs_saved_command_and_result_survives_api_restart(self):
        index=pd.bdate_range('2024-01-01',periods=80)
        close=pd.Series(range(100,180),index=index,dtype=float)
        bars=pd.DataFrame({'open':close,'close':close,'high':close+1,'low':close-1})
        name=sync.history_name({'asset_key':'index:sh000300:'})
        storage.write(name,bars,{'through':'2024-06-01','info':{'source':'offline integration fixture'}})
        jid,_=self.enqueue()
        environment={**os.environ,'FIL_HOME':str(C.ROOT),'FIL_DATA_DIR':self.temp.name,'FIL_AUTO_SERVICES':'0','FIL_SYNC_ENABLED':'0'}
        result=subprocess.run([sys.executable,'-m','market_research.tasks.worker','--once'],cwd=C.ROOT,env=environment,capture_output=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        job=DurableJobStore().get(jid)
        self.assertEqual(job['status'],'done',job['error'])
        self.assertAlmostEqual(job['result']['stats']['total_return'],.79)
        from fastapi.testclient import TestClient
        from market_research.api.app import create_app
        for _ in range(2):
            response=TestClient(create_app()).get('/api/result/'+jid)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['result']['code'],'sh000300')

    def test_http_body_limit_and_openapi_contract(self):
        from fastapi.testclient import TestClient
        from market_research.api.app import create_app
        client=TestClient(create_app())
        response=client.post('/api/analyze',content=b'x'*(8*1024*1024+1),headers={'Content-Type':'application/json'})
        self.assertEqual(response.status_code,413)
        self.assertEqual(response.json()['field'],'body')
        document=client.get('/openapi.json').json()
        self.assertIn('/api/market/analyze',document['paths'])
        self.assertIn('MarketRequest',document['components']['schemas'])
        self.assertEqual(client.post('/api/analyze',json={'code':'110022','refresh':'false'}).status_code,400)

    @unittest.skipUnless(os.name=='nt','Windows job-object process-tree isolation')
    def test_native_exit_cleans_up_model_descendants(self):
        code="import os,sys,subprocess; from market_research.tasks.processes import contain_children; contain_children(); p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); os._exit(23)"
        process=subprocess.Popen([sys.executable,'-c',code],cwd=C.ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            output,error=process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            output,error=process.communicate(timeout=5)
            if output.strip().isdigit():
                subprocess.run(['taskkill','/PID',output.decode().strip(),'/T','/F'],capture_output=True)
            self.fail('Calculation descendant survived owning process exit')
        self.assertEqual(process.returncode,23,error.decode(errors='replace'))
        child_pid=int(output.strip())
        import ctypes
        from ctypes import wintypes as W
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];kernel.OpenProcess.restype=W.HANDLE
        kernel.GetExitCodeProcess.argtypes=[W.HANDLE,ctypes.POINTER(W.DWORD)]
        kernel.CloseHandle.argtypes=[W.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,child_pid)
        if handle:
            try:
                exit_code=W.DWORD()
                self.assertTrue(kernel.GetExitCodeProcess(handle,ctypes.byref(exit_code)))
                self.assertNotEqual(exit_code.value,259,'Model child is still active')
            finally:
                kernel.CloseHandle(handle)


if __name__=='__main__':unittest.main()
