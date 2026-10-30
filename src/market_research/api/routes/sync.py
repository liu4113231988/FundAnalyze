import os
from fastapi import APIRouter
import market_research.tasks.sync as sync
from market_research.api.schemas import SyncAssetRequest, SyncRetryRequest

router=APIRouter()


@router.get('/api/sync')
def status():
    return {'enabled':os.environ.get('FIL_SYNC_ENABLED','1')!='0','time':sync.schedule_time().strftime('%H:%M'),
            'timezone':'Asia/Shanghai','assets':sync.assets()}


@router.post('/api/sync/assets')
def add(payload:SyncAssetRequest):
    key=sync.register(payload.kind,payload.code,payload.basis)
    return {'ok':True,'asset_key':key}


@router.post('/api/sync/retry')
def retry(payload:SyncRetryRequest):
    return {'ok':True,'queued':sync.request_sync(payload.asset_key)}
