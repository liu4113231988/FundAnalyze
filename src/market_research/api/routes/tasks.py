from fastapi import APIRouter
from fastapi.responses import JSONResponse
import market_research.services.analysis as services

router=APIRouter()


@router.get('/api/job/{jid}')
def status(jid:str):
    job=services.JOBS.public(jid)
    return job if job else JSONResponse({'error':'任务不存在或已过期'},status_code=404)


@router.get('/api/result/{jid}')
def result(jid:str):
    job=services.JOBS.get(jid)
    if not job:
        return JSONResponse({'error':'任务不存在'},status_code=404)
    if job['status']=='error':
        return JSONResponse({'error':job['error']},status_code=500)
    if job['status']!='done':
        return JSONResponse({'error':'尚未完成','status':job['status']},status_code=202)
    return {'ok':True,'result':job['result']}


@router.post('/api/job/{jid}/cancel')
def cancel(jid:str):
    return {'ok':services.JOBS.cancel(jid)}
