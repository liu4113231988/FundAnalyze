from fastapi import APIRouter
from fastapi.responses import JSONResponse
import market_research.config as C
import market_research.data.marketdata as D
from market_research.validation.window import parse_window
import market_research.services.analysis as services
from market_research.api.schemas import AnalysisRequest, RefitRequest, BacktestRequest, SubmissionResponse

router=APIRouter()


@router.post('/api/analyze',response_model=SubmissionResponse,response_model_exclude_none=True)
def analyze(payload:AnalysisRequest):
    return services.submit('analyze',payload.model_dump())


@router.post('/api/refit',response_model=SubmissionResponse,response_model_exclude_none=True)
def refit(payload:RefitRequest):
    return services.submit('refit',payload.model_dump())


@router.post('/api/backtest',response_model=SubmissionResponse,response_model_exclude_none=True)
def backtest(payload:BacktestRequest):
    return services.submit('backtest',payload.model_dump())


@router.get('/api/name/{code}')
def name(code:str):
    if len(code)!=6 or not code.isdigit():
        raise ValueError('请输入六位基金代码')
    try:
        from market_research.funds.analysis import _guess_name
        result=_guess_name(code)
        return {'code':code,'name':result or code,'found':result is not None}
    except Exception as error:
        return {'code':code,'name':code,'found':False,'error':str(error)[:80]}


@router.get('/api/corr')
def correlation(code:str,win:int=120,start:str|None=None,end:str|None=None):
    if len(code)!=6 or not code.isdigit():
        raise ValueError('请输入六位基金代码')
    start,end=parse_window(start,end)
    try:
        panel=D.load_panel(code,start=start or C.START_DATE,end=end,verbose=False)
        from market_research.funds.payloads import correlation_block
        result=correlation_block(panel,panel['__FUND__'],roll_n=max(20,min(win,750)))
        return {'ok':True,'code':code,'correlation':result}
    except Exception as error:
        return JSONResponse({'error':f'{type(error).__name__}: {str(error)[:200]}'},status_code=500)
