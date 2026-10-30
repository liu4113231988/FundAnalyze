from fastapi import APIRouter
import market_research.services.analysis as services
from market_research.api.schemas import MarketRequest, SubmissionResponse

router=APIRouter()


@router.post('/api/market/analyze',response_model=SubmissionResponse,response_model_exclude_none=True)
def analyze(payload:MarketRequest):
    return services.submit('market',payload.model_dump())
