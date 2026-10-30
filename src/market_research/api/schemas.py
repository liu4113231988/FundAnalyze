"""Validated public contracts; preserve existing field names and defaults."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator
import market_research.config as C
from market_research.modeling.catalog import MODEL_ORDER
from market_research.market.analysis import normalize_code
from market_research.validation.parameters import _parse_max_rebal, _parse_auto_equity, _check_params
from market_research.validation.window import parse_window


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False, validate_default=True)


class WindowRequest(RequestModel):
    start: str | None = None
    end: str | None = None

    @model_validator(mode="after")
    def window(self):
        self.start, self.end = parse_window(self.start, self.end, min_days=30 if isinstance(self, MarketRequest) else None)
        return self


class FundWindow(WindowRequest):
    code: str = Field(pattern=r"^\d{6}$")
    auto_equity: bool = Field(default_factory=lambda: bool(C.RBSA.get("auto_equity", False)))

    @field_validator("auto_equity", mode="before")
    @classmethod
    def equity(cls, value):
        return _parse_auto_equity({"auto_equity":value})


class AnalysisRequest(FundWindow):
    models: list[str] = Field(default_factory=lambda: list(MODEL_ORDER))
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    max_rebal: int | None = None
    refresh: StrictBool = False

    @field_validator("models", mode="before")
    @classmethod
    def valid_models(cls, value):
        if value is None or value == []:
            return list(MODEL_ORDER)
        if not isinstance(value, (list, tuple)) or any(m not in MODEL_ORDER for m in value):
            raise ValueError("models 必须是有效模型名列表")
        return list(dict.fromkeys(value))

    @field_validator("params")
    @classmethod
    def parameters(cls, value):
        return _check_params(value)

    @field_validator("max_rebal", mode="before")
    @classmethod
    def rebal(cls, value):
        return _parse_max_rebal(value)


class RefitRequest(FundWindow):
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    max_rebal: int | None = None

    @field_validator("max_rebal", mode="before")
    @classmethod
    def rebal(cls, value):
        return _parse_max_rebal(value)

    @model_validator(mode="after")
    def parameters(self):
        _check_params({self.model:self.params})
        return self


class BacktestRequest(FundWindow):
    model: str = "enet_slsqp"
    cost_rate: float | None = Field(default=None, ge=0, le=.05)

    @field_validator("model")
    @classmethod
    def known(cls, value):
        if value not in MODEL_ORDER:
            raise ValueError("未知模型")
        return value

    @field_validator("cost_rate", mode="before")
    @classmethod
    def cost(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int,float))):
            raise ValueError("cost_rate 必须是数值")
        return value


class MarketRequest(WindowRequest):
    kind: Literal["stock", "index"] = "stock"
    code: str
    adjust: Literal["qfq", "hfq", ""] = "qfq"
    benchmark: str = "sh000300"
    refresh: StrictBool = False

    @model_validator(mode="after")
    def codes(self):
        self.code = normalize_code(self.kind, self.code)
        self.benchmark = normalize_code("index", self.benchmark)
        if self.kind == "index":
            self.adjust = ""
        return self


class SyncAssetRequest(RequestModel):
    kind: Literal["stock", "fund", "index"]
    code: str
    basis: str = ""


class SyncRetryRequest(RequestModel):
    asset_key: str = Field(min_length=1, max_length=100)


class SubmissionResponse(BaseModel):
    ok: bool = True
    job_id: str | None = None
    cached: bool = False
    merged: bool | None = None
    result: dict | None = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    field: str | None = None
