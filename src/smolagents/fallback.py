from enum import Enum
from typing import Any


class ErrorClass(str, Enum):
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    NON_RETRYABLE = "non_retryable"


_RATE_LIMIT_KEYWORDS = ("429", "rate limit", "too many requests", "rate_limit")
_TRANSIENT_KEYWORDS = ("502", "503", "504", "500", "timeout", "timed out", "connection", "broken pipe")


def classify_error(exception: BaseException) -> ErrorClass:
    error_str = str(exception).lower()
    if any(kw in error_str for kw in _RATE_LIMIT_KEYWORDS):
        return ErrorClass.RATE_LIMIT
    if any(kw in error_str for kw in _TRANSIENT_KEYWORDS):
        return ErrorClass.TRANSIENT
    return ErrorClass.NON_RETRYABLE


def is_transient_error(exception: BaseException) -> bool:
    return classify_error(exception) == ErrorClass.TRANSIENT


class FallbackMetrics:
    """Records which model served each call, and fallback events, for benchmarking."""

    def __init__(self):
        self._calls_by_model: dict[str, int] = {}
        self._failures_by_model: dict[str, int] = {}
        self._fallbacks: list[dict[str, str]] = []
        self._served_by: list[str] = []

    def record_success(self, model_id: str) -> None:
        self._calls_by_model[model_id] = self._calls_by_model.get(model_id, 0) + 1
        self._served_by.append(model_id)

    def record_failure(self, model_id: str, error: BaseException) -> None:
        self._failures_by_model[model_id] = self._failures_by_model.get(model_id, 0) + 1
        self._fallbacks.append({"from": model_id, "error_class": classify_error(error).value})

    @property
    def fallback_count(self) -> int:
        return len(self._fallbacks)

    def summary(self) -> dict[str, Any]:
        return {
            "calls_by_model": dict(self._calls_by_model),
            "failures_by_model": dict(self._failures_by_model),
            "fallback_count": self.fallback_count,
            "fallbacks": list(self._fallbacks),
            "served_by": list(self._served_by),
        }
