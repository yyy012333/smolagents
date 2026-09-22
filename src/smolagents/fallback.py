from enum import Enum


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
