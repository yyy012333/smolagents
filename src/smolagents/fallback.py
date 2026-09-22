from enum import Enum
from logging import getLogger
from typing import Any, Generator

from .models import MODEL_REGISTRY, ChatMessage, ChatMessageStreamDelta, Model
from .utils import Retrying

logger = getLogger(__name__)


class ErrorClass(str, Enum):
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    NON_RETRYABLE = "non_retryable"


_RATE_LIMIT_KEYWORDS = ("429", "rate limit", "too many requests", "rate_limit")
_TRANSIENT_KEYWORDS = ("502", "503", "504", "500", "timeout", "timed out", "connection", "broken pipe")


def classify_error(exception: BaseException) -> ErrorClass:
    """Classify an exception into rate-limit, transient, or non-retryable.

    Rate-limit errors should be retried with a long backoff; transient errors
    (5xx / timeout / connection) with a short backoff; non-retryable errors
    (e.g. 4xx) should not be retried at all.
    """
    error_str = str(exception).lower()
    if any(kw in error_str for kw in _RATE_LIMIT_KEYWORDS):
        return ErrorClass.RATE_LIMIT
    if any(kw in error_str for kw in _TRANSIENT_KEYWORDS):
        return ErrorClass.TRANSIENT
    return ErrorClass.NON_RETRYABLE


def is_transient_error(exception: BaseException) -> bool:
    """Return True if the exception is a transient error (5xx / timeout / connection)."""
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


class FallbackModel(Model):
    """A model that composes several models and transparently falls back on failure.

    Tries each model in order; if one raises (after exhausting transient retries),
    it tries the next. The agent using this model is unaware of the fallback — it
    simply receives the first successful result. Rate-limit (429) errors are still
    handled by each model's own internal retryer; this layer only retries transient
    errors (5xx / timeout / connection) before falling back.

    Parameters:
        models (`list[Model]`):
            Ordered list of models. `models[0]` is the primary; the rest are fallbacks.
        transient_retries (`int`, default `3`):
            Number of attempts per model for transient errors (5xx / timeout / connection).
        wait_seconds (`float`, default `1.0`):
            Initial backoff in seconds before retrying a transient error (exponential).
        exponential_base (`float`, default `2.0`):
            Base for the exponential backoff.
        jitter (`bool`, default `True`):
            Whether to add random jitter to the backoff delay.
        **kwargs:
            Forwarded to the base `Model` (e.g. `flatten_messages_as_text`).

    Example:
    ```python
    >>> from smolagents import FallbackModel, OpenAIModel
    >>> model = FallbackModel([
    ...     OpenAIModel("gpt-4o-mini"),
    ...     OpenAIModel("gpt-4o"),  # fallback if the primary is down
    ... ])
    ```
    """

    def __init__(
        self,
        models: list[Model],
        transient_retries: int = 3,
        wait_seconds: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        **kwargs,
    ):
        if not models:
            raise ValueError("FallbackModel requires at least one model")
        self.models = list(models)
        self.transient_retries = transient_retries
        self.wait_seconds = wait_seconds
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryer = Retrying(
            max_attempts=transient_retries,
            wait_seconds=wait_seconds,
            exponential_base=exponential_base,
            jitter=jitter,
            retry_predicate=is_transient_error,
            reraise=True,
        )
        self.metrics = FallbackMetrics()
        model_id = ",".join(m.model_id for m in self.models if m.model_id) or None
        super().__init__(model_id=model_id, **kwargs)

    def _model_name(self, model: Model) -> str:
        return model.model_id or model.__class__.__name__

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        last_error: BaseException | None = None
        for model in self.models:
            try:
                result = self.retryer(
                    model.generate,
                    messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
                self.metrics.record_success(self._model_name(model))
                return result
            except Exception as e:
                last_error = e
                self.metrics.record_failure(self._model_name(model), e)
                logger.info(
                    "Model '%s' failed (%s: %s), falling back.",
                    self._model_name(model), e.__class__.__name__, e,
                )
        raise last_error

    def generate_stream(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta, None, None]:
        last_error: BaseException | None = None
        for model in self.models:
            try:
                stream = model.generate_stream(
                    messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
            except Exception as e:
                last_error = e
                self.metrics.record_failure(self._model_name(model), e)
                continue
            try:
                first_delta = next(stream)
            except StopIteration:
                self.metrics.record_success(self._model_name(model))
                return
            except Exception as e:
                last_error = e
                self.metrics.record_failure(self._model_name(model), e)
                continue
            # 首个 delta 成功 → 锁定该模型,不再降级
            self.metrics.record_success(self._model_name(model))
            yield first_delta
            yield from stream
            return
        raise last_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "models": [
                {"class": m.__class__.__name__, "data": m.to_dict()}
                for m in self.models
            ],
            "transient_retries": self.transient_retries,
            "wait_seconds": self.wait_seconds,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "FallbackModel":
        models = []
        for m in model_dictionary["models"]:
            model_cls = MODEL_REGISTRY.get(m["class"])
            if model_cls is None:
                raise ValueError(f"Unknown model class '{m['class']}'")
            models.append(model_cls.from_dict(m["data"]))
        return cls(
            models=models,
            transient_retries=model_dictionary.get("transient_retries", 3),
            wait_seconds=model_dictionary.get("wait_seconds", 1.0),
            exponential_base=model_dictionary.get("exponential_base", 2.0),
            jitter=model_dictionary.get("jitter", True),
        )


MODEL_REGISTRY["FallbackModel"] = FallbackModel

__all__ = [
    "ErrorClass",
    "classify_error",
    "is_transient_error",
    "FallbackMetrics",
    "FallbackModel",
]
