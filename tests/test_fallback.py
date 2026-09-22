import pytest
from unittest.mock import patch

from smolagents.fallback import ErrorClass, FallbackMetrics, FallbackModel, classify_error, is_transient_error
from smolagents.models import MODEL_REGISTRY, ChatMessage, ChatMessageStreamDelta, MessageRole, Model


def test_classify_rate_limit():
    for msg in ["429 Too Many Requests", "rate limit exceeded", "Rate_LIMIT", "too many requests"]:
        assert classify_error(Exception(msg)) == ErrorClass.RATE_LIMIT


def test_classify_transient():
    for msg in [
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "500 Internal Server Error",
        "Request timed out",
        "connection reset by peer",
    ]:
        assert classify_error(Exception(msg)) == ErrorClass.TRANSIENT


def test_classify_non_retryable():
    for msg in ["400 Bad Request", "401 Unauthorized", "invalid argument"]:
        assert classify_error(Exception(msg)) == ErrorClass.NON_RETRYABLE


def test_is_transient_error():
    assert is_transient_error(Exception("502 Bad Gateway")) is True
    assert is_transient_error(Exception("429 too many requests")) is False
    assert is_transient_error(Exception("400 Bad Request")) is False


def test_fallback_metrics_summary():
    m = FallbackMetrics()
    m.record_success("primary")
    m.record_failure("primary", Exception("502 Bad Gateway"))
    m.record_success("backup")

    summary = m.summary()
    assert summary["calls_by_model"] == {"primary": 1, "backup": 1}
    assert summary["failures_by_model"] == {"primary": 1}
    assert summary["fallback_count"] == 1
    assert summary["served_by"] == ["primary", "backup"]
    assert summary["fallbacks"][0]["from"] == "primary"
    assert summary["fallbacks"][0]["error_class"] == "transient"


class FakeModel(Model):
    """可配置成功/失败的模型替身。"""

    def __init__(self, model_id="fake", fail_with=None):
        super().__init__(model_id=model_id)
        self.fail_with = fail_with  # BaseException 实例或 None
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ChatMessage(role=MessageRole.ASSISTANT, content=f"ok-from-{self.model_id}")


def test_fallback_primary_succeeds():
    primary = FakeModel(model_id="primary")
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup])

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-primary"
    assert primary.calls == 1
    assert backup.calls == 0
    assert model.metrics.summary()["served_by"] == ["primary"]


def test_fallback_on_failure():
    primary = FakeModel(model_id="primary", fail_with=Exception("401 Unauthorized"))
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup])

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-backup"
    summary = model.metrics.summary()
    assert summary["served_by"] == ["backup"]
    assert summary["fallback_count"] == 1


def test_fallback_all_fail_raises_last():
    primary = FakeModel(model_id="primary", fail_with=Exception("400 bad"))
    backup = FakeModel(model_id="backup", fail_with=Exception("401 unauthorized"))
    model = FallbackModel([primary, backup])

    with pytest.raises(Exception, match="401 unauthorized"):
        model.generate([ChatMessage(role=MessageRole.USER, content="hi")])


def test_fallback_requires_models():
    with pytest.raises(ValueError, match="at least one model"):
        FallbackModel([])


class FlakyModel(Model):
    """前 fail_times 次调用抛瞬时错误,之后成功。"""

    def __init__(self, model_id="flaky", fail_times=2):
        super().__init__(model_id=model_id)
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Exception("502 Bad Gateway")
        return ChatMessage(role=MessageRole.ASSISTANT, content=f"ok-from-{self.model_id}")


def test_transient_error_is_retried_without_fallback():
    primary = FlakyModel(model_id="primary", fail_times=2)  # 前 2 次 502,第 3 次成功
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup], transient_retries=3, wait_seconds=0)

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-primary"  # 重试后主模型自己成功
    assert primary.calls == 3                   # 被重试了 2 次
    assert backup.calls == 0                    # 没有降级
    assert model.metrics.summary()["fallback_count"] == 0


def test_transient_error_exhausts_then_falls_back():
    primary = FakeModel(model_id="primary", fail_with=Exception("502 Bad Gateway"))  # 一直 502
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup], transient_retries=2, wait_seconds=0)

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-backup"   # 重试耗尽后降级
    assert primary.calls == 2                   # transient_retries=2 → 恰好 2 次
    assert model.metrics.summary()["fallback_count"] == 1


class StreamingFakeModel(Model):
    def __init__(self, model_id="stream", fail_on_first=False):
        super().__init__(model_id=model_id)
        self.fail_on_first = fail_on_first

    def generate_stream(self, messages, **kwargs):
        if self.fail_on_first:
            def gen():
                raise Exception("502 Bad Gateway")
                yield  # pragma: no cover
            return gen()
        def gen():
            yield ChatMessageStreamDelta(content="hello")
            yield ChatMessageStreamDelta(content=" world")
        return gen()


def test_generate_stream_falls_back_on_first_delta_failure():
    primary = StreamingFakeModel(model_id="primary", fail_on_first=True)
    backup = StreamingFakeModel(model_id="backup")
    model = FallbackModel([primary, backup])

    deltas = list(model.generate_stream([ChatMessage(role=MessageRole.USER, content="hi")]))

    assert "".join(d.content for d in deltas) == "hello world"
    assert model.metrics.summary()["fallback_count"] == 1


def test_fallback_model_serialization_roundtrip():
    primary = FakeModel(model_id="primary")
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup], transient_retries=5, wait_seconds=0)

    d = model.to_dict()
    assert d["model_id"] == "primary,backup"
    assert [m["class"] for m in d["models"]] == ["FakeModel", "FakeModel"]
    assert d["transient_retries"] == 5

    with patch.dict(MODEL_REGISTRY, {"FakeModel": FakeModel}):
        restored = FallbackModel.from_dict(d)

    assert [m.model_id for m in restored.models] == ["primary", "backup"]
    assert restored.transient_retries == 5
