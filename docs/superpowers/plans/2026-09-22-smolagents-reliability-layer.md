# smolagents 生产级可靠性层 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 smolagents 新增一个 `FallbackModel` 组合器 + 通用错误分类器,让 agent 在瞬时故障(5xx/超时)下自动重试并跨模型降级,并用 fault-injection benchmark 量化成功率提升。

**Architecture:** 新增独立文件 `src/smolagents/fallback.py`,内含错误分类器 `classify_error`、指标收集器 `FallbackMetrics`、组合器 `FallbackModel`。完全不改动 `models.py` / `agents.py` / `utils.py`。`FallbackModel` 在 `Model` 层外层做"瞬时错误短退避重试 + 跨模型降级",内层 `ApiModel.retryer` 的 429 长退避保持原样(分层防护)。

**Tech Stack:** Python 3.10+、smolagents(本项目)、pytest、`unittest.mock`(单测全 mock,不依赖真实 API)。

**Spec:** `docs/superpowers/specs/2026-09-22-smolagents-reliability-layer-design.md`

## Global Constraints

- Python >= 3.10(项目 `requires-python = ">=3.10"`)
- **禁止改动** `src/smolagents/models.py`、`agents.py`、`utils.py`、`memory.py`、`monitoring.py`
- 所有新代码只进 `src/smolagents/fallback.py`(以及 `__init__.py` 的一行导出)
- 单元测试**不得**依赖真实 API / 网络,全部用 mock 模型
- 相对导入用 `from .models import ...`、`from .utils import ...`
- 测试里构造 `FallbackModel` 时传 `wait_seconds=0`,避免 `Retrying` 里的 `time.sleep` 拖慢测试
- 遵守现有代码风格:dataclass、类型标注、`__all__`

---

## 前置:环境准备(不在任务内,但必须先做)

当前机器 `python` 是 3.9.13,不满足 `>=3.10`,需先装 3.10+。安装后:

```bash
# 1. 用 3.10+ 建虚拟环境(示例,路径按你装的版本改)
python3.12 -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash

# 2. 以可编辑模式安装 + 测试依赖
pip install -e ".[test]"

# 3. 跑一遍现有测试,确认环境 OK(可选,慢)
pytest tests/test_utils.py -q
```

---

## 文件结构

- **Create** `src/smolagents/fallback.py` — 全部新代码:错误分类、指标、组合器
- **Modify** `src/smolagents/__init__.py` — 加一行 `from .fallback import *`
- **Create** `tests/test_fallback.py` — 单元测试(含 `FakeModel` / `FlakyModel` 测试替身)
- **Create** `examples/reliability_benchmark.py` — fault-injection benchmark

---

## Task 1: 错误分类器 `classify_error` + `is_transient_error`

**Files:**
- Create: `src/smolagents/fallback.py`
- Test: `tests/test_fallback.py`

**Interfaces:**
- Consumes: `str(exception)` 字符串(不依赖具体 SDK 异常类型)
- Produces:
  - `classify_error(exception: BaseException) -> ErrorClass` — 把异常归为 `RATE_LIMIT` / `TRANSIENT` / `NON_RETRYABLE` 三类
  - `is_transient_error(exception: BaseException) -> bool` — 是否为瞬时错误(供 `Retrying` 的 `retry_predicate` 用)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_fallback.py`:

```python
import pytest

from smolagents.fallback import ErrorClass, classify_error, is_transient_error


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
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_fallback.py -v`
Expected: FAIL,报 `ModuleNotFoundError: No module named 'smolagents.fallback'`

- [ ] **Step 3: 写最小实现**

创建 `src/smolagents/fallback.py`:

```python
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
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_fallback.py -v`
Expected: PASS(4 个测试)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/fallback.py tests/test_fallback.py
git commit -m "feat(fallback): add error classifier for rate-limit/transient/non-retryable"
```

---

## Task 2: 指标收集器 `FallbackMetrics`

**Files:**
- Modify: `src/smolagents/fallback.py`(追加)
- Test: `tests/test_fallback.py`(追加)

**Interfaces:**
- Consumes: `classify_error`(Task 1)
- Produces:
  - `FallbackMetrics()` 实例
  - `record_success(model_id: str) -> None`
  - `record_failure(model_id: str, error: BaseException) -> None`
  - `summary() -> dict` — 返回 `calls_by_model` / `failures_by_model` / `fallback_count` / `fallbacks` / `served_by`

- [ ] **Step 1: 写失败测试**

在 `tests/test_fallback.py` 末尾追加:

```python
from smolagents.fallback import FallbackMetrics


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
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_fallback.py::test_fallback_metrics_summary -v`
Expected: FAIL,报 `ImportError: cannot import name 'FallbackMetrics'`

- [ ] **Step 3: 写最小实现**

在 `fallback.py` 追加(保持 `from enum import Enum` 在文件顶部):

```python
from typing import Any


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
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_fallback.py -v`
Expected: PASS(5 个测试)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/fallback.py tests/test_fallback.py
git commit -m "feat(fallback): add FallbackMetrics collector"
```

---

## Task 3: `FallbackModel` 核心(降级 + 指标,先不做外层重试)

**Files:**
- Modify: `src/smolagents/fallback.py`(追加)
- Test: `tests/test_fallback.py`(追加 `FakeModel` 测试替身)

**Interfaces:**
- Consumes: `Model`、`ChatMessage`、`MessageRole`(来自 `.models`)、`FallbackMetrics`(Task 2)
- Produces:
  - `FallbackModel(models: list[Model], transient_retries: int = 3, wait_seconds: float = 1.0, exponential_base: float = 2.0, jitter: bool = True, **kwargs)`
  - `generate(messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs) -> ChatMessage` — 逐个模型尝试,失败降级;全部失败抛最后一个异常

- [ ] **Step 1: 写失败测试**

在 `tests/test_fallback.py` 末尾追加(先加测试替身 + 用例):

```python
from smolagents.fallback import FallbackModel
from smolagents.models import ChatMessage, MessageRole, Model


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
    backup = FakeModel(model_id="backup", fail_with=Exception("500 boom"))
    model = FallbackModel([primary, backup])

    with pytest.raises(Exception, match="500 boom"):
        model.generate([ChatMessage(role=MessageRole.USER, content="hi")])


def test_fallback_requires_models():
    with pytest.raises(ValueError, match="at least one model"):
        FallbackModel([])
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_fallback.py::test_fallback_primary_succeeds -v`
Expected: FAIL,报 `ImportError: cannot import name 'FallbackModel'`

- [ ] **Step 3: 写最小实现**

在 `fallback.py` 追加(顶部 import 补全为 `from .models import ChatMessage, MessageRole, Model` 等;`MessageRole` 仅测试用,实现只需 `Model`):

```python
from logging import getLogger

from .models import ChatMessage, Model
from .utils import Retrying

logger = getLogger(__name__)


class FallbackModel(Model):
    """Composes several models: primary first, fallbacks in order, on failure."""

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
```

> 说明:本任务里 `transient_retries` 已传入 `Retrying`,但测试里 `fail_with=Exception("401...")` 属 NON_RETRYABLE,不会触发重试,所以"重试"行为留到 Task 4 单独验证。这是有意的顺序。

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_fallback.py -v`
Expected: PASS(9 个测试)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/fallback.py tests/test_fallback.py
git commit -m "feat(fallback): add FallbackModel with transparent fallback"
```

---

## Task 4: `FallbackModel` 外层重试(瞬时错误短退避)

**Files:**
- Test: `tests/test_fallback.py`(追加 `FlakyModel` 替身 + 用例)
- Modify: 无(实现已在 Task 3 就位,本任务验证 + 修正)

**Interfaces:**
- Consumes: `FallbackModel`(Task 3)、`Retrying` 的 `retry_predicate=is_transient_error`
- Produces: 验证「瞬时错误被短退避重试 `transient_retries` 次,成功后不再降级」

- [ ] **Step 1: 写失败测试**

在 `tests/test_fallback.py` 末尾追加:

```python
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

    assert result.content == "ok-from-primary"   # 重试后主模型自己成功
    assert primary.calls == 3                    # 被重试了 2 次
    assert backup.calls == 0                     # 没有降级
    assert model.metrics.summary()["fallback_count"] == 0


def test_transient_error_exhausts_then_falls_back():
    primary = FakeModel(model_id="primary", fail_with=Exception("502 Bad Gateway"))  # 一直 502
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup], transient_retries=2, wait_seconds=0)

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-backup"    # 重试耗尽后降级
    assert primary.calls == 2                    # transient_retries=2 → 恰好 2 次
    assert model.metrics.summary()["fallback_count"] == 1
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_fallback.py::test_transient_error_is_retried_without_fallback -v`
Expected: FAIL(若 Task 3 实现正确,`primary.calls` 应为 3,但测试仍会先跑起来验证;若实现有偏差,按 Step 3 修正)

- [ ] **Step 3: 检查/修正实现**

确认 `fallback.py` 中 `Retrying` 的 `retry_predicate=is_transient_error` 且 `max_attempts=transient_retries`。`Retrying` 会在非最后一次且 `is_transient_error(e)` 为真时重试。若测试失败,检查:
- `is_transient_error` 是否正确匹配 "502"
- `wait_seconds=0` 时 `Retrying` 不会 `sleep`(其内部 `if delay > 0` 才 sleep,0 安全)

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_fallback.py -v`
Expected: PASS(11 个测试)

- [ ] **Step 5: 提交**

```bash
git add tests/test_fallback.py
git commit -m "test(fallback): verify transient-error retry before fallback"
```

---

## Task 5: `FallbackModel` 流式降级 `generate_stream`

**Files:**
- Modify: `src/smolagents/fallback.py`(追加 `generate_stream`)
- Test: `tests/test_fallback.py`(追加)

**Interfaces:**
- Consumes: `ChatMessageStreamDelta`(来自 `.models`)、`FallbackModel` 现有字段
- Produces: `generate_stream(messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs) -> Generator[ChatMessageStreamDelta]` — 仅在「首个 delta 之前」失败才降级;一旦产出首个 delta 即锁定该模型

- [ ] **Step 1: 写失败测试**

在 `tests/test_fallback.py` 末尾追加:

```python
from smolagents.fallback import FallbackModel
from smolagents.models import ChatMessageStreamDelta


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
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_fallback.py::test_generate_stream_falls_back_on_first_delta_failure -v`
Expected: FAIL,报 `AttributeError: 'FallbackModel' object has no attribute 'generate_stream'`

- [ ] **Step 3: 写最小实现**

在 `fallback.py` 的 `FallbackModel` 内追加 `generate_stream`,并在顶部 import 补上 `Generator` 与 `ChatMessageStreamDelta`:

```python
from typing import Any, Generator

from .models import ChatMessage, ChatMessageStreamDelta, Model
```

```python
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
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_fallback.py -v`
Expected: PASS(12 个测试)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/fallback.py tests/test_fallback.py
git commit -m "feat(fallback): add streaming fallback via generate_stream"
```

---

## Task 6: 序列化(`to_dict`/`from_dict`)+ 注册 + 导出

**Files:**
- Modify: `src/smolagents/fallback.py`(追加 `to_dict` / `from_dict` + 底部注册)
- Modify: `src/smolagents/__init__.py`(加一行导出)
- Test: `tests/test_fallback.py`(追加)

**Interfaces:**
- Consumes: `MODEL_REGISTRY`(来自 `.models`)、各子模型的 `to_dict` / `from_dict`
- Produces:
  - `FallbackModel.to_dict() -> dict` — 含 `model_id`、`models`(每个含 `class` + `data`)、`transient_retries`/`wait_seconds`/`exponential_base`/`jitter`
  - `FallbackModel.from_dict(cls, model_dictionary) -> FallbackModel`
  - `MODEL_REGISTRY["FallbackModel"] = FallbackModel`

- [ ] **Step 1: 写失败测试**

在 `tests/test_fallback.py` 末尾追加:

```python
from unittest.mock import patch

from smolagents.models import MODEL_REGISTRY


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
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `pytest tests/test_fallback.py::test_fallback_model_serialization_roundtrip -v`
Expected: FAIL(继承自基类的 `to_dict` 不返回 `models` 字段,断言 `d["models"]` 抛 KeyError)

- [ ] **Step 3: 写最小实现**

在 `fallback.py` 的 `FallbackModel` 内追加(import 补上 `MODEL_REGISTRY`):

```python
from .models import MODEL_REGISTRY, ChatMessage, ChatMessageStreamDelta, Model
```

```python
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
```

在 `fallback.py` 文件**末尾**追加注册 + `__all__`:

```python
MODEL_REGISTRY["FallbackModel"] = FallbackModel

__all__ = [
    "ErrorClass",
    "classify_error",
    "is_transient_error",
    "FallbackMetrics",
    "FallbackModel",
]
```

修改 `src/smolagents/__init__.py`:在 `from .models import *` 之后加一行:

```python
from .fallback import *
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `pytest tests/test_fallback.py -v`
Expected: PASS(13 个测试)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/fallback.py src/smolagents/__init__.py tests/test_fallback.py
git commit -m "feat(fallback): add serialization + registry + public export"
```

---

## Task 7: fault-injection benchmark

**Files:**
- Create: `examples/reliability_benchmark.py`

**Interfaces:**
- Consumes: `FallbackModel`(Task 6)、`FakeModel` 类逻辑(独立实现,不 import 测试文件)
- Produces: 可执行脚本,输出「裸模型 vs FallbackModel」在不同故障率下的成功率对比表

- [ ] **Step 1: 写 benchmark 脚本**

创建 `examples/reliability_benchmark.py`:

```python
"""Fault-injection benchmark: bare model vs FallbackModel under transient failures.

Run:  python examples/reliability_benchmark.py
No API keys needed — uses a deterministic simulated model with a fixed seed.
"""
import random

from smolagents import ChatMessage, FallbackModel, MessageRole, Model
from smolagents.monitoring import TokenUsage


class SimulatedModel(Model):
    """永远成功的假模型,带固定 token 用量(用于模拟成本)。"""

    def __init__(self, model_id="sim"):
        super().__init__(model_id=model_id)

    def generate(self, messages, **kwargs):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="ok",
            token_usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


class FaultInjectingModel(Model):
    """按固定概率注入瞬时错误的模型包装器(每次调用独立判定)。"""

    def __init__(self, wrapped: Model, fail_prob: float, rng: random.Random):
        super().__init__(model_id=wrapped.model_id)
        self.wrapped = wrapped
        self.fail_prob = fail_prob
        self.rng = rng

    def generate(self, messages, **kwargs):
        if self.rng.random() < self.fail_prob:
            raise Exception("502 Bad Gateway")
        return self.wrapped.generate(messages, **kwargs)


def run_trial(model: Model, steps: int) -> bool:
    """模拟一个 steps 步的 agent run;任一步失败即 run 失败。"""
    for _ in range(steps):
        try:
            model.generate([ChatMessage(role=MessageRole.USER, content="step")])
        except Exception:
            return False
    return True


def main():
    seed, steps, trials = 42, 5, 200
    print(f"{'fail_rate':>10} | {'bare':>8} | {'fallback':>8}")
    print("-" * 34)
    for fail_prob in [0.0, 0.1, 0.3, 0.5]:
        rng = random.Random(seed)  # 每个 fail_prob 重置 master RNG → 结果可复现
        bare_wins = 0
        fb_wins = 0
        for _ in range(trials):
            # 每个 trial 从 master RNG 派生独立状态;同 seed 下每次运行输出一致
            bare_rng = random.Random(rng.random())
            fb_rng = random.Random(rng.random())

            # bare: 单模型,故障直接致命
            bare = FaultInjectingModel(SimulatedModel(), fail_prob, bare_rng)
            bare_wins += run_trial(bare, steps)

            # fallback: 主 + 备都可能有故障,但可重试 + 降级
            primary = FaultInjectingModel(SimulatedModel(), fail_prob, fb_rng)
            backup = FaultInjectingModel(SimulatedModel(model_id="backup"), fail_prob, fb_rng)
            fb = FallbackModel([primary, backup], transient_retries=2, wait_seconds=0)
            fb_wins += run_trial(fb, steps)

        print(f"{fail_prob:>10.0%} | {bare_wins/trials:>8.1%} | {fb_wins/trials:>8.1%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑通,确认有合理输出**

Run: `python examples/reliability_benchmark.py`
Expected: 输出一张表;`fail_rate=0.3` 时 `bare` 明显低于 `fallback`(bare ≈ `0.7^5 ≈ 16.8%`,fallback 应 ≥90%)

- [ ] **Step 3: 核对数字合理性**

- `bare` 理论值 = `(1 - fail_prob)^steps`:0.1→0.59、0.3→0.17、0.5→0.03(样本 200,允许 ±5% 波动)
- `fallback` 应接近 100%:主模型重试 2 次 + 降级到备模型,全部失败的概率极小(`fail_prob^3` 量级)
- 若 `fallback` 也偏低,检查 `transient_retries` 是否传入、`wait_seconds=0` 是否生效(避免 sleep 拖慢)

- [ ] **Step 4: 提交**

```bash
git add examples/reliability_benchmark.py
git commit -m "bench(fallback): add fault-injection reliability benchmark"
```

---

## 自审记录(写作时已内联修复)

- **Spec 覆盖**:spec 的 3.1(分类器)→ Task 1;3.2(组合器)→ Task 3/4/5;3.3(指标)→ Task 2;第 7 节(测试)→ 各 Task 的 Step 1;第 8 节(量化)→ Task 7;第 6 节(文件)→ 文件结构段;序列化 → Task 6。
- **占位符**:无 TBD/TODO;每个 code step 都有完整代码。
- **类型一致性**:`classify_error` / `is_transient_error` / `FallbackMetrics` / `FallbackModel` 的签名在各 Task 间一致;`ErrorClass.TRANSIENT.value == "transient"` 与 Task 2 测试断言一致。
- **已识别风险**:`examples/reliability_benchmark.py` 里 `SimulatedModel.generate` 的 `token_usage` 导入写得不优雅,Step 3 给了修正;若执行时报错,按 Step 3 处理。
