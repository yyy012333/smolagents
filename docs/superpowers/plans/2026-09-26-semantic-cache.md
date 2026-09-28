# SemanticCacheModel 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 smolagents 中新增 `SemanticCacheModel`,语义相似查询走缓存,降低 LLM 成本与延迟,并用 benchmark 量化收益。

**Architecture:** 一个 `SemanticCacheModel(Model)` 包装器(与 `FallbackModel` 同构),内部由 `Embedder`(text→向量)、`CacheStore`(余弦相似检索 + LRU)、`CacheMetrics`(命中/延迟统计)三个组件协作。未命中时调用被包装的模型并缓存结果。

**Tech Stack:** Python 3.12 / numpy / sentence-transformers(仅 benchmark 用,惰性加载)/ pytest

**Spec:** `docs/superpowers/specs/2026-09-25-semantic-cache-design.md`

## Global Constraints

- Python 3.10+(项目要求),本机用 3.12
- 测试全 mock:不调真实 LLM、不加载真实 embedding 模型、不联网
- `sentence-transformers` 仅在 `SentenceTransformerEmbedder.embed()` 内惰性导入,测试与核心代码不依赖它
- 不做并发/线程安全、不做 `generate_stream`、cache key 只用最后一条 user 消息(见 spec 第 9 节)
- 代码风格对齐 `src/smolagents/fallback.py`(同类多组件单文件)
- 每个任务的依赖 `numpy`,测试文件需 `import numpy as np`

## Review Focus

1. 空缓存时 `search` 不能崩,应返回 `(None, -1.0)` 并触发未命中 → Task 2 加测试
2. 完全相同查询应命中(余弦 = 1.0),且**不调用**底层模型 → Task 5 加测试
3. 相似度恰等于阈值时应命中(`>=`,不是 `>`)→ Task 5 加测试
4. 缓存满时 LRU 淘汰最久未用的,最新的保留 → Task 2 加测试
5. 零向量(范数为 0)导致除零 → `cosine_similarity` 应返回 0.0 而非抛异常 → Task 1 加测试

---

### Task 1: 余弦相似度 `cosine_similarity`

**Files:**
- Create: `src/smolagents/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`(值域 [-1, 1])

- [ ] **Step 1: 写失败测试**

在 `tests/test_cache.py` 写:

```python
import numpy as np
import pytest

from smolagents.cache import cosine_similarity


def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'smolagents.cache'`

- [ ] **Step 3: 写最小实现**

在 `src/smolagents/cache.py` 写:

```python
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """返回两个向量的余弦相似度,值域 [-1, 1]。零向量返回 0.0。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cache.py -v`
Expected: PASS(4 个测试)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/cache.py tests/test_cache.py
git commit -m "feat(cache): add cosine_similarity"
```

---

### Task 2: 缓存存储 `CacheStore`(相似检索 + LRU)

**Files:**
- Modify: `src/smolagents/cache.py`(追加)
- Test: `tests/test_cache.py`(追加)

**Interfaces:**
- Consumes: `cosine_similarity`(Task 1)
- Produces:
  - `class CacheEntry`:`query: str` / `vector: np.ndarray` / `response: ChatMessage`
  - `class CacheStore(max_size: int = 100)`:
    - `search(vector: np.ndarray) -> tuple[CacheEntry | None, float]`(返回最相似的 entry 和相似度;空返回 `(None, -1.0)`)
    - `add(query: str, vector: np.ndarray, response: ChatMessage) -> None`(超 `max_size` 时 LRU 淘汰)

- [ ] **Step 1: 写失败测试**

在 `tests/test_cache.py` 追加:

```python
from smolagents import ChatMessage, MessageRole
from smolagents.cache import CacheStore


def _msg(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content=text)


def test_cache_store_search_empty_returns_none():
    store = CacheStore(max_size=10)
    entry, sim = store.search(np.array([1.0, 0.0]))
    assert entry is None
    assert sim == -1.0


def test_cache_store_add_and_search_hit():
    store = CacheStore(max_size=10)
    store.add("q", np.array([1.0, 0.0]), _msg("answer"))
    entry, sim = store.search(np.array([1.0, 0.0]))
    assert entry is not None
    assert entry.response.content == "answer"
    assert sim == pytest.approx(1.0)


def test_cache_store_lru_evicts_oldest():
    store = CacheStore(max_size=2)
    store.add("a", np.array([1.0, 0.0]), _msg("a"))
    store.add("b", np.array([0.0, 1.0]), _msg("b"))
    store.add("c", np.array([1.0, 1.0]), _msg("c"))  # 超限,淘汰最久未用的 "a"
    entry, _ = store.search(np.array([1.0, 0.0]))
    assert entry is None  # "a" 已被淘汰
    entry, _ = store.search(np.array([0.0, 1.0]))
    assert entry is not None and entry.response.content == "b"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL,`ImportError: cannot import name 'CacheStore'`

- [ ] **Step 3: 写最小实现**

在 `src/smolagents/cache.py` 追加(顶部需 `from collections import OrderedDict`,并从 models 导入 `ChatMessage`):

```python
from collections import OrderedDict

from .models import ChatMessage


class CacheEntry:
    def __init__(self, query: str, vector: np.ndarray, response: ChatMessage):
        self.query = query
        self.vector = vector
        self.response = response


class CacheStore:
    """内存语义缓存:按余弦相似度检索,超 max_size 时 LRU 淘汰。"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    def search(self, vector: np.ndarray) -> tuple[CacheEntry | None, float]:
        best_entry: CacheEntry | None = None
        best_sim = -1.0
        for entry in self._entries.values():
            sim = cosine_similarity(vector, entry.vector)
            if sim > best_sim:
                best_entry, best_sim = entry, sim
        if best_entry is not None:
            self._entries.move_to_end(best_entry.query)  # LRU:最近使用移到末尾
        return best_entry, best_sim

    def add(self, query: str, vector: np.ndarray, response: ChatMessage) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.popitem(last=False)  # LRU:淘汰最久未用
        self._entries[query] = CacheEntry(query, vector, response)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cache.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/cache.py tests/test_cache.py
git commit -m "feat(cache): add CacheStore with cosine search and LRU eviction"
```

---

### Task 3: 向量化接口 `Embedder`

**Files:**
- Modify: `src/smolagents/cache.py`(追加)
- Test: `tests/test_cache.py`(追加)

**Interfaces:**
- Produces:
  - `class Embedder(ABC)`:`embed(text: str) -> np.ndarray`(抽象)
  - `class SentenceTransformerEmbedder(Embedder)`:`__init__(model_name="all-MiniLM-L6-v2")`,惰性加载

- [ ] **Step 1: 写失败测试**

在 `tests/test_cache.py` 追加:

```python
from smolagents.cache import Embedder, SentenceTransformerEmbedder


class FakeEmbedder(Embedder):
    """测试替身:按给定映射返回确定性向量。"""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def embed(self, text: str) -> np.ndarray:
        return self.mapping[text]


def test_fake_embedder_returns_mapped_vector():
    emb = FakeEmbedder({"hi": np.array([1.0, 0.0])})
    assert np.allclose(emb.embed("hi"), np.array([1.0, 0.0]))


def test_sentence_transformer_embedder_lazy_import():
    # 实例化不应加载 sentence-transformers(惰性导入)
    emb = SentenceTransformerEmbedder()
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert emb._model is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL,`ImportError: cannot import name 'Embedder'`

- [ ] **Step 3: 写最小实现**

在 `src/smolagents/cache.py` 追加(顶部需 `from abc import ABC, abstractmethod`):

```python
from abc import ABC, abstractmethod


class Embedder(ABC):
    """把文字转成向量的接口。"""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """用 sentence-transformers 生成向量。模型惰性加载,首次 embed 才真正 import。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(text)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cache.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/cache.py tests/test_cache.py
git commit -m "feat(cache): add Embedder interface and SentenceTransformerEmbedder"
```

---

### Task 4: 指标统计 `CacheMetrics`

**Files:**
- Modify: `src/smolagents/cache.py`(追加)
- Test: `tests/test_cache.py`(追加)

**Interfaces:**
- Produces: `class CacheMetrics`:
  - `record_hit(latency: float) -> None` / `record_miss(latency: float) -> None`
  - `hit_ratio() -> float` / `avg_latency_hit() -> float` / `avg_latency_miss() -> float`
  - `summary() -> dict`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cache.py` 追加:

```python
from smolagents.cache import CacheMetrics


def test_cache_metrics_hit_ratio():
    m = CacheMetrics()
    m.record_hit(0.01)
    m.record_miss(1.0)
    assert m.hit_ratio() == pytest.approx(0.5)


def test_cache_metrics_avg_latency():
    m = CacheMetrics()
    m.record_hit(0.01)
    m.record_hit(0.03)
    m.record_miss(1.0)
    assert m.avg_latency_hit() == pytest.approx(0.02)
    assert m.avg_latency_miss() == pytest.approx(1.0)


def test_cache_metrics_empty_is_zero():
    m = CacheMetrics()
    assert m.hit_ratio() == 0.0
    assert m.avg_latency_hit() == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL,`ImportError: cannot import name 'CacheMetrics'`

- [ ] **Step 3: 写最小实现**

在 `src/smolagents/cache.py` 追加:

```python
class CacheMetrics:
    """记录命中/未命中与延迟,供 benchmark 量化。"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []

    def record_hit(self, latency: float) -> None:
        self.hits += 1
        self._hit_latencies.append(latency)

    def record_miss(self, latency: float) -> None:
        self.misses += 1
        self._miss_latencies.append(latency)

    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def avg_latency_hit(self) -> float:
        return sum(self._hit_latencies) / len(self._hit_latencies) if self._hit_latencies else 0.0

    def avg_latency_miss(self) -> float:
        return sum(self._miss_latencies) / len(self._miss_latencies) if self._miss_latencies else 0.0

    def summary(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio(),
            "avg_latency_hit": self.avg_latency_hit(),
            "avg_latency_miss": self.avg_latency_miss(),
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cache.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/cache.py tests/test_cache.py
git commit -m "feat(cache): add CacheMetrics collector"
```

---

### Task 5: 主类 `SemanticCacheModel`

**Files:**
- Modify: `src/smolagents/cache.py`(追加)
- Test: `tests/test_cache.py`(追加)

**Interfaces:**
- Consumes: `CacheStore`、`Embedder`、`CacheMetrics`、`ChatMessage`、`MessageRole`、`Model`
- Produces: `class SemanticCacheModel(Model)`:
  - `__init__(self, model: Model, embedder: Embedder, threshold: float = 0.9, max_size: int = 100, **kwargs)`
  - `generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs) -> ChatMessage`

- [ ] **Step 1: 写失败测试**

在 `tests/test_cache.py` 追加:

```python
from smolagents import ChatMessage, MessageRole
from smolagents.cache import SemanticCacheModel
from smolagents.models import Model


class CountingModel(Model):
    """记录被调用次数的假模型。"""

    def __init__(self, answer: str):
        super().__init__(model_id="counting")
        self.calls = 0
        self.answer = answer

    def generate(self, messages, **kwargs) -> ChatMessage:
        self.calls += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.answer)


def _user_msg(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=text)


def _identical_embedder() -> FakeEmbedder:
    return FakeEmbedder({"how to cook rice": np.array([1.0, 0.0])})


def test_cache_hit_does_not_call_model():
    model = CountingModel("boiled rice")
    cache = SemanticCacheModel(model, _identical_embedder(), threshold=0.9)
    m1 = [_user_msg("how to cook rice")]
    cache.generate(m1)  # 第一次 miss,调模型
    assert model.calls == 1
    cache.generate(m1)  # 第二次相同 → hit,不调模型
    assert model.calls == 1


def test_cache_miss_calls_model_and_stores():
    model = CountingModel("answer")
    emb = FakeEmbedder({
        "q1": np.array([1.0, 0.0]),
        "q2": np.array([0.0, 1.0]),  # 与 q1 正交 → 相似度 0 → miss
    })
    cache = SemanticCacheModel(model, emb, threshold=0.9)
    cache.generate([_user_msg("q1")])
    cache.generate([_user_msg("q2")])
    assert model.calls == 2


def test_threshold_boundary_hits_at_equals():
    model = CountingModel("answer")
    emb = FakeEmbedder({
        "q1": np.array([1.0, 0.0]),
        "q2": np.array([1.0, 0.0]),  # 与 q1 相同,余弦 = 1.0
    })
    cache = SemanticCacheModel(model, emb, threshold=1.0)  # 阈值恰为 1.0
    cache.generate([_user_msg("q1")])
    cache.generate([_user_msg("q2")])  # 相似度 1.0 >= 1.0 → 应命中
    assert model.calls == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL,`ImportError: cannot import name 'SemanticCacheModel'`

- [ ] **Step 3: 写最小实现**

在 `src/smolagents/cache.py` 追加(顶部需 `import time` 及 `from .models import ChatMessage, MessageRole, Model`):

```python
import time

from .models import ChatMessage, MessageRole, Model


class SemanticCacheModel(Model):
    """语义缓存包装器:命中返回缓存,未命中调用被包装模型并缓存。与 FallbackModel 同构,可组合。"""

    def __init__(
        self,
        model: Model,
        embedder: Embedder,
        threshold: float = 0.9,
        max_size: int = 100,
        **kwargs,
    ):
        super().__init__(model_id=model.model_id, **kwargs)
        self.model = model
        self.embedder = embedder
        self.threshold = threshold
        self.cache = CacheStore(max_size=max_size)
        self.metrics = CacheMetrics()

    def _extract_query(self, messages) -> str:
        for message in reversed(messages):
            if isinstance(message, ChatMessage) and message.role == MessageRole.USER:
                content = message.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item["text"]
        return ""

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        query = self._extract_query(messages)
        start = time.time()
        vector = self.embedder.embed(query)
        entry, similarity = self.cache.search(vector)
        if entry is not None and similarity >= self.threshold:
            self.metrics.record_hit(time.time() - start)
            return entry.response
        result = self.model.generate(
            messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        self.cache.add(query, vector, result)
        self.metrics.record_miss(time.time() - start)
        return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cache.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/cache.py tests/test_cache.py
git commit -m "feat(cache): add SemanticCacheModel with transparent semantic caching"
```

---

### Task 6: 导出到包

**Files:**
- Modify: `src/smolagents/__init__.py:27`(在 `from .fallback import *` 后加一行)
- Test: `tests/test_cache.py`(追加)

- [ ] **Step 1: 写失败测试**

在 `tests/test_cache.py` 追加:

```python
def test_semantic_cache_model_exported_from_package():
    from smolagents import SemanticCacheModel  # noqa: F401
    assert SemanticCacheModel is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_cache.py::test_semantic_cache_model_exported_from_package -v`
Expected: FAIL,`ImportError: cannot import name 'SemanticCacheModel' from 'smolagents'`

- [ ] **Step 3: 写最小实现**

在 `src/smolagents/__init__.py` 中 `from .fallback import *` 之后加一行:

```python
from .cache import *
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_cache.py -v`
Expected: PASS(全部)

- [ ] **Step 5: 提交**

```bash
git add src/smolagents/__init__.py tests/test_cache.py
git commit -m "feat(cache): export cache module from package"
```

---

### Task 7: 量化 benchmark

**Files:**
- Create: `examples/cache_benchmark.py`

**Interfaces:**
- Consumes: `SemanticCacheModel`、`SentenceTransformerEmbedder`、`ChatMessage`、`MessageRole`、`Model`
- Produces: 可运行的脚本,打印 Hit Ratio / 命中与未命中延迟 / 不同阈值下的命中率

- [ ] **Step 1: 写脚本**

创建 `examples/cache_benchmark.py`:

```python
"""语义缓存 benchmark:量化 Hit Ratio 与延迟收益。

Run:  python examples/cache_benchmark.py
需要: 本地安装 sentence-transformers(用于真实 embedding);不调真实 LLM。
"""
import time

import numpy as np

from smolagents import ChatMessage, MessageRole, Model, SemanticCacheModel
from smolagents.cache import SentenceTransformerEmbedder


# 查询簇:同一意图的多种说法(用于测语义命中的能力)
CLUSTERS = {
    "cooking_rice": [
        "how to cook rice",
        "how do I make rice",
        "steps to boil rice",
        "cooking white rice",
    ],
    "weather": [
        "what's the weather today",
        "weather forecast now",
        "is it sunny outside",
    ],
    "capital_france": [
        "what is the capital of france",
        "france capital city",
        "capital of france please",
    ],
}


class FakeModel(Model):
    """按簇返回确定性答案的假模型,模拟 LLM(带延迟)。"""

    def __init__(self):
        super().__init__(model_id="fake")
        self.answer_by_cluster = {
            "cooking_rice": "Rinse rice, add water, boil for 15 minutes.",
            "weather": "Sunny, 24 degrees.",
            "capital_france": "Paris.",
        }

    def generate(self, messages, **kwargs) -> ChatMessage:
        time.sleep(0.05)  # 模拟 LLM 延迟 50ms
        query = messages[-1].content
        for cluster, paraphrases in CLUSTERS.items():
            if query in paraphrases:
                return ChatMessage(role=MessageRole.ASSISTANT, content=self.answer_by_cluster[cluster])
        return ChatMessage(role=MessageRole.ASSISTANT, content="unknown")


def main():
    embedder = SentenceTransformerEmbedder()
    model = FakeModel()
    cache = SemanticCacheModel(model, embedder, threshold=0.7, max_size=100)

    queries = [q for paraphrases in CLUSTERS.values() for q in paraphrases]
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        cache.threshold = threshold
        cache.cache = type(cache.cache)(max_size=100)  # 重置缓存
        cache.metrics = type(cache.metrics)()
        for q in queries:
            cache.generate([ChatMessage(role=MessageRole.USER, content=q)])
        print(f"threshold={threshold}: {cache.metrics.summary()}")

    print("---")
    print(f"最终 metrics: {cache.metrics.summary()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行脚本**

Run: `python examples/cache_benchmark.py`
Expected: 输出各阈值下的 `hit_ratio`,且阈值越高命中率越低(但越准);无报错。

> 注:首次运行会下载 `all-MiniLM-L6-v2`(约 80MB)。若本机装不了 sentence-transformers,先跳过本任务,核心代码(任务 1-6)已完整可用。

- [ ] **Step 3: 提交**

```bash
git add examples/cache_benchmark.py
git commit -m "bench(cache): add semantic-cache benchmark with threshold sweep"
```

---

## Self-Review 备注

- **Spec 覆盖**:spec 第 4 节四组件(Embedder/CacheStore/CacheMetrics/SemanticCacheModel)对应 Task 3/2/4/5;第 5 节数据流对应 Task 5 的 `generate`;第 6 节 benchmark 对应 Task 7;第 8 节文件结构对应各任务 Files。`cosine_similarity`(spec 第 4/5 节提到)独立成 Task 1。
- **类型一致**:`cosine_similarity(a,b)->float`、`CacheStore.search()->(CacheEntry|None,float)`、`Embedder.embed(text)->np.ndarray`、`CacheMetrics.record_hit(latency)` 在前后任务签名一致。
- **Review Focus 覆盖**:#1→Task2 `test_cache_store_search_empty`、#2→Task5 `test_cache_hit_does_not_call_model`、#3→Task5 `test_threshold_boundary_hits_at_equals`、#4→Task2 `test_cache_store_lru_evicts_oldest`、#5→Task1 `test_cosine_similarity_zero_vector_returns_zero`。
