import numpy as np
import pytest

from smolagents import ChatMessage, MessageRole
from smolagents.cache import (
    CacheMetrics,
    CacheStore,
    Embedder,
    SentenceTransformerEmbedder,
    SemanticCacheModel,
    cosine_similarity,
)
from smolagents.models import Model


# ---------- Task 1: cosine_similarity ----------

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


# ---------- Task 2: CacheStore ----------

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
    assert list(store._entries.keys()) == ["b", "c"]


# ---------- Task 3: Embedder ----------

class FakeEmbedder(Embedder):
    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def embed(self, text: str) -> np.ndarray:
        return self.mapping[text]


def test_fake_embedder_returns_mapped_vector():
    emb = FakeEmbedder({"hi": np.array([1.0, 0.0])})
    assert np.allclose(emb.embed("hi"), np.array([1.0, 0.0]))


def test_sentence_transformer_embedder_lazy_import():
    emb = SentenceTransformerEmbedder()
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert emb._model is None


# ---------- Task 4: CacheMetrics ----------

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


# ---------- Task 5: SemanticCacheModel ----------

class CountingModel(Model):
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
    cache.generate(m1)
    assert model.calls == 1
    cache.generate(m1)
    assert model.calls == 1


def test_cache_miss_calls_model_and_stores():
    model = CountingModel("answer")
    emb = FakeEmbedder({
        "q1": np.array([1.0, 0.0]),
        "q2": np.array([0.0, 1.0]),
    })
    cache = SemanticCacheModel(model, emb, threshold=0.9)
    cache.generate([_user_msg("q1")])
    cache.generate([_user_msg("q2")])
    assert model.calls == 2


def test_threshold_boundary_hits_at_equals():
    model = CountingModel("answer")
    emb = FakeEmbedder({
        "q1": np.array([1.0, 0.0]),
        "q2": np.array([1.0, 0.0]),
    })
    cache = SemanticCacheModel(model, emb, threshold=1.0)
    cache.generate([_user_msg("q1")])
    cache.generate([_user_msg("q2")])
    assert model.calls == 1


# ---------- Task 6: 包导出 ----------

def test_semantic_cache_model_exported_from_package():
    from smolagents import SemanticCacheModel as Exported
    assert Exported is SemanticCacheModel
