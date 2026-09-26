import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from .models import ChatMessage, MessageRole, Model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """返回两个向量的余弦相似度,值域 [-1, 1]。零向量返回 0.0。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


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
