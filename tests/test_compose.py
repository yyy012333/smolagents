"""组合验证:SemanticCacheModel 包裹 FallbackModel 的嵌套行为。

验证两层包装器的语义:
  1. 缓存 miss 时,请求正常下钻到 FallbackModel,内部主模型故障能降级到备用模型;
  2. 缓存 hit 时,短路返回,完全不触碰 fallback 层与内部模型。
"""
import numpy as np

from smolagents import ChatMessage, MessageRole
from smolagents.cache import Embedder, SemanticCacheModel
from smolagents.fallback import FallbackModel
from smolagents.models import Model


class FakeEmbedder(Embedder):
    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def embed(self, text: str) -> np.ndarray:
        return self.mapping[text]


class CountingModel(Model):
    """记录调用次数、返回固定答案的模型。"""

    def __init__(self, answer: str):
        super().__init__(model_id="counting")
        self.calls = 0
        self.answer = answer

    def generate(self, messages, **kwargs) -> ChatMessage:
        self.calls += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.answer)


class FlakyModel(Model):
    """前 fail_times 次调用抛 non-retryable 错误,之后返回固定答案。

    non-retryable(ValueError 不含 429/5xx/timeout 关键字)保证 Retrying
    不重试、不 sleep,直接抛出 → FallbackModel 立刻降级,测试快速且确定。
    """

    def __init__(self, answer: str, fail_times: int = 1):
        super().__init__(model_id="flaky")
        self.calls = 0
        self.answer = answer
        self.fail_times = fail_times

    def generate(self, messages, **kwargs) -> ChatMessage:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("invalid request")
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.answer)


def _user_msg(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=text)


def _build_composite(primary: Model, backup: Model) -> tuple[SemanticCacheModel, FakeEmbedder]:
    """构造 SemanticCacheModel 包裹 FallbackModel([primary, backup]) 的组合。"""
    fb = FallbackModel([primary, backup])
    emb = FakeEmbedder({"q": np.array([1.0, 0.0])})
    cache = SemanticCacheModel(fb, emb, threshold=0.9)
    return cache, emb


def test_miss_falls_back_to_backup_and_caches():
    """缓存 miss → 下钻 FallbackModel → 主模型故障 → 备用模型顶上 → 结果进缓存。"""
    primary = FlakyModel("from primary", fail_times=1)
    backup = CountingModel("from backup")
    cache, _ = _build_composite(primary, backup)

    first = cache.generate([_user_msg("q")])

    assert first.content == "from backup"
    assert primary.calls == 1          # 主模型被调了一次并失败
    assert backup.calls == 1           # 备用模型顶上
    assert cache.metrics.misses == 1   # 记为一次 miss
    # 结果已入缓存:同向量能检索到
    entry, sim = cache.cache.search(np.array([1.0, 0.0]))
    assert entry is not None and sim >= cache.threshold


def test_hit_short_circuits_fallback():
    """缓存 hit → 短路返回,不触碰 fallback 层,也不触碰内部任何模型。"""
    primary = FlakyModel("from primary", fail_times=0)
    backup = CountingModel("from backup")
    cache, _ = _build_composite(primary, backup)
    # 预置缓存条目,模拟之前已缓存过
    cache.cache.add(
        "q",
        np.array([1.0, 0.0]),
        ChatMessage(role=MessageRole.ASSISTANT, content="cached"),
    )

    out = cache.generate([_user_msg("q")])

    assert out.content == "cached"
    assert primary.calls == 0          # 主模型(即使健康)未被调用
    assert backup.calls == 0           # 备用模型未被调用
    assert cache.metrics.hits == 1     # 记为一次 hit
