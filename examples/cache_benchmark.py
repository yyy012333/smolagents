"""语义缓存 benchmark:量化 Hit Ratio 与延迟收益。

Run:  python examples/cache_benchmark.py
需要: 本地安装 sentence-transformers(用于真实 embedding);不调真实 LLM。
"""
import time

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
    embedder.embed("warmup")  # 预热:提前加载 embedding 模型,避免首个阈值的计时被模型加载污染
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
