# SemanticCacheModel 设计文档(LLM 语义缓存)

> 日期:2026-09-25
> 状态:设计已确认,待实现
> 性质:学习向项目(边做边学),同时作为简历项目,契合"优化推理性能与成本"的岗位要求

---

## 1. 背景与目标

### 背景

LLM 应用规模化后,面临两大生产痛点:

1. **成本高**:每次调 LLM 都花钱,用户反复问语义相似的问题 = 重复花钱
2. **延迟高**:LLM 一次调用 1~10 秒,缓存命中只要毫秒级

语义缓存(Semantic Cache)通过缓存"语义相似查询"的答案,避免重复调用 LLM,同时降低成本和延迟。参考架构来自 [GPTCache](https://github.com/zilliztech/GPTCache),但本项目做**最小化 + 量化**版本。

### 目标

1. 在 smolagents 中新增 `SemanticCacheModel`,与已有的 `FallbackModel` 同构,包装任意 Model
2. 用本地 `sentence-transformers` + `numpy` 实现,零额外服务依赖(不引入 FAISS/Milvus/Redis)
3. 用确定性 benchmark 量化收益(Hit Ratio / Latency / Recall)
4. 边做边学:embedding、余弦相似度、阈值调优、LRU、量化、Model 组合

### 成功标准

- `SemanticCacheModel` 能正确命中语义相似的查询、未命中时正确回源并缓存
- benchmark 能输出 Hit Ratio / Latency / Recall 三个数字,并画出"阈值 vs 命中率/准确率"的权衡曲线
- 单元测试全 mock,不依赖真实 LLM 和网络

---

## 2. 设计概述

`SemanticCacheModel(Model)` 是一个 Model 包装器,和 `FallbackModel` 同一接缝。它拦截 `generate()`:先查语义缓存,命中则返回缓存答案;未命中则调用被包装的模型,并把结果存入缓存。

**关键能力:与 `FallbackModel` 可组合** —— `SemanticCacheModel(FallbackModel([...]))`。缓存未命中 → 走降级链调 LLM → 结果存回缓存。

**核心工程点:相似度阈值(threshold)的调优** —— 太松答非所问(假阳性),太紧命中率低。这是本项目的含金量所在。

---

## 3. 架构

```
SemanticCacheModel(Model)           ← 包装器(与 FallbackModel 同一接缝)
 ├── Embedder                       ← 包 sentence-transformers,text → 向量
 ├── CacheStore                     ← 存(query, 向量, 答案),相似检索 + LRU 淘汰
 └── CacheMetrics                   ← 记命中/未命中/延迟(给 benchmark 用)
      │
      └─ 包装的 model(可能是 FallbackModel,负责降级)
```

### 与 FallbackModel 的关系

| | FallbackModel(已有) | SemanticCacheModel(本项目) |
|---|---|---|
| 解决什么 | **可靠性**(故障时换模型) | **成本/性能**(相似查询走缓存) |
| 接缝 | Model 包装器 | Model 包装器(同一接缝) |
| 可组合 | 是 | 是 —— `SemanticCacheModel(FallbackModel(...))` |

---

## 4. 组件接口

### 4.1 Embedder

```python
class Embedder:
    def embed(self, text: str) -> np.ndarray:
        """把一段文字转成向量。"""
```

- 实现:`SentenceTransformerEmbedder(Embedder)`,模型用 `all-MiniLM-L6-v2`(小、快、本地、免费)
- 测试替身:`FakeEmbedder`,返回确定性向量(用于单测,不加载真实模型)

### 4.2 CacheStore

```python
class CacheStore:
    def __init__(self, max_size: int = 100):
        ...
    def search(self, vector: np.ndarray) -> tuple[CacheEntry | None, float]:
        """遍历所有缓存向量,返回余弦相似度最高的 (entry, similarity)。"""
    def add(self, query: str, vector: np.ndarray, response: ChatMessage) -> None:
        """存入缓存;超过 max_size 时 LRU 淘汰最久未用的。"""
```

- 数据结构:`OrderedDict[query_text, CacheEntry]`,其中 `CacheEntry = (vector, response)`
- LRU:每次 `search` 命中时把该 key `move_to_end()`;淘汰时 `popitem(last=False)`

### 4.3 SemanticCacheModel

```python
class SemanticCacheModel(Model):
    def __init__(
        self,
        model: Model,
        embedder: Embedder,
        threshold: float = 0.9,
        max_size: int = 100,
        **kwargs,
    ):
        ...
    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs) -> ChatMessage:
        ...
```

`generate` 流程(见第 5 节)。`generate_stream` 列为非目标(见第 9 节)。

### 4.4 CacheMetrics

```python
class CacheMetrics:
    hits: int
    misses: int
    # 延迟记录(每次查询的耗时列表)
    def hit_ratio(self) -> float        # hits / (hits + misses)
    def avg_latency_hit(self) -> float  # 命中请求的平均延迟
    def avg_latency_miss(self) -> float # 未命中请求的平均延迟
    def summary(self) -> dict
```

---

## 5. 数据流(`generate` 一次请求)

```
generate(messages)
  → ① 提取 query(最后一条 role=user 消息的 content)
  → ② v = embedder.embed(query)
  → ③ (entry, sim) = cache.search(v)     ← 遍历所有缓存向量算余弦相似度,取最高
  → ④ sim >= threshold ?
       ├─ 是(命中):metrics 记 hit,返回 entry.response  ✅
       └─ 否(未命中):
            result = self.model.generate(messages)   ← 调被包装的模型(可能是 FallbackModel)
            cache.add(query, v, result)              ← 存回缓存
            metrics 记 miss
            return result
```

**余弦相似度**:`cos(a, b) = (a·b) / (‖a‖·‖b‖)`,值域 [-1, 1],语义相近通常 > 0.7。

---

## 6. benchmark 设计(量化)

### 6.1 数据集

构造"查询簇":同一意图的多种说法。例如:

- 簇 1:"怎么煮饭" / "怎么做米饭" / "煮饭的步骤" / "米饭怎么做"
- 簇 2:"今天天气" / "天气怎么样" / "现在外面天气如何"
- ...

每个簇有唯一 ID,作为"该命中哪一簇"的 ground truth(用于算 Recall)。

### 6.2 三个指标(参照 GPTCache)

| 指标 | 定义 | 回答什么 |
|---|---|---|
| **Hit Ratio** | 命中数 / 总请求数 | 缓存省了多少次 LLM 调用 |
| **Latency** | 请求处理耗时(命中 vs 未命中分开) | 缓存快了多少 |
| **Recall** | 该命中的查询里,实际命中同一簇的比例 | 有没有漏掉本该死的 |

### 6.3 阈值扫描

对不同 `threshold`(如 0.5 ~ 0.95),测"命中率"和"准确率"(命中是否命中了正确的簇),画出权衡曲线,选最优阈值。这是核心调优点。

### 6.4 确定性

固定随机种子 + 固定数据集,保证可复现(与 fallback 的 fault-injection benchmark 同思路)。

---

## 7. 测试策略

- 全 mock,不依赖真实 LLM / 真实 embedding 模型 / 网络
- `FakeEmbedder`:返回确定性向量(可让"相似文本 → 相似向量"可控)
- `FakeModel`:返回固定答案
- 用例覆盖:
  1. 相同查询 → 命中,不调底层模型
  2. 不同查询 → 未命中,调底层模型并缓存
  3. 阈值边界(相似度恰等于阈值)
  4. LRU 淘汰(超过 max_size 时淘汰最久未用)
  5. 与 FallbackModel 组合(缓存未命中 → 降级 → 缓存结果)

---

## 8. 文件结构

```
src/smolagents/cache.py           ← 新增:Embedder / CacheStore / CacheMetrics / SemanticCacheModel
src/smolagents/__init__.py        ← 加一行导出(与 fallback 相同)
tests/test_cache.py               ← 新增:单元测试(全 mock)
examples/cache_benchmark.py       ← 新增:量化 benchmark
```

---

## 9. 非目标(out of scope,学习向先砍掉)

1. **并发/线程安全**:单线程足够,不加锁
2. **分布式/持久化缓存**:numpy 内存足够
3. **`generate_stream`**:先只做 `generate`,流式缓存复杂(需要存 token 流),列为未来扩展
4. **cache key 只用最后一条 user 消息**:已知局限 —— 同样的问法但 system prompt 不同(如"用法语答")会误命中。这是"metadata 边界"问题,记录为未来扩展点(面试可主动提)

---

## 10. 学习点清单(边做边学)

1. **embedding**:text → 向量(sentence-transformers)
2. **余弦相似度**:数学与实现
3. **阈值调优**:命中率 vs 准确率的权衡(核心)
4. **LRU 淘汰**:`OrderedDict` 实现
5. **量化 benchmark**:Hit Ratio / Latency / Recall
6. **Model 包装器 + 组合**:与 fallback 同构、可嵌套

---

## 11. 参考

- [GPTCache](https://github.com/zilliztech/GPTCache) —— 架构参考(模块化 + 三个指标)
- [GlyphCache](https://pypi.org/project/glyphcache/) —— 阈值调优参考
- smolagents `fallback.py` —— 同构的 Model 包装器参考(本项目 `FallbackModel`)
