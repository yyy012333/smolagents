# smolagents 生产级可靠性层设计

- 日期:2026-09-22
- 目标:后端 / AI 应用实习简历项目
- 状态:待评审

---

## 1. 背景与问题

smolagents 的 agent 每一步都要调用一次 LLM(`agents.py:1677` 的 `self.model.generate(...)`)。一个多步任务会触发几十次模型调用,**任何一次瞬时故障(5xx / 超时 / 断连)都会让整个 run 崩溃、前功尽弃**。

当前现状(源码实证):

| 能力 | 现状 | 证据 |
|---|---|---|
| 限流重试 | 有,但**只重试 429 / 限流错误** | `ApiModel.retryer`,`retry_predicate=is_rate_limit_error` |
| 限速节流 | 有 | `RateLimiter(requests_per_minute)` |
| 指数退避 + 抖动 | 有 | `utils.py:528` `Retrying` |
| 多模型路由 / 降级 | 仅限 LiteLLM | `LiteLLMRouterModel` 包 LiteLLM `Router` |
| **瞬时故障重试**(5xx/超时/断连) | **无** | `is_rate_limit_error` 只匹配 "429" / "rate limit" |
| **跨 provider fallback**(非 LiteLLM) | **无** | 主模型失败即抛,无备胎 |

**事故链**(一次 502 如何杀死 agent):

```
OpenAI 返回 502
  → create() 抛异常
  → Retrying 判 is_rate_limit_error(502)=False
  → 立刻 raise(不重试)
  → OpenAIModel.generate 抛出
  → agents.py:1699 except Exception → 包成 AgentGenerationError
  → agents.py:594 except AgentGenerationError → raise
  → agent.run() 整个崩溃
```

---

## 2. 设计目标

### Goals

1. 在瞬时故障下,agent 的**任务成功率显著提升**
2. 对现有代码**零侵入**(opt-in,不改默认行为)
3. 可量化(自带指标 + fault-injection benchmark)
4. 兼容 streaming(`generate_stream`)

### Non-goals(明确不做)

1. **不改** `is_rate_limit_error` 的默认行为(尊重官方保守默认)
2. **不做**成本 / 延迟感知路由(留作 Phase 2)
3. **不做** circuit breaker(熔断器,可后续)
4. 不追求"效果更好",只追求"在故障面前更鲁棒"

---

## 3. 核心设计

整体思路:在 `Model` 层新增一个**组合器**,**完全不改动 `models.py` 和任何现有 `Model` 子类**。所有新代码进独立文件 `fallback.py`。

```
agents.py:1677  self.model.generate(...)     ← 上层:不动
        │
        ▼
[FallbackModel.generate]  ← 新增:外层重试(瞬时错误)+ 降级
        │  (内部用 Retrying + classify_error)
        ▼
model.generate(...)         ← 现有:内层 retryer(429 长退避)保持原样
        │
        ▼
models.py:1780  retryer(create, ...)         ← 现有:完全不动
```

**分层防护**:内层(`ApiModel.retryer`)继续负责 429 长退避;外层(`FallbackModel`)负责瞬时错误(5xx/超时)短退避重试 + 跨模型降级。两层职责不重叠、互不干扰。

### 3.1 通用错误分类器 `is_retryable_error`

新增一个工具函数,把异常分成三类(比现有 `is_rate_limit_error` 更完整):

| 类别 | 示例 | 由谁重试 | 退避 |
|---|---|---|---|
| 限流 | 429 / rate limit | 内层 `ApiModel.retryer`(**不动**) | 长(60s 起) |
| 瞬时 | 5xx / timeout / 连接错误 | 外层 `FallbackModel` | 短(1-2s 起) |
| 不可重试 | 4xx(参数错/鉴权错)/ 其他 | 不重试,直接降级 | — |

实现要点:
- **不修改**现有 `is_rate_limit_error`(它在 `models.py`,我们完全不碰该文件)
- 新增 `classify_error(exception) -> ErrorClass`,放在新文件 `fallback.py`
- 由 `FallbackModel` 用它构建自己的 `Retrying` predicate(见 3.2),**不注入、不改动 `ApiModel`**
- 判别手段以异常字符串匹配为主(与现有 `is_rate_limit_error` 一致),避免引入对具体 SDK 异常类型的硬依赖

### 3.2 `FallbackModel(Model)` 组合器

接收一组 model,按顺序尝试:

```python
class FallbackModel(Model):
    def __init__(self, models: list[Model], transient_retries: int = 3, ...):
        self.models = models                      # models[0] 为主,其余为 fallback
        self.retryer = Retrying(
            max_attempts=transient_retries,
            wait_seconds=1.0,                     # 短退避,区别于内层 429 的 60s
            exponential_base=2,
            retry_predicate=is_transient_error,   # 只重试瞬时错误(见 3.1)
        )
        self.metrics = FallbackMetrics()

    def generate(self, messages, **kwargs) -> ChatMessage:
        last_error = None
        for model in self.models:
            try:
                # 外层重试:瞬时错误(5xx/超时)短退避重试 N 次
                result = self.retryer(model.generate, messages, **kwargs)
                self.metrics.record(served_by=model.model_id)
                return result
            except Exception as e:
                last_error = e
                self.metrics.record_fallback(from_model=model.model_id, error=e)
                # 重试耗尽 → 降级到下一个模型
        raise last_error   # 全部模型都失败,抛最后一个错误
```

关键行为:
1. **外层重试 + 降级**:瞬时错误先用短退避重试,耗尽再降级;内层(`ApiModel`)的 429 长退避保持原样,两层不重叠
2. **透明**:上层 `Agent` 完全无感知,拿到的是最终成功的结果
3. **指标收集**:记录每次调用"哪个模型实际服务、重试了几次、降级了几次、耗时、token"
4. **兼容 streaming**:实现 `generate_stream`,流式场景下同样逐模型降级
5. **`model_id` / `to_dict` / `from_dict`**:按 `Model` 基类契约实现,保证可序列化(与 `serialization.py` 兼容)

### 3.3 指标结构 `FallbackMetrics`

用于量化,字段:
- 每个模型被调用次数 / 成功次数 / 失败次数
- 降级(fallback)发生次数、各发生在哪对模型之间
- 每次调用的耗时、token 消耗
- 暴露 `summary() -> dict` 供 benchmark 直接使用

---

## 4. 数据流

```
FallbackModel.generate(messages)
  ├─ for model in [主, 备1, 备2]:
  │    try:
  │        resp = model.generate(messages)     # 内部已有 retryer(扩成通用分类后)
  │        metrics.record(served_by=model)     # 记录"谁服务了"
  │        return resp
  │    except Exception as e:
  │        metrics.record_fallback(from=model) # 记录"从谁降级"
  │        last_error = e                      # 继续下一个
  └─ raise last_error                          # 全部失败
```

---

## 5. 错误处理策略

- **主模型限流**(429):内层 `ApiModel.retryer` 长退避重试(不动);耗尽 → 外层降级
- **主模型瞬时故障**(502/超时):外层 `FallbackModel` 短退避重试;耗尽 → 降级
- **主模型不可重试错误**(4xx 如鉴权/参数错):不重试,直接降级(可能是该 provider 配置问题,换备胎有意义)
- **所有模型都失败**:抛最后一个错误(与裸模型行为一致,不吞错)

---

## 6. 文件改动

| 文件 | 改动 |
|---|---|
| `src/smolagents/fallback.py` | **新增**:`classify_error` / `is_transient_error`、`FallbackModel`、`FallbackMetrics` |
| `src/smolagents/models.py` | **不改**(完全不动) |
| `src/smolagents/__init__.py` | 导出 `FallbackModel` 等新符号 |
| `tests/test_fallback.py` | 单元测试(mock model,不依赖真实 API) |
| `examples/reliability_benchmark.py` | fault-injection benchmark + 对比输出 |
| `docs/superpowers/specs/` | 本设计文档 |

---

## 7. 测试策略

全部用 **mock**,不依赖真实 API(成本可控、可复现):

1. **分类器单测**:429 / 502 / 504 / timeout / connection / 400 / 401 各算哪类
2. **FallbackModel 单测**:
   - 主成功 → 只用主,不降级
   - 主失败 → 降级到备,返回备的结果
   - 全失败 → 抛最后一个错误
   - 指标正确记录(served_by / fallback 次数)
3. **streaming 单测**:主模型流式失败 → 降级到备
4. **序列化单测**:`to_dict` / `from_dict` 往返一致

---

## 8. 量化方法(Fault Injection)

可靠性层不能靠"真实世界的随机失败"来量化,必须**确定性注入故障**:

1. **故障注入器**:`FaultInjector` 按"每第 N 次调用失败"注入 502 / timeout(确定性、可复现)
2. **任务集**:3-5 个**答案已知、需要多步**的任务(多步才暴露可靠性问题)
3. **对比**:裸模型 vs `FallbackModel`,同一故障率(0/10/30/50%)、同一任务、各跑 ≥10 次
4. **指标**:成功率(核心)、平均耗时、平均 token/成本

预期结果形态(示例,实际数字由 benchmark 跑出):

| 故障率 | 方案 | 成功率 | 平均耗时 | 平均成本 |
|---|---|---|---|---|
| 0% | 裸模型 | 100% | 45s | $0.12 |
| 30% 502 | 裸模型 | ~35% | 120s | $0.15 |
| 30% 502 | FallbackModel | ~95% | 88s | $0.14 |

> 关键论证:5 步任务在 30% 注入下,裸模型撞至少一次故障的概率 `1 - 0.7⁵ ≈ 83%`,几乎必死;FallbackModel 靠重试+降级存活。

---

## 9. 风险与取舍

| 风险 | 应对 |
|---|---|
| 重试 5xx 可能非幂等(重复执行/多花钱) | 只对"读类"调用安全;作为 opt-in 特性,由使用者决定是否开启 |
| 分类靠字符串匹配,可能漏判 | 与现有 `is_rate_limit_error` 同法;后续可扩展异常类型映射 |
| streaming 降级实现复杂 | 降级只发生在"首个 delta 之前"的失败;中途失败按最终结果处理(实现时细化) |
| 降级增加总 token(重试多花钱) | benchmark 如实报成本,展示 trade-off |
| PR 不被上游合并 | 简历价值在"分析+实现+量化"过程本身,不在被合并 |

---

## 10. 分阶段计划

- **Phase 1(核心)**:`classify_error` 分类器 + `FallbackModel` + `FallbackMetrics` + 单元测试
- **Phase 2(量化)**:`FaultInjector` + 任务集 + `reliability_benchmark.py`,跑出对比表
- **Phase 3(可选)**:README 用法说明 + 尝试向上游提 PR

---

## 11. 验收标准

- [ ] 裸模型在 30% 注入下成功率明显低于 `FallbackModel`(benchmark 可复现)
- [ ] 单元测试全绿,且不依赖真实 API
- [ ] 现有 `smolagents` 默认行为零变化(所有原有测试通过)
- [ ] `FallbackModel` 可序列化、兼容 streaming
