<!---
Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0.
This README describes a fork of smolagents that adds a reliability layer.
-->

# smolagents + Reliability Layer

> **A fork of [HuggingFace `smolagents`](https://github.com/huggingface/smolagents) adding a production-grade reliability layer: transparent retry + cross-model fallback for transient failures.**

`smolagents` is an agent framework where the LLM is called once per step and a single task can span dozens of steps. Its built-in retry only covers **429 rate-limit** errors — a single transient failure (502 / timeout / broken connection) aborts the entire run.

This fork adds a thin, **opt-in** layer that fixes this, without modifying any existing framework code.

## The result

Measured with deterministic fault injection (30% per-call failure rate, 5-step task, 200 trials):

| Failure rate | Bare model | With `FallbackModel` |
|---|---|---|
| 30% | 20.5% | **96.5%** |

A **~4.7×** improvement in task success rate. Reproduce it:

```bash
python examples/reliability_benchmark.py
```

## What it adds

- **`classify_error`** — classifies any exception as rate-limit / transient / non-retryable.
- **`FallbackModel`** — a drop-in `Model` that retries transient errors with short exponential backoff, then transparently falls back to the next model in an ordered list.
- **`FallbackMetrics`** — records which model served each call and every fallback event, for observability and benchmarking.

## Usage

```python
from smolagents import CodeAgent, FallbackModel, OpenAIModel

model = FallbackModel(
    [
        OpenAIModel("gpt-4o-mini"),   # primary
        OpenAIModel("gpt-4o"),        # fallback on transient failure
    ],
    transient_retries=3,              # retry 5xx/timeout up to 3× per model
)

agent = CodeAgent(tools=[], model=model)
agent.run("What is 2 + 2?")
```

`FallbackModel` implements the same `Model` interface, so it can be passed anywhere a model is expected — the fallback is transparent to the agent.

## How it works (layered protection)

```
Agent
  └─ FallbackModel.generate   ← new: short backoff for transient errors + cross-model fallback
       └─ ApiModel.retryer     ← unchanged: long backoff for 429
            └─ HTTP call
```

The inner layer (each model's own `retryer`) still handles 429 with a long backoff; the outer layer (new) handles 5xx/timeout with a short backoff and adds cross-model fallback. The two layers' responsibilities never overlap.

## Files changed

| File | Change |
|---|---|
| `src/smolagents/fallback.py` | **new** — `FallbackModel`, `classify_error`, `FallbackMetrics` |
| `src/smolagents/__init__.py` | one line — export the new module |
| `tests/test_fallback.py` | **new** — 13 unit tests (all mocked, no real API) |
| `examples/reliability_benchmark.py` | **new** — deterministic fault-injection benchmark |
| `examples/fallback_model_demo.py` | **new** — usage example |

## Development

```bash
pip install -e ".[test]"
pytest tests/test_fallback.py
```

## License

This fork retains the original [Apache License 2.0](LICENSE). Original work is © HuggingFace Inc.; the reliability-layer additions are released under the same license.

## Credit

Built on [HuggingFace `smolagents`](https://github.com/huggingface/smolagents) — the original project and its documentation live there.
