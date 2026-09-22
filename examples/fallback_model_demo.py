"""How to use FallbackModel: transparent retry + fallback across models.

This is a usage example, not a benchmark — see reliability_benchmark.py for the
fault-injection benchmark that quantifies the reliability gain.

To run with real models, set your API keys and swap in your provider's model class.
"""
from smolagents import ChatMessage, FallbackModel, MessageRole, OpenAIModel

# 1. Build a FallbackModel: primary first, then fallbacks in order.
model = FallbackModel(
    [
        OpenAIModel("gpt-4o-mini"),  # primary
        OpenAIModel("gpt-4o"),       # fallback if the primary errors out
    ],
    transient_retries=3,             # retry transient errors (5xx/timeout) 3x per model
)

# 2. Use it exactly like any other model — the fallback is transparent.
messages = [ChatMessage(role=MessageRole.USER, content="What is 2 + 2?")]
reply = model.generate(messages)
print("reply:", reply.content)

# 3. Inspect which model actually served the call, and any fallback events.
print("metrics:", model.metrics.summary())
