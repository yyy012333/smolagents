"""Fault-injection benchmark: bare model vs FallbackModel under transient failures.

Run:  python examples/reliability_benchmark.py
No API keys needed — uses a deterministic simulated model with a fixed seed.
"""
import random

from smolagents import ChatMessage, FallbackModel, MessageRole, Model


class SimulatedModel(Model):
    """永远成功的假模型。"""

    def __init__(self, model_id="sim"):
        super().__init__(model_id=model_id)

    def generate(self, messages, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")


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
            bare_rng = random.Random(rng.random())
            fb_primary_rng = random.Random(rng.random())
            fb_backup_rng = random.Random(rng.random())

            # bare: 单模型,故障直接致命
            bare = FaultInjectingModel(SimulatedModel(), fail_prob, bare_rng)
            bare_wins += run_trial(bare, steps)

            # fallback: 主 + 备都有故障,但可重试 + 降级
            primary = FaultInjectingModel(SimulatedModel(), fail_prob, fb_primary_rng)
            backup = FaultInjectingModel(SimulatedModel(model_id="backup"), fail_prob, fb_backup_rng)
            fb = FallbackModel([primary, backup], transient_retries=2, wait_seconds=0)
            fb_wins += run_trial(fb, steps)

        print(f"{fail_prob:>10.0%} | {bare_wins/trials:>8.1%} | {fb_wins/trials:>8.1%}")


if __name__ == "__main__":
    main()
