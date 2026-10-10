"""Drive the real PPO ``Worker`` (with a no-op Ray stub) and check it is bounded.

This is the regression test for the memory leaks: the reward history must be a
bounded deque and the replay buffer must be drained on the ``rollout_steps``
cadence even when episodes are short.  Gymnasium is needed for the Pendulum env;
the test is skipped when it is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parent.parent


class _FakeRay:
    """Just enough of the Ray API for ``_build_worker_class``."""

    @staticmethod
    def remote(*args, **kwargs):
        return lambda obj: obj  # decorator becomes a no-op


def _load_worker_class():
    spec = importlib.util.spec_from_file_location(
        "rl_ray_ppo_worker", ROOT / "examples" / "rl_ray_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._build_worker_class(_FakeRay)


def test_worker_buffers_stay_bounded() -> None:
    worker_class = _load_worker_class()
    worker = worker_class(
        {
            "batch_size": 64,
            "ctx_size": 8,
            "rollout_steps": 128,
            "lr": 5e-4,
            "dtype": torch.float32,
        }
    )
    worker.reset_initialize()

    max_buffer = 0
    max_rewards = 0
    for _ in range(60):
        worker.rollout(64)
        worker.train_get_weights_infos()
        max_buffer = max(max_buffer, len(worker.memory))
        max_rewards = max(max_rewards, len(worker.rewards))

    assert max_buffer <= 128, "replay buffer grew past the rollout cadence"
    assert max_rewards <= 100, "reward history is not a bounded deque"
    assert len(worker.memory) <= 128


def test_worker_evaluate_is_deterministic_and_finite() -> None:
    worker_class = _load_worker_class()
    worker = worker_class(
        {"ctx_size": 8, "rollout_steps": 128, "lr": 5e-4, "dtype": torch.float32}
    )
    worker.reset_initialize()
    first = worker.evaluate(2)
    second = worker.evaluate(2)
    assert first == second  # fixed seeds + deterministic mean action
    assert first < 0.0  # Pendulum returns are negative
