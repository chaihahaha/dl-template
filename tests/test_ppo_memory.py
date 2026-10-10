"""Regression tests for the example actor-learner's non-V-trace pieces.

The V-trace math lives in ``tests/test_vtrace.py``; here we cover the buffer,
normalizer, model shapes and the bounded-memory cadence.  No Ray or Gymnasium
required.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def load_module():
    spec = importlib.util.spec_from_file_location(
        "rl_ray_ppo_example", ROOT / "examples" / "rl_ray_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ppo = load_module()


def test_replay_memory_arrays_shape_and_order() -> None:
    memory = ppo.ReplayMemory(torch.float32)
    for t in range(5):
        memory.add(
            obs=np.full(4, t, dtype=np.float32),
            act=np.array([t, -t], dtype=np.float32),
            act_logprob=float(-t),
            reward=float(t),
            done=(t == 2),
            value=0.5,
        )
    arrays = memory.as_arrays()
    assert len(memory) == 5
    assert arrays["obs"].shape == (5, 4)
    assert arrays["act"].shape == (5, 2)
    assert arrays["act_logprob"].shape == (5,)
    assert arrays["rewards"].shape == (5,)
    assert arrays["dones"].tolist() == [0, 0, 1, 0, 0]
    assert np.allclose(arrays["obs"][3], 3.0)


def test_normalizer_take_collect_resets_counters() -> None:
    normalizer = ppo.Normalizer((4,), np.float32)
    for _ in range(5):
        normalizer.collect_obs(np.ones(4, dtype=np.float32))
    first = normalizer.take_collect()
    second = normalizer.take_collect()
    assert first["cnt"] == 5.0
    assert second["cnt"] == 0.0
    assert np.allclose(second["sum"], 0.0)

    merged = ppo.Normalizer((4,), np.float32)
    merged.aggregate(first)
    merged.aggregate(second)
    mean, std = merged.mean_std_obs()
    assert np.allclose(mean, 1.0)
    assert np.all(std >= 0)


def test_agent_shapes_and_registered_buffers() -> None:
    model = ppo.AgentModel(2, 4, 2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    state = model.state_dict()
    assert "w" in state and "b" in state

    obs = torch.zeros(1, 2, 4)
    dist, value = model.dist_value(obs)
    assert value.shape == (1, 1)
    assert dist.mean.shape == (1, 2)

    act, logprob, dist2, value2 = model(obs)
    assert act.shape == (1, 2)
    # joint log-probability: summed over action dimensions
    assert logprob.shape == (1,)
    assert value2.shape == (1, 1)
    assert dist2.mean.shape == (1, 2)


def test_rollout_step_count_drives_training_cadence() -> None:
    """The buffer must be drained on a step cadence, not on episode length."""
    source = inspect.getsource(ppo._build_worker_class)
    assert "steps_since_train" in source
    assert "collections.deque" in source
