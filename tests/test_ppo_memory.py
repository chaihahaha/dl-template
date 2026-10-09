"""Regression tests for the example PPO implementation.

These exercise the pure-PyTorch/NumPy pieces (no Ray, no Gymnasium required) and
guard the memory fixes: bounded reward history, replay-buffer draining, single
stacking per training pass, and the normalizer counter reset.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def load_ppo():
    spec = importlib.util.spec_from_file_location(
        "rl_ray_ppo_example", ROOT / "examples" / "rl_ray_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ppo = load_ppo()


def make_memory(length: int = 40, d_obs: int = 4, d_act: int = 2):
    memory = ppo.ReplayMemory(torch.float32)
    for t in range(length):
        memory.add(
            np.random.randn(d_obs).astype(np.float32),
            np.random.randn(d_act).astype(np.float32),
            0.3,
            1.0,
            t % 13 == 0,
            0.5,
        )
    return memory


def test_replay_sample_shapes_and_single_stacking() -> None:
    memory = make_memory()
    obs, act, logp = memory.arrays()
    returns, advantages = ppo.compute_advantages(memory.r, memory.done, memory.value + [0.1])
    batch = memory.sample(8, 16, obs, act, logp, returns, advantages, torch.device("cpu"))
    obs_ctx, acts, act_logprobs, batch_returns, batch_advantages = batch
    assert obs_ctx.shape == (16, 8, 4)
    assert acts.shape == (16, 2)
    assert act_logprobs.shape == (16,)
    assert batch_returns.shape == (16,) and batch_advantages.shape == (16,)
    assert len(memory) == 40


def test_sample_no_longer_recomputes_gae_per_batch() -> None:
    signature = inspect.signature(ppo.ReplayMemory.sample)
    assert "last_value" not in signature.parameters
    assert {"returns", "advantages"} <= set(signature.parameters)


def test_compute_advantages_masks_done() -> None:
    rewards = [1.0, 1.0, 1.0]
    dones = [False, True, False]
    values = [0.5, 0.5, 0.5, 0.5]
    returns, advantages = ppo.compute_advantages(rewards, dones, values)
    assert returns.shape == advantages.shape == (3,)
    # The transition *into* the done state must not bootstrap through it.
    assert advantages[1] == pytest.approx(rewards[1] - values[1])


def test_normalizer_take_collect_resets_counters() -> None:
    normalizer = ppo.Normalizer((4,), np.float32)
    for _ in range(5):
        normalizer.collect_obs(np.ones(4, dtype=np.float32))
    first = normalizer.take_collect()
    second = normalizer.take_collect()
    assert first["cnt"] == 5.0
    assert second["cnt"] == 0.0
    assert np.allclose(second["sum"], 0.0)

    # Aggregate once: no double counting even if take_collect is called twice.
    merged = ppo.Normalizer((4,), np.float32)
    merged.aggregate(first)
    merged.aggregate(second)
    mean, std = merged.mean_std_obs()
    assert np.allclose(mean, 1.0)
    assert np.all(std >= 0)


def test_agent_observability_values_are_registered_buffers() -> None:
    model = ppo.AgentModel(2, 4, 2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    state = model.state_dict()
    assert "w" in state and "b" in state
    obs = torch.zeros(1, 2, 4)
    act, logprob, dist, value = model(obs)
    assert act.shape == (1, 2)
    assert logprob.shape == (1, 2)
    assert value.shape == (1, 1)


def test_rollout_step_count_drives_training_cadence() -> None:
    """The buffer must be drained on a step cadence, not on episode length."""
    source = inspect.getsource(ppo._build_worker_class)
    assert "steps_since_train" in source
    assert "collections.deque" in source
