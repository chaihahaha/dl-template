"""V-trace correctness tests (the IMPALA off-policy correction).

The recursion in ``vtrace_targets`` is checked against a direct O(T^2)
implementation of the paper's equations, plus the on-policy identity, the
truncation behaviour, and episode-boundary handling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def load_module():
    spec = importlib.util.spec_from_file_location(
        "rl_ray_ppo_vtrace", ROOT / "examples" / "rl_ray_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ppo = load_module()


def reference_vtrace(rewards, dones, values, mu_logp, pi_logp, gamma, rho_bar, c_bar):
    """Straightforward O(T^2) transcription of IMPALA eqs. 1-3."""
    r = np.asarray(rewards, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    V = np.asarray(values, dtype=np.float64)
    ratio = np.exp(np.clip(np.asarray(pi_logp) - np.asarray(mu_logp), -20, 20))
    rho = np.minimum(ratio, rho_bar)
    c = np.minimum(ratio, c_bar)
    T = len(r)
    vs = np.zeros(T)
    for s in range(T):
        total = 0.0
        product = 1.0
        for t in range(s, T):
            delta = rho[t] * (r[t] + gamma * V[t + 1] * (1 - d[t]) - V[t])
            total += gamma ** (t - s) * product * delta
            product *= c[t] * (1 - d[t])
        vs[s] = V[s] + total
    next_vs = np.concatenate([vs[1:], V[-1:]])
    advantages = rho * (r + gamma * next_vs * (1 - d) - V[:-1])
    return vs, advantages, rho


def test_matches_reference_equations() -> None:
    rng = np.random.default_rng(0)
    T = 25
    rewards = rng.normal(0, 1, T)
    dones = (rng.random(T) < 0.15).astype(np.float64)
    dones[-1] = 0.0
    values = rng.normal(0, 1, T + 1)
    mu_logp = rng.normal(0, 1, T)
    pi_logp = mu_logp + rng.normal(0, 0.5, T)  # somewhat off-policy

    got = ppo.vtrace_targets(
        rewards, dones, values, mu_logp, pi_logp, gamma=0.99, rho_bar=1.0, c_bar=1.0
    )
    expected = reference_vtrace(rewards, dones, values, mu_logp, pi_logp, 0.99, 1.0, 1.0)
    for actual, wanted in zip(got, expected):
        assert np.allclose(actual, wanted, atol=1e-9)


@pytest.mark.parametrize("rho_bar,c_bar", [(1.0, 1.0), (2.0, 1.0), (5.0, 5.0)])
def test_matches_reference_for_various_truncation(rho_bar: float, c_bar: float) -> None:
    rng = np.random.default_rng(1)
    T = 12
    rewards = rng.normal(0, 1, T)
    dones = np.zeros(T)
    values = rng.normal(0, 1, T + 1)
    mu_logp = np.zeros(T)
    pi_logp = rng.normal(0, 1.5, T)  # ratios from ~0.2 to ~5
    got = ppo.vtrace_targets(
        rewards, dones, values, mu_logp, pi_logp, gamma=0.97, rho_bar=rho_bar, c_bar=c_bar
    )
    expected = reference_vtrace(rewards, dones, values, mu_logp, pi_logp, 0.97, rho_bar, c_bar)
    for actual, wanted in zip(got, expected):
        assert np.allclose(actual, wanted, atol=1e-9)


def test_on_policy_reduces_to_discounted_returns() -> None:
    # pi == mu -> all ratios 1; gamma=1, V=0 -> vs are the Monte-Carlo returns.
    vs, advantages, rho = ppo.vtrace_targets(
        rewards=[1.0, 1.0, 1.0],
        dones=[0.0, 0.0, 0.0],
        values=[0.0, 0.0, 0.0, 0.0],
        behavior_log_probs=[0.0, 0.0, 0.0],
        target_log_probs=[0.0, 0.0, 0.0],
        gamma=1.0,
    )
    assert np.allclose(vs, [3.0, 2.0, 1.0])
    assert np.allclose(advantages, [3.0, 2.0, 1.0])
    assert np.allclose(rho, 1.0)


def test_ratios_are_truncated_to_rho_bar() -> None:
    # pi/mu = 2 everywhere; rho_bar=c_bar=1 must clamp it to 1.
    log2 = float(np.log(2.0))
    vs, advantages, rho = ppo.vtrace_targets(
        rewards=[1.0, 1.0],
        dones=[0.0, 0.0],
        values=[0.0, 0.0, 0.0],
        behavior_log_probs=[0.0, 0.0],
        target_log_probs=[log2, log2],
        gamma=1.0,
        rho_bar=1.0,
        c_bar=1.0,
    )
    assert np.allclose(rho, 1.0)
    assert np.allclose(vs, [2.0, 1.0])  # same as on-policy


def test_done_stops_bootstrapping_and_the_trace() -> None:
    # dones[1] = 1 must break both the value bootstrap and the c-product.
    vs, advantages, _ = ppo.vtrace_targets(
        rewards=[1.0, 1.0, 100.0],
        dones=[0.0, 1.0, 0.0],
        values=[0.0, 0.0, 0.0, 0.0],
        behavior_log_probs=[0.0, 0.0, 0.0],
        target_log_probs=[0.0, 0.0, 0.0],
        gamma=1.0,
    )
    assert np.allclose(vs, [2.0, 1.0, 100.0])
    assert np.allclose(advantages, [2.0, 1.0, 100.0])


def test_empty_input_is_safe() -> None:
    vs, advantages, rho = ppo.vtrace_targets([], [], [0.0], [], [])
    assert vs.shape == advantages.shape == rho.shape == (0,)


def test_values_length_is_validated() -> None:
    with pytest.raises(ValueError):
        ppo.vtrace_targets([1.0, 2.0], [0.0, 0.0], [0.0], [0.0, 0.0], [0.0, 0.0])


# --------------------------------------------------------------------------- #
# context reconstruction
# --------------------------------------------------------------------------- #
def test_build_contexts_stacks_recent_observations() -> None:
    obs = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    windows = ppo.build_contexts(obs, [1, 2, 3, 4], ctx_size=2)
    assert windows.shape == (4, 2, 3)
    assert np.allclose(windows[0], obs[[0, 1]])
    assert np.allclose(windows[3], obs[[3, 4]])


def test_build_contexts_zeroes_across_episode_boundary() -> None:
    obs = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    dones = [0, 0, 1, 0, 0, 0]  # episode ends after step 2
    windows = ppo.build_contexts(obs, [3, 4, 5], ctx_size=3, dones=dones)
    # context for step 3 = [step1, step2, step3], but step1/step2 predate the
    # reset -> they must be zero.
    assert np.allclose(windows[0], [np.zeros(2), np.zeros(2), obs[3]])
    assert np.allclose(windows[2], obs[[3, 4, 5]])


# --------------------------------------------------------------------------- #
# policy losses
# --------------------------------------------------------------------------- #
def test_impala_policy_loss_is_truncated_is_gradient() -> None:
    target_logp = torch.tensor([0.0, 1.0], requires_grad=True)
    advantages = torch.tensor([2.0, -1.0])
    rhos = torch.tensor([0.0, 1.0])  # first sample fully suppressed
    loss = ppo.impala_policy_loss(target_logp, advantages, rhos)
    loss.backward()
    assert torch.allclose(target_logp.grad, torch.tensor([-0.0, 0.5]))


def test_ppo_clipped_loss_clips_the_ratio() -> None:
    behavior = torch.zeros(2)
    advantages = torch.tensor([1.0, 1.0])
    # ratio = 10 -> clipped to 1.2 under the negative-advantage branch
    far = torch.log(torch.tensor([10.0, 10.0]))
    loss = ppo.ppo_clipped_loss(far, behavior, advantages, clip=0.2)
    assert torch.isclose(loss, torch.tensor(-1.2))


# --------------------------------------------------------------------------- #
# PopArt value normalisation
# --------------------------------------------------------------------------- #
def test_popart_first_update_matches_batch_statistics() -> None:
    popart = ppo.PopArtValue(4)
    targets = torch.tensor([1.0, 2.0, 3.0])
    popart.update(targets)
    assert torch.allclose(popart.mean, targets.mean().reshape(1), atol=1e-6)
    assert torch.allclose(popart.var, targets.var(unbiased=False).reshape(1), atol=1e-6)
    normalised = popart.normalize(targets)
    assert abs(float(normalised.mean())) < 1e-5
    assert abs(float(normalised.std(unbiased=False)) - 1.0) < 1e-4


def test_popart_update_preserves_predictions() -> None:
    popart = ppo.PopArtValue(4, beta=0.5)
    h = torch.randn(3, 4)
    before = popart.denormalize(popart(h)).detach().clone()
    popart.update(torch.tensor([-100.0, -200.0, -150.0]))
    after = popart.denormalize(popart(h)).detach()
    # Rescaling the head must not change the value it predicts.
    assert torch.allclose(before, after, atol=1e-4)


def test_popart_adapts_mean_over_repeated_updates() -> None:
    popart = ppo.PopArtValue(4, beta=0.5)
    for value in (10.0, 20.0, 30.0):
        popart.update(torch.full((8,), value))
    # EMA should have moved towards the recent targets.
    assert 10.0 < float(popart.mean) < 30.0
    assert float(popart.std) > 0.0


def test_agent_value_head_uses_popart_buffers() -> None:
    model = ppo.AgentModel(2, 4, 2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    state = model.state_dict()
    assert "dv.mean" in state and "dv.var" in state
    assert model.dv.linear.weight.shape == (1, 128)
