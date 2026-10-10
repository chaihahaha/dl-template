"""Asynchronous actor-learner with Ray + PyTorch, using IMPALA-style **V-trace**.

Why this is not plain PPO anymore
---------------------------------
The workers keep rolling out while the learner keeps updating the weights, so
the data in a worker's buffer was produced by an **older** policy (the behaviour
policy ``mu``) than the one being trained (the target policy ``pi``).  PPO's
clipped ratio assumes the two are close (it is an *on-policy* method); with a
stale ``mu`` the ratio/GAE targets are biased and training becomes unstable.

IMPALA (Espeholt et al., 2018, "Scalable Distributed Deep-RL with Importance
Weighted Actor-Learner Architectures", https://arxiv.org/abs/1802.01561) solves
exactly this with **V-trace**: a truncated importance-sampling correction that

* corrects the value targets for the fact that ``mu != pi``, and
* bounds the correction, so a few very off-policy samples cannot blow the update up.

The equations implemented here (paper eqs. 1-3)
-----------------------------------------------
::

    rho_t = min(rho_bar, pi(a_t|x_t) / mu(a_t|x_t))      # value-target weight
    c_t   = min(c_bar,   pi(a_t|x_t) / mu(a_t|x_t))      # trace truncation
    delta_t = rho_t ( r_t + gamma V(x_{t+1}) - V(x_t) )
    v_s   = V(x_s) + sum_{t=s}^{T-1} gamma^{t-s} (prod_{i=s}^{t-1} c_i) delta_t
    A_s   = rho_s ( r_s + gamma v_{s+1} - V(x_s) )
    policy loss = - E[ rho_s * A_s * log pi(a_s|x_s) ]
    value  loss =   E[ (v_s - V(x_s))^2 ]   # computed in PopArt-normalised space

Typical values are ``rho_bar = c_bar = 1`` (as in the paper).  ``pi`` is the
*current* learner network; ``mu`` is the log-probability stored by the actor at
rollout time.  When ``pi == mu`` all ratios are 1 and V-trace reduces to
n-step/GAE-style on-policy returns, so the correction is a strict generalisation.

The value head uses **PopArt** (``PopArtValue``) as in the IMPALA paper: it keeps
a running mean/variance of the value targets and rescales its linear layer so the
predicted value is preserved while the loss is computed in normalised space.
This keeps the value scale under control when discounted returns are ~100x the
per-step reward.

``--policy-loss ppo`` is also provided: it uses the V-trace advantages but keeps
the PPO clip, which is a common "V-trace + PPO" combination.  ``impala`` (the
default) is the paper's clipped-IS policy gradient.

Everything below the ``Worker`` class is pure numpy/torch and unit-tested in
``tests/test_vtrace.py``; the Ray wiring is unchanged in spirit.

Run::

    pip install "ray[default]" torch gymnasium
    python examples/rl_ray_ppo.py --iterations 200
"""

from __future__ import annotations

import argparse
import collections
import logging
import threading
import time

import numpy as np
import torch
import torch.nn as nn

_LOG = logging.getLogger("dlt.rl_ray_ppo")

__all__ = [
    "vtrace_targets",
    "build_contexts",
    "impala_policy_loss",
    "ppo_clipped_loss",
    "PopArtValue",
    "AgentModel",
    "ReplayMemory",
    "Normalizer",
    "Context",
    "run_parallel",
    "parse_args",
]


# --------------------------------------------------------------------------- #
# V-trace (pure numpy; the core of the IMPALA fix)
# --------------------------------------------------------------------------- #
def vtrace_targets(
    rewards,
    dones,
    values,
    behavior_log_probs,
    target_log_probs,
    *,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
):
    """Truncated importance-sampled (V-trace) value targets and advantages.

    Parameters
    ----------
    rewards, dones:
        length ``T`` arrays for the transitions.
    values:
        length ``T + 1`` array; ``values[t]`` is ``V(x_t)`` and ``values[T]`` is
        the bootstrap value ``V(x_T)``.
    behavior_log_probs:
        ``log mu(a_t | x_t)`` recorded by the actor at rollout time.
    target_log_probs:
        ``log pi(a_t | x_t)`` under the *current* learner network.
    gamma, rho_bar, c_bar:
        discount and the two truncation levels (``rho_bar >= c_bar``).

    Returns
    -------
    (vs, advantages, rhos) as float64 numpy arrays of length ``T``.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    dones = np.asarray(dones, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    behavior_log_probs = np.asarray(behavior_log_probs, dtype=np.float64).reshape(-1)
    target_log_probs = np.asarray(target_log_probs, dtype=np.float64).reshape(-1)

    T = rewards.shape[0]
    if T == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if values.shape[0] != T + 1:
        raise ValueError(f"values must have length T+1={T + 1}, got {values.shape[0]}")

    # Importance ratios. Clipping the exponent only guards against overflow when
    # the two policies are astronomically far apart; the truncation below is
    # what actually bounds the correction.
    log_ratio = np.clip(target_log_probs - behavior_log_probs, -20.0, 20.0)
    ratio = np.exp(log_ratio)
    rho = np.minimum(ratio, rho_bar)
    c = np.minimum(ratio, c_bar)

    mask = 1.0 - dones
    deltas = rho * (rewards + gamma * values[1:] * mask - values[:-1])

    # v_s = V_s + sum_{t>=s} gamma^{t-s} (prod c) delta_t, computed backwards.
    vs = np.empty(T, dtype=np.float64)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = deltas[t] + gamma * c[t] * mask[t] * acc
        vs[t] = values[t] + acc

    next_vs = np.concatenate([vs[1:], values[-1:]])
    advantages = rho * (rewards + gamma * next_vs * mask - values[:-1])
    return vs, advantages, rho


def build_contexts(obs, indices, ctx_size, dones=None):
    """Stack the ``ctx_size`` observations ending at each index.

    ``obs`` is the flat sequence of per-step observations stored by the worker.
    When ``dones`` is given, observations from *before the most recent episode
    boundary* are zeroed, matching the zero-padded context the actor actually
    used after an episode reset (the old stacking ignored the boundary).
    """
    obs = np.asarray(obs, dtype=np.float32)
    indices = np.asarray(indices)
    windows = np.stack(
        [obs[indices - offset] for offset in range(ctx_size - 1, -1, -1)], axis=1
    )
    if dones is None:
        return windows

    dones = np.asarray(dones).astype(bool).reshape(-1)
    last_done = np.full(obs.shape[0], -1, dtype=np.int64)
    current = -1
    for i in range(obs.shape[0]):
        last_done[i] = current
        if dones[i]:
            current = i

    windows = windows.copy()
    for row, t in enumerate(indices):
        boundary = last_done[t]
        if boundary < 0:
            continue
        start = max(0, int(t) - ctx_size + 1)
        for slot, position in enumerate(range(start, int(t) + 1)):
            if position <= boundary:
                windows[row, slot] = 0.0
    return windows


def impala_policy_loss(target_log_probs, advantages, rhos):
    """``-E[ rho_s * A_s * log pi(a_s|x_s) ]`` (truncated IS policy gradient)."""
    return -(rhos * advantages * target_log_probs).mean()


def ppo_clipped_loss(target_log_probs, behavior_log_probs, advantages, clip=0.2):
    """PPO clip, but with V-trace advantages (optional ``--policy-loss ppo``)."""
    ratio = torch.exp(target_log_probs - behavior_log_probs)
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


# --------------------------------------------------------------------------- #
# model / rollout bookkeeping
# --------------------------------------------------------------------------- #
class PopArtValue(nn.Module):
    """PopArt: adaptively normalise value targets while preserving predictions.

    The head predicts a *normalised* value ``v_norm``; the real value is
    ``v_norm * std + mean``.  We keep a running mean/variance of the (V-trace)
    value targets and, whenever the statistics change, rescale the linear
    weights so the predicted value is unchanged.  The loss is then computed in
    normalised space, which is what keeps the value scale under control when the
    discounted returns are ~100x the per-step reward (IMPALA uses PopArt for the
    same reason).
    """

    def __init__(self, d_in, beta=0.99):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)
        self.beta = beta
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @property
    def std(self):
        # Floor the scale: if the targets briefly have (near) zero variance the
        # normalised targets would explode.  Kept well above the epsilon used
        # when learning the variance.
        return torch.sqrt(self.var + 1e-4).clamp(min=1e-3)

    def forward(self, h):
        return self.linear(h)  # normalised value

    def denormalize(self, v_norm):
        return v_norm * self.std + self.mean

    def normalize(self, values):
        return (values - self.mean) / self.std

    @torch.no_grad()
    def update(self, targets):
        """Adapt mean/var to ``targets`` and rescale weights to preserve outputs."""
        targets = targets.reshape(-1).float()
        if targets.numel() == 0:
            return
        batch_mean = targets.mean()
        batch_var = targets.var(unbiased=False)
        old_mean, old_std = self.mean.clone(), self.std.clone()
        if not bool(self.initialized):
            new_mean, new_var = batch_mean, batch_var
            self.initialized.fill_(True)
        else:
            new_mean = self.beta * old_mean + (1 - self.beta) * batch_mean
            new_var = self.beta * self.var + (1 - self.beta) * batch_var
        new_std = torch.sqrt(new_var + 1e-4).clamp(min=1e-3)
        # Preserve outputs: v * old_std + old_mean == v' * new_std + new_mean.
        scale = (old_std / new_std).reshape(1)
        self.linear.weight.data.mul_(scale)
        self.linear.bias.data.mul_(scale).add_(((old_mean - new_mean) / new_std).reshape(1))
        self.mean.copy_(new_mean.reshape(1))
        self.var.copy_(new_var.reshape(1))


class AgentModel(nn.Module):
    def __init__(self, ctx_size, d_obs, d_act, low, high, popart_beta=0.99):
        super().__init__()
        self.flatten = nn.Flatten()
        self.act = nn.LeakyReLU(0.01)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.d1 = nn.Linear(ctx_size * d_obs, 128)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)
        self.d2 = nn.Linear(128, 128)
        self.dv = PopArtValue(128, beta=popart_beta)
        self.dloc = nn.Linear(128, d_act)
        self.dcov = nn.Linear(128, d_act)
        # Buffers so they move with .to(device) and show up in state_dict.
        self.register_buffer("w", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def features(self, obs):
        obs = self.flatten(obs)
        h = self.act(self.d1(obs))
        h = self.ln1(h)
        h = self.act(self.d2(h))
        h = self.ln2(h)
        return h

    def dist_value_norm(self, obs):
        """Return ``(distribution, normalised value)`` for training (PopArt space)."""
        h = self.features(obs)
        aloc = self.w * self.tanh(self.dloc(h)) + self.b
        # Floor the scale: sigmoid can saturate to 0, which would make the
        # Gaussian entropy -inf and poison the loss.
        acov = self.sig(self.dcov(h)) + 1e-3
        return torch.distributions.Normal(aloc, acov), self.dv(h)

    def dist_value(self, obs):
        """Return ``(distribution, value)`` in real units, without sampling."""
        dist, v_norm = self.dist_value_norm(obs)
        return dist, self.dv.denormalize(v_norm)

    def forward(self, obs):
        dist, value = self.dist_value(obs)
        act = dist.sample()
        # Sum over action dimensions -> joint log-probability (scalar per step).
        act_logprob = dist.log_prob(act).sum(-1)
        return act, act_logprob, dist, value


class ReplayMemory:
    """Flat rollout buffer: one entry per transition, in time order."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.clear()

    def add(self, obs, act, act_logprob, reward, done, value=None):
        self.obs.append(obs)
        self.act.append(act)
        self.act_logprob.append(act_logprob)
        self.r.append(reward)
        self.done.append(done)
        self.value.append(value)

    def clear(self):
        self.obs = []
        self.act = []
        self.act_logprob = []
        self.r = []
        self.done = []
        self.value = []

    def __len__(self):
        return len(self.r)

    def as_arrays(self):
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "act": np.asarray(self.act, dtype=np.float32),
            "act_logprob": np.asarray(self.act_logprob, dtype=np.float32).reshape(-1),
            "rewards": np.asarray(self.r, dtype=np.float32).reshape(-1),
            "dones": np.asarray(self.done, dtype=np.float32).reshape(-1),
        }


class Normalizer:
    """Running observation mean/std with per-worker delta accumulation.

    Two *separate* sets of counters are essential:

    * ``sum_obs / sumsq_obs / cnt_obs`` are the **global** statistics used for
      normalisation (shared with workers via :meth:`adopt`);
    * ``sum_obs_collect / sumsq_obs_collect / cnt_obs_collect`` accumulate only
      the observations this worker has seen since the last :meth:`take_collect`.

    If a single set were used for both, the driver would re-send the global
    totals to every worker, ``take_collect`` would hand them back, and
    ``aggregate`` would add them again -- the statistics would grow like
    ``(1 + n_workers)^iterations`` until they overflow to NaN.
    """

    def __init__(self, obs_shape, dtype):
        self.dtype = dtype
        self.obs_shape = obs_shape
        self.sum_obs = np.zeros(obs_shape, dtype=dtype)
        self.sumsq_obs = np.zeros(obs_shape, dtype=dtype)
        self.cnt_obs = 0.0
        self.reset_collect()

    def reset_collect(self):
        self.sum_obs_collect = np.zeros(self.obs_shape, dtype=self.dtype)
        self.sumsq_obs_collect = np.zeros(self.obs_shape, dtype=self.dtype)
        self.cnt_obs_collect = 0.0

    def take_collect(self):
        """Return this worker's freshly collected deltas and reset them."""
        stats = {
            "sum": self.sum_obs_collect.copy(),
            "sumsq": self.sumsq_obs_collect.copy(),
            "cnt": self.cnt_obs_collect,
        }
        self.reset_collect()
        return stats

    def aggregate(self, stats):
        """Fold another normalizer's *deltas* into these global statistics."""
        self.sum_obs = self.sum_obs + stats["sum"]
        self.sumsq_obs = self.sumsq_obs + stats["sumsq"]
        self.cnt_obs = self.cnt_obs + stats["cnt"]

    def adopt(self, other):
        """Use ``other``'s running statistics for normalisation (reset deltas)."""
        self.sum_obs = np.array(other.sum_obs, dtype=self.dtype, copy=True)
        self.sumsq_obs = np.array(other.sumsq_obs, dtype=self.dtype, copy=True)
        self.cnt_obs = float(other.cnt_obs)
        self.reset_collect()

    def mean_std_obs(self):
        if self.cnt_obs <= 0:
            return np.zeros(self.obs_shape, dtype=self.dtype), np.ones(
                self.obs_shape, dtype=self.dtype
            )
        mean_obs = self.sum_obs / self.cnt_obs
        var_obs = np.maximum(self.sumsq_obs / self.cnt_obs - mean_obs**2, 0.0)
        return mean_obs, np.sqrt(var_obs)

    def normalize_obs(self, obs):
        mean_obs, std_obs = self.mean_std_obs()
        return (obs - mean_obs) / (1e-3 + std_obs)

    def collect_obs(self, obs):
        """Accumulate one observation into this worker's delta counters."""
        self.sum_obs_collect = self.sum_obs_collect + obs
        self.sumsq_obs_collect = self.sumsq_obs_collect + obs**2
        self.cnt_obs_collect = self.cnt_obs_collect + 1


class Context:
    def __init__(self, ctx_size, d_obs, dtype):
        self.ctx_size = ctx_size
        self.d_obs = d_obs
        self.dtype = dtype
        self.obs_ctx = [np.zeros(d_obs, dtype=np.float32) for _ in range(ctx_size)]
        self.normalizer = Normalizer((d_obs,), np.float32)

    def add(self, obs):
        self.obs_ctx.append(self.normalizer.normalize_obs(obs))
        self.obs_ctx.pop(0)

    def get(self, device):
        return torch.as_tensor(
            np.stack(self.obs_ctx, 0), dtype=self.dtype, device=device
        ).unsqueeze(0)

    def reset(self):
        self.obs_ctx = [np.zeros(self.d_obs, dtype=np.float32) for _ in range(self.ctx_size)]

    def set_normalizer(self, normalizer):
        # Adopt the global statistics but keep collecting our own deltas.
        self.normalizer.adopt(normalizer)


# --------------------------------------------------------------------------- #
# Ray worker
# --------------------------------------------------------------------------- #
def _build_worker_class(ray, num_gpus=0.5):
    @ray.remote(num_gpus=num_gpus)
    class Worker:
        def __init__(self, params):
            import gymnasium as gym

            self.ctx_size = params["ctx_size"]
            self.rollout_steps = params.get("rollout_steps", 128)
            self.dtype = params["dtype"]
            self.lr = params["lr"]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # V-trace / optimisation hyper-parameters.
            self.gamma = params.get("gamma", 0.99)
            self.rho_bar = params.get("rho_bar", 1.0)
            self.c_bar = params.get("c_bar", 1.0)
            self.epochs = params.get("epochs", 2)
            self.value_coef = params.get("value_coef", 0.5)
            self.entropy_coef = params.get("entropy_coef", 0.01)
            self.max_grad_norm = params.get("max_grad_norm", 0.5)
            self.clip = params.get("clip", 0.2)
            self.policy_loss = params.get("policy_loss", "impala")
            self.normalize_advantages = params.get("normalize_advantages", True)
            self.popart_beta = params.get("popart_beta", 0.99)

            self.env = gym.make("Pendulum-v1")
            self.d_obs = self.env.observation_space.shape[0]
            self.d_act = self.env.action_space.shape[0]
            self.low = self.env.action_space.low
            self.high = self.env.action_space.high
            self.agent = AgentModel(
                self.ctx_size,
                self.d_obs,
                self.d_act,
                self.low,
                self.high,
                popart_beta=self.popart_beta,
            ).to(self.device)
            self.opt = torch.optim.Adam(self.agent.parameters(), params["lr"])
            self.memory = ReplayMemory(self.dtype)
            self.context = Context(self.ctx_size, self.d_obs, self.dtype)
            self.T = 0
            # Bounded reward history: only the recent window is ever used, so a
            # plain list here used to grow without limit for the whole run.
            self.rewards = collections.deque(maxlen=100)
            self.done = False
            # Train every ``rollout_steps`` transitions and then drop the buffer.
            self.steps_since_train = 0
            self.last_loss = 0.0

        # -- introspection ------------------------------------------------- #
        def get_info_dims(self):
            return self.d_obs, self.d_act

        def get_weights(self):
            return {k: v.detach().cpu() for k, v in self.agent.state_dict().items()}

        def get_avg_reward(self):
            # Average over the recent window (deque maxlen=100).  Averaging only
            # the last 4 steps is far too noisy to judge learning.
            return float(np.mean(self.rewards)) if self.rewards else 0.0

        def get_last_loss(self):
            return self.last_loss

        def train_get_weights_infos(self):
            if self.steps_since_train >= self.rollout_steps and len(self.memory) > self.ctx_size:
                self.train_policy()
                self.memory.clear()
                self.steps_since_train = 0
            return (
                self.get_weights(),
                self.get_avg_reward(),
                self.context.normalizer.take_collect(),
                self.last_loss,
            )

        def set_weights(self, weights):
            # Recover from a previously diverged state: Adam's moments would
            # still be NaN, so rebuild the optimiser before loading good weights.
            if not all(bool(torch.isfinite(v).all()) for v in self.agent.state_dict().values()):
                _LOG.warning("received non-finite weights; resetting the optimiser")
                self.opt = torch.optim.Adam(self.agent.parameters(), self.lr)
            self.agent.load_state_dict(weights)

        def set_normalizer(self, normalizer):
            self.context.set_normalizer(normalizer)

        def set_lr(self, lr):
            """Update the learning rate (used for annealing by the driver)."""
            self.lr = float(lr)
            for group in self.opt.param_groups:
                group["lr"] = self.lr

        def reset_initialize(self):
            self.context.reset()
            obs, _ = self.env.reset()
            self.context.normalizer.collect_obs(obs)
            self.context.add(obs)
            self.T = 0

        @torch.no_grad()
        def evaluate(self, episodes=5):
            """Mean episode return of the deterministic (mean-action) policy.

            Episodes use fixed seeds so successive evaluations measure the
            policy instead of the (large) variation of Pendulum's random
            starting angle; otherwise a 5-episode estimate is very noisy.
            """
            returns = []
            for episode in range(episodes):
                obs, _ = self.env.reset(seed=1000 + episode)
                context = [np.zeros(self.d_obs, dtype=np.float32) for _ in range(self.ctx_size)]
                context.append(self.context.normalizer.normalize_obs(obs))
                context.pop(0)
                episode_return = 0.0
                done = False
                while not done:
                    batch = torch.as_tensor(
                        np.stack(context, 0), dtype=self.dtype, device=self.device
                    ).unsqueeze(0)
                    distribution, _ = self.agent.dist_value(batch)
                    action = distribution.mean.detach().cpu().numpy()[0]
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    context.append(self.context.normalizer.normalize_obs(obs))
                    context.pop(0)
                    episode_return += float(reward)
                    done = bool(terminated or truncated)
                returns.append(episode_return)
            self.done = True  # the next rollout must restart the training episode
            return float(np.mean(returns))

        # -- learning (the IMPALA/V-trace part) ---------------------------- #
        def train_policy(self):
            data = self.memory.as_arrays()
            obs_flat = data["obs"]
            size = obs_flat.shape[0]
            if size <= self.ctx_size:
                return

            index = np.arange(self.ctx_size - 1, size)
            ctx_np = build_contexts(obs_flat, index, self.ctx_size, data["dones"])

            ctx = torch.as_tensor(ctx_np, dtype=self.dtype, device=self.device)
            if not bool(torch.isfinite(ctx).all()):
                _LOG.warning("non-finite observations; skipping training")
                return
            acts = torch.as_tensor(data["act"][index], dtype=self.dtype, device=self.device)
            behavior_logp = torch.as_tensor(
                data["act_logprob"][index], dtype=self.dtype, device=self.device
            )
            rewards = data["rewards"][index]
            dones = data["dones"][index]

            last_loss = torch.tensor(0.0, device=self.device)
            for _ in range(self.epochs):
                # Normalised point of view (PopArt): v_norm is the raw head output,
                # values are the real-unit values fed to V-trace.
                dist, v_norm = self.agent.dist_value_norm(ctx)
                target_logp = dist.log_prob(acts).sum(-1)
                values = self.agent.dv.denormalize(v_norm).reshape(-1)
                _, boot_norm = self.agent.dist_value_norm(self.context.get(self.device))
                bootstrap = self.agent.dv.denormalize(boot_norm).reshape(-1)
                values_np = (
                    torch.cat([values, bootstrap]).detach().cpu().numpy()
                )

                # Targets are constants: no gradient flows through V-trace.
                with torch.no_grad():
                    vs, advantages, rhos = vtrace_targets(
                        rewards,
                        dones,
                        values_np,
                        behavior_logp.detach().cpu().numpy(),
                        target_logp.detach().cpu().numpy(),
                        gamma=self.gamma,
                        rho_bar=self.rho_bar,
                        c_bar=self.c_bar,
                    )
                    vs_t = torch.as_tensor(vs, dtype=self.dtype, device=self.device)
                    adv_t = torch.as_tensor(advantages, dtype=self.dtype, device=self.device)
                    rho_t = torch.as_tensor(rhos, dtype=self.dtype, device=self.device)
                if self.normalize_advantages and adv_t.numel() > 1:
                    # A per-batch affine transform of the advantages does not
                    # change the V-trace correction, it just scales the gradient.
                    # Only standardise when there is variance to divide by --
                    # otherwise 0/0 produces NaNs.
                    scale = adv_t.std()
                    if float(scale) > 1e-6:
                        adv_t = (adv_t - adv_t.mean()) / scale

                if self.policy_loss == "ppo":
                    policy_loss = ppo_clipped_loss(
                        target_logp, behavior_logp, adv_t, clip=self.clip
                    )
                else:
                    policy_loss = impala_policy_loss(target_logp, adv_t, rho_t)

                # Value loss in PopArt's normalised space (scale-free).
                normalised_target = self.agent.dv.normalize(vs_t).detach()
                value_loss = torch.mean((v_norm.reshape(-1) - normalised_target) ** 2)
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss):
                    # A single pathological batch must not poison the weights.
                    _LOG.warning("non-finite loss; skipping this update")
                    self.opt.zero_grad(set_to_none=True)
                    continue

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grads_finite = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.agent.parameters()
                )
                if not grads_finite:
                    # Clipping inf/nan would turn it into nan; skip instead.
                    _LOG.warning("non-finite gradients; skipping this update")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.opt.step()
                # Adapt the value statistics; weights are rescaled so the
                # predicted value itself does not jump.
                if torch.isfinite(vs_t).all():
                    self.agent.dv.update(vs_t)
                last_loss = loss

            self.last_loss = float(last_loss.detach().cpu())

        def rollout(self, T_rollout):
            for _ in range(T_rollout):
                if self.done:
                    self.done = False
                    self.reset_initialize()
                obs_ctx = self.context.get(self.device)
                act, act_logprob, _, value = self.agent(obs_ctx)
                act = act.detach().cpu().numpy()[0]
                act_logprob = float(act_logprob.detach().cpu().reshape(-1)[0])
                value = float(value.detach().cpu().reshape(-1)[0])

                obs_, reward, terminated, truncated, _ = self.env.step(act)
                self.done = bool(terminated or truncated)

                self.memory.add(
                    self.context.obs_ctx[-1], act, act_logprob, reward, self.done, value
                )
                self.context.normalizer.collect_obs(obs_)
                self.context.add(obs_)
                self.rewards.append(reward)
                self.T += 1
                self.steps_since_train += 1

    return Worker


def _build_caller_class(ray):
    @ray.remote
    class WorkerCaller:
        """Pump rollouts continuously until told to stop.

        The pump runs on a background *thread* inside the actor: a plain
        ``while running:`` loop inside the actor method would occupy the actor's
        only execution slot, so ``stop()`` could never be scheduled and the
        program would hang on shutdown.
        """

        def __init__(self, workers, rollout_steps):
            self.workers = workers
            self.n_workers = len(workers)
            self.rollout_steps = rollout_steps
            self._stop = threading.Event()
            self._thread = None

        def start(self):
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self):
            pending = [w.rollout.remote(self.rollout_steps) for w in self.workers]
            while not self._stop.is_set():
                ready, _ = ray.wait(pending, num_returns=1, timeout=1.0)
                for obj in ready:
                    index = pending.index(obj)
                    pending[index] = self.workers[index].rollout.remote(self.rollout_steps)

        def stop(self):
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5)

    return WorkerCaller


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_parallel(args):
    import ray

    ray.init(
        num_gpus=args.num_gpus,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    Worker = _build_worker_class(ray, num_gpus=args.worker_gpus)
    WorkerCaller = _build_caller_class(ray)

    params = {
        "ctx_size": args.ctx_size,
        "rollout_steps": args.rollout_steps,
        "lr": args.lr,
        "dtype": torch.float32,
        "gamma": args.gamma,
        "rho_bar": args.rho_bar,
        "c_bar": args.c_bar,
        "epochs": args.epochs,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "clip": args.clip,
        "policy_loss": args.policy_loss,
        "normalize_advantages": args.normalize_advantages,
        "popart_beta": args.popart_beta,
    }
    workers = [Worker.remote(params) for _ in range(args.n_workers)]
    avg_weight = ray.get(workers[0].get_weights.remote())
    d_obs, _ = ray.get(workers[0].get_info_dims.remote())
    ray.get([w.reset_initialize.remote() for w in workers])

    normalizer = Normalizer((d_obs,), np.float32)
    caller = WorkerCaller.remote(workers, args.rollout_steps)
    caller.start.remote()

    try:
        for iteration in range(args.iterations):
            infos = ray.get([w.train_get_weights_infos.remote() for w in workers])
            workers_weights, workers_reward, workers_stats, workers_loss = zip(*infos)
            for stats in workers_stats:
                normalizer.aggregate(stats)

            # Averaging every single iteration is "local SGD with a sync every
            # step": workers barely train before their weights are replaced.
            # Syncing every --sync-every iterations lets each worker make real
            # progress and is closer to true async IMPALA (stale actors), which
            # is exactly what V-trace corrects for.
            sync_now = args.sync_every <= 1 or iteration % args.sync_every == 0
            if sync_now:
                # Ignore any worker that diverged, so one bad actor cannot
                # poison the averaged weights everyone else receives.
                healthy = [
                    weights
                    for weights in workers_weights
                    if all(bool(torch.isfinite(v).all()) for v in weights.values())
                ]
                if healthy:
                    avg_weight = {
                        key: sum(weights[key] for weights in healthy) / len(healthy)
                        for key in avg_weight
                    }
                else:
                    _LOG.warning("all workers returned non-finite weights; keeping previous")

            handles = []
            if args.lr_final is not None:
                # Linear annealing: the standard cure for late-phase oscillation.
                lr_now = args.lr + (args.lr_final - args.lr) * (
                    iteration / max(1, args.iterations - 1)
                )
            else:
                lr_now = args.lr
            for worker in workers:
                if sync_now:
                    handles.append(worker.set_weights.remote(avg_weight))
                handles.append(worker.set_normalizer.remote(normalizer))
                handles.append(worker.set_lr.remote(lr_now))
            ray.get(handles)

            line = (
                f"[{iteration}] avg_reward={sum(workers_reward) / args.n_workers:.3f} "
                f"loss={sum(workers_loss) / args.n_workers:.3f}"
            )
            if args.eval_every and (iteration + 1) % args.eval_every == 0:
                eval_returns = ray.get(
                    [w.evaluate.remote(args.eval_episodes) for w in workers]
                )
                line += f" eval_return={float(np.mean(eval_returns)):+.1f}"
            print(line)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("manually stopped")
    finally:
        try:
            ray.get(caller.stop.remote(), timeout=15)
        except Exception:
            pass
        ray.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Async actor-learner (IMPALA V-trace) with Ray"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="if set, linearly anneal the learning rate from --lr to this value",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rho-bar", type=float, default=1.0, help="V-trace rho truncation")
    parser.add_argument("--c-bar", type=float, default=1.0, help="V-trace c truncation")
    parser.add_argument("--epochs", type=int, default=2, help="optimisation passes per batch")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument(
        "--policy-loss", choices=("impala", "ppo"), default="impala",
        help="impala = V-trace IS policy gradient; ppo = V-trace advantages + PPO clip",
    )
    parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="standardise V-trace advantages per batch before the policy loss",
    )
    parser.add_argument(
        "--popart-beta",
        type=float,
        default=0.99,
        help="EMA decay for PopArt value statistics (first update is exact)",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=1,
        help="average worker weights every N iterations (1 = every iteration)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="evaluate deterministic episode return every N iterations (0 = off)",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--worker-gpus",
        type=float,
        default=0.5,
        help="GPU fraction requested per worker (n_workers * worker_gpus should be <= num_gpus)",
    )
    parser.add_argument("--local-mode", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_parallel(parse_args())
