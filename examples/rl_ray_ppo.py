"""Asynchronous multi-worker PPO with Ray + PyTorch (optional heavy example).

This is the original ``rl_ray_ppo.py`` example, moved under ``examples/`` and
bug-fixed.  It is **not** part of the minimal template and its dependencies are
heavy::

    pip install "ray[default]" torch gymnasium

Run (a real GPU is recommended)::

    python examples/rl_ray_ppo.py --iterations 200

Fixes applied compared with the original
----------------------------------------
* ``ray.init()`` no longer runs at import time (importing the module had the
  side effect of starting a cluster).
* ``gym`` -> ``gymnasium`` (``gym`` is unmaintained and the code already used
  the new 5-tuple ``step`` API).
* Observability scalars ``w``/``b`` are registered as buffers, so they follow
  the module across devices and appear in ``state_dict``.
* The model and all tensors are moved to the correct device; ``.numpy()`` is
  replaced by ``.detach().cpu()``.
* Normalizer collection counters are consumed and reset (``take_collect``),
  instead of being aggregated again and again every iteration.
* The worker pump can be stopped; the training loop handles ``Ctrl+C`` and
  shuts the cluster down.
* Reward averaging and hyper-parameters come from ``argparse``.
* **Bounded memory** (second pass): ``self.rewards`` is a ``deque(maxlen=100)``
  instead of an ever-growing list; the replay buffer is drained on a
  ``rollout_steps`` cadence rather than an episode-length condition (short
  episodes used to make it grow forever); and the rollout is stacked / GAE is
  computed once per training pass instead of once per PPO minibatch.
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np
import torch
import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self, ctx_size, d_obs, d_act, low, high, gamma=0.95):
        super().__init__()
        self.gamma = gamma
        self.flatten = nn.Flatten()
        self.act = nn.LeakyReLU(0.01)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.d1 = nn.Linear(ctx_size * d_obs, 128)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)
        self.d2 = nn.Linear(128, 128)
        self.dv = nn.Linear(128, 1)
        self.dloc = nn.Linear(128, d_act)
        self.dcov = nn.Linear(128, d_act)
        # Buffers so they move with .to(device) and show up in state_dict.
        self.register_buffer("w", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def forward(self, obs):
        obs = self.flatten(obs)
        h = self.act(self.d1(obs))
        h = self.ln1(h)
        h = self.act(self.d2(h))
        h = self.ln2(h)
        aloc = self.w * self.tanh(self.dloc(h)) + self.b
        acov = self.sig(self.dcov(h))
        a_dist = torch.distributions.Normal(aloc, acov)
        act = a_dist.sample()
        act_logprob = a_dist.log_prob(act)
        value = self.dv(h) / (1 - self.gamma)
        return act, act_logprob, a_dist, value


class ReplayMemory:
    def __init__(self, dtype):
        self.dtype = dtype
        self.clear()

    def add(self, obs, act, act_logprob, reward, done, value):
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

    def arrays(self):
        """Stack the rollout once (instead of once per PPO minibatch)."""
        return (
            np.asarray(self.obs, dtype=np.float32),
            np.asarray(self.act, dtype=np.float32),
            np.asarray(self.act_logprob, dtype=np.float32),
        )

    def sample(self, ctx_size, batch_size, obs_np, act_np, act_logprob_np, returns, advantages, device):
        index = np.random.choice(range(ctx_size - 1, len(self.r)), batch_size)
        obs_ctx = np.stack(
            [obs_np[index - ctx_idx] for ctx_idx in range(ctx_size - 1, -1, -1)],
            axis=1,
        )
        return (
            torch.as_tensor(obs_ctx, dtype=self.dtype, device=device),
            torch.as_tensor(act_np[index], dtype=self.dtype, device=device),
            torch.as_tensor(act_logprob_np[index], dtype=self.dtype, device=device),
            torch.as_tensor(returns[index], dtype=self.dtype, device=device),
            torch.as_tensor(advantages[index], dtype=self.dtype, device=device),
        )


class Normalizer:
    """Running mean/std with separate "collection" counters for async workers."""

    def __init__(self, obs_shape, dtype):
        self.dtype = dtype
        self.obs_shape = obs_shape
        self.sum_obs = np.zeros(obs_shape, dtype=dtype)
        self.sumsq_obs = np.zeros(obs_shape, dtype=dtype)
        self.cnt_obs = 0.0

    def take_collect(self):
        """Return this worker's freshly collected sums and reset them."""
        stats = {
            "sum": self.sum_obs.copy(),
            "sumsq": self.sumsq_obs.copy(),
            "cnt": self.cnt_obs,
        }
        self.sum_obs = np.zeros(self.obs_shape, dtype=self.dtype)
        self.sumsq_obs = np.zeros(self.obs_shape, dtype=self.dtype)
        self.cnt_obs = 0.0
        return stats

    def aggregate(self, stats):
        self.sum_obs += stats["sum"]
        self.sumsq_obs += stats["sumsq"]
        self.cnt_obs += stats["cnt"]

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
        self.sum_obs = self.sum_obs + obs
        self.sumsq_obs = self.sumsq_obs + obs**2
        self.cnt_obs = self.cnt_obs + 1

    # numpy pickling helpers ------------------------------------------------
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


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
        self.normalizer = normalizer


def compute_advantages(rewards, dones, values, gamma=0.95, lambd=0.95):
    advantages = np.zeros(len(rewards), dtype=np.float64)
    adv = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        mask = 1 - int(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        adv = delta + gamma * lambd * mask * adv
        advantages[t] = adv
    returns = advantages + np.asarray(values[:-1], dtype=np.float64)
    return returns, advantages


def ppo_loss(pred_act_dist, pred_value, acts, act_logprobs, returns, advantages, clip=0.2):
    act_logprobs = act_logprobs.view(-1, 1)
    pred_value = pred_value.view(-1, 1)
    returns = returns.view(-1, 1)
    advantages = advantages.view(-1, 1)
    ratio = torch.exp(pred_act_dist.log_prob(acts) - act_logprobs)
    clipped_ratio = torch.clip(ratio, 1 - clip, 1 + clip)
    loss_policy = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
    loss_value = torch.mean((pred_value - returns) ** 2)
    return loss_policy + loss_value


def _build_worker_class(ray):
    @ray.remote(num_gpus=0.5)
    class Worker:
        def __init__(self, params):
            import gymnasium as gym

            self.batch_size = params["batch_size"]
            self.ctx_size = params["ctx_size"]
            self.rollout_steps = params.get("rollout_steps", 128)
            self.dtype = params["dtype"]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.env = gym.make("Pendulum-v1")
            self.d_obs = self.env.observation_space.shape[0]
            self.d_act = self.env.action_space.shape[0]
            self.low = self.env.action_space.low
            self.high = self.env.action_space.high
            self.agent = AgentModel(
                self.ctx_size, self.d_obs, self.d_act, self.low, self.high
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
            # Keying the trigger on episode length (the old ``self.T``) meant the
            # buffer was never cleared for short episodes -> unbounded growth.
            self.steps_since_train = 0

        def get_info_dims(self):
            return self.d_obs, self.d_act

        def get_weights(self):
            return {k: v.detach().cpu() for k, v in self.agent.state_dict().items()}

        def get_avg_reward(self):
            return float(np.mean(list(self.rewards)[-4:])) if self.rewards else 0.0

        def train_get_weights_infos(self):
            if self.steps_since_train >= self.rollout_steps and len(self.memory) > self.ctx_size:
                self.train_policy()
                self.memory.clear()
                self.steps_since_train = 0
            return self.get_weights(), self.get_avg_reward(), self.context.normalizer.take_collect()

        def set_weights(self, weights):
            self.agent.load_state_dict(weights)

        def set_normalizer(self, normalizer):
            self.context.set_normalizer(normalizer)

        def reset_initialize(self):
            self.context.reset()
            obs, _ = self.env.reset()
            self.context.normalizer.collect_obs(obs)
            self.context.add(obs)
            self.T = 0
            # Note: ``self.memory`` intentionally survives episode boundaries
            # (it is cleared after each training pass), so a new episode must
            # not reset ``steps_since_train``.

        def train_policy(self):
            n_batches = max(1, int(self.T / self.batch_size) + 1)
            obs_ctx = self.context.get(self.device)
            _, _, _, last_value = self.agent(obs_ctx)
            last_value = float(last_value.detach().cpu().reshape(-1)[0])

            # Compute the rollout arrays and GAE once per training pass instead
            # of once per minibatch (the old code re-stacked and re-computed
            # them inside every ``sample`` call).
            obs_np, act_np, act_logprob_np = self.memory.arrays()
            returns, advantages = compute_advantages(
                self.memory.r, self.memory.done, self.memory.value + [last_value]
            )

            self.opt.zero_grad()
            for _ in range(n_batches):
                batch = self.memory.sample(
                    self.ctx_size,
                    self.batch_size,
                    obs_np,
                    act_np,
                    act_logprob_np,
                    returns,
                    advantages,
                    self.device,
                )
                obs_ctxs, acts, act_logprobs, batch_returns, batch_advantages = batch
                _, _, pred_act_dist, pred_value = self.agent(obs_ctxs)
                loss = ppo_loss(
                    pred_act_dist, pred_value, acts, act_logprobs, batch_returns, batch_advantages
                )
                (loss / n_batches).backward()
            self.opt.step()

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
        """Pump rollouts continuously until told to stop."""

        def __init__(self, workers, rollout_steps):
            self.workers = workers
            self.n_workers = len(workers)
            self.rollout_steps = rollout_steps
            self.running = True

        def start(self):
            pending = [w.rollout.remote(self.rollout_steps) for w in self.workers]
            while self.running:
                ready, _ = ray.wait(pending, num_returns=1, timeout=1.0)
                for obj in ready:
                    index = pending.index(obj)
                    pending[index] = self.workers[index].rollout.remote(self.rollout_steps)
                if not ready:
                    time.sleep(0.01)

        def stop(self):
            self.running = False

    return WorkerCaller


def run_parallel(args):
    import ray

    ray.init(
        num_gpus=args.num_gpus,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    Worker = _build_worker_class(ray)
    WorkerCaller = _build_caller_class(ray)

    params = {
        "batch_size": args.batch_size,
        "ctx_size": args.ctx_size,
        "rollout_steps": args.rollout_steps,
        "lr": args.lr,
        "dtype": torch.float32,
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
            workers_weights, workers_reward, workers_stats = zip(*infos)
            avg_weight = {
                k: sum(w[k] for w in workers_weights) / args.n_workers for k in avg_weight
            }
            for stats in workers_stats:
                normalizer.aggregate(stats)

            handles = []
            for worker in workers:
                handles.append(worker.set_weights.remote(avg_weight))
                handles.append(worker.set_normalizer.remote(normalizer))
            ray.get(handles)

            print(f"[{iteration}] avg_reward={sum(workers_reward) / args.n_workers:.3f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("manually stopped")
    finally:
        try:
            ray.get(caller.stop.remote())
        except Exception:
            pass
        ray.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Async multi-worker PPO with Ray")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ctx-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--local-mode", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_parallel(parse_args())
