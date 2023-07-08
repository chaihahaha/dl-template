import ray
import torch
import torch.nn as nn
import gym
import numpy as np
import time

ray.init(num_gpus=1)

class AgentModel(nn.Module):
    def __init__(self, ctx_size, d_obs, d_act, low, high):
        super().__init__()
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
        self.w = torch.tensor((high - low)/2)
        self.b = torch.tensor((high + low)/2)

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

        value = self.dv(h)/(1 - 0.95)
        return act, act_logprob, a_dist, value

class ReplayMemory:
    def __init__(self):
        self.clear()
    def add(self, obs, act, act_logprob, r, done, value):
        self.obs.append(obs)
        self.act.append(act)
        self.act_logprob.append(act_logprob)
        self.r.append(r)
        self.done.append(done)
        self.value.append(value)
    def clear(self):
        self.obs = []
        self.act = []
        self.act_logprob = []
        self.r = []
        self.done = []
        self.value = []
    def sample(self, ctx_size, batch_size, last_value):
        M = len(self.r)
        i = np.random.choice(range(ctx_size - 1, M), batch_size)

        returns, advantages = compute_advantages(self.r, self.done, self.value + [last_value])
        returns = np.array(returns)[i]
        advantages = np.array(advantages)[i]

        obs_np = np.array(self.obs)
        act_np = np.array(self.act)
        act_logprob_np = np.array(self.act_logprob)
        obs_ctx = []
        for ctx_idx in range(ctx_size - 1, -1, -1):
            obs_ctx.append(obs_np[i - ctx_idx])
        obs_ctx = np.stack(obs_ctx, axis=1)

        obs_ctx = torch.tensor(obs_ctx)
        act = torch.tensor(act_np[i])
        act_logprob = torch.tensor(act_logprob_np[i])
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        return obs_ctx, act, act_logprob, returns, advantages

class Context:
    def __init__(self, ctx_size, d_obs, dtype):
        self.ctx_size = ctx_size
        self.d_obs = d_obs
        self.obs_ctx = [np.zeros(d_obs) for i in range(ctx_size)]
        self.dtype = dtype
    def add(self, obs):
        self.obs_ctx.append(obs)
        self.obs_ctx.pop(0)
    def get(self):
        return torch.tensor(np.stack(self.obs_ctx, 0), dtype=self.dtype).unsqueeze(0)
    def reset(self):
        self.obs_ctx = [np.zeros(self.d_obs) for i in range(self.ctx_size)]

def compute_advantages(rs, dones, values):
    gamma = 0.95
    lambd = 0.95
    advantages = []
    returns = []
    adv = 0
    for tt in range(len(rs)-1, -1, -1):
        m = 1 - int(dones[tt])
        delta = rs[tt] + gamma * values[tt+1] * m - values[tt]
        adv = delta + gamma * lambd * m * adv
        R = adv + values[tt]
        advantages.append(adv)
        returns.append(R)
    advantages.reverse()
    returns.reverse()
    returns = torch.tensor(np.array(returns))
    advantages = torch.tensor(np.array(advantages))
    return returns, advantages

def ppo_loss(pred_act_dist, pred_value, acts, act_logprobs, returns, advantages):
    act_logprobs = act_logprobs.view(-1, 1)
    pred_value = pred_value.view(-1, 1)
    returns = returns.view(-1, 1)
    advantages = advantages.view(-1, 1)
    ratio = torch.exp(pred_act_dist.log_prob(acts) - act_logprobs)
    clipped_ratio = torch.clip(ratio, 1-0.2, 1+0.2)
    #g = torch.where(A>=0, (1+0.2)*advantages, (1-0.2)*advantages)
    #loss_policy = torch.mean(torch.min(ratio*advantages, g))
    loss_policy = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
    loss_value = torch.mean( (pred_value - returns)**2 )
    loss = loss_policy + loss_value
    return loss

@ray.remote(num_gpus=0.5)
class Worker:
    def __init__(self, params):
        #self.n_batches = params['n_batches']
        self.batch_size = params['batch_size']
        self.ctx_size = params['ctx_size']
        self.dtype = params['dtype']
        self.env = gym.make('Pendulum-v1')
        self.d_obs = self.env.observation_space.shape[0]
        self.d_act = self.env.action_space.shape[0]
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        # 用于决策的神经网络
        self.agent = AgentModel(self.ctx_size, self.d_obs, self.d_act, self.low, self.high)
        self.opt = torch.optim.Adam(self.agent.parameters(), params['lr'])
        self.memory = ReplayMemory()
        self.context = Context(self.ctx_size, self.d_obs, self.dtype)
        self.T = 0
        self.rewards = [0.0]
        self.done = False
    def get_weights(self):
        # 异步收集每个worker的权重用于平均
        return self.agent.state_dict()
    def get_avg_reward(self):
        # 异步收集当前任务成功率等信息
        avg_reward_finished = np.mean(self.rewards[-4:])
        return avg_reward_finished
    def train_get_weights_infos(self):
        # 合并多个异步收集任务，防止时间不同步
        if self.T > self.ctx_size:
            # 若episode时间够长，则训练
            self.train_policy()
            self.memory.clear()
        return self.get_weights(), self.get_avg_reward()
    def set_weights(self, w):
        # 为每个worker分发平均后的权重
        self.agent.load_state_dict(w)
    def reset_initialize(self):
        # 初始化仿真环境，上下文和log信息
        self.context.reset()
        obs, _ = self.env.reset()
        self.context.add(obs)
        self.T = 0
    def train_policy(self):
        # episode结束，训练策略网络
        n_batches = int(self.T / self.batch_size) + 1

        obs_ctx = self.context.get() # 获取状态上下文
        _, _, _, last_value = self.agent(obs_ctx) # 获取不完全轨迹最后一个value用于bootstrap
        last_value = float(last_value.detach().numpy()[0])

        # 计算Generalized Advantage Estimation
        self.opt.zero_grad()
        obs_ctxs, acts, act_logprobs, returns, advantages = self.memory.sample(self.ctx_size, self.batch_size, last_value)   # 从重放记忆中采样经验
        for _ in range(n_batches):
            _, _, pred_act_dist, pred_value = self.agent(obs_ctxs)
            loss = ppo_loss(pred_act_dist, pred_value, acts, act_logprobs, returns, advantages)
            (loss/n_batches).backward()
        self.opt.step()
    def rollout(self, T_rollout):
        # 仿真循环，一直展开仿真到done为True
        for _ in range(T_rollout):
            if self.done:
                self.done = False
                self.reset_initialize()
            obs_ctx = self.context.get() # 获取状态上下文

            # 根据状态上下文决策，得到动作，概率，和价值
            act, act_logprob, _, value = self.agent(obs_ctx)
            act = act.detach().numpy()[0]
            act_logprob = float(act_logprob.detach().numpy()[0])
            value = float(value.detach().numpy()[0])

            # 仿真一步
            obs_, r, terminated, truncated, _ = self.env.step(act)
            self.done = terminated or truncated

            # 将历史经验加入重放记忆中
            self.memory.add(self.context.obs_ctx[-1], act, act_logprob, r, self.done, value)
            # 将需要累积的状态向量加入上下文
            self.context.add(obs_)
            self.rewards.append(r)
            self.T += 1
        return

@ray.remote
class WorkerCaller:
    def __init__(self, workers, rollout_steps):
        # 设置一个对应的worker
        self.workers = workers
        self.n_workers = len(workers)
        self.rollout_steps = rollout_steps
    def start(self):
        # 对workers持续不断地触发rollout函数
        finish_indicators = [worker.rollout.remote(self.rollout_steps) for worker in self.workers]
        while True:
            for i in range(self.n_workers):
                if is_ready(finish_indicators[i]):
                    finish_indicators[i] = self.workers[i].rollout.remote(self.rollout_steps)

def is_ready(obj):
    ready_oids, _ = ray.wait([obj])
    if ready_oids:
        return True
    else:
        return False

def run_parallel():
    params = {'batch_size':64, 'ctx_size':8, 'lr':5e-4, 'n_episodes':99999999, 'n_workers':2, 'rollout_steps':128, 'dtype':torch.float32 }
    n_episodes = params['n_episodes']
    n_workers = params['n_workers']

    # 初始化worker
    workers = [Worker.remote(params) for i in range(n_workers)]
    avg_weight = ray.get(workers[0].get_weights.remote())
    ray.get([worker.reset_initialize.remote() for worker in workers])

    # 初始化持续调用worker的caller
    worker_caller = WorkerCaller.remote(workers, params['rollout_steps'])

    # 启动worker的caller，开始持续异步触发worker的rollout函数
    worker_caller.start.remote()
    time.sleep(1)

    # 主循环
    for i_episodes in range(n_episodes):
        # 收集worker的权重，只要有一个未收集完就会阻塞在这里
        weights_infos = ray.get([worker.train_get_weights_infos.remote() for worker in workers])
        workers_weights, workers_reward = zip(*weights_infos)
        # 计算平均权重
        avg_weight = {k:sum([workers_weights[wid][k] for wid in range(n_workers)])/n_workers for k in avg_weight.keys()}

        # 非阻塞异步地分发权重给每个worker
        for worker in workers:
            worker.set_weights.remote(avg_weight)

        # 处理所有worker的log信息
        avg_reward = sum(workers_reward)/n_workers
        print(avg_reward)
        time.sleep(0.5)
if __name__ == '__main__':
    run_parallel()
