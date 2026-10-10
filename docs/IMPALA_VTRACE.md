# 异步 actor-learner 的异策略问题与 V-trace 修复

对应文件：`examples/rl_ray_ppo.py`（默认 `--policy-loss impala`）。

## 1. 问题：为什么异步 PPO 会不稳定

当前架构是 actor-learner：

* actor（`Worker.rollout`）持续用**当前权重** `mu` 采样并写入 buffer；
* learner（`Worker.train_policy`）稍后才用**已经变过的权重** `pi` 去训练这份数据。

也就是说，buffer 里的 `(s, a, r)` 由旧策略 `mu` 产生，而更新用的是新策略 `pi`。PPO 的
clip 比例 `pi(a|s)/mu(a|s)` 和 GAE 目标都假设 `mu ≈ pi`（on-policy）；当 rollout 与训练
之间存在权重滞后时，这个假设被打破：

* 价值目标 `r + γV(s')` 用的是**当前** `V`，但轨迹来自旧策略 → 目标有偏；
* 少数很偏离的样本会产生巨大的比例，梯度被放大 → 训练震荡/发散。

## 2. IMPALA 的解法：V-trace

IMPALA（Espeholt et al., ICML 2018）用**截断重要性采样**修正价值目标，并用它做策略梯度。
实现见 `vtrace_targets()`：

```
rho_t = min(rho_bar, pi(a_t|x_t) / mu(a_t|x_t))     # 价值目标的权重
c_t   = min(c_bar,   pi(a_t|x_t) / mu(a_t|x_t))     # 迹的截断系数
delta_t = rho_t ( r_t + gamma V(x_{t+1}) - V(x_t) )
v_s  = V(x_s) + sum_{t=s}^{T-1} gamma^{t-s} (prod_{i=s}^{t-1} c_i) delta_t
A_s  = rho_s ( r_s + gamma v_{s+1} - V(x_s) )
policy loss = - E[ rho_s * A_s * log pi(a_s|x_s) ]
value  loss =   E[ (v_s - V(x_s))^2 ]
```

要点：

* `v_s` 是修正后的价值目标（`vs`）；`A_s` 是修正后的优势；
* `rho_bar` / `c_bar` **截断**重要性权重，单个很离策略的样本无法把更新炸掉；论文默认
  两者都为 1；
* 当 `pi == mu` 时所有比例为 1，V-trace 退化为 on-policy 的多步回报，因此它是 on-policy
  方法的严格推广。

代码对应：

| 论文符号 | 代码 |
| --- | --- |
| `mu` | 采样时存下的 `act_logprob`（`ReplayMemory.act_logprob`） |
| `pi` | 训练时用当前网络重算的 `dist.log_prob(acts).sum(-1)` |
| `V` | `AgentModel.dist_value()` 的 value 输出 |
| `v_s, A_s, rho` | `vtrace_targets(...)` 返回值 |
| 策略损失 | `impala_policy_loss(...)`；`--policy-loss ppo` 时用 `ppo_clipped_loss(...)` |

## 3. 相对原实现的改动

1. **损失换成 V-trace**：删掉 GAE + `ppo_loss`，改为 `vtrace_targets` +
   `impala_policy_loss`（默认）；保留 `--policy-loss ppo` 作为“V-trace 优势 + PPO clip”
   的选项。
2. **按轨迹整段训练**：V-trace 的迹是沿时间递推的，按 minibatch 随机采样会打断它，
   所以每个 batch 对整段 rollout、做 `--epochs` 遍（默认 2）。
3. **正确存储 log-prob**：原来 `a_dist.log_prob(act)` 只取了第一个维度，多维动作会错；
   现在对动作维求和（`sum(-1)`），得到联合 log 概率。
4. **上下文重建修复**：`build_contexts(..., dones=...)` 会在 episode 边界把旧 episode 的
   观测清零，和 actor 重置后的零填充上下文一致（原来跨 episode 直接堆叠）。
5. **PopArt 价值归一化**：`PopArtValue` 维护价值目标的滑动均值/方差，价值头预测
   *归一化* 价值 `v_norm`，实际价值为 `v_norm * std + mean`；每次更新统计量时同步缩放
   linear 层的权重/偏置，保证输出不变；损失在归一化空间计算。折扣回报量级约为
   `reward/(1-gamma)`（`gamma=0.99` 时是每步奖励的 ~100 倍），不做归一化会让价值损失高达
   `1e4~1e5` 且价值头学不动；PopArt 后归一化损失回到 O(1)，这也是 IMPALA 论文的做法。
   `--popart-beta`（默认 0.99）控制统计量的 EMA；首次更新直接用整批统计量。
6. **优势标准化**：`--normalize-advantages`（默认开）对每个 batch 做仿射标准化，纯粹是
   梯度尺度控制，不改变 V-trace 修正本身。
7. 参数全部走 `argparse`：`--gamma --rho-bar --c-bar --epochs --value-coef
   --entropy-coef --max-grad-norm --policy-loss --popart-beta
   --normalize-advantages`。

### PopArt 更新公式

```
batch_mean, batch_var = mean(targets), var(targets)
new_mean = beta*mean + (1-beta)*batch_mean
new_var  = beta*var  + (1-beta)*batch_var
scale    = sqrt(var+eps) / sqrt(new_var+eps)
W <- W * scale                     # 保持输出不变
b <- b * scale + (mean - new_mean) / sqrt(new_var+eps)
mean, var <- new_mean, new_var
```

保持输出不变很关键：策略分布不会因为价值尺度变化而被“惊到”，只是损失被重新缩放。

## 4. 怎么用 / 怎么调

```bash
pip install "ray[default]" torch gymnasium
python examples/rl_ray_ppo.py --iterations 500            # 默认 impala
python examples/rl_ray_ppo.py --policy-loss ppo           # V-trace + PPO clip
```

* **滞后越严重**（worker 越多、rollout 越长、训练越慢）越应该用 V-trace；
* `rho_bar`/`c_bar` 调小（如 0.5）会更保守、方差更低，但偏差更大；默认 1.0；
* `--epochs` 越大越 off-policy（同一批数据被反复用），V-trace 仍能纠正，但建议 ≤ 3；
* 如果任务是 dense reward 且量级很大，先用 `--gamma` 和奖励缩放把回报控制住。

## 5. 验证

纯函数部分有完整单测（`tests/test_vtrace.py`）：

* 与论文公式的 **O(T²) 直接实现逐步对齐**（随机数据 + 多种 `rho_bar/c_bar`）；
* on-policy 时退化为折扣回报；
* 比例截断到 `rho_bar`；
* `done` 会同时中断 bootstrap 和迹的乘积；
* 空输入、长度校验、策略损失梯度/裁剪行为；
* **PopArt**：首次更新等于整批统计量、更新统计量时输出保持不变、归一化后目标
  `mean≈0, std≈1`、EMA 随重复更新收敛。

另外用 no-op Ray stub 驱动真实 `Worker`（Gymnasium Pendulum）跑训练，确认损失有限、
缓冲区有界（`tests/test_ppo_worker.py`）。

### 真实 Ray 多进程 + GPU 实跑

```
python examples/rl_ray_ppo.py --iterations 1200 --n-workers 4 --num-gpus 1 \
    --worker-gpus 0.25 --rollout-steps 128 --lr 5e-4
```

GTX 1070 / Ray 2.59 上连续 1200 次迭代**无 NaN、无异常**，per-step reward 从 -6.11 提升到
-4.47（趋势约 +0.17/100 迭代），loss 稳定在 ~0.3。期间修掉了真实多进程才会暴露的问题：
`WorkerCaller` 退出死锁、`Normalizer` 全局统计与 worker 增量混用导致的指数爆炸、PopArt
eps 不一致等（见 `PITFALL.md` 第 25–30 条）。复现绘图的脚本：
`scripts/plot_rl_log.py`。

> 说明：这里验证的是**修正的正确性与训练稳定性**（以及指标确实上升），不是“多少步解出
> Pendulum”。Pendulum 完全收敛需要远多于示例默认步数；示例的定位是可读、可扩展的分布式骨架。

## 6. 收敛慢 / 看着不稳定，原因与调法

第一版 1200 迭代的曲线（`reports/rl_impala_reward.png`）在 per-step reward ≈ -4.3 处“平台化”，
看起来既不收敛也不稳定。逐项排查后的结论：

1. **训练量不够**：`rollout_steps=128, epochs=2, 4 workers, 1200 iters` ≈ **0.6M 环境步**、
   每个 worker 只有 **~2400 次优化**。Pendulum 用 IMPALA 通常要百万级步 / 上万次更新。
2. **指标太噪**：`avg_reward` 是最近 100 个 **per-step** reward 的均值；Pendulum 的初始角度
   方差极大，5 个随机 episode 的评估噪声就有 ±200。于是“抖动”很多是**评估噪声**，不是策略崩。
   → 新增 `Worker.evaluate()`：用**固定种子**跑 K 个 episode、取确定性动作（`dist.mean`）的
   **episode return**，才能干净地看收敛。
3. **常数学习率 + 每步同步**：把训练量加到 3000 迭代（rollout 256、epochs 4、~6M 步、
   ~48k 更新）后，策略能到近最优（eval **-177**），但会在 **-177 ↔ -1055** 之间来回振荡。
   原因：没有信任域/KL 约束（V-trace 只修正偏差、不降方差）；worker 每步（或每 4 步）平均权重、
   但 Adam 的动量没有一起平均，每次同步都会踢一下优化轨迹；多 epoch 复用旧数据。
4. **修复：学习率退火**。新增 `--lr-final`，driver 每次迭代把线性衰减后的 lr 下发给 worker。
   同样 2000 迭代、`3e-4 → 3e-5`，固定种子评估的曲线平滑收敛到 **-208 ~ -290**（近最优，
   见 `reports/rl_impala_annealed.png`），对比同迭代数下常数 lr 的 **-888 ~ -1033**。

可调项：

| 参数 | 作用 |
| --- | --- |
| `--lr-final` | 线性退火（**最有效的稳定手段**） |
| `--sync-every N` | 每 N 次迭代才平均一次权重（默认 1；调大更接近真正异步 IMPALA，减少同步踢动） |
| `--policy-loss ppo` | 用 V-trace 优势 + PPO clip 作为信任域 |
| `--epochs` | 每批数据复用次数；越大越 off-policy，靠 V-trace 纠正，建议 ≤ 4 |
| `--eval-every/--eval-episodes` | 固定种子 episode-return 评估 |

推荐配置：

```bash
python examples/rl_ray_ppo.py \
  --iterations 2000 --n-workers 4 --worker-gpus 0.25 \
  --rollout-steps 256 --epochs 4 --lr 3e-4 --lr-final 3e-5 \
  --sync-every 4 --eval-every 50 --eval-episodes 15
```

## 7. 保存权重并把策略渲染成动画

训练时加 `--checkpoint-dir runs/impala-pendulum`：每次评估若刷新最好成绩，就把
**权重 + 观测归一化统计** 存到 `best.pt`（原子写，并用 `never_overwrite` 保护旧文件）。
注意存的是“评估最好”而不是“最后一次”——本示例后期仍会波动，best 往往远好于 last。

```bash
python scripts/render_policy.py runs/impala-pendulum/best.pt --episodes 4 \
    -o reports/pendulum_best.mp4
```

`render_policy.py` 用 `render_mode="rgb_array"` + pygame 无头渲染，imageio-ffmpeg 编码
H.264；画面上叠加 iteration / eval_return / episode return。`--random` 可渲染未训练策略做对比。
实测收敛策略在 seed 0/1/2/3 上的回报为 **-136.8 / -7.9 / -123.3 / -361.9**（随机策略约 -1500）。

## 参考

* IMPALA 论文：Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance
  Weighted Actor-Learner Architectures*, ICML 2018,
  <https://arxiv.org/abs/1802.01561>（V-trace 见第 2 节）。
* 你给出的参考实现：
  <https://github.com/RLOpensource/IMPALA-Scalable-Distributed-Deep-RL-with-Importance-Weighted-Actor-Learner-Architectures>
