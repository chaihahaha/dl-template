# 性能与资源分析（内存 / 磁盘 / GPU 等待 / Lightning）

针对四个具体疑问的排查结论与修复。结论均给出代码位置，能实测的给出了实测数据。

---

## 1. RL PPO（`examples/rl_ray_ppo.py`）会不会内存泄漏？

**会，原实现有两处会随训练无限增长，另有一处“隐性”的 CPU/内存放大。** 已修复。

### 1.1 `self.rewards` 无界增长（确定泄漏）

原代码：

```python
self.rewards = [0.0]          # __init__
...
self.rewards.append(reward)   # rollout，每一步都 append，永不清理
```

而它只被用到最后 4 个：

```python
def get_avg_reward(self):
    return float(np.mean(self.rewards[-4:]))
```

所以每走一步就多存一个 float，训练越久内存越大。**修复**：改为
`collections.deque(maxlen=100)`（见 `Worker.__init__`），并且 `get_avg_reward`
先 `list(...)` 再切片（`deque` 不支持切片，直接用 `[-4:]` 会 `TypeError`）。

### 1.2 Replay buffer 在“短 episode”下永不清理（确定泄漏）

原触发条件是**当前 episode 的长度**：

```python
def train_get_weights_infos(self):
    if self.T > self.ctx_size:      # T 在 reset_initialize() 里被清零
        self.train_policy()
        self.memory.clear()
```

`self.T` 每回合重置，但 `self.memory` 不重置。如果环境经常在 `ctx_size` 步内终止
（`done=True`），`T` 永远到不了 `ctx_size`，`memory` 就越攒越多，永不清空。

**修复**：用独立的 `steps_since_train` 计数，攒够 `rollout_steps`（默认 128）就训练并
`clear()`（见 `Worker.train_get_weights_infos` / `rollout`）。内存上界 ≈ `rollout_steps`
条 transition。

### 1.3 每个 minibatch 重复整段 rollout（CPU/瞬时内存放大）

原 `ReplayMemory.sample` 每次调用都：

* `np.array(self.obs/act/act_logprob)` 重新堆叠整段 buffer；
* 重新计算一次 GAE（`compute_advantages`）；
* 再转成 tensor。

而 `train_policy` 会循环 `n_batches` 次，等于把上述工作重复 `n_batches` 遍。

**修复**：`ReplayMemory.arrays()` 与 `compute_advantages` 在每次 `train_policy` 里
只做一次，`sample` 只负责按 `ctx_size`/`batch_size` 取索引（见 `ReplayMemory.sample`
与 `Worker.train_policy`）。

### 1.4 尚未覆盖的部分

* Ray object store / actor 堆外内存没有实测（本机无 ray/gymnasium，只做了纯
  PyTorch 部分的回归测试 `tests/test_ppo_memory.py`）。
* `WorkerCaller` 的 `pending` 列表大小 = worker 数，不增长，无泄漏。

---

## 2. `never_overwrite` 在某些情况下会不会把硬盘写爆？

**会**：只要反复用同一个文件名覆盖**大文件**（最典型的就是每个 epoch 存同一个
`ckpt.pt`），备份就会线性累积，最终写满磁盘。已加入多道防线。

### 2.1 机制与数量级

备份做法是把旧文件**改名**到同目录：

```
model.pt  ->  model.bak-<UTC时间戳>-<id>.pt
```

`os.replace` 本身几乎不占额外空间，但**旧版本会一直留着**。反复覆盖同一个大小 S 的
文件 N 次，磁盘占用 ≈ (N+1)·S。一个 2 GB 的 checkpoint 存 100 次就是 ~200 GB。

### 2.2 现在提供的防线

| 参数 / 环境变量 | 作用 |
| --- | --- |
| `BackupPolicy(keep=N)` / `DLT_NEVER_OVERWRITE_KEEP` | 每个文件最多保留 N 份备份 |
| `BackupPolicy(max_total_bytes=B)` / `DLT_NEVER_OVERWRITE_MAX_TOTAL_MB` | 单个文件的备份总量上限，超了删最旧的 |
| `BackupPolicy(min_free_bytes=B)`（默认 512 MiB）/ `DLT_NEVER_OVERWRITE_MIN_FREE_MB` | **磁盘快满时自动删除最旧的备份**并打 warning，防止备份把盘写爆 |
| `BackupPolicy(backup_dir=...)` | 备份集中到指定目录，便于统一清理/挂到另一块盘 |
| `DLT_NEVER_OVERWRITE=0` | 彻底关闭 |

`min_free_bytes` 的剪枝实现见 `_enforce_free_space`：只删除符合备份命名规则的文件，
按 mtime 从旧到新删，并逐条 warning，绝不会误删用户数据。

### 2.3 使用建议

* 对“每次都会重写的日志/配置文件”，用 `dlt` 的原子写（`_atomic.py`）或
  `async_save`，它们走 `os.replace`，**不经过备份逻辑**，不会产生 `.bak`。
* 对 checkpoint：设 `keep`（如 3~5）或 `max_total_bytes`；`torch.save` 已由
  `protect_torch()` 纳入保护。
* 备份留在数据旁边，注意 `glob("*.pt")` 之类的通配可能把 `.bak-*.pt` 也扫进去，建议
  加载时用精确文件名，或把备份放到 `backup_dir`。

---

## 3. `dlt` 输出日志会不会让 GPU 等 CPU，拖慢训练？

**会，原同步保存确实可能造成 GPU 空转。** 两个来源，都已处理/给出开关。

### 3.1 同步 `save()` 在训练线程里做 O(n) 的 YAML dump

`save()` 每次都会 `yaml_dump(整个 metrics 历史)` 再写盘。历史越长越慢，而它发生在
训练主线程里——这段时间没有新的 kernel 入队，GPU 可能就空了。

**实测**（40,000 个数据点，`save_every=2000`，本机 CPU）：

```
async_save=False  loop-time=2.91s   (训练线程被阻塞)
async_save=True   loop-time=0.07s   incl-flush=0.27s
```

即 `async_save=True` 把训练线程里的保存开销从 ~2.9s 降到 ~0.07s（~40×），序列化改到
后台 `dlt-yaml-writer` 线程完成。

**修复/用法**：`YamlLogger(..., async_save=True)`；`save()` 只发信号，写盘由后台线程
合并完成（多次请求会 coalesce 成一次最新快照）。`with` 退出、`close()`、`atexit`
以及 `flush()` 都会等待落盘，不会丢数据。`examples/train_demo.py` 与
`examples/lightning_demo.py` 默认已开启。

### 3.2 `.item()` / `.cpu()` 会同步 CUDA

`YamlLogger._coerce` 对 0 维/单元素 **CUDA tensor** 会调用 `.cpu().item()`，这会强制
设备同步。如果每个 micro-step 都 `log(loss=cuda_loss)`，就是每步一次同步，损失吞吐。

**建议**：

* 在热循环里记录**已经是 Python float** 的值（例如累计若干 step 再
  `float(loss.detach())` 一次），不要每步传 tensor；
* 或降低 `log()` 频率、增大 `save_every`；
* 超长训练用 `jsonl_metrics=True`（每步只追加一行，避免整体重写）配合
  `keep_last=N` 控制内存。

> 注意：`jsonl` 追加仍是同步的小写入，但只有一行，代价远小于整体 dump。

### 3.3 结论

默认（`async_save=False`）是“保存后立刻落盘”的可预测语义，但会阻塞；**对 GPU 训练
建议显式开启 `async_save=True`**，并避免每步同步 tensor。

---

## 4. 有没有“充分利用 PyTorch Lightning 的训练加速”？

**模板核心没有用 Lightning。** 需要分清“加速”来自哪里：

* dlt 本身只是日志/命名/防覆盖工具，**不参与训练**；
* Lightning 的加速全部来自 `Trainer`：DDP/FSDP 多卡、混合精度
  （`precision="16-mixed"/"bf16-mixed"`）、梯度累积
  （`accumulate_grad_batches`）、`torch.compile`、梯度裁剪、checkpoint/早停回调等。
  不使用 `Trainer` 就谈不上“利用”。

因此这次补上了一个**可选集成**，让你能同时用 Lightning 的加速和 dlt 的日志：

* `dlt/integrations/lightning.py`：`DLTLogger`（`LightningLogger` 的实现），把
  `log_metrics` / `log_hyperparams` 写进 `YamlLogger`，`finalize` 时 flush + close，
  `after_save_checkpoint` 记录 checkpoint 路径。
* `examples/lightning_demo.py`：可运行示例，包含推荐的加速开关：

```python
trainer = Trainer(
    logger=DLTLogger("resnet50", save_dir="runs", save_every=100, async_save=True),
    accelerator="gpu", devices=4, strategy="ddp",
    precision="16-mixed",            # 或 bf16-mixed（Ampere+）
    accumulate_grad_batches=2,
    # compile=True,                  # torch>=2.0
)
```

* `tests/test_lightning.py`：在本机 Lightning 2.6.6 上实跑 `Trainer.fit`，验证日志落盘。

补充：Lightning 只在 rank 0 调用 `log_metrics`，配合 `async_save` 不会引入每卡 I/O；
`train_demo.py` 仍是纯 PyTorch，保持框架无关。
