# PITFALL

本轮重构踩到的坑与结论，供后续维护参考。

## 1. 文件沙箱只允许写工作区

`pip install` / `pip install --user` 都因权限失败：

```
[Errno 13] 权限不够: '/home/ordhktvu3862/miniconda3/envs/tch/lib/python3.13/site-packages/...'
[Errno 13] 权限不够: '/home/ordhktvu3862/.local/lib/python3.13'
```

**结论**：不要往工作区外写文件（也正是 AGENTS.md 的要求）。需要装依赖时在项目内建
venv：`python3 -m venv --system-site-packages envs/venv`，再用 `envs/venv/bin/pip install`。

## 2. matplotlib 默认往 `~/.cache` 写字体缓存

首次 `import matplotlib` 会因沙箱无法写 `~/.cache/matplotlib` 而报权限警告。
**结论**：设 `MPLCONFIGDIR=<项目>/cache/matplotlib`（`plot_utils` 已在装饰器与重绘路径
里 `setdefault`），并把 `MPLBACKEND=Agg` 用于非交互场景。

## 3. PyYAML 的 `CDumper` 与自定义 representer

可以在 `yaml.CSafeDumper` 子类上 `add_multi_representer(object, ...)`，用于兜底序列化
numpy/torch/任意对象；否则遇到未知类型会抛 `RepresenterError`。已实测可用。

## 4. PNG 的 EXIF 前缀陷阱

旧代码统一 `img.info['exif'][6:]`，那个 6 字节 `Exif\x00\x00` 前缀是 **JPEG** 专有。
PNG 用 `PngInfo.add_itxt()` 写入、从 `img.info['dlt:plot']` 读回才正确（已验证无损往返）。
JPEG 的 `Image.Exif()[0x9286] = blob` 读回来是**裸字节**，不带前缀。两种要分开处理。

## 5. `torch.save` 不经过 Python `open`

现代 PyTorch 对字符串路径用 C++ writer，`torch.save(obj, "m.pt")` **完全不会调用**
`builtins.open`（实测 open 调用数为 0）。patch `open` 保护不了它。
**结论**：单独 `protect_torch()` 包装 `torch.save`。

## 6. “始终备份”而非“按磁盘空间决定”

旧实现 `shutil.disk_usage('/')...free > 2**34` 决定是否备份，磁盘紧张时静默覆盖。
同目录 `os.replace` 改名不额外占用空间，**没有理由不备份**。

## 7. pytest 默认 tmp 在 `/tmp`

AGENTS.md 要求所有文件落在项目内。**结论**：`pyproject.toml` 里
`addopts = "--basetemp=tmp/pytest"`。

## 8. `open` 包装函数的签名

`def safe_open(filename, mode='r', **kwargs)` 会吞掉第 3 个及之后的位置参数
（`open(f, 'w', 1, 'utf-8')` 直接 `TypeError`）。必须写成
`def safe_open(file, mode='r', *args, **kwargs)`。

## 9. 策略对象泄漏

`uninstall()` 若不重置模块级 `_POLICY`，前一个测试设置的 `enabled=False` 会泄漏到后续
`install()`（因为 `policy or _POLICY`）。**结论**：`uninstall` 置 `_POLICY=None`，
`install(policy=None)` 在未安装时给默认策略。

## 10. 兼容 shim 的 import 副作用（已改变）

早期 `dlt/never_overwrite.py` 在 import 时自动 `install()`，会全局改写 `open`。现已改为
**默认不安装**：`import` 无副作用，用 `@protect` / `protect_scope` / `install()` 按需开启。
测试里不再需要靠环境变量规避，直接 `uninstall()` 即可。

---

# 第二轮补充

## 11. `collections.deque` 不支持切片

把无界的 `self.rewards` 列表改成 `deque(maxlen=100)` 后，原来的
`np.mean(self.rewards[-4:])` 会 `TypeError: sequence index must be integer`。
必须 `list(...)[-4:]`。回归测试见 `tests/test_ppo_memory.py`。

## 12. 异步保存必须把最后一次写刷出去

`async_save=True` 时 `save()` 只发信号，如果进程在后台线程写完前退出，最后一段日志会丢。
因此 `__exit__` / `close()` / `catch` 都调用 `flush()`，并且 `atexit` 注册的是
`_finalize`（save + flush）而不是裸 `save`。多个保存请求会 coalesce 成一次最新快照。

## 13. 备份目录里的 `.bak` 可能被 glob 误加载

`never_overwrite` 把备份留在原目录（`model.bak-<ts>-<id>.pt`）。加载 checkpoint 时若用
`glob("*.pt")` 可能把备份也扫进去。建议用精确文件名，或设 `backup_dir` 集中存放。

## 14. Lightning 的 `log_every_n_steps` 默认 50

用 2 个 batch 做集成测试时发现日志里 `metrics` 为空——因为 `Trainer` 默认
`log_every_n_steps=50`，不足 50 步不会调用 `log_metrics`。测试里需显式设
`log_every_n_steps=1`。

## 15. Lightning 的 dataloader 传参

`train_dataloaders=[(x, y), ...]`（普通 list）会被当成“一个 batch”，训练时报
`linear(): argument 'input' must be Tensor, not tuple`。要用
`DataLoader(TensorDataset(...), batch_size=...)`。

---

# 第三轮补充

## 16. pickle 不是“只读”，它本身就会执行代码

`plot_utils` 原本用 pickle/dill 存参数，`retrieve_data` 看起来只是“读数据”，实际上
`pickle.loads` 遇到恶意 `__reduce__` 就会执行任意代码。所以“只有 `retrieve_plot` 危险”
的说法是错的。已改为白名单 JSON（v2），读取端只做 JSON 解析；v1 pickle payload 默认拒绝。

## 17. pickle 无法限制大小，会把整个数据集塞进图片

用 pickle 时只能在序列化*之后*才知道多大；大数组/大 bytes 会被整个写进 PNG/EXIF。
JSON 编码可以在类型层面先看 `nbytes`/`len`，超限直接转摘要。这是“会不会把大文件写进
exif”的根本修复。

## 18. `except Exception: 写 sidecar` 会污染非图片文件

原 `embed_metadata` 只要 `Image.open` 失败就走 sidecar，于是画图函数误写 `.csv`/`.pt`
也会在它旁边生成 `.meta`。现在先用 `_image_format` 判断：不是可读图片就 raise，绝不碰该
文件；只有“是图片但格式不支持元信息”（TIFF/BMP）才用 sidecar。

## 19. `os.replace` 会吃掉符号链接

对符号链接调用 `os.replace(tmp, link)` 会把链接本身替换成普通文件。改为先
`os.path.realpath()` 再替换目标，链接保持不动。

## 20. zlib 解压要防 zip bomb

解压前检查 header 声明的 `raw_len`，再用 `decompressobj().decompress(data, max_length)`
限制最大输出并检查 `unconsumed_tail`，避免攻击者用小文件解压出巨量内存。

---

# 第五轮补充

## 21. V-trace 的迹不能按 minibatch 打乱

V-trace 的 `v_s` 是沿时间反向递推的（`acc = delta_t + gamma*c_t*mask*acc`）。像 PPO 那样
随机采样 minibatch 会打断这个迹，算出来的目标就是错的。必须对整段 rollout 计算（本例
rollout 只有 ~128 步，整段很便宜）。

## 22. 多维动作的 log-prob 要按维求和

原代码 `a_dist.log_prob(act)` 返回每个动作维度的 log-prob（形状 `(B, d_act)`），而它只取
了 `reshape(-1)[0]`，多维动作时是错的。对角高斯应取 `log_prob(act).sum(-1)` 作为联合
log 概率。

## 23. 上下文跨 episode 会串味

用“把连续若干条 obs 堆起来”重建 context 时，必须在 episode 边界清零；否则新 episode 的
context 里混着上一个 episode 的观测，价值/策略输入就错了。`build_contexts(..., dones=...)`
在每个 batch 内先求“最近一次 done”，把边界之前的 slot 置零。

## 24. 价值尺度：别让价值头从两个数量级之外开始学

折扣回报量级 ≈ `reward/(1-gamma)`（`gamma=0.99` 时约为每步奖励的 100 倍）。旧代码用
`value = dv(h)/(1-gamma)` 硬编码了这个尺度（当时 gamma=0.95）。重构时如果直接把除以
`(1-gamma)` 删掉，价值头会比目标小 ~200 倍，MSE 巨大、学习极慢。最终采用 **PopArt**
（滑动均值/方差 + 更新时缩放输出层保持输出不变），损失在归一化空间计算——实测归一化损失
从 O(1e4) 降到 O(0.1)，统计量自动收敛到回报量级（mean≈-259, std≈113）。

---

# 第六轮补充

## 25. Ray actor 里的 `while` 循环会让 `stop()` 永远排不上

`WorkerCaller.start()` 原来在 actor 方法里 `while running: ...`，而 `running` 只能由
`stop()` 修改；actor 单线程，`start()` 不返回就一直占着执行槽，`stop()` 永远调度不到，
`ray.get(caller.stop.remote())` 永久阻塞 → 退出挂死（表现为 GPU 一直 90%+）。
**修复**：`start()` 里起一个后台线程跑 pump，方法立即返回，actor 才能处理 `stop()`。

## 26. PopArt 的 eps 要一致，否则缩放系数会爆

`normalize()` 用带 eps/下限的 `std`，但 `update()` 里如果用自己的 `sqrt(var+1e-8)` 算
`scale = old_std/new_std`，当 `new_var≈0` 时 `scale` 可达 1e4 量级，输出层权重瞬间炸成
NaN。两处必须用同一个 `sqrt(var+1e-4).clamp(min=1e-3)`。

## 27. 多 worker 权重平均会把一个 NaN 传染给所有人

`avg = sum(weights)/n`：只要有一个 worker 出现 NaN，平均值就是 NaN。驱动端应先过滤掉
非有限的权重再平均；worker 端在 `set_weights` 前若发现自身参数非有限，要**重建优化器**
（Adam 的 exp_avg/exp_avg_sq 也是 NaN，只换权重没用）。

## 28. `clip_grad_norm_` 不能救 inf/nan 梯度

`clip_grad_norm_` 遇到含 inf/nan 的梯度会算出 NaN 系数，反而把梯度变成 NaN。
在裁剪前先判断 `torch.isfinite(grad).all()`，不满足就跳过这一步。

## 29. 数值护栏清单（真实多进程实跑踩出来的）

* 动作标准差 `sigmoid(...)` 会饱和到 0 → 高斯熵 `-inf`，要加下限 `+1e-3`；
* 优势标准化在 batch 方差≈0 时是 0/0，要判 `std > 1e-6` 再除；
* loss 非有限时跳过该步；观测/权重/梯度非有限时都要有兜底；
* 多 worker 共享一张卡用 `--worker-gpus`（如 4×0.25），否则默认 0.5 会因 GPU 资源不足排不上。

## 30. 全局统计和“本 worker 增量”必须分开存（否则指数爆炸）

`Normalizer` 一开始用同一组 `sum_obs/sumsq_obs/cnt_obs` 既做归一化用的全局统计，又做
`take_collect` 的增量累加器。于是每轮同步：driver 把全局统计发给 worker → worker 在其上
继续 collect → `take_collect` 把“全局+增量”交回 → driver 再加一遍。统计量按
`(1+n_workers)^iterations` 增长，~50 轮后溢出成 inf/NaN，表现为训练中途
`Normal(expected ... loc nan)`。

**修复**：`sum_obs/sumsq_obs/cnt_obs` 只存全局统计；`sum_obs_collect/...` 只存本 worker 增量，
`take_collect` 只取后者并清零，`set_normalizer` 用 `adopt()` 复制全局统计并清空增量。
回归测试 `test_normalizer_aggregation_does_not_compound` 断言 50 轮后 cnt == 1000（线性）。
