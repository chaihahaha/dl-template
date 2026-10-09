# 重构修复清单

本文件记录相对旧代码（`yaml_logging.py` / `never_overwrite.py` / `plot_utils.py` /
`rl_ray_ppo.py`）修复的具体问题，便于回溯。

## `never_overwrite.py`

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | 只在根分区剩余 > 16 GiB 时备份，否则直接覆盖 | **磁盘紧张时反而丢数据** | 始终备份；同目录 `os.replace` 改名几乎不占空间 |
| 2 | `safe_open(filename, mode='r', **kwargs)` | 丢失 `open` 的第 3+ 个位置参数（`buffering` 等），调用即 `TypeError` | `safe_open(file, mode='r', *args, **kwargs)` 完整转发 |
| 3 | 只 patch `builtins.open` | `pathlib.write_text`、`io.open` 不受保护 | 同时 patch `io.open`，覆盖 pathlib |
| 4 | 未处理文件描述符 / 非路径对象 | `open(fd, "w")` 可能报错 | 非路径直接透传 |
| 5 | 跟随符号链接 `os.path.isfile` | 会对链接本身改名，语义错乱 | 跳过符号链接并记录日志 |
| 6 | 备份名用 `ctime + time + uuid`，无去重 | 极端情况下可能撞名 | `unique_path` 保证唯一 |
| 7 | 备份无限增长 | 长期实验产生海量 `.bak` | `BackupPolicy(keep=N)` 可限制数量 |
| 8 | 无开关、无日志 | 无法关闭、出了问题查不到 | `DLT_NEVER_OVERWRITE` 环境变量 + logging |
| 9 | 无法保护 `torch.save` | 现代 PyTorch 走 C++ 写出，绕过 `open` | 新增 `protect_torch()` |

## `yaml_logging.py`

旧文件其实是**无法运行/导入的伪代码**（引用了未定义的 `name`、`config`、`N`、
`dataset`、`model`、`info`、`postprocess`），且 `os.environ['VAR_NAME']=sys.argv[1]`
在无参数时直接 `IndexError`。重写为可导入的 `YamlLogger` 类，并修复：

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | `logger['env_vars'] = os.environ` | **把所有环境变量（含密钥）写进日志** | 默认不记录；`log_env=True` 也会过滤敏感键 |
| 2 | 每次 `open(log, 'w')` 重写 | 配合 `never_overwrite` 每次保存都产生一个 `.bak`，且崩溃可致日志损坏 | 原子写（临时文件 + `os.replace`），不触发备份 |
| 3 | 仅在 `batch_cnt % log_freq == 0` 时保存 | `Ctrl+C` 会丢掉最后一段数据 | 退出/异常/`atexit`/`catch` 上下文都会保存 |
| 4 | `pdb.set_trace()` 放在 `finally` | **非交互环境（CI、重定向）永久挂起** | 调试钩子默认关闭，仅 `DLT_DEBUG=1` 且 TTY 时启用 |
| 5 | `handle_exception` 无条件 `shell.interact()` | 同样会挂起 | 不再默认开启交互 |
| 6 | `yaml.dump` 直接序列化 numpy/torch | `RepresenterError` | 内置 numpy/torch/pathlib/dataclass/Enum representer，未知对象降级为 `repr` |
| 7 | `uuid.uuid1()[:8]` | 泄露 MAC/时间，8 位碰撞概率不低 | `uuid4().hex[:8]` / `secrets` |
| 8 | 无时间戳命名、字段不统一 | 多次实验难以区分 | 可排序文件名 + 结构化字段 |
| 9 | 无系统/git/包版本信息 | 复现困难 | `system` / `git` / `packages` / `code(sha256)` |
| 10 | `datetime.now()` 本地时间 | 跨时区混乱 | 统一 UTC ISO 时间 |
| 11 | 无读取 API | 只能手写 yaml | `load_log` / `load_logs` / `latest_log` |
| 12 | loss 列表无限增长 / 每次全量重写 | 长训练内存与 I/O 爆炸 | `keep_last=N`、`jsonl_metrics=True` |

## `plot_utils.py`

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | 始终读 `img.info['exif'][6:]` | 6 字节前缀是 **JPEG** 专有；PNG 元信息读取错误 | PNG 用 iTXt，JPEG/WEBP 用 EXIF，其它用 sidecar |
| 2 | 先 `os.remove` 再保存 | 保存失败则**原图彻底丢失** | 先写临时文件，成功后再 `os.replace` |
| 3 | `tmp-<hash(filename)>` 拼接路径 | `tmp-123./imgs/a.png` 是非法路径 | 在同一目录生成合法唯一文件名 |
| 4 | `hash()` | 每进程随机加盐，不可复现 | SHA-256 校验 + magic + 版本号 |
| 5 | 参数不可 pickle 时抛异常 | 画图直接失败 | 不可序列化的参数替换为占位符并写入 `warnings` |
| 6 | `clean_code` 手写缩进推断 | 空行/制表符/短行易错 | `textwrap.dedent` + 稳健去除装饰器 |
| 7 | 无脑 `exec` 图片里的代码 | 图片可被投毒执行任意代码 | `retrieve_plot` 需显式 `allow_execute=True` |
| 8 | 手工按 `\n` 切掉装饰器行 | 换行/注释一变就失效 | 规范去缩进后定位 `def` 行 |
| 9 | 位置参数转 kwargs 手写 | `*args`、位置限定参数会出错 | `inspect.signature().bind()` |
| 10 | 不在 `filename` 参数时报错含糊 | 难排查 | 明确 `TypeError` / `FileNotFoundError` |
| 11 | 重新绘制时未关 matplotlib 图 | 内存泄漏 | `plt.close("all")` |
| 12 | 无格式/大小兜底 | 大 payload 或特殊格式静默失败 | sidecar 兜底 + logging 警告 |

## `rl_ray_ppo.py`

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | 模块顶层 `ray.init(num_gpus=1)` | import 就启动集群，副作用极强 | 移入 `run_parallel()`，`__main__` 才执行 |
| 2 | `import gym` | `gym` 已停止维护，且代码用的是新 5 元组 API | 改用 `gymnasium` |
| 3 | `self.w = torch.tensor(...)` | 不是 buffer，不随 `.to(device)`/`state_dict` | `register_buffer` |
| 4 | `torch.tensor` 固定 CPU，`.numpy()` | 有 GPU 时设备不匹配 | 统一 `device`，`.detach().cpu()` |
| 5 | `aggregate_collection` 累加 worker 的收集计数后不清零 | **统计量重复累加，normalizer 失真** | `take_collect()` 取走即清零 |
| 6 | `WorkerCaller.start` 死循环无法停止 | 无法优雅退出 | 增加 `stop()`，主循环 `finally` 调用并 `ray.shutdown()` |
| 7 | `compute_advantages` 返回 tensor 又转 numpy | 冗余且易错 | 统一 numpy |
| 8 | 无 `Ctrl+C` 处理 | 只能强杀 | 捕获后优雅关闭 |
| 9 | 超参数硬编码 | 无法调参 | `argparse` |

## 其它

* 旧根目录脚本 `rl_ray_ppo.py` 移动到 `examples/rl_ray_ppo.py`。
* 旧 `never_overwrite.py` / `plot_utils.py` / `yaml_logging.py` 改为转发 shim，
  保证既有 `import never_overwrite` 等写法继续可用。
* 新增 `pyproject.toml`、`tests/`（60 个用例）、`docs/`。
* `.gitignore` 补齐 `AGENTS.md` 要求的生成物目录（`/tmp/`、`/runs/`、`/envs/`、`/cache/`
  等）以及 `*.bak`。

---

# 第二轮：性能与资源问题

详见 [`PERFORMANCE.md`](PERFORMANCE.md)。摘要：

## RL PPO 内存（`examples/rl_ray_ppo.py`）

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | `self.rewards = [0.0]` 每步 append，永不清理 | **无界内存增长**（只用到最后 4 个） | `collections.deque(maxlen=100)` |
| 2 | 仅当 `self.T > ctx_size`（episode 长度）才 `memory.clear()` | 短 episode 下 replay buffer **永不清理** | 用 `steps_since_train >= rollout_steps` 触发并 drain |
| 3 | `sample()` 每个 minibatch 重新 `np.array` 整段 buffer 并重算 GAE | CPU/瞬时内存放大 `n_batches` 倍 | `arrays()` + GAE 每次训练只算一次 |
| 4 | `np.mean(self.rewards[-4:])` | 换 deque 后切片会 `TypeError` | `list(...)[-4:]` |

## `never_overwrite` 磁盘占用

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | 备份默认无上限 | 反复覆盖大文件（如每个 epoch 同一个 `ckpt.pt`）会写爆磁盘 | `keep` / `max_total_bytes` 上限 |
| 2 | 无磁盘水位保护 | 备份可能把盘写满 | `min_free_bytes`（默认 512 MiB）磁盘吃紧时删最旧备份并 warning |
| 3 | 只能放在原目录 | 不便统一清理 | `backup_dir` 可集中存放 |
| 4 | 无环境变量 | 无法快速调参 | `DLT_NEVER_OVERWRITE_KEEP/_MAX_TOTAL_MB/_MIN_FREE_MB` |

## 日志阻塞训练（`dlt/yaml_logging.py`）

| # | 旧行为 | 问题 | 新行为 |
| --- | --- | --- | --- |
| 1 | `save()` 在训练线程里 `yaml_dump(全量历史)` + 写盘 | GPU 可能等 CPU（实测 40k 点、`save_every=2000`：2.91s） | `async_save=True` 后台线程写盘（同场景 0.07s），多次保存自动合并 |
| 2 | 训练线程 append 与写盘并发 | 数据竞争 | `threading.RLock` 保护 metrics/info/events；`snapshot()` 加锁拷贝 |
| 3 | 退出时异步写可能来不及落盘 | 丢最后一段日志 | `__exit__`/`close()`/`atexit`/`flush()` 都会等待落盘 |
| 4 | `_coerce` 对 CUDA tensor `.cpu().item()` | 每步强制设备同步 | 文档明确警告；建议记录 Python float / 降低频率 |

## PyTorch Lightning

* 模板本身不用 Lightning，因此谈不上“利用 Trainer 的加速”。
* 新增 `dlt/integrations/lightning.py`（`DLTLogger`）与 `examples/lightning_demo.py`，
  可在使用 `Trainer`（DDP / 混合精度 / 梯度累积 / `compile`）的同时获得 dlt 日志。
* `tests/test_lightning.py` 在 Lightning 2.6.6 上实跑 `Trainer.fit` 验证。

---

# 第三轮：`plot_utils` 极端情况加固

核心改动：**payload 从 pickle/dill 换成有大小上限的、白名单式的 JSON（版本 2）**。
pickle 能装下任意对象（整个数据集、打开的文件、密钥），而且反序列化本身就会执行代码，
所以“读取端安全”是假的。JSON 可以测量/限制大小、脱敏、按固定白名单解码。

| # | 极端情况 | 旧行为 | 新行为 |
| --- | --- | --- | --- |
| 1 | 画图函数把结果写到了**非图片**文件（`.txt`/`.csv`/`.pt`） | 在文件旁写 `.meta`，等于污染非代码文件 | 识别到不是可读图片就 **raise**，文件保持原样、不写 sidecar |
| 2 | 参数是**大数组/大 tensor**（如整个数据集） | 直接序列化进 EXIF/PNG，图片可达数 GB | `max_arg_bytes`（默认 256 KiB）超出则存 shape/dtype/统计量摘要 |
| 3 | 参数是**大 bytes/长字符串** | 全量嵌入 | 超过 `max_bytes`/`max_string` 只存长度 + SHA-256 摘要 |
| 4 | 参数总量过大 | 无上限 | `max_payload_bytes`（默认 2 MiB），按大小从大到小用 `omitted` 占位 |
| 5 | 参数含**密钥/Token** | 原样嵌入 | 敏感键名（TOKEN/SECRET/PASSWORD/...）自动 `<redacted>` |
| 6 | **递归/超深/超长容器** | 可能递归爆栈或巨大 payload | 深度、元素数、`id()` 环检测，超限转摘要 |
| 7 | **不受支持的对象**（模块、文件句柄、模型、函数） | 依赖 pickle 成败 | 统一存 `type` + 截断 `repr` 摘要并记 warning |
| 8 | PNG 里嵌入任意大小 payload | 无硬上限 | `_MAX_EMBED_BYTES`（8 MiB）超出转 sidecar；JPEG EXIF 仍受 60 KB 限制 |
| 9 | **zip bomb**：篡改压缩流声明超大解压尺寸 | 直接 `zlib.decompress` | 解压前检查声明大小、`decompressobj` 限制最大输出并校验 |
| 10 | 读取端 `retrieve_data/retrieve_code` | 实际会 `pickle.loads`（可执行代码） | 只解析 JSON，**不执行任何代码**；只有 `retrieve_plot(allow_execute=True)` 执行源码 |
| 11 | 旧版（v1）pickle payload | 直接加载 | 默认拒绝，需显式 `unpack(path, allow_pickle=True)` |
| 12 | `filename` 是**符号链接** | `os.replace` 会把链接替换成普通文件 | 先 `realpath`，写目标文件，保留链接 |
| 13 | 源码函数过大 | 全量嵌入 | `max_code`（默认 200 KB）截断 |
| 14 | 需要精细控制 | 无接口 | `packplot(include=..., exclude=..., limits=EncodeLimits(...))` |

依赖变化：`plot_utils` 现在**只依赖 Pillow**（不再需要 dill）。
