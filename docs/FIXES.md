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
* 新增 `pyproject.toml`、`tests/`（45 个用例）、`docs/`。
* `.gitignore` 补齐 `AGENTS.md` 要求的生成物目录（`/tmp/`、`/runs/`、`/envs/`、`/cache/`
  等）以及 `*.bak`。
