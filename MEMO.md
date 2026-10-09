# MEMO

## 当前状态（2026-10-10）

对仓库做了一轮完整重构，从“根目录散落的片段脚本”变成可安装、可测试、文档齐全的
`dlt` 包。

已完成：

- [x] `dlt/` 包：`naming`、`yaml_logging`、`never_overwrite`、`plot_utils`、`_atomic`
- [x] 旧根文件改为兼容 shim（`never_overwrite` / `plot_utils` / `yaml_logging`）
- [x] `rl_ray_ppo.py` → `examples/rl_ray_ppo.py`（现代 gymnasium + 修 bug）
- [x] `examples/train_demo.py` 端到端示例（已验证可运行）
- [x] `tests/` 71 个用例全绿（含 async 日志、PPO 内存回归、plot 极端情况、Lightning 集成）
- [x] `pyproject.toml`、`README.md`、`docs/FIXES.md`、`docs/DESIGN.md`
- [x] `.gitignore` 补齐 AGENTS.md 要求的目录

## 第三轮（plot_utils 加固 + README 精简）

- [x] payload 从 pickle/dill 换成**有大小上限的 JSON（v2）**：大数组/大 bytes/超长字符串
      转摘要，总预算超出按大小 omitted，敏感键脱敏，递归/超深容器保护
- [x] 非图片输出直接 raise（不再往非代码文件旁写 `.meta`）；符号链接写目标不破坏链接
- [x] zip-bomb 防护；v1 pickle payload 默认拒绝（`allow_pickle=True` 才读）
- [x] `retrieve_data` / `retrieve_code` 现在只解析 JSON，**不会执行代码**
- [x] plot 只依赖 Pillow（移除 dill）；新增 `EncodeLimits`、`include`/`exclude`
- [x] README 改为“如何使用示例 / 如何改网络结构 / 如何加数据集 / 如何扩展”

## 第四轮（never_overwrite 去全局化）

- [x] `import never_overwrite` 不再 patch 任何东西（默认零副作用）
- [x] 新增 `@protect`（函数级作用域）、`protect_scope(only=...)`（代码块级，可限定路径）、
      `@protect_paths("path")`（只备份参数，完全不改 `open`）
- [x] 全局 patch 改为显式 `install()` 或 `DLT_NEVER_OVERWRITE=1`
- [x] 示例 `train_demo.py` 改用 `@protect_paths("path")`

## 第二轮（性能与资源）

- [x] PPO：`rewards` 改 deque、replay buffer 按 `rollout_steps` drain、rollout 只堆叠/算 GAE 一次
- [x] `never_overwrite`：`keep` / `max_total_bytes` / `min_free_bytes`（默认 512 MiB）防写爆盘
- [x] `YamlLogger`：`async_save` 后台写盘 + 线程锁 + `flush()`（实测阻塞开销 ~40×↓）
- [x] PyTorch Lightning：`DLTLogger` + `examples/lightning_demo.py`（Lightning 2.6.6 实跑验证）
- [x] `docs/PERFORMANCE.md`：四个疑问的排查结论与实测数据

## 关键决策

1. **包名 `dlt`**，根目录保留同名 shim，兼顾“复制文件即用”和“正经 import”。
2. **导入零副作用**：`dlt/__init__.py` 不启动 Ray、不 patch `open`；
   `never_overwrite` 导入时不安装任何东西，靠 `@protect` / `protect_scope` /
   `@protect_paths` / 显式 `install()` 按需开启。
3. **原子写贯穿始终**：日志/配置用临时文件 + `os.replace`，从而与 `never_overwrite`
   解耦（不会每次保存都产生 `.bak`）。
4. **防覆盖始终备份**，不再按剩余磁盘空间决定是否备份。
5. **plot 元信息按格式分流**（PNG iTXt / JPEG EXIF / sidecar 兜底），并做 SHA-256 校验。
6. **执行图片内代码需显式 `allow_execute=True`**（安全）。

## 待办 / 可继续改进

- [ ] `examples/rl_ray_ppo.py` 只用 no-op Ray stub 驱动过真实 `Worker`（缓冲/RSS 有界，
      见 `tests/test_ppo_worker.py`），尚未用真实 Ray 多进程实跑。
- [ ] 可选：为 `YamlLogger` 增加 `tensorboard` 导出适配器（不引入硬依赖）。
- [ ] 可选：`never_overwrite` 支持 `numpy.save`、`safetensors` 等绕过 `open` 的写出 API。
- [ ] 可选：CI（GitHub Actions）跑 `pytest`。

## 环境备注

- 本机 Python 3.13，`yaml`/`numpy`/`torch`/`tqdm` 已有；`Pillow`/`dill`/`matplotlib`/`pytest`
  装在项目内 venv `envs/venv`（`--system-site-packages`）。
- 跑测试：`envs/venv/bin/python -m pytest -q` 或装好依赖后 `python -m pytest -q`。
