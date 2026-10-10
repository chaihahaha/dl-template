# MEMO

## 当前状态（2026-10-10）

对仓库做了一轮完整重构，从“根目录散落的片段脚本”变成可安装、可测试、文档齐全的
`dlt` 包。

已完成：

- [x] `dlt/` 包：`naming`、`yaml_logging`、`never_overwrite`、`plot_utils`、`_atomic`
- [x] 旧根文件改为兼容 shim（`never_overwrite` / `plot_utils` / `yaml_logging`）
- [x] `rl_ray_ppo.py` → `examples/rl_ray_ppo.py`（现代 gymnasium + 修 bug）
- [x] `examples/train_demo.py` 端到端示例（已验证可运行）
- [x] `tests/` 101 个用例全绿（含 async 日志、PPO 内存、V-trace、PopArt、plot 极端情况、Lightning）
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

## 第五轮（IMPALA V-trace）

- [x] `examples/rl_ray_ppo.py` 的损失从 GAE+PPO 换成 **V-trace 截断重要性采样**
      （`vtrace_targets` + `impala_policy_loss`），修异步异策略不稳定
- [x] 修 log-prob（动作维求和）、episode 边界上下文清零、价值尺度先验、优势标准化
- [x] 新增 `tests/test_vtrace.py`（与论文公式 O(T²) 实现逐步对齐），全套 93 用例
- [x] 文档 `docs/IMPALA_VTRACE.md`（问题/公式/代码对应/调参/参考）

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

- [x] `examples/rl_ray_ppo.py` 用**真实 Ray 多进程 + GPU** 实跑（见下方第六轮）
- [ ] 可选：为 `YamlLogger` 增加 `tensorboard` 导出适配器（不引入硬依赖）。
- [x] `never_overwrite` 支持 `numpy.save` / `safetensors` 等写出 API（`protect_numpy()` /
      `protect_safetensors()`；safetensors 实测绕过 Python `open`）
- [ ] 可选：CI（GitHub Actions）跑 `pytest`。

## 第六轮（PopArt + 真实 Ray GPU 实跑 + 更多写出 API）

- [x] **PopArt** 价值归一化（`PopArtValue`）：滑动均值/方差 + 更新时缩放输出层保持输出不变；
      归一化损失从 O(1e4) 降到 O(0.1~0.9)，统计量自动收敛到回报量级
- [x] 修真实多进程实跑时发现的 **WorkerCaller 退出死锁**：`start()` 的 `while` 占满 actor
      执行槽导致 `stop()` 永远排不上 → 改为 actor 内后台线程 pump，`stop()` 可被调度
- [x] `--worker-gpus` 参数化每个 worker 的 GPU 份额（支持多 worker 共享一张卡）
- [x] 数值稳定护栏：动作 std 下限、优势标准化方差过小时跳过、PopArt std 下限、
      非有限 loss/梯度/权重兜底、驱动端过滤非有限权重
- [x] **修 `Normalizer` 全局统计与 worker 增量混用导致的指数爆炸**（真实多进程跑到
      ~50 轮后 NaN 的根因）：拆分全局 `sum_obs/...` 与增量 `sum_obs_collect/...`，
      `adopt()` 复制全局并清空增量；新增线性增长回归测试
- [x] `never_overwrite.protect_numpy()` / `protect_safetensors()` + `_auto_protect_optional`
- [x] 新增 PopArt / numpy / safetensors 测试；全套 100 用例
- [x] 环境：`ray 2.59`、`safetensors 0.8` 装入 `envs/venv`（PyPI 直连被限速，
      改用腾讯云镜像下载 wheel）

## 第七轮（收敛诊断：为什么曲线不好看/不稳）

用户看 `reports/rl_impala_reward.png` 觉得没收敛、不稳定。排查结论：

- [x] 定位三点：① 第一版训练量不够（0.6M 步 / 2400 次更新）；② `avg_reward`（100 步
      per-step 均值）在 Pendulum 上噪声极大；③ 补足训练量后仍有真实后期振荡（-177↔-1055），
      根因是常数 lr + 权重平均但 Adam 动量不同步 + 无信任域。
- [x] 新增 `Worker.evaluate()`（固定种子、确定性动作、episode return）——评估曲线立刻平滑
- [x] 新增 `--lr-final` 线性退火（driver 每步 `set_lr`），2000 迭代 3e-4→3e-5 平滑收敛到
      **-208 ~ -290**（对比同迭代数常数 lr 的 -888~-1033）
- [x] 新增 `--sync-every`（权重平均频率）；`--eval-every/--eval-episodes`
- [x] `scripts/plot_rl_log.py` 支持画 eval_return
- [x] 报告 `reports/rl_impala_run.md` + 三张对比图

## 环境备注

- 本机 Python 3.13，`yaml`/`numpy`/`torch`/`tqdm` 已有；`Pillow`/`dill`/`matplotlib`/`pytest`
  装在项目内 venv `envs/venv`（`--system-site-packages`）。
- 跑测试：`envs/venv/bin/python -m pytest -q` 或装好依赖后 `python -m pytest -q`。
