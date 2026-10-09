# MEMO

## 当前状态（2026-10-10）

对仓库做了一轮完整重构，从“根目录散落的片段脚本”变成可安装、可测试、文档齐全的
`dlt` 包。

已完成：

- [x] `dlt/` 包：`naming`、`yaml_logging`、`never_overwrite`、`plot_utils`、`_atomic`
- [x] 旧根文件改为兼容 shim（`never_overwrite` / `plot_utils` / `yaml_logging`）
- [x] `rl_ray_ppo.py` → `examples/rl_ray_ppo.py`（现代 gymnasium + 修 bug）
- [x] `examples/train_demo.py` 端到端示例（已验证可运行）
- [x] `tests/` 45 个用例全绿
- [x] `pyproject.toml`、`README.md`、`docs/FIXES.md`、`docs/DESIGN.md`
- [x] `.gitignore` 补齐 AGENTS.md 要求的目录

## 关键决策

1. **包名 `dlt`**，根目录保留同名 shim，兼顾“复制文件即用”和“正经 import”。
2. **导入零副作用**：`dlt/__init__.py` 不启动 Ray、不 patch `open`；
   `never_overwrite` 只有显式 import 子模块才生效。
3. **原子写贯穿始终**：日志/配置用临时文件 + `os.replace`，从而与 `never_overwrite`
   解耦（不会每次保存都产生 `.bak`）。
4. **防覆盖始终备份**，不再按剩余磁盘空间决定是否备份。
5. **plot 元信息按格式分流**（PNG iTXt / JPEG EXIF / sidecar 兜底），并做 SHA-256 校验。
6. **执行图片内代码需显式 `allow_execute=True`**（安全）。

## 待办 / 可继续改进

- [ ] `examples/rl_ray_ppo.py` 只做了静态检查（环境缺 ray/gymnasium，未实跑）。
- [ ] 可选：为 `YamlLogger` 增加 `tensorboard` 导出适配器（不引入硬依赖）。
- [ ] 可选：`never_overwrite` 支持 `numpy.save`、`safetensors` 等绕过 `open` 的写出 API。
- [ ] 可选：CI（GitHub Actions）跑 `pytest`。

## 环境备注

- 本机 Python 3.13，`yaml`/`numpy`/`torch`/`tqdm` 已有；`Pillow`/`dill`/`matplotlib`/`pytest`
  装在项目内 venv `envs/venv`（`--system-site-packages`）。
- 跑测试：`envs/venv/bin/python -m pytest -q` 或装好依赖后 `python -m pytest -q`。
