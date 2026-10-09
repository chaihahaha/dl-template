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

## 10. 兼容 shim 的 import 副作用

`dlt/never_overwrite.py` 在 import 时自动 `install()`（受 `DLT_NEVER_OVERWRITE` 控制）。
测试里通过 conftest 设 `DLT_NEVER_OVERWRITE=0`，避免整套测试期间 `open` 被全局改写。

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
