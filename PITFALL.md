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
