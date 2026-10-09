# 设计说明

## 为什么是 YAML 而不是 TensorBoard / W&B / JSONL？

* **可读**：训练中途直接 `grep '^latest:' -A3` 就能看到当前 loss，不需要启服务、开网页。
* **可携带**：一个文件里同时有超参数、代码、git commit、环境、曲线，`scp` 一个文件即可复现。
* **极简**：唯一硬依赖是 `PyYAML`。研究代码最怕“为了记日志引入半个生态”。

代价是每次保存要重写整个文件。因此：

* 保存是**原子**的（临时文件 + `os.replace`），且有 `save_every` 控制频率；
* 曲线极长时可用 `jsonl_metrics=True`，把每步追加到 `*.jsonl`，YAML 只留元数据与
  `latest`；
* `keep_last=N` 可限制内存里保留的历史长度。

## 为什么 `never_overwrite` 用“改名为备份”而不是“拒绝写入”？

研究中最常见的事故不是“误删”，而是“用旧脚本重跑，把上次的结果覆盖”。
直接抛异常会打断训练；把旧文件改名则既保留数据又不打断流程，且同目录 `os.replace`
是原子、几乎零成本、不消耗额外磁盘。旧实现用“剩余空间 > 16 GiB”作为开关，方向反了：
磁盘越紧张越不能覆盖，因此新实现始终备份。

`torch.save` 在现代 PyTorch 中走 C++ 写出路径，不经过 Python `open`，所以单独提供
`protect_torch()` 包装。类似地，如果将来用到其它绕过 `open` 的写出 API，应在
`_auto_protect_optional()` 中补充，而不是去 patch 更多底层函数。

## 命名规则 `<UTC时间戳>_<slug>_<随机id>`

* 时间在前 → `ls` 结果天然按时间排序；
* slug → 人一眼能看出实验内容（保留中文）；
* 随机 id → 同一秒内并发启动也不会撞名；
* 全部用 UTC，避免服务器时区不同导致排序错乱；
* `unique_path(..., create=True)` 用 `O_CREAT|O_EXCL` 原子占位，跨进程安全。

## 图片元信息嵌入格式选择

| 格式 | 方案 | 原因 |
| --- | --- | --- |
| PNG | `iTXt` 文本块 | 无损、无需重编码、容量大、UTF-8 |
| JPEG/WEBP | EXIF `UserComment` | 标准位置；用 `quality="keep"` 尽量减少重编码损失 |
| 其它 | 同名 `.meta` sidecar | 兜底，绝不丢 payload |

payload 结构：`MAGIC(4) + version(1) + pickle-tag(1) + compressed(1) + raw_len(4) +
sha256[:16] + body`。压缩用 zlib，校验和不匹配立即报错，避免“读出乱七八糟的东西”。

**安全边界**：图片可被恶意替换，元信息里的代码在 `retrieve_plot` 时会被 `exec`。
因此默认拒绝执行，必须先 `allow_execute=True`；只读的 `retrieve_code`/`retrieve_data`
/`unpack` 不会执行代码。

## 依赖分层

* 核心（`naming`、`_atomic`、`never_overwrite`）：仅标准库；
* 日志（`yaml_logging`）：+ PyYAML；
* 画图（`plot_utils`）：+ Pillow，`dill` 可选（支持 lambda/闭包序列化）；
* `rl_ray_ppo` 示例：+ ray / gymnasium / torch，属于可选的“重”示例，不参与最小模板。

`dlt/__init__.py` 导入时不做任何副作用（不启动集群、不 patch `open`），
`plot_utils` 通过 `__getattr__` 懒加载，因此没装 Pillow 也不影响使用核心功能。
