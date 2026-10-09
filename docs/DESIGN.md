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

## 为什么默认不 patch `open`、改成装饰器/作用域？

全局 patch `builtins.open` 会波及**进程里的所有代码**：第三方库正常的 `write_text`、
临时文件、缓存都会莫名其妙产生 `.bak-*`，甚至打断依赖文件语义的逻辑。因此：

* 默认**不安装**任何 patch（`import dlt.never_overwrite` 无副作用）；
* `@protect_paths("path")` 完全不改 `open`，只是在调用前把指定参数备份走，语义最清晰，
  而且对 `torch.save` 这种绕过 `open` 的写法同样有效；
* `@protect` / `protect_scope(only=[...])` 只在函数/代码块执行期间替换 `open`，退出即还原，
  还可以用 `only` 把范围限制到具体文件/目录，避免影响被调用的第三方库；
* 只有确实需要时才 `install()`（或 `DLT_NEVER_OVERWRITE=1`）做进程级 patch。

代价是作用域内仍然是全局替换（Python 没有更细的拦截点），所以 `only` 过滤和“退出即还原”
是这道设计的关键。

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

payload 结构：`MAGIC(4) + version(1) + 类型/压缩标志(2) + raw_len(4) + sha256[:16] + body`。
body 是 **JSON**（v2，可选 zlib 压缩）而不是 pickle：JSON 能在编码时就按类型测量并限制
大小（`nbytes`/`len`）、对敏感键脱敏，解码只需白名单，不会像 `pickle.loads` 那样执行代码。
读取端限制解压后的大小（防 zip bomb），校验和不匹配立即报错。v1 的 pickle payload 默认
拒绝，需显式 `allow_pickle=True`。

**安全边界**：图片可被恶意替换，元信息里的代码在 `retrieve_plot` 时会被 `exec`。
因此默认拒绝执行，必须先 `allow_execute=True`；只读的 `retrieve_code`/`retrieve_data`
/`unpack` 不会执行代码。

## 依赖分层

* 核心（`naming`、`_atomic`、`never_overwrite`）：仅标准库；
* 日志（`yaml_logging`）：+ PyYAML；
* 画图（`plot_utils`）：+ Pillow；payload 是**有大小上限的 JSON**（不再用 pickle/dill）；
* `rl_ray_ppo` 示例：+ ray / gymnasium / torch，属于可选的“重”示例，不参与最小模板。

`dlt/__init__.py` 导入时不做任何副作用（不启动集群、不 patch `open`），
`plot_utils` 通过 `__getattr__` 懒加载，因此没装 Pillow 也不影响使用核心功能。

## 日志的同步/异步取舍

默认 `async_save=False`：`save()` 返回时文件一定已经原子落盘，语义最简单、最可预测。
但 YAML 序列化是 O(历史长度)，在训练线程里做会让 GPU 队列断供。因此提供
`async_save=True`：`save()` 只置一个“待写”标志，后台线程取最新快照写盘（多次请求自动
合并），`__exit__`/`close()`/`atexit`/`flush()` 负责等待落盘。线程间通过
`threading.RLock` 保护 metrics/info/events。

这是“可预测性 vs 吞吐”的显式开关，不替用户做决定；但示例默认开启异步。

## 与第三方框架的关系

核心保持框架无关。需要 Lightning 时，用 `dlt.integrations.lightning.DLTLogger` 把
`Trainer` 的 metrics/hyper-parameters 接到 `YamlLogger`；这是**可选子包**，不 import 就
不会引入 lightning 依赖。训练加速（DDP/AMP/累积/compile）仍由 `Trainer` 负责。
