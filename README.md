# 深度学习训练项目模板（dlt）

面向深度学习研究者的**极简**初始化模板：用最少的依赖，解决训练中真正会让人丢数据、
丢实验、复现不了的几个问题。

| 工具 | 作用 | 依赖 |
| --- | --- | --- |
| `dlt.yaml_logging` | 崩溃安全的 YAML 实验日志（随时 `grep` 当前 loss，`Ctrl+C` 不丢数据） | PyYAML |
| `dlt.never_overwrite` | 把 `open(..., "w")` 改成“先备份再覆盖”，并额外保护 `torch.save` | 无 |
| `dlt.naming` | 实验/文件自动命名：可排序、人类可读、永不撞名 | 无 |
| `dlt.plot_utils` | 图片自带“配方”：把画图代码和数据嵌进 PNG/JPEG，随时重新绘制 | Pillow（`dill` 可选） |

> 设计原则：**核心零第三方依赖，重依赖全部可选**。只想要日志时只装 `PyYAML` 即可；
> 用到画图再装 `Pillow`。

---

## 目录结构

```
.
├── dlt/                     # 库代码（核心）
│   ├── naming.py            # 自动命名 / 实验目录
│   ├── yaml_logging.py      # YamlLogger
│   ├── never_overwrite.py   # 防覆盖 open
│   ├── plot_utils.py        # packplot / retrieve_*
│   └── _atomic.py           # 原子写（临时文件 + os.replace）
├── examples/
│   ├── train_demo.py        # 端到端可运行示例（推荐先看）
│   └── rl_ray_ppo.py        # Ray 多卡异步 PPO（可选，重依赖）
├── tests/                   # pytest 测试
├── docs/FIXES.md            # 相对旧代码修复了哪些问题
├── never_overwrite.py       # 兼容旧用法的转发文件（import 即生效）
├── plot_utils.py            # 兼容旧用法的转发文件
├── yaml_logging.py          # 兼容旧用法的转发文件
└── pyproject.toml
```

## 安装

```bash
# 只想要核心（日志 + 命名 + 防覆盖）
pip install pyyaml

# 需要画图功能
pip install pillow dill matplotlib

# 或者以可编辑方式安装本仓库
pip install -e ".[plot,dev]"
```

## 快速开始

```bash
python examples/train_demo.py --epochs 30
```

一次运行会生成：

```
runs/20261010T002915Z_demo-mlp_ab12cd34/
├── config.yaml                 # 本次超参数快照
├── logs/20261010T002915Z_demo-mlp_deadbeef.yaml
├── ckpts/final.pt
└── figures/loss.png            # 内含画图代码与数据
```

---

## 1. 实验日志 `YamlLogger`

```python
import math
from dlt import ExperimentNamer, YamlLogger

run = ExperimentNamer("runs", name="resnet50-cifar").new_run({"lr": 1e-3, "bs": 256})

with YamlLogger("resnet50-cifar", run=run, config={"lr": 1e-3}, save_every=100) as log:
    for step in range(1000):
        loss = math.exp(-step / 200)
        log.log(step=step, loss=loss, acc=1 - loss)   # 自动每 100 步落盘
    log.add_info("训练结束")
```

日志文件的关键字段：

```yaml
experiment: {id, name, created_at, save_count, ...}
system:     {python, platform, hostname, cwd, ...}
packages:   {numpy: ..., torch: ...}
git:        {commit, branch, dirty}       # log_git=True
config:     {...}                          # 超参数
latest:     {step: 999, loss: 0.0067}      # 直接 grep 当前值
metrics:    {loss: [...], acc: [...]}      # 完整曲线
info:       ["..."]                        # 文本信息
code:       {path, sha256, source}         # 本次训练脚本
```

要点：

* **崩溃安全**：每次保存都是“临时文件 + `os.replace`”原子替换，断电/被杀不会写出半截 YAML。
* **`Ctrl+C` 不丢数据**：`with YamlLogger(...)` 退出、未捕获异常、解释器退出时都会保存；
  也可用 `with log.catch():` 显式包裹训练循环。
* **默认不记录环境变量**：`log_env=True` 时也会过滤含 `TOKEN/SECRET/PASSWORD/...` 的变量，
  避免把密钥写进日志。这修复了旧实现直接 `os.environ` 全量落盘的问题。
* **NumPy / torch 直接可写**：`np.float32`、0 维 tensor 会自动转成 Python 标量。
* **长训练可选 `jsonl_metrics=True`**：每个 step 追加一行 JSON 到 `*.jsonl`，避免反复重写整个 YAML。
* **调试钩子默认关闭**：只有 `DLT_DEBUG=1` 且 stdin 是 TTY 时才会进入 `pdb`，CI 不会卡住。

读取日志：

```python
from dlt import load_log, latest_log
data = load_log("runs/.../logs/xxx.yaml")
print(data["latest"]["loss"])
```

## 2. 防覆盖 `never_overwrite`

在脚本第一行：

```python
import never_overwrite          # 旧写法，仍然可用
# 等价于
import dlt.never_overwrite      # noqa: F401
```

此后**任何**以 `"w"` 打开已存在文件的操作，都会先把旧文件重命名为
`原名.bak-<UTC时间戳>-<随机id><后缀>`，再执行写入：

```python
with open("result.csv", "w") as fh:   # result.csv -> result.bak-20261010T002915Z-ab12.csv
    fh.write("new")
```

覆盖范围与细节：

* 同时 patch `builtins.open` 与 `io.open`，因此 `pathlib.Path.write_text/write_bytes`
  也受保护。
* `torch.save` 走的是 C++ 写出路径、绕开 Python `open`，需显式调用
  `dlt.never_overwrite.protect_torch()`（若 import 时 torch 已加载则自动完成）。
* `"a"`（追加）和 `"x"`（独占创建）不会丢数据，不做备份。
* 符号链接跳过备份并记录日志（移动链接会改变语义）。
* 空文件不备份；`backup_dir` 可把备份集中到某个目录；`keep=N` 可只保留最近 N 份。
* 用环境变量关闭：`DLT_NEVER_OVERWRITE=0`。

与旧实现的区别（重要）：旧代码只在根分区剩余空间 > 16 GiB 时才备份，否则**静默覆盖**；
而 `os.replace` 同目录改名几乎不占额外空间，所以新实现**始终备份**。

## 3. 自动命名 `naming`

```python
from dlt import ExperimentNamer, auto_name

namer = ExperimentNamer("runs", name="my exp")
run = namer.new_run({"lr": 1e-3})        # 也可以 new_run(name=..., config=...)
print(run.run_dir)                        # runs/20261010T002915Z_my-exp_ab12cd34
print(run.logs, run.ckpts, run.figures, run.results)

p = auto_name("loss curve", ext=".png", directory="imgs")
# imgs/20261010T002915Z_loss-curve_9f3a.png
```

* 名字 = `<UTC时间戳>_<slug>_<随机id>`：**可排序**（时间在前）、**可读**、**不撞名**。
* `unique_path(path, create=True)` 用 `O_CREAT|O_EXCL` 原子占位，多进程也不会撞名。
* slug 保留中文，空白/标点折叠为 `-`。

## 4. 会“自我描述”的图片 `plot_utils`

```python
from dlt import packplot, retrieve_data, retrieve_code, retrieve_plot

@packplot
def plot_loss(history, filename):
    import matplotlib.pyplot as plt
    plt.plot(history)
    plt.savefig(filename)
    plt.close()

plot_loss(history, "loss.png")

retrieve_data("loss.png")     # -> {"history": [...], "filename": ...}
retrieve_code("loss.png")     # -> 画图函数源码
retrieve_plot("loss.png", allow_execute=True)   # 重新画一张
```

* PNG 用 `iTXt` 文本块（无损、不重编码）；JPEG/WEBP 用 EXIF `UserComment`；
  其它格式自动退化为同名 `.meta` sidecar，绝不丢数据。
* payload 带 magic + 版本 + SHA-256 校验，损坏会明确报错。
* **安全**：`retrieve_plot` 会执行图片里的代码，必须显式 `allow_execute=True`；
  只想看内容请用 `retrieve_code` / `retrieve_data` / `unpack`。

---

## 测试

```bash
python -m pytest -q
```

## 进一步阅读

* [`docs/FIXES.md`](docs/FIXES.md)：本次重构修复的完整问题清单。
* [`docs/DESIGN.md`](docs/DESIGN.md)：关键设计决策与取舍。
