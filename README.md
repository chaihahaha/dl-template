# 深度学习训练项目模板（dlt）

给深度学习研究者的一个极简起点：`dlt` 提供日志、防覆盖、自动命名、可自描述图片四个小
工具，外加几个可运行的训练示例。**网络结构、数据集、训练循环都由你自己写**，模板不绑定
任何框架。

| 工具 | 一句话 | 依赖 |
| --- | --- | --- |
| `dlt.yaml_logging.YamlLogger` | 崩溃安全的 YAML 实验日志（可 `grep`、`Ctrl+C` 不丢数据） | PyYAML |
| `dlt.never_overwrite` | 覆盖前先备份，保护 `open` 和 `torch.save` | 无 |
| `dlt.naming` | 可排序、不撞名的实验/文件命名 | 无 |
| `dlt.plot_utils.packplot` | 把画图代码和数据嵌进图片，随时重画 | Pillow |
| `dlt.integrations.lightning.DLTLogger` | 给 PyTorch Lightning `Trainer` 用上面这套日志 | lightning（可选） |

## 安装

```bash
pip install pyyaml            # 核心
pip install pillow            # 需要画图时
pip install ".[dev]"          # 跑测试（pytest + matplotlib）
pip install ".[lightning]"    # 用 Lightning 示例
```

也可以直接 `pip install -e .`。

## 目录结构

```
dlt/                 # 工具库
examples/            # 可运行示例（从改这里开始）
  train_demo.py      #   纯 PyTorch 训练 + 四个工具的完整用法
  lightning_demo.py  #   PyTorch Lightning Trainer 用法
  rl_ray_ppo.py      #   Ray 多卡异步 PPO（可选，重依赖）
tests/               # pytest
docs/                # 详细设计/修复/性能说明
```

## 先跑起来

```bash
python examples/train_demo.py --epochs 30
python examples/lightning_demo.py --epochs 3
```

会生成一个实验目录：

```
runs/20261010T002915Z_demo-mlp_ab12cd34/
├── config.yaml          # 超参数快照
├── logs/*.yaml          # 训练日志
├── ckpts/final.pt       # 权重
└── figures/loss.png     # 图片内含画图代码和数据
```

## 改成你自己的实验

复制 `examples/train_demo.py` 当作起点即可。下面三处是最常改的。

### 1. 修改神经网络结构

替换示例里的 `MLP` 就行，其余训练代码不用动：

```python
class MyNet(nn.Module):                       # dlt 对模型没有任何约束
    def __init__(self, d_in, d_out, hidden=256, n_layers=4):
        super().__init__()
        layers, prev = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.GELU()]
            prev = hidden
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MyNet(args.d_in, 1).to(device)        # 换成你的结构
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
```

用 Lightning 时改 `LightningModule` 的 `__init__`/`forward` 即可，见
`examples/lightning_demo.py`。

### 2. 换数据集

示例里是 `make_batch()` 造的合成数据。换成真实数据用 `Dataset` + `DataLoader`：

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, root, split="train"):
        self.files = sorted(Path(root).glob(f"{split}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        return torch.as_tensor(data["x"]), torch.as_tensor(data["y"])

train_loader = DataLoader(MyDataset("data/mydata"), batch_size=64,
                          shuffle=True, num_workers=8, pin_memory=True)

for x, y in train_loader:        # 把原来 make_batch() 的地方换成这个循环
    ...
```

### 3. 添加自己的代码 / 新工具

`dlt` 只是普通包，不限制你的工程结构。推荐把训练相关代码放在项目里，按需 import：

```
mydata/                  # 数据
configs/                 # yaml/json 配置
src/models.py  src/data.py  src/losses.py     # 你的代码
scripts/train.py                              # 入口
outputs/  runs/          # 运行产物
```

* 直接 `from src.models import MyNet` 即可；
* 想给 `dlt` 加自己的通用工具，放进 `dlt/` 并在 `dlt/__init__.py` 里导出；
* 配置建议写成 dict 传给 `YamlLogger(config=...)`，会一起存进日志和 `config.yaml`。

## 把四个工具接进你的训练循环

```python
import never_overwrite                       # 第一行：覆盖前自动备份
from dlt import ExperimentNamer, YamlLogger
from dlt.plot_utils import packplot

config = {"lr": 1e-3, "batch_size": 64, "model": "MyNet"}
run = ExperimentNamer("runs", name="mynet").new_run(config)

with YamlLogger("mynet", run=run, config=config, save_every=100, async_save=True) as log:
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader)
        log.log(epoch=epoch, loss=loss)      # 落盘交给后台线程，不阻塞训练
    torch.save(model.state_dict(), run.ckpts / "final.pt")   # 已受防覆盖保护
    log.add_artifact(run.ckpts / "final.pt")
```

画一张会“自我描述”的图（图片里存了画图函数、参数，删除脚本后仍能重画）：

```python
@packplot
def plot_loss(history, filename):
    import matplotlib.pyplot as plt
    plt.plot(history); plt.savefig(filename); plt.close()

plot_loss(history, run.figures / "loss.png")
```

Lightning 用户：

```python
from lightning.pytorch import Trainer
from dlt.integrations.lightning import DLTLogger

trainer = Trainer(
    logger=DLTLogger("mynet", save_dir="runs"),
    accelerator="gpu", devices=4, strategy="ddp", precision="16-mixed",
)
trainer.fit(model, train_dataloaders=train_loader)
```

## 常用 API 速查

| 需求 | 写法 |
| --- | --- |
| 建实验目录 | `ExperimentNamer("runs", name="exp").new_run(config)` |
| 记录指标 | `log.log(step=i, loss=0.1)` / `log.add_info("...")` |
| 强制落盘 | `log.save()` / `log.flush()` / `log.close()` |
| 读日志 | `from dlt import load_log, latest_log` |
| 备份策略 | `BackupPolicy(keep=5, max_total_bytes=..., min_free_bytes=...)` |
| 关掉防覆盖 | 环境变量 `DLT_NEVER_OVERWRITE=0` |
| 自定义命名 | `from dlt import auto_name; auto_name("loss curve", ext=".png", directory="imgs")` |
| 重画图片 | `from dlt import retrieve_plot; retrieve_plot("loss.png", allow_execute=True)` |
| 只看数据/代码 | `retrieve_data("loss.png")` / `retrieve_code("loss.png")` |

参数含义与边界（大小上限、脱敏、异常情况）见
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) 和 [`docs/FIXES.md`](docs/FIXES.md)。

## 测试

```bash
python -m pytest -q
```
