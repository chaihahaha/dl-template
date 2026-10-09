# AGENTS.md

本文件是本目录（项目根）下工作的 Agent 必须遵守的约定。请先完整阅读再动手。
**除非用户明确要求，否则不要修改本文件**（原因见第 4 节）。

---

## 1. 文件与目录存放约定

### 1.1 禁止使用 `/tmp`

`/tmp`（以及 `/var/tmp`、`/run`、系统临时目录）在机器/容器**重启后会被清空**，代码和产物会丢失。
**所有文件一律保存到当前项目目录的子目录中**，不要写到项目目录之外。

### 1.2 推荐的子目录

不存在时按需创建（`mkdir -p`）。按用途命名，不要把所有东西堆在根目录：

| 目录 | 用途 |
| --- | --- |
| `./tmp` | 临时中间文件（本质是项目内的“临时区”，重启不丢） |
| `./scripts` | 可执行脚本（shell / python / 工具脚本） |
| `./src` | 主要源代码 |
| `./tests` | 单元测试 / 集成测试 |
| `./assets` | 静态资源（字体、模板、小素材） |
| `./binaries` | 编译产物、第三方可执行文件、二进制发布包 |
| `./build` / `./dist` | 构建中间产物 / 最终分发产物 |
| `./data` / `./datasets` | 数据集、输入数据 |
| `./configs` | 配置文件（yaml/json/toml/ini） |
| `./docs` | 文档、说明、设计稿 |
| `./logs` | 运行日志 |
| `./imgs` | 图片 |
| `./videos` | 视频 |
| `./musics` | 音频 |
| `./models` / `./ckpts` / `./weights` | 模型文件、检查点、权重 |
| `./outputs` / `./results` | 程序输出、实验结果、评测报告 |
| `./reports` | 汇总报告（html/md/pdf） |
| `./notebooks` | Jupyter notebook |
| `./cache` | 可重建的缓存 |
| `./envs` | 虚拟环境（venv/conda） |
| `./vendor` / `./third_party` | 第三方依赖或 vendored 代码 |
| `./backups` / `./archive` | 备份、归档、历史版本 |
| `./state` | 运行时状态、数据库文件、持久化数据 |

命名原则：**目录名用复数英文小写**；若项目已有既定目录结构，优先沿用，不要另起炉灶。

### 1.3 硬性要求

- 生成的大文件（日志、图片、视频、模型、数据集）必须落到上面对应的子目录，不要放在项目根目录。
- 临时文件用 `./tmp` 而非 `/tmp`，命令示例：`TMPDIR=./tmp mktemp -p ./tmp`。
- 不要再把 `./tmp` 当成一次性目录而随意删除整个项目根；只清理 `./tmp` 内部内容。

---

## 2. Git 使用约定

### 2.1 初始化

- 开始改动前先检查是否存在 `.git`：
  - 不存在 → 运行 `git init`。
  - 存在 → 直接使用。
- 建议先设置基础信息（若未配置）：`git config user.name` / `git config user.email`。

### 2.2 `.gitignore`

- 若根目录没有 `.gitignore`，**必须创建**。
- 至少忽略大文件 / 生成物目录：

```gitignore
# 临时与生成物
/tmp/
/assets/
/logs/
/imgs/
/videos/
/musics/
/build/
/dist/
/cache/
/outputs/
/results/
/reports/
/envs/

# 大文件 / 模型
/models/
/weights/
/ckpts/
*.ckpt
*.pth
*.pt
*.safetensors
*.bin
*.onnx
*.h5
*.pb
*.gguf

# 归档与数据
/backups/
/archive/
/data/
/datasets/
*.tar
*.tar.gz
*.tgz
*.zip
*.7z

# 系统与编辑器
.DS_Store
Thumbs.db
*.swp
*~
.idea/
.vscode/
__pycache__/
.pytest_cache/
*.pyc
*.log
```

- **后续开发中一旦出现新的大文件类型（如 `*.ckpt`、`*.safetensors`、`*.parquet`、`*.bag`、`*.mp4` 等），必须同步把对应后缀 / 目录补进 `.gitignore`**，避免误提交大文件。
- 对确实需要保留的空目录，放一个 `.gitkeep` 占位。

### 2.3 提交

- 在**重要修改完成后**及时提交：

```bash
git add .
git commit -m "descriptions"
```

- commit message 要具体描述改动（做了什么、为什么），避免 `update`、`fix` 之类无信息量的信息。
- 不要提交密钥、token、密码、证书等敏感信息；必要时加入 `.gitignore` 并改用环境变量。
- 除非用户明确要求，不要执行 `git push`、`git reset --hard`、`git push --force`、改写历史等破坏性操作。

---

## 3. 禁止 / 慎用的危险命令（死循环、死锁、卡死）

Agent 运行命令时**没有交互式终端**，以下情况会导致命令永久挂起、把自身杀死或产生死循环。
**默认禁止直接运行**；确需执行时必须加 `timeout`、重定向输入、或放入 tmux 后台运行。若判断可能卡死，先停下来向用户确认。

**通用防护：**
```bash
timeout 30 <cmd>                       # 加超时
<cmd> < /dev/null                      # 断开 stdin，避免等待输入
tmux new -d -s <name> '<cmd>'          # 后台/常驻任务统一用 tmux（见 3.3）
export PAGER=cat GIT_PAGER=cat          # 关闭分页器
```

### 3.1 自匹配导致自杀 / 卡死

- ❌ `pkill -f "string"` / `pkill "string"` / `killall name`
  - `-f` 会匹配整条命令行；执行该命令的 shell 自身的命令行里往往就包含这个 `string`，于是 **pkill 把发起它的 shell（或自己的父进程）也杀掉**，命令表现为“卡死”或会话中断。
  - ❌ `kill 0`（杀掉整个进程组，含自己）、`kill -9 -1`（杀掉该用户所有进程）、`pkill -9 -f .`
  - ✅ 正确做法：先 `ps`/`pgrep` 看清 PID，再 `kill <具体PID>`；或用精确匹配 `pkill -x <name>`，并避免在命令串里直接写目标字符串（可借助 `[s]tring` 写法）。
- ❌ `ps aux | grep foo` —— 结果里永远包含 `grep foo` 自己，容易误判/误杀；改 `pgrep -fl foo`。

### 3.2 等待 stdin / 交互输入（永不返回）

- ❌ 裸 `cat`、`grep`（无文件参数）、脚本里的 `read`：无输入时永久阻塞。
- ❌ 交互式程序：`ssh`（密码提示）、`sudo`/`su`/`passwd`、`ftp`/`telnet`、`mysql`/`psql`/`redis-cli`、`python`/`node`/`irb`（无脚本进 REPL）、`vim`/`nano`/`less`/`more`/`man`。
- ❌ 会打开编辑器的命令：不带 `-m` 的 `git commit`、`git rebase -i`、`git merge`（冲突时）、`git config -e`。
- ❌ 会等待确认的包管理/工具：`npm init`（无 `-y`）、`apt-get`（无 `-y`）、`pip` 交互、`ssh-keygen`、`openssl` 交互、`docker login`。
- ❌ 带确认的删除/覆盖：`rm -i`、`mv -i`、`cp -i`、`git clean -i`。
- ✅ 正确做法：所有命令显式加非交互参数（`-y`、`--yes`、`--non-interactive`、`-m "msg"`、`-c`/`-e` 传代码），并加 `< /dev/null` 与 `timeout`。

### 3.3 后台 / 常驻 / 交互任务：统一用 tmux（禁止 nohup 与 `&`）

**禁止**用 `nohup <cmd> &`、`<cmd> &`、`disown` 等方式后台启动任务：
这种方式会**污染 stdout/stderr**、日志与提示混在一起，**后期查看进度、判断是否结束、定位报错都非常困难**，进程还可能随会话退出而丢失，无法可靠管理。
**所有后台任务的启动与结束一律通过 tmux**，交互式程序也放进 tmux 操作。

```bash
tmux ls                                    # 查看已有 session
tmux new -d -s <session> '<cmd>'           # 新开后台 session 运行任务
tmux new -s <session>                      # 新开并进入可交互 session
tmux send-keys -t <session> '<cmd>' Enter  # 向 session 发送命令并回车
tmux capture-pane -t <session> -p          # 抓取当前面板输出，查看进度
tmux capture-pane -t <session> -p -S -300  # 回看最近 300 行历史输出
tmux kill-session -t <session>             # 结束/杀死任务（进入后 Ctrl-C 亦可）
```

- ❌ 永不退出的 follow 命令：`tail -f` / `tail -F`、`journalctl -f`、`docker logs -f`、`docker attach`、`kubectl logs -f` / `kubectl port-forward`。改用 `tail -n 100`、`docker logs --tail 100`，或在 tmux 中运行并用 `capture-pane` 查看。
- ❌ 不加 `-d` 的 `docker run` / `docker compose up`；`docker exec -it`。
- ❌ `watch <cmd>`、`top`/`htop`、`vmstat 1`/`iostat 1`/`mpstat 1`（不带次数）。
- ❌ 常驻服务：`python -m http.server`、`flask run`、`npm run dev`、`next dev`、`nc -l`、`socat`、gunicorn/uvicorn 前台运行。
- ✅ 正确做法：常驻/长任务 `tmux new -d -s <session> '<cmd>'` 启动，进度用 `tmux capture-pane -t <session> -p` 查看，结束用 `tmux kill-session -t <session>`；短期命令用 `timeout` 限定。

**交互式程序（如 Miniconda 安装包、需要交互确认的安装/配置命令）**：也在 tmux 中运行，用 `tmux send-keys` 发送 `yes` / 回车 / 选项完成交互，再用 `tmux capture-pane` 确认结果，避免命令卡在等待输入。

**子 agent**：可新开一个 tmux session 在其中启动 `opencode`；用 `tmux send-keys` 发送子代理名字并回车切换，再用 `tmux send-keys` 发送 prompt 并说明该子代理应如何执行；用 `tmux capture-pane` 查看其输出。

### 3.4 忙等待 / 无界循环

- ❌ `while true; do ...; done`（无 `break`/`exit`/`sleep`）；`until`、`for ((;;))` 同理。
- ❌ `yes`；`yes | <cmd>`（cmd 不消费输入时瞬间刷屏/占满）。
- ❌ 脚本/函数递归调用自身而不设终止条件；Makefile 目标互相调用形成环。
- ❌ **Fork bomb**：`:(){ :|:& };:`。
- ❌ 对永不结束的流做管道：`while read ...` 读一个永不 EOF 的管道。
- ✅ 正确做法：循环必须有明确终止条件与 `sleep`；用 `timeout` 包裹所有可能长时间运行的命令。

### 3.5 等待锁 / 资源导致死锁

- ❌ 对一个已被其他进程持有的文件执行 `flock`；等待已被占用的锁文件 / 互斥量。
- ❌ `cat <named_fifo>` 读一个没有写端的命名管道（永久阻塞）。
- ❌ `dd if=/dev/zero of=... ` / `dd if=/dev/urandom of=...`：无限写入，可能瞬间写满磁盘。
- ❌ 无超时的网络命令连不上/无响应时会卡死：`curl`、`wget`、`git clone`、`pip install`、`scp`、`rsync`、DNS 解析、NFS 挂载。
- ✅ 正确做法：网络命令加 `--connect-timeout` / `--max-time` 或外包 `timeout`；后台/远程命令加超时。

### 3.6 递归 / 扫描爆炸

- ❌ `find /`、`find . ` 递归进 `/proc`、`/sys`、网络挂载盘；`find -L` 跟随符号链接遇到环 → 无限递归。
- ❌ `grep -r` / `du -sh` / `ls -R` 作用于 `/` 或超大目录树；巨大 `**/*` glob。
- ✅ 正确做法：限定目录与深度（`-maxdepth`、`-P` 不跟随链接），并加 `timeout`。

### 3.7 输出阻塞 / 分页器

- ❌ 在 tty 下 `git log`、`git diff`、`git show` 会自动调用分页器 `less` 并**等你按 `q`**，在非交互环境表现为卡死。
- ✅ 正确做法：`export PAGER=cat GIT_PAGER=cat`，或用 `git --no-pager ...`；输出重定向到文件而不是淹没终端。

### 3.8 破坏性 / 不可逆命令

- ❌ 未加引号且变量可能为空的 `rm -rf $VAR`（会变成 `rm -rf /`）；`rm -rf /*`；`rm -rf ~`。
- ❌ `cmd > file` 同时又从同一 `file` 读取；`sort file > file`（先截断再读，数据损坏）。
- ❌ `chmod -R 777 /`、`chown -R` 到系统目录、`dd of=/dev/...`。
- ✅ 正确做法：删除前 `echo`/`ls` 预览目标；变量加引号并加空值保护 `${VAR:?}`；确认路径无误。

> 兜底原则：**任何可能耗时或阻塞的命令，默认加 `timeout` + `< /dev/null`；后台 / 常驻 / 交互任务一律放进 tmux，用 `tmux capture-pane` 查看进度。** 拿不准就先问用户，不要盲目执行。

---

## 4. 文档与记忆维护

**不要轻易修改本文件 `AGENTS.md`。** opencode 会在 `AGENTS.md` 被修改后**自动重新导入其内容**，导致已缓存的 KV cache 失效（cache miss），从而显著变慢、浪费 token。

因此：

- `AGENTS.md`：只放**稳定不变**的通用约定（目录、git、危险命令、记忆规则）。**非必要不修改**。
- `MEMO.md`：保存**记忆类**信息——待办事项、用户要求、todo list、背景信息、当前进度、关键决策。
- `FYI.md`：保存**外部知识类**信息——参考链接、文档 URL、API 说明、资料出处、名词解释。
- `PITFALL.md`：保存**踩坑记录**——之前遇到的陷阱、错误、报错信息、失败的尝试与原因、以及修复方法。

新增或更新信息时，请写进上述对应的 `.md`，而不是往 `AGENTS.md` 里塞。
若确需修改 `AGENTS.md`，先与用户确认，尽量合并成一次改动，避免频繁触发重导入。
