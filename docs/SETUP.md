# 陌生机器安装与启动

## 1. 支持边界

推荐环境是 Windows 10/11 x64、PowerShell 5.1 或 7、64 位 Python 3.11。项目支持 Python 3.10–3.14；仍需按目标机器核对 Ripser 与科学计算 wheel 的可用性。

硬件建议：

| 资源 | 最低建议 | 5027 支股票推荐 | 影响 |
|---|---:|---:|---|
| 逻辑内核 | 4 | 8–16 | 持续同调和瓶颈距离速度 |
| 内存 | 8 GiB | 16–32 GiB | 匹配进程各自持有历史持续同调数据 |
| 工作盘空间 | 原始 CSV 大小 + 2 GiB | 10 GiB 以上余量 | SQLite、日志和导出结果 |
| 网络 | 首次安装需要 | 可用离线 wheelhouse | 下载 Python 包 |

不需要安装 PostgreSQL、MySQL、Node.js、Jupyter、Java、CUDA 或 Visual Studio。SQLite 随 Python 自带。正常情况下 NumPy、pandas、Ripser 和 Topp 都通过 wheel 安装，不应在新机器上手工编译 C/C++。

## 2. Python 库

运行时有五个直接依赖：

| 库 | 版本范围 | 用途 | 是否可移除 |
|---|---|---|---|
| NumPy | `>=1.24,<3` | 数组、标准化和持续同调数值对 | 否 |
| pandas | `>=2,<4` | CSV、日期和行情表处理 | 否 |
| Ripser | `>=0.6,<0.7` | Vietoris–Rips 持续同调 | 否 |
| Topp | `==0.1.0` | exact Bottleneck 距离 | 否 |
| Rich | `>=13,<14` | 预检表格、进度条和结果终端界面 | 可替换，但当前 CLI 需要 |

开发/测试额外使用 `pytest>=8,<10`。

## 3. 在线安装（推荐）

先从 [Python 官方网站](https://www.python.org/downloads/windows/)安装 64 位 Python 3.11。建议勾选 Python Launcher。然后在项目根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

脚本会按 3.12 → 3.11 → 3.10 的顺序寻找解释器，创建项目独占的 `.venv`，安装依赖和项目，并执行 CLI 自检。指定解释器或同时安装测试依赖：

```powershell
.\scripts\bootstrap.ps1 -PythonExe "C:\Python311\python.exe" -Dev
```

只检查 Python 是否合格而不安装任何内容，可使用 `-CheckOnly`。

脚本不会把依赖装进系统 Python，也不会删除已有数据或实验目录。

## 4. 数据与配置放置

推荐目录结构：

```text
项目根目录/
├─ config.example.json
├─ data/
│  └─ stock/
│     ├─ 000001.SZ.csv
│     ├─ 000002.SZ.csv
│     └─ 600000.SH.csv
└─ runs/
   └─ 20240628/          # 程序自动创建
```

复制 `config.example.json` 为自己的配置，例如 `config.20240628.json`。相对路径以配置文件所在目录为基准；数据也可以放在其他盘并使用绝对路径。详细字段和数据质量要求见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

原始 5027 支股票行情并未随仓库提供，需要用户从有授权的数据源导出。不要把切片 CSV、归一化 CSV、持续同调图或 notebook 临时结果放进 `data/stock`。

## 5. 启动验收顺序

```powershell
# 快速检查环境、所有文件名和所有 CSV 表头
.venv\Scripts\python -m topoquant --config config.20240628.json preflight

# 首次换数据时完整扫描日期、重复行、数值和窗口数量
.venv\Scripts\python -m topoquant --config config.20240628.json validate-data

# 正式运行
.venv\Scripts\python -m topoquant --config config.20240628.json run

# 随时查看实验库和准确率
.venv\Scripts\python -m topoquant --config config.20240628.json status
```

`preflight` 每次现场检测 CPU、内存、磁盘、依赖、数据位置和表头，然后计算实际并发数；检查失败不会进入计算。`validate-data` 会完整读取全部行情，结果写入 `runs/<实验>/outputs/data_validation.csv` 和 `data_validation.json`。

匹配默认使用 `matching_pivots=8` 建立可复用的精确距离下界缓存。首次匹配会额外计算历史候选到 pivots 的距离；同一实验目录后续直接复用。机器较慢或只做一次查询时可设为 `0` 禁用，候选库较大且会重复运行时可尝试 `8` 或 `16`。

## 6. 离线安装

在一台具有相同 Windows 架构和 Python 小版本的联网机器上准备 wheelhouse：

```powershell
python -m pip download -d wheelhouse `
  "numpy>=1.24,<3" "pandas>=2,<4" "ripser>=0.6,<0.7" "topp==0.1.0" `
  "rich>=13,<14" "setuptools>=68" wheel
```

复制整个项目和 `wheelhouse` 到离线机器，然后运行：

```powershell
.\scripts\bootstrap.ps1 -Wheelhouse .\wheelhouse
```

不同 Python 小版本或不同 CPU/操作系统的二进制 wheel 不能混用。

## 7. 常见失败

- `No matching distribution found for ripser`：通常是 Python 版本、位数或平台没有对应 wheel；优先改用官方 64 位 Python 3.11 或 3.12。
- `No matching distribution found for topp`：Topp 首发只提供 Windows x64 的 CPython 3.10–3.14 wheel；确认 Python 版本、系统位数和平台符合要求。
- `python/py 不是命令`：重新安装 Python Launcher，或通过 `-PythonExe` 指定完整路径。
- 预检报告 CSV 为 0：检查 `source_dir` 是配置文件相对路径还是绝对路径。
- 文件契约错误：文件名必须带 SZ/SH，列名大小写必须与契约一致。
- 内存压力大：优先调小 `topology_workers`；匹配阶段已改为 `mmap` 只读共享持续同调数据，进程数不会成倍放大内存。
- 换了行情数据后被签名检查拦住：加 `--reset`（无条件重算）或 `--force-rebuild`（失配时授权清空），也可在 `run.py` 的"重算控制"里选择。
- 预测失败：确认原始 CSV 包含截止日之后至少 `forecast_horizon` 个交易日，而不只是截至截止日的数据。

安装完成的最低验收是：CLI 帮助可显示、`preflight` 无红色错误、`validate-data` 的 `error=0`。这三项通过后再启动完整计算。

## 8. 可选的 MLflow 实验记录

MLflow 不属于核心运行依赖，只在需要上传实验参数和指标时安装。项目使用 uv 环境时执行：

```powershell
uv pip install --python .\.venv\Scripts\python.exe "mlflow==3.16.0"
.\.venv\Scripts\python.exe .\uplooad_mlflow.py
```

上传脚本写入实验参数、准确率和平均策略对数收益率，同时上传 `config.json` 与 `outputs/` 结果；源码不再上传。旧实验结果需先重新执行 `forecast` 和 `report` 生成带 `log_return` 的 `metrics.json`。本地脚本已被 Git 忽略，其中的连接设置不得提交或作为 artifact 上传。
