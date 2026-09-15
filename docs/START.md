# TopoQuant 启动说明

## 1. 安装

需要 Windows 10/11 和 64 位 Python 3.10–3.12，推荐 Python 3.11。

在项目根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

脚本会创建 `.venv` 并安装 NumPy、pandas、Ripser、Topp、Rich 和本项目。Ripser 负责持续同调，Topp `0.1.0` 负责 exact Bottleneck 距离。

## 2. 放置数据

把每支股票一个 CSV 放到：

```text
data\stock\
```

文件名示例：`000001.SZ.csv`、`600000.SH.csv`。

每个 CSV 至少包含：

```text
EventDate,money,volume,high,close,prev_close
```

数据需要包含截止日前至少 240 个交易日，以及截止日后至少 5 个交易日。详细要求见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

## 3. 修改配置

复制配置文件：

```powershell
Copy-Item .\config.example.json .\config.20240628.json
```

确认以下字段正确：

```json
{
  "source_dir": "./data/stock",
  "work_dir": "./runs/20240628",
  "as_of_date": "2024-06-28"
}
```

并发参数保持 `0` 即可，程序会在启动前根据运行机器自动选择。

## 4. 检查并运行

```powershell
# 检查环境、依赖和 CSV 表头
.venv\Scripts\python -m topoquant --config .\config.20240628.json preflight

# 首次使用一批新数据时执行完整检查
.venv\Scripts\python -m topoquant --config .\config.20240628.json validate-data

# 执行全部流程
.venv\Scripts\python -m topoquant --config .\config.20240628.json run

# 查看进度和结果
.venv\Scripts\python -m topoquant --config .\config.20240628.json status
```

## 5. 结果位置

结果保存在 `runs\20240628\`：

- `artifacts.sqlite3`：持续同调、匹配和预测实验库；
- `outputs\selected_matches.csv`：相似点云；
- `outputs\predictions.csv`：预测明细；
- `outputs\metrics.json`：准确率；
- `outputs\report.txt`：文本报告。

安装或离线部署问题见 [SETUP.md](SETUP.md)。
