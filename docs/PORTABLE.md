# TopoQuant Windows 便携版

这个目录已经包含 Python、NumPy、pandas、Ripser、Topp 和程序本身。目标电脑不需要安装 Python、Visual Studio、Jupyter、CUDA 或数据库。

## 第一次使用

1. 把整个 `TopoQuant` 文件夹复制到目标电脑本地磁盘，不要只复制 `TopoQuant.exe`。
2. 双击 `check_runtime.cmd`。看到 `Portable runtime is ready.` 后关闭窗口。
3. 把行情 CSV 放进 `data\stock\`。交付包若已带数据，可直接使用或替换。
4. 双击 `start_topoquant.cmd`，按提示确认数据、实验协议和结果目录。
5. 结果在所选 `runs\...\outputs\` 中。

不要从压缩包内部直接运行，也不要移动或删除 `runtime` 目录。

便携版面向 64 位 Windows 10/11。程序目前没有代码签名，首次复制到另一台电脑时 Windows SmartScreen 可能提示“未知发布者”；请先确认文件来自本项目，再选择“更多信息 → 仍要运行”。若目标电脑是 32 位 Windows、Windows 7/8，或被单位安全策略禁止运行未签名 EXE，需要针对那台机器另做兼容构建或代码签名，不能用安装更多 Python 环境来绕过。

## 更换数据

`data\stock\` 是外置数据目录，每支股票一个 CSV。文件至少包含：

```text
EventDate,money,volume,high,close,prev_close
```

程序会对数据内容计算指纹。数据变化后，旧实验目录不会被静默复用。最稳妥的做法是给新数据或新实验协议指定新的结果目录；只有明确不再保留旧结果时，才选择清空重算。

## 修改实验协议

双击启动时可以修改并保存以下参数：

- 数据目录、基准日期和结果目录；
- 窗口长度、回看交易日数和最少窗口数；
- 特征列、最大边长和最大同调维度；
- H0/H1 距离阈值和 Top-K；
- 预测天数与三个阶段的并发数。

设置保存在外置 `config.json` 中，下次启动会自动作为默认值。也可以用文本编辑器直接修改它。同一个基准日期如果更改实验协议，请使用新的 `work_dir`，这样不同实验可以并排比较。

当前固定在程序里的算法定义包括：总体标准差 Z-score、Ripser 持续同调、Topp exact Bottleneck 距离、H0/H1 双阈值筛选和多数投票。`distance_dimensions` 因此必须保持 `[0, 1]`。若要修改这些公式、距离维度或算法，需要在开发电脑修改源码、运行测试并重新生成便携包；目标电脑无需承担编译。

## 老电脑建议

并发参数填 `0` 会按 CPU 自动选择。如果机器内存较小或运行不稳定，可先使用：

```text
topology_workers = 1
matching_workers = 1
forecast_workers = 2
```

这会变慢，但更稳。程序支持断点续算，意外关闭后使用相同数据、协议和结果目录再次启动即可。

## 构建便携包（开发电脑）

在项目虚拟环境中安装一次构建工具：

```powershell
.venv\Scripts\python -m pip install -e ".[portable]"
```

生成包含 `data\stock` 的完整交付目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_portable.ps1
```

只生成程序、暂不复制行情数据：

```powershell
.\scripts\build_portable.ps1 -WithoutData
```

输出目录是 `dist\TopoQuant\`。构建脚本会运行真实的 Ripser 和瓶颈距离自检；只有自检通过的目录才应交付。
