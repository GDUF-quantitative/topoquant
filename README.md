# TopoQuant

> 当前 Bottleneck 距离由公开 Python 包 [Topp](https://pypi.org/project/topp/0.1.0/) `0.1.0` 提供；历史 GUDHI/Hera 基准见 [Bottleneck 后端记录](docs/BOTTLENECK_KERNEL.md)。

> 需要交付给没有 Python 环境的 Windows 电脑时，使用[便携版说明](docs/PORTABLE.md)。便携目录包含完整运行时，数据和 `config.json` 实验协议仍保持外置可修改。

这是对 `quant260803.ipynb` 和 `第三次实验流程图.drawio` 的工程化实现。核心实验口径保持为：60 个交易日构成一个点云，使用 `money / volume / high / close` 四列的 Z-score 标准化数据构建 Vietoris–Rips 复形，保存 H0/H1/H2 持续同调结果，以 H0/H1 瓶颈距离进行历史类比，再用 Top-N 相似点云的未来走势多数投票。

第一次运行可直接按照 [启动文档](docs/START.md) 操作；环境与离线部署细节见 [陌生机器安装与启动](docs/SETUP.md)，字段要求见 [行情数据契约](docs/DATA_CONTRACT.md)。仓库不附带商业行情数据；默认把原始 CSV 放在 `data/stock/`，实验产物写到 `runs/<截止日>/`。

## 流程承接关系

1. 从每支股票截至实验日的最近 480 个交易日中，按从新到旧切成最多 8 个、不重叠的 60 日窗口；至少有 4 个完整窗口的股票才进入实验。
2. 每个窗口只在内存中抽取四个特征并做 Z-score 标准化。notebook 中先对 C-L 列做 Min-Max、随后又对四个特征做 Z-score；对非恒定列而言前一步不会改变最终 Z-score，因此不再保存归一化 CSV。
3. 对每个窗口计算 H0/H1/H2。数值对写入 `artifacts.sqlite3`，不生成持续同调图，也不再生成 39318 个 TXT。
4. `cloud_date == as_of_date` 是目标点云，`cloud_date < as_of_date` 才是历史候选（不使用未来数据）。对 H0、H1 分别计算瓶颈距离，两个距离都严格小于阈值才合格。
5. 合格历史点云数量不少于 `top_k` 时，按 H1 距离、H0 距离、点云 ID 稳定排序并保留前 `top_k`。
6. 对目标及相似点云分别读取其窗口日期之后的 5 个交易日，以首个未来交易日的 `prev_close` 为基准计算累计涨跌；相似点云多数投票生成预测，并与目标真实方向比较。
7. 从实验库导出匹配表、逐日预测、机器可读指标和文本报告。

## 中间产物

所有可恢复的计算状态集中在 `<work_dir>/artifacts.sqlite3`：

- `clouds`：点云 ID、日期、股票代码、来源和处理状态；
- `diagrams`：H0/H1/H2 的 birth-death 数值对，以紧凑的 `float64` 二进制保存；
- `match_runs` / `matches`：候选数、阈值内数量、Top-N 及两个距离；
- `forecast_runs` / `forecasts`：未来日期、价差、实际方向、投票和预测正确性；
- `metadata`：配置、输入文件指纹和阶段签名，防止把不同实验混进同一目录。

匹配阶段还会在 `<work_dir>/mmap/` 下派生一份只读的持续同调镜像（`diagrams_h0.npy`、`diagrams_h1.npy`、`diagram_counts.npy`、`diagram_index.json`、`diagram_signature.txt`），由 SQLite 原子导出，供多进程以 `mmap` 零拷贝共享。这些文件可随时删除，下次运行会按签名自动重建，因此不纳入版本控制。

最终结果位于 `<work_dir>/outputs/`：

- `selected_matches.csv`
- `predictions.csv`
- `metrics.json`
- `report.txt`

原始 CSV 是唯一外部输入；窗口 CSV、归一化 CSV、持续同调 TXT 和持续同调图都不再保存。SQLite 是 Python 自带组件，避免为单一列式格式引入 PyArrow/HDF5 等较重依赖，同时保留随机读取、事务、断点续算和审计能力。

## 安装与运行

建议 Python 3.10–3.12。复制并修改 `config.example.json`，然后在项目目录执行：

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\topoquant --config config.example.json --verbose run
```

`run` 会先执行正式的预启动检查：现场检测逻辑内核、物理内存、工作盘空间、行情文件、实验库和依赖版本，计算本次实际并发数；存在阻断错误时不会进入计算。也可以只做预检：

```powershell
.venv\Scripts\topoquant --config config.example.json preflight
```

换机器或更换整批行情后，建议在正式计算前完整验证一次数据：

```powershell
.venv\Scripts\topoquant --config config.example.json validate-data
```

运行期间终端会显示三个阶段各自的实时进度条、完成/错误数量、实际并发数和耗时。查看已写入实验库的点云、匹配、预测数量及逐日准确率：

```powershell
.venv\Scripts\topoquant --config config.example.json status
```

需要脚本消费时可增加全局参数 `--json`，关闭动态界面并输出机器可读 JSON。

也可以分阶段执行，便于断点恢复：

```powershell
.venv\Scripts\topoquant --config config.example.json --verbose build
.venv\Scripts\topoquant --config config.example.json --verbose match
.venv\Scripts\topoquant --config config.example.json --verbose forecast
.venv\Scripts\topoquant --config config.example.json report
```

持续同调阶段会跳过实验库中已经完整保存的点云。若影响点云定义的配置或原始文件发生变化，程序默认会拒绝在旧实验库上继续。

## 换数据重算

`source_signature` 是原始 CSV 的内容哈希（sha256 over bytes），只要行情内容改变就会失配，覆盖了"改数据但不改文件大小/时间戳"的情况。两个开关用于处理失配：

```powershell
.venv\Scripts\topoquant --config config.json --verbose run --reset
.venv\Scripts\topoquant --config config.json --verbose run --force-rebuild
```

- `--reset`：无条件清空实验库中的 clouds/diagrams/matches/forecasts 与 mmap 镜像，从头重算；
- `--force-rebuild`：仅在检测到签名失配时授权就地清空并重算，签名一致则等价于普通增量运行。

两个开关对 `run` 和 `build` 子命令都可用；`run.py` 交互界面的第 7 组"重算控制"提供同样的两个选项，并会在确认摘要中高亮显示。不加任何开关时行为与旧版一致：签名失配直接报错退出，不会静默丢数据。

修改距离语义（例如本质类的处理方式）会改变 `matching_signature` 中的 `distance_algo` 字段，从而使旧匹配结果自动失效，不需要人工干预。

## 依赖取舍与性能

持续同调由 Ripser 生成；有限持续图的 exact Bottleneck 距离由 Topp `0.1.0` 计算，其余直接依赖为 NumPy、pandas 和 Rich。外层空图、本质类和严格阈值语义保持不变。历史 GUDHI/Giotto/Hera 对比只作为研究证据保留，不再是运行时后端。

三个计算阶段分别并发，配置值 `0` 表示根据机器逻辑内核数自动选择：

- `topology_workers`：按股票使用进程池计算持续同调，自动值为 `min(8, 逻辑内核数 - 1)`；
- `matching_workers`：按目标点云使用进程池计算瓶颈距离，自动值为 `min(16, 逻辑内核数 - 1)`；
- `forecast_workers`：按目标点云使用线程池读取行情并预测，自动值为 `min(32, 逻辑内核数 × 2)`。

单核机器的进程数会退化为 1。三个值都可以设置为正整数手动覆盖。旧配置中的 `workers` 仍会作为 `matching_workers` 读取，但建议改用新字段。

匹配阶段的历史持续同调数据以 `np.load(..., mmap_mode="r")` 只读映射进每个工作进程，由操作系统页缓存统一承载，不再按进程复制，因此提高 `matching_workers` 不会线性放大内存占用，自动上限从 4 提高到 16。

匹配阶段采用“廉价下界 → pivot 三角下界 → Topp exact Bottleneck”的结构：

1. H0/H1 的本质类个数、前 8 秩有限持续度谱和本质类 birth 下界通过 NumPy 整批筛选，不再逐候选执行 Python 判断；
2. 默认选择 8 个 farthest-point pivots，持久化历史候选到 pivot 的精确 H0/H1 距离矩阵；查询时用 `max_j |d(Q,P_j)-d(H_i,P_j)|` 严格排除不可能满足阈值的候选；
3. 只有剩余候选才调用 Topp 精确瓶颈距离，最终 H0/H1 `< threshold`、合格数量和 Top-K 排序语义不变。

剪枝摘要和 pivot 矩阵保存在 `work_dir`，所有 worker 以 mmap 共享，避免每个进程重复扫描全部持续图。第一次 `match` 需要支付 `候选数 × pivot 数` 的离线精确距离成本，后续在源持续图、候选集合和 pivot 数不变时直接复用。`matching_pivots` 默认 8，可设为 0 禁用，允许范围 0–32。`match_runs.candidate_count` 表示全部下界剪枝后实际计算过 H0 距离的候选数。

终端界面仅增加 Rich 这一项轻量纯 Python 依赖，不引入 Qt、Electron 或浏览器运行时；在新机器上的安装成本远低于桌面 GUI，同时可通过 SSH、PowerShell 和自动化脚本使用。

## 明确的数据契约

- 行情文件名推荐为 `000001.SZ.csv` 或 `000001SZ.csv`。
- 点云计算至少需要 `EventDate, money, volume, high, close`。
- 预测还需要 `prev_close`；每个目标和相似点云在其日期之后必须恰好取得至少 5 个有效、无重复的交易日。
- 若 H0 或 H1 任一持续图为空，其瓶颈距离按无穷大处理，与原 notebook 的筛选行为一致。
- 本质类（`death=+inf`）继续按既有语义精确处理：有限部分交给 Topp，本质类个数不同判为 `+inf`，个数相同时按 birth 降序逐位配对。
- 距离语义已写入 `matching_signature`，Giotto 版本产生的匹配缓存不会被静默复用。
- Pivot 只用于严格下界筛选，最终距离仍由 Topp 精确计算；单元测试会对拍启用/禁用 pivot 的最终匹配结果。
- 多数投票的门槛是 `floor(top_k / 2) + 1`；建议使用奇数 `top_k`。

## 测试

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

单元测试覆盖窗口顺序、标准化、未来涨跌基准、预启动检测、按内核自动选取并发数、进度事件、Ripser 的 H0/H1/H2 输出、Topp 有限距离与本质类精确语义、pivot 缓存及启用/禁用 pivot 的结果对拍，以及使用多进程/多线程的微型端到端流水线。GUDHI 仅作为可选差分 oracle。完整实盘验证仍需原始 `stock` 行情目录。
