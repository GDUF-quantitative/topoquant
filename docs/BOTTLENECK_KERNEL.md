# Bottleneck 后端记录

## 当前运行后端

TopoQuant 现在通过公开 Python 包 Topp `0.1.0` 调用 exact Bottleneck 距离。外层的空图筛选、本质类 birth 配对、pivot 下界和严格 `<0.1` 判定保持不变。更换后端会更新匹配签名，因此旧匹配缓存不会被静默复用。

安装或修复当前后端：

```powershell
.venv\Scripts\python -m pip install "topp==0.1.0"
```

下文是此前 GUDHI small-N 和 Hera 实验的历史证据，不再描述当前运行时依赖。

## 历史 GUDHI small-N 实验

当 `Persistence_graph::size() <= 256` 时，候选点保存在连续数组中，以闭合的 L-infinity 方框做线性扫描，命中后通过 swap-pop 删除；更大的图仍使用原 CGAL kd-tree。阈值可在编译时通过 `GUDHI_BOTTLENECK_LINEAR_SCAN_MAX_SIZE` 覆盖。

### 历史本机产物

- 安装版本：`gudhi 3.13.0+topoquant.smalln2`
- Python ABI：CPython 3.12 / Windows x64
- wheel：`build/wheelhouse/gudhi-3.13.0+topoquant.smalln2-cp312-cp312-win_amd64.whl`
- wheel SHA-256：`6642AC9B9F6A776B78235ECB69D9B1786AA3D7A573B0D585CD9338B21D22419B`
- 上游补丁：`patches/gudhi-3.13-small-n-neighbors.patch`

该 wheel 使用 GUDHI 3.13.0 标签、Nanobind 2.13.0、NumPy 2.5.0、CGAL 6.2，并关闭 TBB，以尽量对齐官方 wheel 的构建条件。它只适用于 CPython 3.12 Windows x64，不能用于 Python 3.10、3.11 或其他平台。

### 验证结果

真实已有持久图按流水线口径剥离本质类后固定抽样 128 对，使用 GUDHI 默认 `e=None` 路径：

| 维度 | 图总点数 | 官方 3.13.0 | small-N | 加速 |
|---|---:|---:|---:|---:|
| H0 | 中位数 118，范围 58–118 | 895.976 μs | 558.716 μs | 1.60x |
| H1 | 中位数 18，范围 6–26 | 153.778 μs | 102.412 μs | 1.50x |

两组各 128 个距离的二进制 SHA-256 在官方版与 small-N 版之间完全一致。另有 250 组随机图的自动后端/强制 kd-tree 差分对拍，差异数为 0；GUDHI Bottleneck_distance 的 8 项测试及 TopoQuant 的 12 项测试均通过。

这些数字是内核及 Python 入口的隔离基准，不代表完整 `match` 阶段会获得同等倍数加速；完整阶段仍包含进程调度、mmap、pivot 和 SQLite 开销。本次没有运行真实行情流水线。

### Hera 对比

2026-08-10 使用 GUDHI 3.13 wheel 自带的 Hera C++ 扩展，对同一批真实有限持久图做了对比。该扩展与 Giotto 使用的 Hera 算法内核属于同一路线，因此无需为测试额外安装整套 Giotto-TDA。

| 维度 | 当前 small-N GUDHI | Hera `delta=0` | Hera `delta=0.01` |
|---|---:|---:|---:|
| H0 | 558.716 μs | 2748.643 μs | 1968.354 μs |
| H1 | 102.412 μs | 406.170 μs | 357.677 μs |

Hera 精确模式在 128 对 H0/H1 图上的距离哈希与当前实现一致，但 H0 约慢 4.9 倍、H1 约慢 4.0 倍。Hera 默认的 1% 相对近似模式仍慢约 3.5 倍，而且会改变距离。

进一步对 H0/H1 各抽样 2048 对真实图：Hera 默认模式在 H1 上产生了 3 次严格 `<0.1` 判定翻转，均为当时 exact GUDHI 距离小于 0.1、Hera 近似值大于 0.1；例如 `0.099738895893096924` 被估为 `0.10071811097441241`。因此近似 Hera 没有进入正式流程。

### 历史安装与回退

安装本机 wheel：

```powershell
.venv\Scripts\python -m pip install --force-reinstall --no-deps `
  build\wheelhouse\gudhi-3.13.0+topoquant.smalln2-cp312-cp312-win_amd64.whl
```

恢复官方 GUDHI：

```powershell
.venv\Scripts\python -m pip install --force-reinstall "gudhi==3.13.0"
```

重新构建时，在 GUDHI 3.13.0 源码上应用补丁后构建 wheel。不要对 3.14 开发分支直接打包，也不要把默认调用改成 `e=0`；后者会进入 exact `sorted_distances()` 路径，算法和性能边界都不同。
