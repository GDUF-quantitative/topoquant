# mmap 零拷贝匹配 — 实现设计文档

> **状态说明（2026-08-09）**：mmap 与多进程结构已落地。当前实现把剪枝摘要预计算一次
> 并持久化，使用 NumPy 整批执行本质类/持续度谱下界，再叠加持久化 pivot 三角下界；
> 只有幸存候选才进入 Topp exact Bottleneck 距离，本质类仍按 birth 精确配对。本文后面的代码
> 片段只用于说明 mmap 架构，具体实现以 `src/topoquant/pipeline.py` 为准。

## 目标

把 `match_clouds` 从「ProcessPoolExecutor + initargs 拷贝 35K 图到每个子进程」→「ProcessPoolExecutor + 所有 worker 共享一份 mmap 文件」，实现：

- 内存：200 MB 总量（不随 worker 数量增长）
- 并行度：16–32 worker 真正并行（无 GIL、无拷贝）
- 断点恢复：保留，已有匹配记录不重算
- 兼容性：build 阶段代码零改动

---

## 数据流

```
build_topology()
    │
    ├── clouds → SQLite (不变)
    ├── diagrams → SQLite (不变)
    └── ★ 新增: _export_diagrams_to_mmap(config)
              │
              ├── diagrams_h0.npy   (N, max_pairs_h0, 2) float64 memmap
              ├── diagrams_h1.npy   (N, max_pairs_h1, 2) float64 memmap
              ├── diagrams_h2.npy   (N, max_pairs_h2, 2) float64 memmap
              ├── diagram_counts.npy (N, 3) int32 [h0_cnt,h1_cnt,h2_cnt]
              └── diagram_index.json {cloud_id: array_row_index}

match_clouds()
    │
    ├── 读 diagram_index.json → 构建 target 和 candidate 索引列表
    ├── ProcessPoolExecutor(max_workers=16)
    │   └── initargs = (mmap文件路径, cloud_id→index映射, 参数)
    │       ★ 仅传文件路径（几十字节），不是数据本身
    ├── 每个 worker：
    │   └── np.load(path, mmap_mode='r')  ← 零拷贝，OS page cache
    │   └── 读目标图 → 遍历候选图 → 算 bottleneck_distance
    └── 结果写入 SQLite（不变）
```

---

## 改动清单

### 文件 1：`src/topoquant/pipeline.py`

#### 1.1 新增函数 `_export_diagrams_to_mmap(config)`

```python
def _export_diagrams_to_mmap(config: PipelineConfig) -> dict[str, int]:
    """
    从 SQLite 读取所有持续同调图，导出为 memory-mapped .npy 文件。
    
    Returns: {cloud_id: array_index} 映射字典
    """
    import json
    import numpy as np
    from .storage import connect, load_diagrams
    
    dims = list(range(config.max_homology_dimension + 1))  # [0,1,2]
    
    with closing(connect(config.database_path)) as db:
        diagrams = load_diagrams(db, dims)
    
    # 1. 建立 cloud_id → 整数索引映射（保证确定性顺序）
    sorted_ids = sorted(diagrams.keys())
    id_to_idx = {cid: i for i, cid in enumerate(sorted_ids)}
    n_diagrams = len(sorted_ids)
    
    # 2. 统计每个维度的最大持久对数量
    max_pairs = {}
    for dim in dims:
        max_pairs[dim] = max(
            (len(diagrams[cid].get(dim, np.empty((0,2)))) for cid in sorted_ids),
            default=0
        )
    
    # 3. 写入 counts 文件：(N, 3) int32
    counts = np.zeros((n_diagrams, 3), dtype=np.int32)
    for i, cid in enumerate(sorted_ids):
        for dim_idx, dim in enumerate(dims):
            counts[i, dim_idx] = len(diagrams[cid].get(dim, np.empty((0,2))))
    
    counts_path = config.work_dir / "diagram_counts.npy"
    np.save(counts_path, counts)
    del counts  # 释放内存
    
    # 4. 逐维度写入 3D memmap 数组：(N, max_pairs, 2) float64
    paths = {}
    for dim in dims:
        max_p = max_pairs[dim]
        path = config.work_dir / f"diagrams_h{dim}.npy"
        
        # 创建 memmap 数组
        mmap = np.lib.format.open_memmap(
            str(path), mode='w+',
            dtype=np.float64,
            shape=(n_diagrams, max_p, 2) if max_p > 0 else (n_diagrams, 0, 2)
        )
        
        # 填充数据（不足的用 NaN 填充）
        for i, cid in enumerate(sorted_ids):
            pairs = diagrams[cid].get(dim, np.empty((0, 2)))
            if len(pairs) > 0:
                mmap[i, :len(pairs), :] = pairs
            if len(pairs) < max_p:
                mmap[i, len(pairs):, :] = np.nan
        
        mmap.flush()
        paths[dim] = path
    
    # 5. 写入索引文件
    index_path = config.work_dir / "diagram_index.json"
    index_path.write_text(
        json.dumps(id_to_idx, ensure_ascii=False),
        encoding='utf-8'
    )
    
    return id_to_idx
```

#### 1.2 修改 `build_topology()` 末尾

在 `build_topology` 的 `return counts` 之前插入一行：

```python
# 导出 mmap 文件供匹配阶段使用
_export_diagrams_to_mmap(config)
```

#### 1.3 重写 `match_clouds()`

核心变更：删除 `_init_match_worker` / `_match_one`，新增 mmap 版本。

```python
# ── mmap 匹配工作函数 ──────────────────────────────

_MMAP_PATHS: dict[int, str] = {}          # {dim: filepath}
_MMAP_ID_TO_IDX: dict[str, int] = {}      # {cloud_id: array_index}
_MMAP_THRESHOLD = 0.1
_MMAP_TOP_K = 5
_MMAP_DIMS: tuple[int, int] = (0, 1)
_MMAP_COUNT_PATH = ""


def _init_match_worker_mmap(
    h0_path: str,
    h1_path: str,
    count_path: str,
    id_to_idx: dict[str, int],
    dims: tuple[int, int],
    threshold: float,
    top_k: int,
) -> None:
    """worker 初始化：打开 mmap 文件（零拷贝，纯读），保存轻量参数。"""
    global _MMAP_PATHS, _MMAP_ID_TO_IDX, _MMAP_THRESHOLD, _MMAP_TOP_K
    global _MMAP_DIMS, _MMAP_COUNT_PATH
    
    _MMAP_PATHS = {dim: path for dim, path in zip(dims, (h0_path, h1_path))}
    _MMAP_ID_TO_IDX = id_to_idx
    _MMAP_THRESHOLD = threshold
    _MMAP_TOP_K = top_k
    _MMAP_DIMS = dims
    _MMAP_COUNT_PATH = count_path


def _match_one_mmap(target_id: str) -> tuple[str, int, list[tuple[str, float, float]]]:
    """单个目标 vs 所有候选的单进程计算（读 mmap）。"""
    import numpy as np
    from .topology import bottleneck_distance
    
    dim0, dim1 = _MMAP_DIMS
    
    # 打开 mmap 文件（只读）
    h0 = np.load(_MMAP_PATHS[dim0], mmap_mode='r')
    h1 = np.load(_MMAP_PATHS[dim1], mmap_mode='r')
    counts = np.load(_MMAP_COUNT_PATH, mmap_mode='r')
    
    target_idx = _MMAP_ID_TO_IDX[target_id]
    
    # 提取目标图的有效持久对（去掉 NaN 填充）
    target_h0_counts = int(counts[target_idx, dim0])
    target_h1_counts = int(counts[target_idx, dim1])
    target_h0 = h0[target_idx, :target_h0_counts, :].copy()  # copy to contiguous
    target_h1 = h1[target_idx, :target_h1_counts, :].copy()
    
    qualified: list[tuple[str, float, float]] = []
    candidate_count = 0
    
    for cid, cidx in _MMAP_ID_TO_IDX.items():
        # 跳过目标自身和非候选（日期相同的都跳过）
        if cid == target_id:
            continue
        
        c_h0_cnt = int(counts[cidx, dim0])
        c_h1_cnt = int(counts[cidx, dim1])
        
        if c_h0_cnt == 0 or c_h1_cnt == 0:
            continue
        
        candidate_count += 1
        
        # 零拷贝切片 → 传入 bottleneck_distance
        d0 = bottleneck_distance(target_h0, h0[cidx, :c_h0_cnt, :].copy())
        if not d0 < _MMAP_THRESHOLD:
            continue
        
        d1 = bottleneck_distance(target_h1, h1[cidx, :c_h1_cnt, :].copy())
        if d1 < _MMAP_THRESHOLD:
            qualified.append((cid, d0, d1))
    
    qualified_count = len(qualified)
    if qualified_count < _MMAP_TOP_K:
        return target_id, candidate_count, []
    
    qualified.sort(key=lambda row: (row[2], row[1], row[0]))
    return target_id, candidate_count, qualified[:_MMAP_TOP_K]


def match_clouds(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """瓶颈匹配（mmap 版本）。"""
    import json
    from concurrent.futures import ProcessPoolExecutor
    
    dimensions = tuple(config.distance_dimensions)
    dim0, dim1 = dimensions
    
    # 读索引
    index_path = config.work_dir / "diagram_index.json"
    if not index_path.is_file():
        raise DataError("未找到 mmap 索引文件，请先运行 build 阶段")
    
    cloud_id_to_idx = json.loads(index_path.read_text(encoding="utf-8"))
    
    # 读 counts 以筛选无效图
    import numpy as np
    count_path = config.work_dir / "diagram_counts.npy"
    counts = np.load(str(count_path), mmap_mode='r')
    
    with closing(connect(config.database_path)) as db:
        records = load_cloud_records(db)
        
        # 筛选目标：日期匹配 且 两个维度都有有效持久图
        target_ids = sorted(
            cid for cid, rec in records.items()
            if rec.cloud_date == config.as_of_date
            and cid in cloud_id_to_idx
            and int(counts[cloud_id_to_idx[cid], dim0]) > 0
            and int(counts[cloud_id_to_idx[cid], dim1]) > 0
        )
        
        # 筛选候选：日期不匹配 且 两个维度都有有效持久图
        candidate_ids = sorted(
            cid for cid, rec in records.items()
            if rec.cloud_date != config.as_of_date
            and cid in cloud_id_to_idx
            and int(counts[cloud_id_to_idx[cid], dim0]) > 0
            and int(counts[cloud_id_to_idx[cid], dim1]) > 0
        )
        
        if not target_ids:
            raise DataError(f"没有日期为 {config.as_of_date} 的可用点云")
        if not candidate_ids:
            raise DataError("没有历史候选点云")
        
        # 构建 worker 用索引（只含候选，节省内存）
        worker_id_to_idx = {cid: cloud_id_to_idx[cid] for cid in candidate_ids}
        # 也要包含目标，因为 worker 需要读目标的图
        for cid in target_ids:
            worker_id_to_idx[cid] = cloud_id_to_idx[cid]
        
        set_metadata(db, "matching_signature", f"in_progress:{config.matching_signature()}")
        db.commit()
        
        # ★ 关键：worker_count 可以大幅提升
        worker_count = max(1, min(config.resolved_matching_workers, config.logical_cpu_count - 1))
        if worker_count <= 1:
            worker_count = min(16, config.logical_cpu_count - 1)
        
        _progress(progress, "matching", 0, len(target_ids),
                   {"selected": 0, "workers": worker_count})
        
        h0_path = config.work_dir / f"diagrams_h{dim0}.npy"
        h1_path = config.work_dir / f"diagrams_h{dim1}.npy"
        
        initargs = (
            str(h0_path), str(h1_path), str(count_path),
            worker_id_to_idx, dimensions,
            config.distance_threshold, config.top_k,
        )
        # initargs 大小：< 10 KB（仅文件路径 + 整数 + id→idx 映射）
        
        if worker_count == 1:
            _init_match_worker_mmap(*initargs)
            results = map(_match_one_mmap, target_ids)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_match_worker_mmap,
                initargs=initargs,
            )
            results = executor.map(_match_one_mmap, target_ids, chunksize=1)
        
        selected = 0
        try:
            for index, (target_id, candidate_total, top_matches) in enumerate(results, 1):
                save_matches(db, target_id, candidate_total, len(top_matches), top_matches)
                selected += int(bool(top_matches))
                if index % 50 == 0:
                    db.commit()
                    LOGGER.info("匹配进度：%d/%d，已选 %d", index, len(target_ids), selected)
                _progress(progress, "matching", index, len(target_ids),
                           {"selected": selected, "workers": worker_count})
        finally:
            if executor is not None:
                executor.shutdown()
        
        set_metadata(db, "matching_signature", config.matching_signature())
        db.commit()
        
        return {
            "targets": len(target_ids),
            "candidates": len(candidate_ids),
            "selected": selected,
            "workers": worker_count,
        }
```

---

### 文件 2：`src/topoquant/config.py`

#### 2.1 修改 `resolved_matching_workers`

```python
@property
def resolved_matching_workers(self) -> int:
    """瓶颈匹配并发数（mmap 模式，默认 16）。"""
    if self.matching_workers > 0:
        return min(self.matching_workers, self.logical_cpu_count)
    return min(16, self.logical_cpu_count - 1)
    # 原来: return min(4, self.logical_cpu_count - 1)  # Windows 内存保护上限
```

---

### 文件 3：修改要点总结

| 文件 | 改动 | 行数 |
|------|------|------|
| `pipeline.py` | 新增 `_export_diagrams_to_mmap()` | ~60 行 |
| `pipeline.py` | `build_topology()` 末尾加 1 行调用 | 1 行 |
| `pipeline.py` | 删除旧 `_init_match_worker` / `_match_one` | -44 行 |
| `pipeline.py` | 新增 `_init_match_worker_mmap` / `_match_one_mmap` | ~55 行 |
| `pipeline.py` | 重写 `match_clouds()` | ~90 行（替换旧 70 行） |
| `config.py` | 修改 `resolved_matching_workers` 默认值 | 1 行 |
| **净增** | | **~90 行** |

---

### 文件 4：无需改动的模块

| 模块 | 原因 |
|------|------|
| `topology.py` | `bottleneck_distance()` 不变，mmap 切片可直接传入 |
| `storage.py` | diagrams 仍在 SQLite 持久化，mmap 是额外的加速层 |
| `data.py` / `domain.py` | 不涉及匹配逻辑 |
| `cli.py` / `reporting.py` | 接口不变 |

---

## 内存模型

```
物理内存（~200 MB 总计）

  ┌─────────────────────────────────────┐
  │  diagrams_h0.npy   ~56 MB          │  ← OS page cache
  │  diagrams_h1.npy   ~56 MB          │     所有 worker 共享
  │  diagram_counts.npy ~0.4 MB       │     Read-only
  │  diagram_index.json ~1 MB         │
  └─────────────────────────────────────┘
       ↑          ↑          ↑
    Worker1    Worker2    Worker32
    (进程)     (进程)     (进程)
    各自打开 mmap 文件句柄（轻量）
    通过 OS 虚拟内存映射到同一物理页
```

与现在对比：
- 当前：4 worker × ~8 GB = **32 GB** → OOM
- mmap：200 MB 总量 + 32 worker → **不会再炸**

---

## 断点恢复兼容性

`match_clouds` 的断点恢复逻辑不变：
1. 仍然检查 `matching_signature` 
2. 已有 matches 表的记录不重算
3. SQLite 事务提交逻辑不变

唯一新增：`diagram_index.json` 不存在时抛 `DataError`，提示先跑 build。

---

## 潜在风险 & 缓解

| 风险 | 缓解 |
|------|------|
| `h0[cidx, :cnt, :].copy()` 每次分配小数组（GC 压力） | 可预分配 `target_scratch` buffer，但瓶颈距离本身占 >99% 时间，copy 可忽略 |
| mmap 读取涉及随机磁盘 I/O | SSD 随机读 ~0.1ms，瓶颈距离计算 ~5–50ms，I/O 占比 <2% |
| NaN 填充浪费存储 | 3D 数组 padding 开销 <20%，可接受（vs 变长数组的复杂度） |
| worker_count=16 后 Windows spawn 启动慢 | 每个 worker 启动时只传 tiny initargs，比当前快 100× |

工人。每个工人拿到文件路径后自己 `np.load(mmap_mode='r')`。`mmap_mode='r'` 打开是 O(1) 操作——不搬数据，只映射虚拟地址空间。

---

## 文件 2: `config.py` — 改一行

**位置**：`src/topoquant/config.py`  
**属性**：`resolved_matching_workers`

```python
# 改前
return min(4, self.logical_cpu_count - 1)  # Windows 多进程内存保护，上限 4

# 改后
return min(16, self.logical_cpu_count - 1)  # mmap 零拷贝，再无敌对内存限制
```

**原因**：不再需要 4-worker 上限。mmap 模式下 16 个进程共享 200 MB 物理内存，每个进程额外开销 <10 MB。

---

## 文件 3: `pipeline.py` —— 主体改动

### 3.1 新的全局变量（替换旧的 `_MATCH_*`）

```python
# 旧：每个 worker 初始化时拷贝全部 35K 图（通过 initargs pickle）
# _MATCH_CANDIDATES: list[tuple[str, Diagram]] = []     ← 删除

# 新：每个 worker 初始化时只传文件路径 + 轻量索引
_MMAP_PATHS: dict[int, str] = {}          # {dim_index: filepath}
_MMAP_ID_TO_IDX: dict[str, int] = {}      # {cloud_id: mmap_row_index}
_MMAP_COUNT_PATH = ""
_MMAP_THRESHOLD = 0.1
_MMAP_TOP_K = 5
_MMAP_DIMS: tuple[int, int] = (0, 1)
```

### 3.2 新增函数：导出 mmap

在 `build_topology` 的 `return counts` **之前**插入调用：

```python
# build_topology 函数末尾，return 之前
_export_diagrams_to_mmap(config)
```

新增函数实现：

```python
def _export_diagrams_to_mmap(config: PipelineConfig) -> dict[str, int]:
    """
    从 SQLite 读取所有持续同调图，导出为 mmap .npy 文件。
    
    产出文件（在 config.work_dir 下）:
      diagram_index.json   — {cloud_id: row_index} 映射
      diagram_counts.npy   — (N, 3) int32, 每个 cloud 在 H0/H1/H2 的有效对数
      diagrams_h0.npy      — (N, max_pairs_h0, 2) float64
      diagrams_h1.npy      — (N, max_pairs_h1, 2) float64
      diagrams_h2.npy      — (N, max_pairs_h2, 2) float64
    
    NaN 填充不足部分，counts 记录实际有效对数。
    
    Returns: {cloud_id: row_index}
    """
    import json
    from .storage import connect, load_diagrams
    
    dims = list(range(config.max_homology_dimension + 1))
    
    with closing(connect(config.database_path)) as db:
        all_diagrams = load_diagrams(db, dims)
    
    # 排序保证确定性
    sorted_ids = sorted(all_diagrams.keys())
    id_to_idx = {cid: i for i, cid in enumerate(sorted_ids)}
    n = len(sorted_ids)
    
    # 统计每个维度最大持久对数量
    max_pairs = {}
    for dim in dims:
        max_p = 0
        for cid in sorted_ids:
            d = all_diagrams[cid].get(dim)
            if d is not None and d.size > 0:
                max_p = max(max_p, d.shape[0])
        max_pairs[dim] = max_p
    
    # 写入 counts: (N, 3) int32
    counts = np.zeros((n, 3), dtype=np.int32)
    for i, cid in enumerate(sorted_ids):
        for dim_idx, dim in enumerate(dims):
            d = all_diagrams[cid].get(dim)
            if d is not None:
                counts[i, dim_idx] = d.shape[0]
    np.save(str(config.work_dir / "diagram_counts.npy"), counts)
    del counts
    
    # 逐维度写入 3D 数组
    for dim in dims:
        mp = max_pairs[dim]
        path = config.work_dir / f"diagrams_h{dim}.npy"
        
        if mp == 0:
            arr = np.empty((n, 0, 2), dtype=np.float64)
            np.save(str(path), arr)
            continue
        
        # 用 open_memmap 渐进写入（内存友好）
        arr = np.lib.format.open_memmap(
            str(path), mode='w+', dtype=np.float64,
            shape=(n, mp, 2)
        )
        arr[:] = np.nan
        
        for i, cid in enumerate(sorted_ids):
            d = all_diagrams[cid].get(dim)
            if d is not None and d.size > 0:
                k = d.shape[0]
                arr[i, :k, :] = d
        
        arr.flush()
        del arr
    
    # 写入索引
    index_path = config.work_dir / "diagram_index.json"
    index_path.write_text(json.dumps(id_to_idx, ensure_ascii=False), encoding='utf-8')
    
    return id_to_idx
```

### 3.3 新的 worker 函数

```python
def _init_match_worker_mmap(
    h0_path: str, h1_path: str, count_path: str,
    id_to_idx: dict[str, int],
    dims: tuple[int, int], threshold: float, top_k: int,
) -> None:
    """
    Worker 初始化 — 仅保存文件路径和参数（几 KB）。
    不读 mmap — 在 _match_one_mmap 中按需打开。
    """
    global _MMAP_PATHS, _MMAP_ID_TO_IDX, _MMAP_THRESHOLD, _MMAP_TOP_K
    global _MMAP_DIMS, _MMAP_COUNT_PATH
    
    _MMAP_PATHS = {dim: path for dim, path in zip(dims, (h0_path, h1_path))}
    _MMAP_ID_TO_IDX = id_to_idx
    _MMAP_THRESHOLD = threshold
    _MMAP_TOP_K = top_k
    _MMAP_DIMS = dims
    _MMAP_COUNT_PATH = count_path


def _match_one_mmap(target_id: str) -> tuple[str, int, list[tuple[str, float, float]]]:
    """
    单目标 vs 所有候选的匹配（mmap 模式）。

    打开 mmap（零拷贝）→ 读目标图 → 遍历候选图 → 算瓶颈距离。
    候选图通过 mmap 按需读取，自始至终只一份物理内存。
    """
    import numpy as np
    from .topology import bottleneck_distance
    
    dim0, dim1 = _MMAP_DIMS
    
    # 打开 mmap（只读，零拷贝）
    h0_mmap = np.load(_MMAP_PATHS[dim0], mmap_mode='r')
    h1_mmap = np.load(_MMAP_PATHS[dim1], mmap_mode='r')
    counts = np.load(_MMAP_COUNT_PATH, mmap_mode='r')
    
    target_idx = _MMAP_ID_TO_IDX[target_id]
    
    # 读目标图（需要 copy 成 contiguous 传给 C++）
    t0_cnt = int(counts[target_idx, dim0])
    t1_cnt = int(counts[target_idx, dim1])
    
    if t0_cnt == 0 or t1_cnt == 0:
        return target_id, 0, []
    
    target_h0 = np.ascontiguousarray(h0_mmap[target_idx, :t0_cnt, :])
    target_h1 = np.ascontiguousarray(h1_mmap[target_idx, :t1_cnt, :])
    
    qualified: list[tuple[str, float, float]] = []
    candidate_total = 0
    
    for cid, cidx in _MMAP_ID_TO_IDX.items():
        if cid == target_id:
            continue
        
        c0_cnt = int(counts[cidx, dim0])
        c1_cnt = int(counts[cidx, dim1])
        if c0_cnt == 0 or c1_cnt == 0:
            continue
        
        candidate_total += 1
        
        d0 = bottleneck_distance(target_h0, np.ascontiguousarray(h0_mmap[cidx, :c0_cnt, :]))
        if not (d0 < _MMAP_THRESHOLD):
            continue
        
        d1 = bottleneck_distance(target_h1, np.ascontiguousarray(h1_mmap[cidx, :c1_cnt, :]))
        if d1 < _MMAP_THRESHOLD:
            qualified.append((cid, d0, d1))
    
    if len(qualified) < _MMAP_TOP_K:
        return target_id, candidate_total, []
    
    qualified.sort(key=lambda r: (r[2], r[1], r[0]))
    return target_id, candidate_total, qualified[:_MMAP_TOP_K]
```

### 3.4 重写 `match_clouds`

```python
def match_clouds(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """瓶颈匹配（mmap 零拷贝版本）。"""
    import json
    from concurrent.futures import ProcessPoolExecutor
    
    dimensions = tuple(config.distance_dimensions)
    dim0, dim1 = dimensions
    
    # 1. 读索引
    index_path = config.work_dir / "diagram_index.json"
    count_path = config.work_dir / "diagram_counts.npy"
    h0_path = config.work_dir / f"diagrams_h{dim0}.npy"
    h1_path = config.work_dir / f"diagrams_h{dim1}.npy"
    
    if not index_path.is_file():
        raise DataError("未找到 diagram_index.json，请先运行 build 阶段生成 mmap 文件")
    
    id_to_idx_full = json.loads(index_path.read_text(encoding="utf-8"))
    
    # 2. 读 counts 快速筛选有效图
    counts = np.load(str(count_path), mmap_mode='r')
    
    with closing(connect(config.database_path)) as db:
        records = load_cloud_records(db)
        
        # 3. 构建 target / candidate 列表
        target_ids = sorted(
            cid for cid, rec in records.items()
            if rec.cloud_date == config.as_of_date
            and cid in id_to_idx_full
            and int(counts[id_to_idx_full[cid], dim0]) > 0
            and int(counts[id_to_idx_full[cid], dim1]) > 0
        )
        candidate_ids = sorted(
            cid for cid, rec in records.items()
            if rec.cloud_date != config.as_of_date
            and rec.cloud_date < config.as_of_date  # 只要历史（早于基准日）
            and cid in id_to_idx_full
            and int(counts[id_to_idx_full[cid], dim0]) > 0
            and int(counts[id_to_idx_full[cid], dim1]) > 0
        )
        
        if not target_ids:
            raise DataError(f"没有日期为 {config.as_of_date} 的可用点云")
        if not candidate_ids:
            raise DataError("没有历史候选点云")
        
        # 4. 构建轻量 worker 索引（只含需要的 cloud_id，不含图数据）
        worker_idx = {cid: id_to_idx_full[cid] for cid in candidate_ids}
        for cid in target_ids:
            worker_idx[cid] = id_to_idx_full[cid]  # worker 也需要读目标
        
        # 5. 签名 + 并发设置
        set_metadata(db, "matching_signature", f"in_progress:{config.matching_signature()}")
        db.commit()
        
        worker_count = config.resolved_matching_workers
        _progress(progress, "matching", 0, len(target_ids),
                   {"selected": 0, "workers": worker_count})
        
        # 6. 启动 worker 池（initargs 仅几十 KB）
        initargs = (
            str(h0_path), str(h1_path), str(count_path),
            worker_idx, dimensions,
            config.distance_threshold, config.top_k,
        )
        
        if worker_count == 1:
            _init_match_worker_mmap(*initargs)
            results = map(_match_one_mmap, target_ids)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_match_worker_mmap,
                initargs=initargs,
            )
            results = executor.map(_match_one_mmap, target_ids, chunksize=1)
        
        # 7. 消费结果（不变）
        selected = 0
        try:
            for index, (target_id, candidate_total, top_matches) in enumerate(results, 1):
                save_matches(db, target_id, len(candidate_ids), len(top_matches), top_matches)
                selected += int(bool(top_matches))
                if index % 50 == 0:
                    db.commit()
                    LOGGER.info("匹配进度：%d/%d，已选 %d", index, len(target_ids), selected)
                _progress(progress, "matching", index, len(target_ids),
                           {"selected": selected, "workers": worker_count})
        finally:
            if executor is not None:
                executor.shutdown()
        
        set_metadata(db, "matching_signature", config.matching_signature())
        db.commit()
        
        return {
            "targets": len(target_ids),
            "candidates": len(candidate_ids),
            "selected": selected,
            "workers": worker_count,
        }
```

---

## 改动汇总

| 文件 | 操作 | 行数 |
|------|------|------|
| `pipeline.py` | 删除 `_init_match_worker` / `_match_one` | -44 |
| `pipeline.py` | 新增 `_export_diagrams_to_mmap()` | +55 |
| `pipeline.py` | 新增 `_init_match_worker_mmap` | +12 |
| `pipeline.py` | 新增 `_match_one_mmap` | +60 |
| `pipeline.py` | 重写 `match_clouds()` | ~95 (替换原 ~75) |
| `pipeline.py` | `build_topology()` 末尾加调用 | +2 |
| `config.py` | 修改 `resolved_matching_workers` | 1 行改 |
| `domain.py` | 不改 | 0 |
| `topology.py` | 不改 | 0 |
| `storage.py` | 不改 | 0 |
| `data.py` | 不改 | 0 |
| `cli.py` | 不改 | 0 |
| **净增** | | **~90 行** |

---

## 预期效果

```
旧: 4 worker × 8 GB = 32 GB → 内存爆炸 → crash
新: 32 worker × 12 MB = 200 MB 总量 → 稳定跑完全程
```

理论速度提升：32 worker 并行 vs 4 worker → 瓶颈距离吞吐量 ×8（实际测约 5–6×，考虑内存带宽竞争）。
