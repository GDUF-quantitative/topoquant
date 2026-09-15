from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar


class ConfigError(ValueError):
    """配置无效。"""


def _coerce_int(name: str, value: object) -> int:
    """把配置字段安全转换为 ``int``，拒绝布尔与带小数部分的浮点（模式 A fail-fast）。

    ``int(True)`` 会静默得到 1、``int(60.5)`` 会静默截断为 60——二者都属于「脏配置
    静默通过」的隐患，这里显式拦下并带字段名报错。
    """
    if isinstance(value, bool):
        raise ConfigError(f"{name} 不能为布尔值（收到 {value!r}），请使用整数")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"{name} 必须为整数，收到小数 {value!r}（请改用整数）")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是整数，收到 {value!r}") from exc


def _coerce_float(name: str, value: object) -> float:
    """把配置字段安全转换为 ``float``，拒绝布尔（模式 A fail-fast）。"""
    if isinstance(value, bool):
        raise ConfigError(f"{name} 不能为布尔值（收到 {value!r}），请使用数值")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是数值，收到 {value!r}") from exc


# 数据集指纹进度上报槽（模块级，仅在 from_json 构造期间临时置位，自动复位）。
# 设计为模块级而非实例属性，是因为指纹在 __post_init__ 内计算，无法经构造参数透传；
# 以 contextmanager 包住构造调用，保证多调用交错时互不污染、构造结束后一定复位。
# 回调签名：(current_name: str, done: int, total: int) -> None。
_FP_PROGRESS_SINK: Callable[[str, int, int], None] | None = None


@contextlib.contextmanager
def fingerprint_progress(sink: Callable[[str, int, int], None] | None) -> Iterator[None]:
    """在 ``from_json`` 计算数据集指纹期间上报进度 (current_name, done, total)。"""
    global _FP_PROGRESS_SINK
    prev = _FP_PROGRESS_SINK
    _FP_PROGRESS_SINK = sink
    try:
        yield
    finally:
        _FP_PROGRESS_SINK = prev


def _sha256_signature(
    fields: dict,
    *,
    ensure_ascii: bool = True,
    truncate: int | None = None,
) -> str:
    """JSON 序列化 + SHA256 的单一签名原语（M5：消除 5 处重复 hash/sign）。

    所有签名方法（topology/matching/forecast/_dir_hash 及 pipeline.matching_resume_key）
    共用本原语；参数严格透传，保证序列化字节与旧内联实现逐字节一致 → 签名键（INV-2）不变。
    """
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=ensure_ascii).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest if truncate is None else digest[:truncate]


def _warn_deprecated_fields(raw: dict[str, Any]) -> int | None:
    """from_json 的废弃字段警告簇（M4：提取自 from_json，逻辑逐字平移）。

    返回 legacy ``workers`` 值（供 matching_workers 回退使用）；无该字段返回 None。
    ``stacklevel=3``：经本函数中转后仍指向 from_json 的调用者（与内联版一致）。
    """
    legacy_workers = raw.get("workers")
    if legacy_workers is not None:
        warnings.warn(
            "配置字段 'workers' 已废弃，请改用 'matching_workers'；"
            "'workers' 现仅作为 matching_workers 的回退值。",
            DeprecationWarning,
            stacklevel=3,
        )
    if "max_homology_dimension" in raw:
        # 现在同调计算维度由 distance_dimensions 唯一决定（见 max_homology_dimension 属性），
        # 独立配置该字段只会与实际需求冲突（配大了浪费、配小了缺维），故废弃并忽略。
        warnings.warn(
            "配置字段 'max_homology_dimension' 已废弃并被忽略：最高同调维度现由 "
            "'distance_dimensions' 自动推导为 max(distance_dimensions)。",
            DeprecationWarning,
            stacklevel=3,
        )
    if "distance_threshold" in raw:
        if "distance_threshold_h0" in raw or "distance_threshold_h1" in raw:
            warnings.warn(
                "配置字段 'distance_threshold' 已废弃，请勿与 'distance_threshold_h0' / "
                "'distance_threshold_h1' 混用；它现在仅作为双阈值的回退值。",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                "配置字段 'distance_threshold' 已废弃，请改用 'distance_threshold_h0' 与 "
                "'distance_threshold_h1'（当前自动拆为两者相等）。",
                DeprecationWarning,
                stacklevel=3,
            )
    return legacy_workers


@dataclass(frozen=True)
class PipelineConfig:
    source_dir: Path
    # work_dir 现为「可选覆盖」：显式提供（含 None）时，为 None 由 work_root + 参数
    # 在 __post_init__ 中派生嵌套路径，非 None 则原样使用（legacy / 手动 pin 目录）。
    # 无默认值以兼容既有位置参数构造（source_dir, work_dir, as_of_date, ...）。
    work_dir: Path | None
    as_of_date: date
    # 实验产物根目录；派生路径 = work_root / 日期 / 数据指纹 / 拓扑段 / 匹配段 / 预测段。
    work_root: Path = Path("./runs")
    window_size: int = 60
    lookback_trading_days: int = 480
    min_windows: int = 4
    features: tuple[str, ...] = ("money", "volume", "high", "close")
    max_edge_length: float = 3.0
    distance_dimensions: tuple[int, int] = (0, 1)
    # 双阈值：H0（dim0）与 H1（dim1）各自的瓶颈距离合格阈值。
    # 匹配合格判定为 (d0 < distance_threshold_h0) AND (d1 < distance_threshold_h1)，
    # 旧单值 distance_threshold 作为 legacy 别名，自动拆给两者（见 from_json）。
    distance_threshold_h0: float = 0.1
    distance_threshold_h1: float = 0.1
    top_k: int = 5
    forecast_horizon: int = 5
    topology_workers: int = 0
    matching_workers: int = 0
    forecast_workers: int = 0
    data_quality_mode: str = "permissive"
    # 融合升级：同调/瓶颈内核后端（唯一切换点经 topology.py 门面）。
    # 默认 ripser_topp（Ripser + Topp 1.0.0 高速瓶颈距离，已替换自研 C-DLL 内核）；
    # native_c_dll 保留为已注册回退；gudhi 仅作可选 extra 用于对拍/兜底。
    topology_backend: str = "ripser_topp"
    # 融合升级：瓶颈匹配第 4 层剪枝（pivot）所用代表候选数；0 表示禁用该层剪枝。
    # 通过三角不等式下界无损剪掉必不合格候选，不改变任何最终匹配结果。
    matching_pivots: int = 8

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> PipelineConfig:
        config_path = Path(path).resolve()
        try:
            raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"配置文件不存在：{config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"配置文件不是合法 JSON：{exc}") from exc
        except OSError as exc:  # 权限不足 / 读取失败等，统一转为 ConfigError 带路径
            raise ConfigError(f"无法读取配置文件 {config_path}：{exc}") from exc

        required = {"source_dir", "as_of_date"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ConfigError(f"配置缺少字段：{', '.join(missing)}")

        base_dir = config_path.parent

        def resolve_path(value: str | None) -> Path | None:
            # 修复 null→"None" 错位：JSON ``null`` 直接映射为 ``None``（不解析成字面量
            # "None" 路径，后者会静默生成一个名为 "None" 的目录，导致续算键误判）。
            if value is None:
                return None
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            return candidate.resolve()

        try:
            parsed_date = date.fromisoformat(str(raw["as_of_date"]))
        except ValueError as exc:
            raise ConfigError("as_of_date 必须是 YYYY-MM-DD") from exc

        legacy_workers = _warn_deprecated_fields(raw)
        # work_dir 仍可作为显式覆盖（legacy / 手动 pin）；缺省时由 work_root + 参数派生。
        # work_dir 仍可作为显式覆盖（legacy / 手动 pin）；缺省时由 work_root + 参数派生。
        override_work_dir = resolve_path(raw["work_dir"]) if "work_dir" in raw else None
        work_root = resolve_path(raw.get("work_root") or "./runs")
        config_kwargs = dict(  # noqa: C408
            source_dir=resolve_path(raw["source_dir"]),
            work_dir=override_work_dir,
            as_of_date=parsed_date,
            work_root=work_root,
            # 数值字段经 fail-fast 强类型转换：拒绝布尔、拒绝带小数部分的浮点（模式 A）。
            window_size=_coerce_int("window_size", raw.get("window_size", 60)),
            lookback_trading_days=_coerce_int(
                "lookback_trading_days", raw.get("lookback_trading_days", 480)
            ),
            min_windows=_coerce_int("min_windows", raw.get("min_windows", 4)),
            features=tuple(raw.get("features", ["money", "volume", "high", "close"])),
            max_edge_length=_coerce_float("max_edge_length", raw.get("max_edge_length", 3.0)),
            distance_dimensions=tuple(
                int(value) for value in raw.get("distance_dimensions", [0, 1])
            ),
            distance_threshold_h0=_coerce_float(
                "distance_threshold_h0",
                raw.get("distance_threshold_h0", raw.get("distance_threshold", 0.1)),
            ),
            distance_threshold_h1=_coerce_float(
                "distance_threshold_h1",
                raw.get("distance_threshold_h1", raw.get("distance_threshold", 0.1)),
            ),
            top_k=_coerce_int("top_k", raw.get("top_k", 5)),
            forecast_horizon=_coerce_int("forecast_horizon", raw.get("forecast_horizon", 5)),
            topology_workers=_coerce_int("topology_workers", raw.get("topology_workers", 0)),
            matching_workers=_coerce_int(
                "matching_workers",
                raw.get("matching_workers", legacy_workers if legacy_workers is not None else 0),
            ),
            forecast_workers=_coerce_int("forecast_workers", raw.get("forecast_workers", 0)),
            data_quality_mode=str(raw.get("data_quality_mode", "permissive")),
            topology_backend=str(raw.get("topology_backend", "ripser_topp")),
            matching_pivots=_coerce_int("matching_pivots", raw.get("matching_pivots", 8)),
        )
        # 指纹扫描在 __post_init__ 内触发；用 contextmanager 临时置位进度槽，
        # 构造结束后自动复位，避免跨调用/并发污染（O-1）。
        with fingerprint_progress(progress):
            config = cls(**config_kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        # 模式 A：消除「nosrc 假通过」——source_dir 必须是一个真实存在的目录，
        # 否则下游指纹/枚举会静默退化成空数据集（"nosrc"），极难排查。
        if self.source_dir is None or not self.source_dir.is_dir():
            raise ConfigError(f"source_dir 不是已存在的目录：{self.source_dir}")
        if self.window_size < 2:
            raise ConfigError("window_size 必须至少为 2")
        if self.lookback_trading_days < self.window_size:
            raise ConfigError("lookback_trading_days 不能小于 window_size")
        if self.min_windows < 1:
            raise ConfigError("min_windows 必须为正整数")
        if self.min_windows * self.window_size > self.lookback_trading_days:
            raise ConfigError("最少窗口所需交易日超过 lookback_trading_days")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ConfigError("features 不能为空或重复")
        if self.max_edge_length <= 0:
            raise ConfigError("max_edge_length 必须大于 0")
        self._validate_distance_dimensions()
        self._validate_runtime_params()

    def _validate_runtime_params(self) -> None:
        """validate 的运行参数守卫簇（M4：提取自 validate，守卫顺序逐字不变）。"""
        if self.distance_threshold_h0 <= 0 or self.distance_threshold_h1 <= 0:
            raise ConfigError("distance_threshold_h0 / distance_threshold_h1 必须大于 0")
        if self.top_k < 1 or self.forecast_horizon < 1:
            raise ConfigError("top_k、forecast_horizon 必须为正整数")
        if any(
            value < 0
            for value in (self.topology_workers, self.matching_workers, self.forecast_workers)
        ):
            raise ConfigError("并发数不能为负数；0 表示根据 CPU 内核数自动选择")
        if self.data_quality_mode not in ("strict", "permissive"):
            raise ConfigError(
                "data_quality_mode 只能是 'strict' 或 'permissive'，收到 "
                f"{self.data_quality_mode!r}"
            )
        if not 0 <= self.matching_pivots <= 32:
            raise ConfigError("matching_pivots 必须在 0 到 32 之间；0 表示禁用 pivot 剪枝")
        if self.topology_backend not in ("gudhi", "ripser_topp", "native_c_dll"):
            raise ConfigError(
                f"topology_backend 只能是 'gudhi' / 'ripser_topp' / 'native_c_dll'，"
                f"收到 {self.topology_backend!r}"
            )

    def _validate_distance_dimensions(self) -> None:
        """校验 ``distance_dimensions``：恰好两个互不相同的非负整数。

        旧契约把维度硬编码为 ``(0, 1)``，与 ``run.py`` 交互提示允许 ``1,2`` 的行为矛盾，
        导致用户填完全部参数后才在校验处崩溃（P1-1）。现放开为任意两个不同维度，
        实际计算的最高同调维度由 :attr:`max_homology_dimension` 自动推导，
        因此不存在"维度值超出计算范围"的情形——上界随配置一起浮动。
        """
        dimensions = self.distance_dimensions
        if not isinstance(dimensions, tuple) or len(dimensions) != 2:
            raise ConfigError("distance_dimensions 必须恰好包含两个维度，如 [0, 1] 或 [1, 2]")
        for value in dimensions:
            # bool 是 int 的子类，显式排除以免 [true, false] 被当成 [1, 0]。
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"distance_dimensions 的元素必须是整数，收到 {value!r}")
            if value < 0:
                raise ConfigError(f"distance_dimensions 的维度不能为负数，收到 {value}")
        if dimensions[0] == dimensions[1]:
            raise ConfigError(
                f"distance_dimensions 的两个维度必须互不相同，收到 {list(dimensions)}"
            )
        if self.max_homology_dimension > 1:
            warnings.warn(
                f"distance_dimensions={list(dimensions)} 需要计算到 "
                f"H{self.max_homology_dimension}；H2 及以上的持续同调在点数较多时"
                "耗时与内存开销显著上升，请预留充足资源。",
                UserWarning,
                stacklevel=3,
            )

    @property
    def max_homology_dimension(self) -> int:
        """实际需要计算的最高同调维度 = ``max(distance_dimensions)``。

        由距离维度唯一推导，避免"配置了 H2 却从不使用"的纯浪费（P1-1）：
        GUDHI 只需算到匹配阶段真正读取的最高维度即可。
        因两个维度互不相同且非负，该值恒 >= 1（原 ``max_homology_dimension >= 1``
        的校验因此被结构性保证，无需再单独检查）。
        """
        return max(self.distance_dimensions)

    @property
    def database_path(self) -> Path:
        return self.work_dir / "artifacts.sqlite3"

    @property
    def output_dir(self) -> Path:
        """预测终产物目录，按 ``top_k`` 命名空间。

        昂贵的拓扑/距离缓存（``artifacts.sqlite3``）仍在 ``work_dir`` 根共享、不随
        ``top_k`` 分区；仅把廉价的终产物（metrics/predictions/report/selected_matches）
        按 ``top_k`` 分目录，避免同 ``work_dir`` 内不同 ``top_k`` 互相覆盖，从而兼顾
        缓存复用与参数可追溯性（详见 compare_runs 配套改动）。
        """
        return self.work_dir / "outputs" / f"topk{self.top_k}"

    @property
    def logical_cpu_count(self) -> int:
        return max(1, os.cpu_count() or 1)

    @property
    def resolved_topology_workers(self) -> int:
        if self.topology_workers:
            # 手动设定上限不应超过物理核心数：超过只会徒增进程争用，无实际收益（P3-4）。
            return min(self.topology_workers, self.logical_cpu_count)
        available = self.logical_cpu_count
        return max(1, min(8, available - 1 if available > 1 else 1))

    @property
    def resolved_matching_workers(self) -> int:
        """瓶颈匹配并发数。

        mmap 零拷贝模式下所有 worker 共享同一份持久图物理内存，
        不再需要旧版为防止 Windows 多进程内存爆炸而设的 4 进程上限。
        """
        if self.matching_workers:
            return min(self.matching_workers, self.logical_cpu_count)
        available = self.logical_cpu_count
        return max(1, min(16, available - 1 if available > 1 else 1))

    @property
    def resolved_forecast_workers(self) -> int:
        # 预测阶段为 I/O 密集型（读取未来行情），并发上限取「核心数*2 与 32 的较小值」。
        # 手动设定值同样不得超过该上限：超过只会徒增线程切换与内存开销，无实际收益
        # （与 topology_workers / matching_workers 的钳制保持一致，P4-4）。
        ceiling = min(32, self.logical_cpu_count * 2)
        if self.forecast_workers:
            return min(self.forecast_workers, ceiling)
        return max(1, ceiling)

    def serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_dir"] = str(self.source_dir)
        data["work_dir"] = str(self.work_dir)
        data["work_root"] = str(self.work_root)
        data["as_of_date"] = self.as_of_date.isoformat()
        data["features"] = list(self.features)
        data["distance_dimensions"] = list(self.distance_dimensions)
        # max_homology_dimension 已改为派生属性，不再出现在 asdict() 中；
        # 这里显式补回，保持落库配置快照与外部消费者看到的字段集不变。
        data["max_homology_dimension"] = self.max_homology_dimension
        data["data_quality_mode"] = self.data_quality_mode
        data["topology_backend"] = self.topology_backend
        data["resolved_workers"] = {
            "topology": self.resolved_topology_workers,
            "matching": self.resolved_matching_workers,
            "forecast": self.resolved_forecast_workers,
        }
        return data

    def topology_signature(self) -> str:
        """拓扑签名：仅含改变点云 / 持续同调结果的参数，**不再含位置信息（source_dir）**。

        挪动数据目录（内容不变）时本签名保持稳定，使整管线重跑可复用既有拓扑结果，
        而非在 ``build_topology`` 处因路径变更大声失败。数据完整性由
        ``storage.check_or_set_identity`` 的 ``source_signature`` 兜底。
        """
        topology_fields = {
            "as_of_date": self.as_of_date.isoformat(),
            "window_size": self.window_size,
            "lookback_trading_days": self.lookback_trading_days,
            "min_windows": self.min_windows,
            "features": self.features,
            "max_edge_length": self.max_edge_length,
            "max_homology_dimension": self.max_homology_dimension,
            # strict 模式会跳过含重复交易日/非数值行的股票，产出的点云集合与
            # permissive 模式不同，必须纳入拓扑签名以免误用对方的缓存（P2-2）。
            "data_quality_mode": self.data_quality_mode,
            # 不同后端产出的持久图/瓶颈距离可能不同（尤其 gudhi 与 ripser/toppopp），
            # 纳入拓扑签名以换后端时自动失效旧拓扑缓存（不手动清）。
            "topology_backend": self.topology_backend,
        }
        return _sha256_signature(topology_fields, ensure_ascii=False)

    def matching_signature(self) -> str:
        # 注：top_k 已从匹配签名中移除（见步骤 A）——它只决定最终保留几条，
        # 不改变瓶颈距离计算，故不应进入匹配缓存键/目录派生。
        fields = {
            "topology": self.topology_signature(),
            "distance_dimensions": self.distance_dimensions,
            "distance_threshold_h0": self.distance_threshold_h0,
            "distance_threshold_h1": self.distance_threshold_h1,
            # 距离语义版本（v4）：源码 exact-essential-v4-topp，便携 exact-essential-v4-native。
            # 改变距离定义时必须递增，否则旧结果会被误当成有效缓存。
            "distance_algo": _distance_algo_label(self.topology_backend),
            # 拓扑后端：换后端即自动失效旧匹配缓存（不手动清缓存）。
            "topology_backend": self.topology_backend,
        }
        return _sha256_signature(fields)

    def forecast_signature(self) -> str:
        fields = {
            "matching": self.matching_signature(),
            "forecast_horizon": self.forecast_horizon,
        }
        return _sha256_signature(fields)

    def _dir_hash(self, fields: dict) -> str:
        """目录派生专用哈希：仅用于段名，剔除 source_dir 等位置相关项。

        与 topology_signature / matching_signature / forecast_signature 不同，
        本函数刻意**不含 source_dir**，使挪动数据目录（内容不变）时目录段名稳定，
        从而复用既有 work_dir 与库内匹配结果（步骤 B）。内容完整性仍由
        storage.check_or_set_identity 的 source_signature 兜底。
        """
        return _sha256_signature(fields, ensure_ascii=False, truncate=8)

    # ── 分层工作目录（按参数派生） ───────────────────────────────────────
    # 让不同参数组合的实验产物互不冲突、可横向对比；只改匹配/预测参数而拓扑参数
    # 不变时，topology_signature 一致 → artifacts.sqlite3 与 mmap 缓存复用，
    # 不必重跑昂贵的持续同调计算。层级正好对应既有的签名依赖链：
    #   topology_signature ⊃ matching_signature ⊃ forecast_signature
    # 命名采用「可读片段 + 签名短哈希」混合风格：可读片段便于人眼定位，
    # 短哈希保证不同参数组合绝不重名。

    # ── 数据集指纹：内容寻址（CAS），剔除 mtime / 时间元数据依赖 ────────────
    # 设计要点（面向未来演进，集中可调）：
    # - 指纹仅由「CSV 文件名集合 + 各文件内容摘要」决定，与路径、mtime/atime/ctime
    #   等时间元数据无关；挪动/复制/不同时刻重生成（内容不变）→ 指纹恒定 → work_dir
    #   复用，消除 relocate 失效与跨秒偶发 flake。
    # - 下方 _FP_* 为常数化可调旋钮：变更任一即改变派生命名空间（属策略 A 一次性重算，
    #   不引发静默混库）。未来新增文件类型 / 换哈希算法 / 纳入额外元数据，只需改对应
    #   旋钮或 _compute_dataset_fingerprint，调用方与 work_dir 派生不受影响。
    # - 内容完整性权威校验仍由 storage.check_or_set_identity 的 source_signature 兜底。

    _FP_CSV_SUFFIX: ClassVar[str] = ".csv"  # 纳入指纹的文件类型（扩展更多后缀改此处）
    _FP_CHUNK_SIZE: ClassVar[int] = 1 << 20  # 流式哈希分块（字节），恒定内存，按 IO 调优
    _FP_HASH_NAME: ClassVar[str] = "sha256"  # 单文件内容摘要算法（未来可换 blake3 等）
    _FP_HASH_HEX_LEN: ClassVar[int] = 16  # 截断长度（64-bit）；受 Windows MAX_PATH 约束
    _FP_READ_BUDGET_S: ClassVar[float] = 300.0  # 单文件哈希软看门狗（秒）；仅诊断 WARN，绝不中止

    def _file_content_digest(self, path: Path) -> str:
        """单文件流式内容摘要（恒定内存），指纹的内容寻址原子单元。

        算法经 ``_FP_HASH_NAME`` / ``_FP_CHUNK_SIZE`` 集中可调；不跟随符号链接、
        不越出目录读取，规避路径穿越类风险。
        """
        h = hashlib.new(self._FP_HASH_NAME)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(self._FP_CHUNK_SIZE), b""):
                h.update(chunk)
        return h.hexdigest()

    def _compute_dataset_fingerprint(self) -> str:
        """扫描 source_dir 的 CSV，按文件名排序后折叠出内容寻址数据集指纹。

        枚举口径统一收敛到模块级 ``iter_source_csv``（O-9），与 data.list_stock_files /
        preflight 审计一致；因语义与旧内联枚举完全相同，指纹输出逐字节不变。
        多文件哈希经线程池并行（O-4，IO 等待释放 GIL），但主线程按文件名排序组装，
        指纹 payload 与串行一致；进度仅在主线程经 ``as_completed`` 更新（线程安全）。
        单文件哈希异常一律跳过并计数（O-7，放宽自仅 OSError），日志就绪后 WARN；
        单文件耗时超 ``_FP_READ_BUDGET_S`` 仅记录慢文件（诊断 WARN，绝不中止——
        指纹是 work_dir 派生必需输入，中止即无目录可用）。
        空目录 / 无 CSV 返回哨兵 ``"nosrc"``。
        """
        entries = iter_source_csv(self.source_dir)
        total = len(entries)
        if total == 0:
            return "nosrc"
        sink = _FP_PROGRESS_SINK
        skipped = 0
        slow: list[str] = []

        def _digest_or_none(path: Path) -> tuple[str | None, int]:
            start = time.monotonic()
            try:
                stat = path.stat(follow_symlinks=False)
                digest = self._file_content_digest(path)
            except Exception:
                # O-7：放宽自仅 OSError → 任何异常都跳过，避免单文件崩溃拖垮整轮；
                # 指纹因跳过而变化会切换 work_dir，不会静默用错数据。
                return None, 0
            elapsed = time.monotonic() - start
            if elapsed > self._FP_READ_BUDGET_S:
                slow.append(path.name)  # 仅诊断；GIL 下 list.append 原子，读在池关闭后
            return digest, stat.st_size

        # O-4：线程池并行哈希；主线程按完成序累加 done 并回调，最终按文件名排序组装。
        records: dict[str, tuple[int, str]] = {}
        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 3)) as pool:
            future_to_name = {pool.submit(_digest_or_none, entry): entry.name for entry in entries}
            for done, future in enumerate(as_completed(future_to_name), start=1):
                name = future_to_name[future]
                digest, size = future.result()
                if digest is None:
                    skipped += 1
                else:
                    records[name] = (size, digest)
                if sink is not None:
                    sink(name, done, total)

        object.__setattr__(self, "_dataset_fp_skipped", skipped)
        object.__setattr__(self, "_dataset_fp_slow", slow)
        if not records:
            return "nosrc"
        # 按文件名排序组装 → 与串行逐字节一致（O-4 确定性保证）。
        payload = json.dumps(
            [[name, size, digest] for name, (size, digest) in sorted(records.items())],
            ensure_ascii=False,
        ).encode("utf-8")
        h = hashlib.new(self._FP_HASH_NAME)
        h.update(payload)
        return h.hexdigest()[: self._FP_HASH_HEX_LEN]

    def _dataset_fingerprint(self) -> str:
        """内容寻址数据集指纹（记忆化）。

        同 source_dir 下多次派生 work_dir 时避免重复扫盘 + 哈希；frozen 数据类借
        ``object.__setattr__`` 挂载私有缓存，不影响既有字段 / 相等性 / 序列化。
        前置：``source_dir`` 在构造后不可变（frozen 约束），故缓存恒有效。
        """
        cached = self.__dict__.get("_dataset_fp_cache")
        if cached is None:
            value = self._compute_dataset_fingerprint()
            object.__setattr__(self, "_dataset_fp_cache", value)
            return value
        return cached

    @property
    def dataset_fingerprint_skipped_count(self) -> int:
        """被指纹扫描跳过（占用 / 无权限 / 哈希异常）的 CSV 数，供日志就绪后告警（C3/O-7）。"""
        return self.__dict__.get("_dataset_fp_skipped", 0)

    @property
    def dataset_fingerprint_slow_count(self) -> int:
        """超过单文件读预算（疑似超大 / 异常）的 CSV 数，供日志就绪后告警（O-7）。"""
        return len(self.__dict__.get("_dataset_fp_slow", []))

    def _format_thresholds(self) -> str:
        """把双距离阈值格式化为最短精确十进制（0.1,0.1→'H0{0.1}_H1{0.1}'），避免科学计数法。"""
        return (
            f"H0{{{float(self.distance_threshold_h0)!r}}}"
            f"_H1{{{float(self.distance_threshold_h1)!r}}}"
        )

    def topo_segment(self) -> str:
        """拓扑段：覆盖点云与持续同调定义的全部参数（剔除 source_dir）。

        段名哈希由 ``_dir_hash`` 生成，刻意不含 ``source_dir``，使挪动数据目录
        （内容不变）时拓扑段名稳定，从而复用既有 work_dir 与拓扑产物（步骤 B）。
        """
        feat = f"feat{len(self.features)}"
        label = (
            f"w{self.window_size}_lb{self.lookback_trading_days}"
            f"_mw{self.min_windows}_{feat}"
            f"_mrl{self.max_edge_length}_{self.data_quality_mode}"
        )
        h = self._dir_hash(
            {
                "as_of_date": self.as_of_date.isoformat(),
                "window_size": self.window_size,
                "lookback_trading_days": self.lookback_trading_days,
                "min_windows": self.min_windows,
                "features": list(self.features),
                "max_edge_length": self.max_edge_length,
                "max_homology_dimension": self.max_homology_dimension,
                "data_quality_mode": self.data_quality_mode,
            }
        )
        return f"topo_{label}__{h}"

    def match_segment(self) -> str:
        """匹配段：仅随距离参数走（剔除 source_dir）。

        top_k 只影响最终保留几条、不改距离计算/缓存键（步骤 A），故不进段名；
        段名现已改为仅依赖距离参数并剔除 ``source_dir``（诊断 B 原话「match 目录
        只随距离键走」）。内容复用由 ``match_clouds`` 的 ``matching_resume_key``
        （按 mmap 内容签名 + 距离参数）负责，段名与续算键已解耦。
        """
        dims = "-".join(str(value) for value in self.distance_dimensions)
        label = f"d{dims}_thr{self._format_thresholds()}"
        h = self._dir_hash(
            {
                "distance_dimensions": self.distance_dimensions,
                "distance_threshold_h0": self.distance_threshold_h0,
                "distance_threshold_h1": self.distance_threshold_h1,
                "distance_algo": _distance_algo_label(self.topology_backend),
                "topology_backend": self.topology_backend,
            }
        )
        return f"match_{label}__{h}"

    def forecast_segment(self) -> str:
        """预测段：覆盖预测跨度（剔除 source_dir）。

        段名哈希由 ``_dir_hash`` 生成，仅依赖距离参数与预测跨度，不再沿签名链
        卷入 ``source_dir``。
        """
        h = self._dir_hash(
            {
                "distance_dimensions": self.distance_dimensions,
                "distance_threshold_h0": self.distance_threshold_h0,
                "distance_threshold_h1": self.distance_threshold_h1,
                "distance_algo": _distance_algo_label(self.topology_backend),
                "forecast_horizon": self.forecast_horizon,
                "topology_backend": self.topology_backend,
            }
        )
        return f"fc_h{self.forecast_horizon}__{h}"

    def _derived_work_dir(self) -> Path:
        """由 work_root + 参数派生嵌套工作目录。"""
        rel = (
            self.as_of_date.isoformat()
            + f"/data_{self._dataset_fingerprint()}"
            + f"/{self.topo_segment()}"
            + f"/{self.match_segment()}"
            + f"/{self.forecast_segment()}"
        )
        return (self.work_root / rel).resolve()

    def __post_init__(self) -> None:
        # work_dir 未显式提供时，按参数派生嵌套路径（见上）。frozen 数据类需借
        # object.__setattr__ 在 __post_init__ 内赋值。
        if self.work_dir is None:
            object.__setattr__(self, "work_dir", self._derived_work_dir())


def _distance_algo_label(backend: str) -> str:
    """距离语义版本标签（v4）。

    - 默认后端（ripser_topp / Topp 1.0.0）：``exact-essential-v4-topp``；
    - 自研回退（native_c_dll）：``exact-essential-v4-native``。

    改变距离定义时必须递增版本，否则旧结果会被误当成有效缓存。
    v4：引入 H0/H1 双阈值，合格判定由单一 r 改为 d0<r0 AND d1<r1。
    """
    return "exact-essential-v4-native" if backend == "native_c_dll" else "exact-essential-v4-topp"


def iter_source_csv(source_dir: Path) -> list[Path]:
    """权威的 source_dir CSV 枚举器（统一口径，O-9）。

    语义固定：``os.scandir`` + 不跟随符号链接 + 文件名小写后缀（``.csv``）+ 按 basename
    排序；与 ``PYTHONHASHSEED`` 无关。目录缺失 / 无 CSV 返回空列表（是否抛错由调用方决定）：
    config 指纹据此返回 ``"nosrc"``，data.list_stock_files 据此抛 ``FileNotFoundError`` /
    ``DataError``，preflight 审计据此得到空列表。三处原本不一致的枚举收敛于此，消除
    「同一目录被不同模块看到不同文件集」的隐患（B3）。
    """
    if source_dir is None or not source_dir.is_dir():
        return []
    return [
        Path(entry.path)
        for entry in sorted(os.scandir(source_dir), key=lambda item: item.name)
        if entry.is_file(follow_symlinks=False)
        and entry.name.lower().endswith(PipelineConfig._FP_CSV_SUFFIX)
    ]
