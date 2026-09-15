"""
TopoQuant 交互式运行脚本
在 VSCode 终端中交互输入参数 → 自动更新 config.json → 运行全流程并监控进度。

⚠ 重要：请在 VSCode 中右键 → "Run Python File in Terminal" 运行整个文件，
  不要逐行执行（流水线使用多进程 spawn，需要 if __name__ == "__main__" 保护）。
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

# ── 便携/源码双模式根目录与虚拟环境检测 ─────────────────
# _FROZEN: PyInstaller 冻结为 EXE 后为 True；直接 `python run.py` 为 False。
# _PROJECT_ROOT: 冻结版取 exe 所在目录（便携包根），源码版取 run.py 所在目录。
_FROZEN = bool(getattr(sys, "frozen", False))
_PROJECT_ROOT = (
    Path(sys.executable).resolve().parent if _FROZEN else Path(__file__).resolve().parent
)
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# 模式 B：re-exec 探针仅当 run.py 作为入口（__main__）时执行；mp.Pool(spawn) 的子进程
# 以 __mp_main__ 重新导入本模块，必须跳过探针，避免子进程重复触发 .venv 切换甚至
# 无限重 spawn。探针仍位于 topoquant 等重导入之前，保证「先切 .venv 再导入」语义不变。
if __name__ == "__main__" and (
    not _FROZEN
    and _VENV_PYTHON.is_file()
    and Path(sys.executable).resolve() != _VENV_PYTHON.resolve()
):
    import subprocess as _sp

    print("\033[33m检测到未使用项目虚拟环境，自动切换到 .venv ...\033[0m")

    # 先用一次性探针确认 .venv 解释器本身没坏（DLL 缺失时会 C 级卡死，
    # 普通 try/except 抓不住），探针才需要超时保护。
    try:
        _probe = _sp.run(
            [str(_VENV_PYTHON), "-c", "import sys; sys.exit(0)"],
            capture_output=True,
            timeout=20,
        )
    except _sp.TimeoutExpired:
        print("\033[31m\n错误: .venv 解释器启动超时（>20s），可能是运行时 DLL 卡死。")
        print(f"请执行: cd {_PROJECT_ROOT} && .venv\\Scripts\\python setup_env.py\033[0m")
        sys.exit(1)
    if _probe.returncode != 0:
        print("\033[31m\n错误: .venv 解释器不可用。")
        print(f"请执行: cd {_PROJECT_ROOT} && python setup_env.py\033[0m")
        sys.exit(1)

    # 探针通过后转交实际运行 —— 流水线可能跑几十分钟，绝不能加超时。
    result = _sp.run([str(_VENV_PYTHON), __file__] + sys.argv[1:])
    sys.exit(result.returncode)

# ── 继续正常的导入 ───────────────────────────────────

import json
import time
from datetime import date, datetime
from typing import Any, NamedTuple
import subprocess
from topoquant._logging import configure_logging, ensure_root_handler

# ── 颜色与打印助手（统一收敛到 topoquant._term，见 A1）──
# 模式 B：仅入口进程（__main__）绑定颜色助手为模块全局；spawn 子进程以 __mp_main__
# 重新导入本模块时跳过此导入，避免子进程重复导入 _term 及其潜在导入期副作用。
# 这些符号仅在函数体内使用（无模块级执行依赖），故不影响任何功能路径。
if __name__ == "__main__":
    from topoquant._term import C, c, sep, ok, warn, err, info, hint

# 实例日志文件格式（权威日志真源）现统一在 topoquant._logging.LOG_FORMAT 定义（O-6 / P1-2）。


# ── 默认值 ────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "source_dir": "./data/stock",
    "as_of_date": "2024-06-28",
    "window_size": 60,
    "lookback_trading_days": 480,
    "min_windows": 4,
    "features": ["money", "volume", "high", "close"],
    "max_edge_length": 3.0,
    "distance_dimensions": [0, 1],
    "distance_threshold_h0": 0.1,
    "distance_threshold_h1": 0.1,
    "top_k": 5,
    "forecast_horizon": 5,
    "topology_workers": 0,
    "matching_workers": 0,
    "forecast_workers": 0,
    "data_quality_mode": "permissive",
}

VALIDATORS: dict[str, str] = {
    "as_of_date": r"\d{4}-\d{2}-\d{2}",
    "features": r"^[a-z_]+(,[a-z_]+)*$",
    # 只约束"两个逗号分隔的非负整数"这一形状；具体取值（是否为 0,1）不再限定，
    # 维度是否互不相同、是否合法由 PipelineConfig.validate() 统一裁决（P1-1）。
    "distance_dimensions": r"^\d+,\d+$",
    "data_quality_mode": r"^(strict|permissive)$",
}


def load_defaults() -> dict[str, Any]:
    """从 config.example.json 加载默认值，缺失项用内置 DEFAULTS 兜底。

    以 DEFAULTS 为基座、example 覆盖其上：example 可能滞后于代码（缺少新增键，
    如 data_quality_mode），用 DEFAULTS 填补可确保 collect_params 引用的键始终存在，
    避免 KeyError（见 issue: config.example.json 缺 data_quality_mode）。
    """
    base = dict(DEFAULTS)
    example_path = Path(__file__).parent / "config.example.json"
    if example_path.is_file():
        try:
            example = json.loads(example_path.read_text(encoding="utf-8"))
            base.update(example)  # example 覆盖默认值；DEFAULTS 填补缺失键
        except (json.JSONDecodeError, OSError):
            pass
    return base


# ── 交互式输入 ────────────────────────────────────────


def prompt_str(label: str, default: Any, hint_text: str = "") -> str:
    """字符串输入（Enter 使用默认值）。"""
    d_str = str(default)
    extra = f"  {hint(hint_text)}" if hint_text else ""
    prompt = f"  {c(label, C['bold'])} [{c(d_str, C['cyan'])}]{extra}: "
    value = input(prompt).strip()
    return value if value else d_str


def prompt_int(label: str, default: int, hint_text: str = "") -> int:
    """整数输入，带校验。"""
    while True:
        raw = prompt_str(label, default, hint_text)
        if raw == str(default):
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"    {err('请输入整数')}")


def prompt_float(label: str, default: float, hint_text: str = "") -> float:
    """浮点数输入，带校验。"""
    while True:
        raw = prompt_str(label, default, hint_text)
        if raw == str(default):
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"    {err('请输入数字')}")


def prompt_date(label: str, default: str) -> str:
    """日期输入 YYYY-MM-DD。"""
    while True:
        raw = prompt_str(label, default, "格式 YYYY-MM-DD")
        try:
            date.fromisoformat(raw)
            return raw
        except ValueError:
            print(f"    {err(f'无效日期: {raw}，请用 YYYY-MM-DD 格式')}")


def prompt_list(label: str, default: list[Any], hint_text: str = "") -> list[Any]:
    """逗号分隔列表输入。"""
    while True:
        d_str = ",".join(str(x) for x in default)
        raw = prompt_str(label, d_str, hint_text)
        if raw == d_str:
            return list(default)
        raw = raw.strip()
        if not raw:
            return list(default)
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        # 尝试转为数字
        result: list[Any] = []
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                try:
                    result.append(float(p))
                except ValueError:
                    result.append(p)
        return result


def prompt_dimensions(label: str, default: list[Any], hint_text: str = "") -> list[int]:
    """瓶颈距离维度输入：恰好两个互不相同的非负整数。

    就地校验形状，避免用户填完全部参数后才在 ``PipelineConfig.validate()`` 处崩溃（P1-1）。
    这里只做"两个不同的非负整数"这一形状校验，与 :data:`VALIDATORS` 保持同一口径；
    具体维度是否可算由配置层统一裁决，交互层不再自行限定取值。
    """
    while True:
        values = prompt_list(label, default, hint_text)
        if len(values) != 2:
            print(f"    {err('请输入两个维度，用逗号分隔，如 0,1 或 1,2')}")
            continue
        if not all(isinstance(value, int) and value >= 0 for value in values):
            print(f"    {err('维度必须是非负整数，如 0,1 或 1,2')}")
            continue
        if values[0] == values[1]:
            print(f"    {err('两个维度必须互不相同')}")
            continue
        if max(values) > 1:
            print(f"    {warn(f'将计算到 H{max(values)}，耗时与内存开销明显高于 H0/H1')}")
        return [int(value) for value in values]


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """是/否确认。"""
    yn = "Y/n" if default else "y/N"
    raw = input(f"\n  {c(question, C['bold'])} [{c(yn, C['cyan'])}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "是")


def _has_flag(name: str) -> bool:
    """手工解析 sys.argv，判断是否存在某标志。

    与 run.py 现有风格一致，不引 argparse，避免破坏其用 subprocess re-exec 自身时传参方式。
    """
    return name in sys.argv


def _maybe_migrate_legacy(config: PipelineConfig) -> None:  # noqa: F821  (延迟导入；main() 内已 import)
    """首次运行自动检测旧版平铺实验目录，询问/按标志决定是否迁移到新嵌套结构。

    复用 tools.migrate_legacy_run.migrate_run 完成实际搬移，run.py 内不重复实现搬移逻辑。
    幂等：已迁移过的真实目录再次运行 need=False，不打扰用户。
    """
    legacy_dir = config.work_root / config.as_of_date.isoformat()
    need = (legacy_dir / "artifacts.sqlite3").exists() and not (
        config.work_dir / "artifacts.sqlite3"
    ).exists()
    if not need:
        return

    info(f"发现旧版平铺实验目录：{legacy_dir}")
    info(f"将迁移到新结构目录：{config.work_dir}")

    if sys.stdin.isatty():
        if prompt_yes_no("发现旧版平铺实验目录，是否迁移到新结构？", default=True):
            try:
                migrate_run(config)  # noqa: F821  (延迟导入；main() 内已 import)
            except Exception as _exc:
                err(f"迁移失败（不影响继续运行，旧平铺目录仍在）：{_exc}")
    else:
        # 非交互环境：安全优先，默认不迁移，仅 --migrate 才自动执行。
        if _has_flag("--migrate"):
            try:
                migrate_run(config)  # noqa: F821  (延迟导入；main() 内已 import)
            except Exception as _exc:
                err(f"迁移失败（不影响继续运行，旧平铺目录仍在）：{_exc}")
        elif _has_flag("--no-migrate"):
            hint("已指定 --no-migrate，跳过迁移。")
        else:
            hint("非交互环境：默认跳过迁移。如需自动迁移，请加 --migrate 参数。")


# ── 参数输入流程 ──────────────────────────────────────


def collect_params(defaults: dict[str, Any]) -> dict[str, Any]:
    """分组交互式收集全部参数。"""
    print(f"\n{c('╔══════════════════════════════════════════════════╗', C['cyan'])}")
    print(
        f"{c('║', C['cyan'])}     {c('TopoQuant 交互式参数配置', C['bold'])}{' ' * 21}{c('║', C['cyan'])}"
    )
    print(
        f"{c('║', C['cyan'])}     {c('Enter 直接回车 = 使用方括号内默认值', C['dim'])}{' ' * 10}{c('║', C['cyan'])}"
    )
    print(f"{c('╚══════════════════════════════════════════════════╝', C['cyan'])}")

    params: dict[str, Any] = {}

    # ── 第1组：数据与日期 ──
    sep("数据与日期")
    params["source_dir"] = prompt_str("行情数据目录", defaults["source_dir"], "CSV 文件所在路径")
    params["as_of_date"] = prompt_date("预测基准日", str(defaults["as_of_date"]))

    # ── 第2组：窗口参数 ──
    sep("窗口参数")
    params["window_size"] = prompt_int(
        "点云窗口大小(交易日)", int(defaults["window_size"]), "建议 40~80"
    )
    params["lookback_trading_days"] = prompt_int(
        "回溯交易日数", int(defaults["lookback_trading_days"]), f"需 ≥ {params['window_size']}"
    )
    params["min_windows"] = prompt_int(
        "最少窗口数", int(defaults["min_windows"]), "每支股票至少需几个完整窗口"
    )

    # ── 第3组：特征与拓扑 ──
    sep("特征与拓扑")
    params["features"] = prompt_list(
        "特征列", list(defaults["features"]), "逗号分隔，如 money,volume,high,close"
    )
    params["max_edge_length"] = prompt_float(
        "VR复形最大边长", float(defaults["max_edge_length"]), "通常 2.0~5.0"
    )
    # 最大同调维度不再单独询问：它已由「瓶颈距离维度」唯一推导（max(distance_dimensions)），
    # 单独配置只会与实际需求冲突（配大了白算、配小了缺维），故从交互中移除（P1-1）。

    # ── 第4组：匹配参数 ──
    sep("匹配参数")
    params["distance_dimensions"] = prompt_dimensions(
        "瓶颈距离维度",
        list(defaults["distance_dimensions"]),
        "请输入两个不同的维度，如 0,1 或 1,2",
    )
    params["distance_threshold_h0"] = prompt_float(
        "H0 距离阈值(r0)", float(defaults["distance_threshold_h0"]), "H0 瓶颈距离需 < r0"
    )
    params["distance_threshold_h1"] = prompt_float(
        "H1 距离阈值(r1)", float(defaults["distance_threshold_h1"]), "H1 瓶颈距离需 < r1"
    )
    params["top_k"] = prompt_int("Top-K 相似点云数", int(defaults["top_k"]), "多数投票用，建议奇数")

    # ── 第5组：预测 ──
    sep("预测参数")
    params["forecast_horizon"] = prompt_int(
        "预测天数", int(defaults["forecast_horizon"]), "取未来 N 天涨跌"
    )

    # ── 第6组：并发 ──
    sep("并发控制（0=自动）")
    params["topology_workers"] = prompt_int(
        "持续同调并发数", int(defaults["topology_workers"]), "进程数，0=自动(min(8,cpu-1))"
    )
    params["matching_workers"] = prompt_int(
        "瓶颈匹配并发数",
        int(defaults["matching_workers"]),
        "进程数，0=自动(min(16,cpu-1))；mmap 共享内存可放心调高",
    )
    params["forecast_workers"] = prompt_int(
        "行情预测并发数", int(defaults["forecast_workers"]), "线程数，0=自动(min(32,cpu×2))"
    )
    params["data_quality_mode"] = prompt_str(
        "数据质量模式",
        str(defaults["data_quality_mode"]),
        "重复交易日/缺列始终跳过；strict=额外跳过含非数值行的股票，permissive=非数值行照常构建",
    )

    # ── 第7组：重算控制 ──
    sep("重算控制")
    params["reset"] = prompt_yes_no(
        "清空实验库与 mmap 后从头重算（换/改股票数据时用）？", default=False
    )
    params["force_rebuild"] = prompt_yes_no(
        "若行情/配置已变化，是否授权就地清空旧数据并重算？", default=False
    )

    # 不再手动拼 work_dir；改为设置 work_root，完整路径由参数（日期/数据指纹/
    # 拓扑/匹配/预测）在 PipelineConfig 内自动派生，不同参数组合自动分流。
    params["work_root"] = "./runs"

    return params


def show_summary(params: dict[str, Any]) -> None:
    """打印参数摘要供确认。"""
    sep("参数摘要")

    groups = [
        (
            "数据与日期",
            [
                ("source_dir", "行情目录"),
                ("as_of_date", "基准日期"),
            ],
        ),
        (
            "窗口",
            [
                ("window_size", "窗口大小"),
                ("lookback_trading_days", "回溯天数"),
                ("min_windows", "最少窗口"),
            ],
        ),
        (
            "特征与拓扑",
            [
                ("features", "特征列"),
                ("max_edge_length", "最大边长"),
            ],
        ),
        (
            "匹配",
            [
                ("distance_dimensions", "距离维度"),
                ("distance_threshold_h0", "H0 距离阈值"),
                ("distance_threshold_h1", "H1 距离阈值"),
                ("top_k", "Top-K"),
            ],
        ),
        (
            "预测",
            [
                ("forecast_horizon", "预测天数"),
            ],
        ),
        (
            "并发",
            [
                ("topology_workers", "同调并发"),
                ("matching_workers", "匹配并发"),
                ("forecast_workers", "预测并发"),
                ("data_quality_mode", "数据质量模式"),
            ],
        ),
    ]

    # 最高同调维度是派生量（max(distance_dimensions)），不再是独立参数，
    # 但它直接决定 build 的计算量，必须让用户在确认前看见（P1-1）。
    derived_max_dimension = max(int(value) for value in params["distance_dimensions"])

    # 派生量单独展示，明确标注「派生」，避免与用户主动输入参数混淆。
    print(f"\n  {c('派生量', C['bold'], C['yellow'])}")
    print(
        f"    {'最高同调维度':<12} {c(str(derived_max_dimension), C['cyan'])}"
        f"  {hint('（= max(distance_dimensions)，决定 build 计算量）')}"
    )

    for group_name, fields in groups:
        print(f"\n  {c(group_name, C['bold'], C['yellow'])}")
        for key, label in fields:
            value = params[key]
            if isinstance(value, list):
                value = ", ".join(str(x) for x in value)
            print(f"    {label:<12} {c(str(value), C['cyan'])}")

    # 重算控制单独展示：reset 是破坏性动作，必须让用户在确认前看见。
    print(f"\n  {c('重算控制', C['bold'], C['yellow'])}")
    _reset = bool(params.get("reset", False))
    _force = bool(params.get("force_rebuild", False))
    print(
        f"    {'清空重算':<12} "
        + (
            c("是（将删除实验库与 mmap，从头计算）", C["red"], C["bold"])
            if _reset
            else c("否", C["cyan"])
        )
    )
    print(
        f"    {'授权就地清空':<12} "
        + (c("是（签名不匹配时自动清空旧数据）", C["yellow"]) if _force else c("否", C["cyan"]))
    )

    print(f"\n  {hint('工作根目录:')} {c(params.get('work_root', './runs'), C['cyan'])}")
    print(
        f"  {hint('完整路径:')} {c('按 日期/数据指纹/拓扑/匹配/预测 自动拼接（见运行日志）', C['cyan'])}"
    )


# ── 主流程 ────────────────────────────────────────────


def _try_sqlite_access(db_path: Path) -> bool:
    """尝试打开 SQLite 数据库，返回是否成功。"""
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path), timeout=1)
        # 对齐 storage.connect 的写竞争退避，避免 WAL 并发下 SELECT 1 误报
        # OperationalError 而被上层误判为「数据库不可访问」（仅探针，不建表/不改 schema）。
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("SELECT 1")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


# ── 进程发现与终止 ────────────────────────────────────
# 【P0 安全修复】原实现用 `tasklist /FI "IMAGENAME eq python.exe"` 取到机器上**全部**
# python.exe 的 PID，仅排除自身后就 `taskkill /F`，会连带杀掉 Jupyter、IDE、其它项目
# 的 Python 进程，造成不可逆的数据丢失。现改为三重保险：
#   1) 精确识别——进程命令行必须同时命中「本项目路径」与「run.py/pipeline.py/topoquant」
#      特征关键字，且排除自身及其祖先进程、系统保留 PID；
#   2) 逐项确认——终止前打印完整进程信息（PID/命令行/创建时间/状态）并要求用户确认；
#   3) 优雅退出——先请求正常终止，等待 5 秒仍存活才强制杀死。
# 依赖策略：若环境中恰好装有 psutil 则优先使用（信息最全）；未安装时退回系统命令
# （Windows: PowerShell + Win32_Process；POSIX: ps），不新增任何硬依赖。

_PROCESS_KEYWORDS = ("run.py", "pipeline.py", "topoquant")
# 系统保留 PID，任何情况下都不得终止（0=Idle/内核，1=init/systemd，4=Windows System）
_PROTECTED_PIDS = frozenset({0, 1, 4})
_KILL_GRACE_SECONDS = 5.0


class ProcInfo(NamedTuple):
    """进程快照的最小信息集合（用于展示与判定）。"""

    pid: int
    ppid: int
    name: str
    cmdline: str
    created: str
    status: str

    def describe(self) -> str:
        command = self.cmdline if len(self.cmdline) <= 160 else self.cmdline[:157] + "..."
        return (
            f"PID {self.pid} (父 {self.ppid})  状态 {self.status or '未知'}  "
            f"创建于 {self.created or '未知'}\n      命令行: {command}"
        )


def _snapshot_processes() -> list[ProcInfo]:
    """采集当前机器上的 Python 进程快照；任何失败都返回空列表（宁可不杀，不可错杀）。"""
    for collector in (_snapshot_via_psutil, _snapshot_via_system):
        try:
            infos = collector()
        except Exception:  # 采集失败不应中断主流程
            infos = []
        if infos:
            return infos
    return []


def _snapshot_via_psutil() -> list[ProcInfo]:
    """可选路径：环境中已安装 psutil 时使用（信息最完整、无需起子进程）。"""
    import psutil  # 可选依赖：未安装时由调用方回退到系统命令

    infos: list[ProcInfo] = []
    fields = ["pid", "ppid", "name", "cmdline", "create_time", "status"]
    for proc in psutil.process_iter(fields):
        try:
            data = proc.info
            name = str(data.get("name") or "")
            parts = data.get("cmdline") or []
            cmdline = " ".join(str(item) for item in parts)
            if "python" not in name.lower() and "python" not in cmdline.lower():
                continue
            created = data.get("create_time")
            infos.append(
                ProcInfo(
                    pid=int(data.get("pid") or 0),
                    ppid=int(data.get("ppid") or 0),
                    name=name,
                    cmdline=cmdline,
                    created=""
                    if not created
                    else datetime.fromtimestamp(created).isoformat(timespec="seconds"),
                    status=str(data.get("status") or ""),
                )
            )
        except Exception:  # 进程在遍历期间退出属于常态，跳过即可
            continue
    return infos


def _snapshot_via_system() -> list[ProcInfo]:
    """兜底路径：仅用标准库 + 系统自带命令采集，不引入新依赖。"""
    import subprocess as _sp

    if sys.platform == "win32":
        # Win32_Process 才带 CommandLine；tasklist 没有命令行，无法做精确匹配。
        script = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -like 'python*' } | "
            "Select-Object ProcessId,ParentProcessId,Name,CommandLine,"
            "@{n='Created';e={ if ($_.CreationDate) { $_.CreationDate.ToString('s') } else { '' } }} | "
            "ConvertTo-Json -Compress"
        )
        result = _sp.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
        payload = (result.stdout or "").strip()
        if not payload:
            return []
        data = json.loads(payload)
        if isinstance(data, dict):  # 只有一个进程时 PowerShell 返回对象而非数组
            data = [data]
        infos: list[ProcInfo] = []
        for item in data:
            try:
                infos.append(
                    ProcInfo(
                        pid=int(item.get("ProcessId") or 0),
                        ppid=int(item.get("ParentProcessId") or 0),
                        name=str(item.get("Name") or ""),
                        cmdline=str(item.get("CommandLine") or ""),
                        created=str(item.get("Created") or ""),
                        status="running",
                    )
                )
            except (TypeError, ValueError):
                continue
        return infos

    result = _sp.run(
        ["ps", "-eo", "pid=,ppid=,etime=,stat=,args="],
        capture_output=True,
        text=True,
        timeout=20,
    )
    infos = []
    for line in (result.stdout or "").splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid_text, ppid_text, elapsed, state, command = parts
        if "python" not in command.lower():
            continue
        try:
            pid, ppid = int(pid_text), int(ppid_text)
        except ValueError:
            continue
        infos.append(ProcInfo(pid, ppid, "python", command, f"已运行 {elapsed}", state))
    return infos


def _ancestor_pids(infos: list[ProcInfo], pid: int) -> set[int]:
    """回溯自身的祖先 PID 集合。

    run.py 会在非 .venv 解释器下自我重启，此时父进程的命令行同样是
    ``python .../run.py``，若不排除就会「自杀式」终止自己的父进程。
    """
    parents = {info.pid: info.ppid for info in infos}
    seen: set[int] = set()
    current = parents.get(pid, 0)
    while current and current not in seen:
        seen.add(current)
        current = parents.get(current, 0)
    return seen


def _is_topoquant_process(info: ProcInfo, work_dir: Path | None = None) -> bool:
    """判定某进程是否确属本项目（命中项目路径 **且** 命中特征关键字）。"""
    if info.pid in _PROTECTED_PIDS or info.pid == os.getpid():
        return False
    cmdline = (info.cmdline or "").replace("\\", "/").lower()
    if not cmdline:
        # 拿不到命令行就无法证明它属于本项目，一律不作为候选（宁可漏杀）。
        return False

    # 项目路径锚点：项目根目录 + run.py 规范路径 +（可选）本次工作目录。
    anchors = {
        os.path.realpath(str(_PROJECT_ROOT)).replace("\\", "/").lower(),
        os.path.realpath(str(_PROJECT_ROOT / "run.py")).replace("\\", "/").lower(),
    }
    if work_dir is not None:
        anchors.add(os.path.realpath(str(work_dir)).replace("\\", "/").lower())
    if not any(anchor and anchor in cmdline for anchor in anchors):
        return False
    return any(keyword in cmdline for keyword in _PROCESS_KEYWORDS)


def _find_topoquant_process_infos(work_dir: Path | None = None) -> list[ProcInfo]:
    """查找确属本项目的 Python 进程（带完整信息，供确认与日志使用）。"""
    infos = _snapshot_processes()
    if not infos:
        return []
    excluded = _ancestor_pids(infos, os.getpid()) | {os.getpid()}
    return sorted(
        (
            info
            for info in infos
            if info.pid not in excluded and _is_topoquant_process(info, work_dir)
        ),
        key=lambda item: item.pid,
    )


def _find_topoquant_processes(work_dir: Path | None = None) -> list[int]:
    """查找占用当前数据库的 run.py / topoquant 进程 PID。

    返回值结构保持不变（PID 列表）；``work_dir`` 为新增的可选参数，
    传入后会把「命令行包含本次工作目录」也作为项目归属的判据之一。
    """
    return [info.pid for info in _find_topoquant_process_infos(work_dir)]


def _pid_alive(pid: int) -> bool:
    """跨平台判断进程是否存活（仅用于终止后的存活轮询）。"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
        if not handle:
            return False
        exit_code = ctypes.c_uint32()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


# ── 单实例互斥：同一 work_dir 只允许一个 run.py 实例运行 ───────────────
# 环境根因：用户两次运行 run.py 指向同一 work_dir 时，两个进程会同时争抢
#   同一 SQLite 与 mmap 导出锁，触发 "database is locked" → 匹配阶段卡死/崩溃
#   （对应 B 项「瓶颈匹配阶段卡死」的环境诱因，已通过 pipeline 层重试/隔离缓解，
#    此处从源头杜绝：不让第二个实例启动）。
# 做法：在 work_dir 下用 O_CREAT|O_EXCL 原子创建 .run_instance.lock 并写入 PID；
#   - 创建成功           → 持锁，进程退出时（atexit）释放；
#   - 已存在且持有者存活 → 直接拒绝启动（避免误入 5.5 的「DB 锁」提示去杀掉对方）；
#   - 已存在但持有者已死 → 视为崩溃残留锁抢占，保证不会永久死锁。
INSTANCE_LOCK_NAME = ".run_instance.lock"


class _InstanceLock:
    """持锁句柄；``release()`` 幂等，可安全注册到 atexit。"""

    __slots__ = ("fd", "path")

    def __init__(self, path: Path, fd: int):
        self.path = path
        self.fd = fd

    def release(self) -> None:
        fd = getattr(self, "fd", -1)
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            self.fd = -1
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _read_instance_owner(lock_path: Path) -> int:
    try:
        with open(str(lock_path), encoding="utf-8") as fh:
            return int((fh.read() or "0").strip() or 0)
    except (OSError, ValueError):
        return 0


def _process_is_run_instance(pid: int) -> bool:
    """尽力判断 PID 是否为本项目的 run.py 实例。

    无法判定时保守返回 True：宁可误拒第二个实例，也绝不误抢一个正在运行的实例锁
    （误抢会让两个实例同时跑 → 正中本次要杜绝的「双实例争锁」死局）。
    """
    try:
        import psutil
    except Exception:
        return True
    try:
        proc = psutil.Process(pid)
        cmd = " ".join(proc.cmdline())
    except Exception:
        return True
    if not cmd:
        return True
    return ("run.py" in cmd) or ("topoquant" in cmd.lower()) or (_PROJECT_ROOT.name in cmd)


def _acquire_instance_lock(work_dir: Path) -> _InstanceLock:
    """获取 work_dir 级单实例锁；若已有存活实例则直接退出，不返回。

    Returns:
        持锁句柄；调用方应将其 ``release()`` 注册到 atexit（见 main()）。
    """
    lock_path = work_dir / INSTANCE_LOCK_NAME
    work_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    # 至多重试一次：处理「抢占到过期锁后重新原子创建」的竞态。
    for _attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{pid}\n".encode())
            return _InstanceLock(lock_path, fd)
        except FileExistsError:
            owner = _read_instance_owner(lock_path)
            owner_alive = bool(owner) and _pid_alive(owner)
            is_ours = owner_alive and _process_is_run_instance(owner)
            if is_ours:
                print(f"\n{err('检测到另一个 TopoQuant 实例正在运行（work_dir 单实例锁被占用）')}")
                print(f"  {hint(f'持有实例 PID: {owner}')}")
                print(f"  {hint(f'锁文件: {lock_path}')}")
                print(f"  {hint('同一 work_dir 不允许并发运行，否则会争抢数据库锁导致卡死。')}")
                print(f"  {hint('请先结束已有实例；若确为上次崩溃残留，可删除上述锁文件后重试。')}")
                sys.exit(1)
            # 过期/残留/不可判定 → 抢占后重试创建。
            try:
                os.unlink(str(lock_path))
            except OSError:
                pass
            continue
    raise RuntimeError(f"无法获取单实例锁（持续被占用）: {lock_path}")


def _signal_process(pid: int, *, force: bool) -> None:
    """向进程发送终止信号：``force=False`` 为优雅退出，``force=True`` 为强杀。"""
    import signal as _signal
    import subprocess as _sp

    if sys.platform == "win32":
        # Windows 没有 SIGTERM 语义：不带 /F 的 taskkill 相当于请求正常关闭，
        # 带 /F 才是强制终止。/T 一并处理子进程（流水线的 worker 进程）。
        command = ["taskkill", "/PID", str(pid), "/T"] + (["/F"] if force else [])
        proc = _sp.run(command, capture_output=True, timeout=10)
        # 不再静默吞掉退出码。此前本行的 "/T" 曾被误改成 "/TD:\\TDA_IMP\\..."（一个数据
        # 路径被粘进了参数里），taskkill 每次都以 returncode=1「无效参数/选项」立即失败；
        # 而调用方只凭 _pid_alive 判断结果，于是表现为「优雅终止 → 干等 5s → 强制终止 →
        # 仍在运行，请手动处理」，残留进程永远杀不掉，且屏幕上没有任何线索。
        # 128 = 目标进程已不存在（对方自己退了），属正常竞态，不算失败。
        if proc.returncode not in (0, 128):
            detail = (proc.stderr or proc.stdout or b"").decode("mbcs", errors="replace").strip()
            raise _sp.SubprocessError(
                f"taskkill 返回 {proc.returncode}：{detail or '(无输出)'}；命令={command}"
            )
        return
    os.kill(pid, _signal.SIGKILL if force else _signal.SIGTERM)


def _confirm_kill(info: ProcInfo, approve_all: bool) -> str:
    """逐项确认，返回 ``yes`` / ``no`` / ``all`` / ``cancel``。"""
    if approve_all:
        return "yes"
    print(f"\n  {warn('待终止进程：')}")
    print(f"      {info.describe()}")
    print(f"  {hint('[y] 终止  [n] 跳过  [a] 全部终止  [c] 取消整个操作')}")
    raw = input(f"  {c('请选择', C['bold'])} [{c('n', C['cyan'])}]: ").strip().lower()
    if raw in ("y", "yes", "是"):
        return "yes"
    if raw in ("a", "all", "全部"):
        return "all"
    if raw in ("c", "cancel", "q", "quit", "取消", "退出"):
        return "cancel"
    return "no"


def _kill_processes(
    pids: list[int],
    infos: list[ProcInfo] | None = None,
    assume_yes: bool = False,
) -> list[ProcInfo]:
    """在用户逐项确认后，优雅（必要时强制）终止指定 PID 的进程。

    保留原有调用形式 ``_kill_processes(pids)``；``infos`` 为可选的进程详情
    （缺省时现场补采，用于展示命令行）；``assume_yes`` 供非交互场景跳过确认。

    Returns:
        被成功终止的进程信息列表（新增返回值，仅用于日志，不影响既有调用方）。
    """
    if not pids:
        return []
    detail = {info.pid: info for info in (infos or _snapshot_processes())}
    killed: list[ProcInfo] = []
    approve_all = assume_yes
    for pid in pids:
        if pid in _PROTECTED_PIDS or pid == os.getpid():
            continue  # 双保险：即便调用方传入也拒绝终止系统进程与自身
        info = detail.get(pid) or ProcInfo(pid, 0, "", "(命令行未知)", "", "")
        decision = _confirm_kill(info, approve_all)
        if decision == "cancel":
            print(f"  {warn('已取消进程终止操作')}")
            return killed
        if decision == "all":
            approve_all = True
        elif decision == "no":
            print(f"  {hint(f'已跳过 PID {pid}')}")
            continue
        try:
            _signal_process(pid, force=False)  # 第一步：请求优雅退出
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  {err(f'PID {pid} 优雅终止失败：{exc}')}")
        deadline = time.monotonic() + _KILL_GRACE_SECONDS
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _pid_alive(pid):  # 第二步：宽限期内仍存活才强杀
            print(f"  {hint(f'PID {pid} 未在 {int(_KILL_GRACE_SECONDS)}s 内退出，改为强制终止')}")
            try:
                _signal_process(pid, force=True)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"  {err(f'PID {pid} 强制终止失败：{exc}')}")
        if _pid_alive(pid):
            print(f"  {err(f'PID {pid} 仍在运行，请手动处理')}")
        else:
            killed.append(info)
            print(f"  {ok(f'PID {pid} 已终止')}")
    return killed


def _check_dependencies(dep_checks: list[tuple[str, str, str]]) -> bool:
    """前置健康检查：逐个子进程导入依赖（带超时，规避 GUDHI DLL 卡死）。

    用 rich 进度条实时展示「已检查 / 总共」的检查进度；任一依赖失败即打印原因与
    解决方式并返回 False，由调用方决定是否退出（保留原交互式退出语义，不静默吞错）。
    """
    from rich.console import Console as _Console
    from rich.progress import (
        BarColumn,
        Progress as _RichProgress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    with _RichProgress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=_Console(),
    ) as prog:
        task = prog.add_task("环境依赖检测", total=max(1, len(dep_checks)))
        for pkg, desc, fix in dep_checks:
            prog.update(task, description=f"检查 {pkg} ({desc})")
            try:
                check = subprocess.run(
                    [sys.executable, "-c", f"import {pkg}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except subprocess.TimeoutExpired:
                prog.stop()
                print(f"  {err(f'{pkg} 导入超时（>15s，可能是 DLL 卡死）')}")
                print(f"    {hint(f'解决: {fix}')}")
                return False
            except Exception as _exc:  # 非 KeyboardInterrupt 的意外
                prog.stop()
                print(f"  {err(f'{pkg} 检查异常: {_exc}')}")
                return False
            if check.returncode != 0:
                stderr_tail = check.stderr.strip()[-200:] or "(无错误输出)"
                prog.stop()
                print(f"  {err(f'{pkg} 导入失败')}")
                print(f"    {hint(f'原因: {stderr_tail}')}")
                print(f"    {hint(f'解决: {fix}')}")
                return False
            prog.advance(task)
    return True


def _run_prelaunch_checks(config, console) -> object:
    """预启动非交互检查，用进度条呈现，消除「工作目录打印后长时间无反馈」的假死观感。

    - 遗留迁移（_maybe_migrate_legacy）：进度条内；
    - 数据库锁检测：保留原逻辑、置于进度条之外——该路径可能交互 input()，
      若夹在 Live 重绘中间会干扰输入提示，且其自身已有明确横幅输出；
    - 预启动环境检测（inspect_environment + render_preflight）：进度条内。

    返回 inspect_environment 的 report，供调用方做 ensure_ready。
    """
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TaskProgressColumn,
    )

    # 本函数由 main() 调用，而 main() 对 topoquant 的导入是「局部变量」，
    # 模块级（本函数可见的全局）命名空间里并不存在 inspect_environment /
    # render_preflight。嵌套函数调用不继承调用方的局部变量，故此处显式导入。
    # 此时 main() 已加载完顶层 topoquant 包，sys.modules 命中、开销极小。
    from topoquant.cli import render_preflight
    from topoquant.preflight import inspect_environment

    # 1) 遗留迁移（进度条内）
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as _pre:
        _pt = _pre.add_task("预启动：遗留迁移", total=1)
        _maybe_migrate_legacy(config)
        _pre.advance(_pt, 1)

    # 2) 数据库锁检测（原逻辑；可能交互 input()，故在进度条之外以纯文本呈现）
    _db_path = config.database_path
    _wal_path = Path(str(_db_path) + "-wal")
    _shm_path = Path(str(_db_path) + "-shm")
    if _wal_path.exists() or _shm_path.exists():
        # 先判定「是否真有活动锁」，而非仅凭文件存在就报警。
        # Windows + WAL 模式下，干净结束的运行常残留空的 -wal/-shm，属无害残留，
        # 不应每次重跑都打断用户。
        _locked = False
        try:
            _locked = not _try_sqlite_access(_db_path)
        except Exception:
            _locked = False
        # _try_sqlite_access 以最后一个连接打开又关闭时，SQLite 可能已自行
        # checkpoint 并删除 WAL/SHM；此处重算 WAL 是否为空（无已提交未落盘帧）。
        _wal_is_empty = (not _wal_path.exists()) or (_wal_path.stat().st_size == 0)

        if (not _locked) and _wal_is_empty:
            # 无害残留：静默清理后继续，不打断用户。
            for _f in (_wal_path, _shm_path):
                try:
                    _f.unlink(missing_ok=True)
                except OSError:
                    pass
            print(f"  {hint('检测到 WAL/SHM 残留（无活动锁、WAL 为空），已自动清理并继续')}\n")
        else:
            # ── 原 P0 危险路径：仅在「库被占用」或「WAL 非空(疑似崩溃热日志)」时进入 ──
            print(f"\n{warn('检测到数据库锁文件（可能上次进程崩溃或仍在运行）：')}")
            print(f"  {hint(str(_wal_path))}")
            print(f"  {hint(str(_shm_path))}")
            print()
            print(f"  {c('[1]', C['cyan'])} 清理残留 → 继续运行（丢弃上次未提交的数据）")
            print(f"  {c('[q]', C['cyan'])} 退出 → 稍后手动处理")
            choice = input(f"\n  {c('请选择', C['bold'])} [{c('1', C['cyan'])}]: ").strip().lower()
            if choice in ("q", "quit", "退出"):
                print(f"\n{warn('已取消')}")
                sys.exit(0)

            if _locked:
                # 只在「命令行确实指向本项目」的进程里挑选候选，绝不波及其它 Python 进程。
                _procs = _find_topoquant_process_infos(config.work_dir)
                if _procs:
                    _pid_str = ", ".join(str(p.pid) for p in _procs)
                    print(
                        f"\n  {warn(f'检测到 {len(_procs)} 个本项目进程可能持有数据库锁 (PID: {_pid_str})')}"
                    )
                    print(f"  {c('[1]', C['cyan'])} 逐个确认并终止这些进程 + 清理锁文件")
                    print(f"  {c('[q]', C['cyan'])} 退出")
                    choice2 = (
                        input(f"\n  {c('请选择', C['bold'])} [{c('1', C['cyan'])}]: ")
                        .strip()
                        .lower()
                    )
                    if choice2 in ("q", "quit", "退出"):
                        print(f"\n{warn('已取消')}")
                        sys.exit(0)
                    _killed = _kill_processes([p.pid for p in _procs], infos=_procs)
                    if _killed:
                        print(f"  {ok(f'已终止 {len(_killed)} 个进程')}")
                    else:
                        print(f"  {warn('未终止任何进程；若锁仍存在，清理锁文件可能失败')}")
                else:
                    # 找不到本项目进程 ≠ 可以乱杀，仅提示由用户自行判断。
                    print(f"\n  {warn('数据库仍被占用，但未发现属于本项目的 Python 进程。')}")
                    print(f"  {hint('可能是其它用户/程序打开了该文件；请手动确认后重试。')}")

            # 清理 WAL/SHM 文件
            for _f in (_wal_path, _shm_path):
                try:
                    _f.unlink(missing_ok=True)
                except OSError:
                    pass
            print(f"  {ok('锁文件已清理')}\n")

    # 3) 预启动环境检测（进度条内）
    #    进度条总任务量 = 行情 CSV 文件数，由 inspect_environment 内部的表头审计
    #    按文件节流上报（每 ~2% 一次 + 末次），从而环境检测阶段不再恒为 0%、
    #    假死观感消除。轻量步骤（依赖版本/内存/磁盘）瞬时完成，不单独计帧。
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as _pre:
        _pt = _pre.add_task("预启动：环境检测", total=1)

        def _on_progress(stage, current, total, stats):
            if stage != "audit" or not total:
                return
            # 首个审计帧时把任务总工作量从占位 1 校正为真实文件数。
            if _pre.tasks[_pt].total != total:
                _pre.update(_pt, total=total)
            _pre.update(_pt, completed=current)

        report = inspect_environment(
            config,
            require_source=True,
            require_database=False,
            require_gudhi=(config.topology_backend == "gudhi"),
            progress=_on_progress,
        )
        # 兜底：若审计被跳过（无 CSV / require_source=False），保证收敛到 100%。
        _pre.update(_pt, completed=(_pre.tasks[_pt].total or 1))
        render_preflight(report)
        _pre.advance(_pt, 1)
    return report


def main() -> None:
    print(c("\n  TopoQuant · 持续同调股票点云实验流水线\n", C["bold"], C["cyan"]))

    # 1. 加载默认值
    defaults = load_defaults()

    # 2. 交互式收集参数
    try:
        params = collect_params(defaults)
    except (KeyboardInterrupt, EOFError):
        print(f"\n{warn('已取消')}")
        return

    # 3. 显示摘要并确认
    show_summary(params)
    if not prompt_yes_no("确认以上参数并开始运行？"):
        print(f"\n{warn('已取消')}")
        return

    # 4. 写入 config.json
    #    reset / force_rebuild 是"本次运行的动作"，不是实验口径配置，
    #    落盘会误导下次运行（看到 reset=true 以为是常设项），故剔除。
    config_path = Path(__file__).parent / "config.json"
    persisted = {k: v for k, v in params.items() if k not in ("reset", "force_rebuild")}
    config_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ok(f'配置已写入 {config_path}')}")
    print(f"  {hint('如需恢复默认: 删除或修改此文件后重新运行本脚本')}")

    # 5. 导入 topoquant 并运行
    print(f"\n{c('[阶段 1/4] 环境与依赖检查', C['bold'], C['cyan'])}")
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "src"))

    # ── 前置健康检查：子进程导入 + 超时保护 ──
    # 原因: import gudhi 在 Windows 缺 MSVC 运行时时会 C 级卡死，
    #       普通 try/except 抓不住。用子进程+超时来检测。
    # 融合升级：拓扑后端默认 ripser_topp，不再硬要求 GUDHI；
    # 仅当 config.json 显式选用 gudhi 后端时才检查 GUDHI。
    _dep_backend = "ripser_topp"
    try:
        _dep_cfg_path = Path(__file__).parent / "config.json"
        if _dep_cfg_path.is_file():
            _dep_cfg_raw = json.loads(_dep_cfg_path.read_text(encoding="utf-8"))
            _dep_backend = str(_dep_cfg_raw.get("topology_backend", "ripser_topp"))
    except Exception:
        pass

    _DEP_CHECKS = [
        ("numpy", "数组与标准化", "pip install numpy"),
        ("pandas", "CSV 与数据处理", "pip install pandas"),
        ("rich", "终端进度显示", "pip install rich"),
    ]
    if _dep_backend == "gudhi":
        _DEP_CHECKS.append(
            (
                "gudhi",
                "持续同调计算（GUDHI 后端）",
                "GUDHI DLL 缺失，请安装 Visual C++ Redistributable:\n"
                "  https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "  安装后重启终端再试",
            )
        )
    elif _dep_backend == "ripser_topp":
        _DEP_CHECKS.append(("ripser", "持续同调计算（Ripser 后端）", "pip install ripser>=0.6.15"))
        _DEP_CHECKS.append(("topp", "瓶颈距离（Topp 高速后端）", "pip install topp==1.0.0"))
    elif _dep_backend == "native_c_dll":
        _DEP_CHECKS.append(("ripser", "持续同调计算（Ripser 后端）", "pip install ripser>=0.6.15"))

    if not _check_dependencies(_DEP_CHECKS):
        print(f"\n{err('依赖检查未通过，无法继续。')}")
        print(f"  {hint('也可运行 setup_env.py 自动修复环境')}")
        sys.exit(1)

    # ── 逐模块导入 topoquant（前置检查已确认依赖可用） ──
    # 首次导入会拉起 pipeline.py(156KB)+numpy/pandas/scipy/ripser/topp，可能耗时数秒且
    # 无任何输出；用 console.status 显示 spinner，消除「依赖检查 100% 后黑屏」观感（O-3）。
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    try:
        with console.status("[bold blue]正在加载流水线模块...[/]", spinner="dots"):
            from topoquant.cli import render_preflight, run_pipeline_from_config  # noqa: F401
            from topoquant.config import PipelineConfig
            from topoquant.preflight import ensure_ready, inspect_environment  # noqa: F401
            from tools.migrate_legacy_run import migrate_run  # noqa: F401  (可用性探针：仅验证可导入)
    except ImportError as _exc:
        print(f"\n{err('导入失败，请运行 setup_env.py 修复环境')}")
        print(f"  {hint(str(_exc))}")
        sys.exit(1)

    # [阶段 2/4] 扫描数据集指纹（横幅在指纹 Progress 之前、且独立于任何 Live，避免撕裂）。
    print(f"\n{c('[阶段 2/4] 扫描数据集指纹', C['bold'], C['cyan'])}")

    # 5.5 数据库锁检测所需的配置加载（锁检测本身已迁入 _run_prelaunch_checks，
    #       防止上一个失败进程残留锁；单实例互斥见 5.4）。
    # 指纹计算（对 source_dir 全部 CSV 做内容哈希）是一段沉默、代价高昂的 I/O+CPU 段，
    # 此前无任何进度输出，造成「依赖检查 100% 后黑屏假死」观感。这里用独立的 rich Progress
    # 实时展示「N/total」，不走 logging（此刻根 logger 尚未挂 handler，日志会被整段丢弃）。
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as _fp_progress:
        _fp_task = _fp_progress.add_task("数据指纹：扫描 CSV 内容哈希", total=None)

        def _fp_sink(name: str, done: int, total: int) -> None:
            # 每次调用都重设 total（首调用即确定），completed 实时推进；
            # 文件名取末尾 40 字符，避免长路径冲掉进度条（O-1）。
            _fp_progress.update(
                _fp_task,
                total=total,
                completed=done,
                description=f"数据指纹 {done}/{total} · {name[-40:]}",
            )

        config = PipelineConfig.from_json(config_path, progress=_fp_sink)
    # C3：被锁/无权限的 CSV 被静默跳过会污染内容寻址指纹（后续 source_signature 可能拒收）；
    # 此刻实例日志与控制台均已就绪，显式告警被跳过数，不改变「跳过」语义。
    _fp_skipped = config.dataset_fingerprint_skipped_count
    if _fp_skipped:
        console.print(
            warn(
                f"有 {_fp_skipped} 个 CSV 因被其它程序占用/无权限被跳过（未纳入数据指纹）；"
                "若该目录正被写入，本次结果可能不一致。"
            )
        )

    # 5.4 单实例互斥：在一切 DB 访问之前先占锁，从源头阻止「双实例争锁」。
    #      若已有存活实例持锁，此处直接退出，不会进入下面的 DB 锁提示去误杀对方。
    _instance_lock = _acquire_instance_lock(config.work_dir)
    atexit.register(_instance_lock.release)
    # 每实例日志文件：按 work_dir 隔离，多实例并发时日志互不交错、各自有序。
    _instance_log_handler = configure_logging("instance", work_dir=config.work_dir)
    if _instance_log_handler is not None:
        atexit.register(_instance_log_handler.close)

    print(f"\n  {ok('工作目录（按参数派生）:')} {config.work_dir}")
    # [阶段 3/4] 预启动非交互检查（横幅在预检 Progress 之外，避免撕裂）。
    print(
        f"\n{c('[阶段 3/4] 预启动检查（遗留迁移 / 数据库锁检测 / 环境预检）', C['bold'], C['cyan'])}"
    )
    # 预启动非交互检查（遗留迁移 / 数据库锁检测 / 环境预检）用进度条呈现，
    # 消除「工作目录打印后长时间无反馈」的假死观感；详见 _run_prelaunch_checks。
    report = _run_prelaunch_checks(config, console)

    try:
        ensure_ready(report)
    except Exception as e:
        console.print(Panel(str(e), title="预检失败", border_style="red"))
        sys.exit(2)

    # 7. 运行全流程：健康检查通过后将执行委托给 topoquant.cli.run_pipeline_from_config，
    #    复用 CLI 的进度条 / 渲染逻辑，避免与 CLI 重复实现（见 A2）。
    sep()
    # [阶段 4/4] 运行全流程（横幅在 run_pipeline_from_config 启动其 Live 之前，避免撕裂）。
    print(f"\n{c('[阶段 4/4] 运行全流程', C['bold'], C['cyan'])}\n")
    run_pipeline_from_config(
        config,
        reset=bool(params.get("reset", False)),
        force_rebuild=bool(params.get("force_rebuild", False)),
        force_content_hash=False,
        # O-8：复用本路径已在 _run_prelaunch_checks 算好的 report，避免重复审计 source_dir。
        preflight_report=report,
    )


def _run_special_mode() -> int | None:
    """处理便携版依赖探针和原生扩展自检，不进入交互界面。

    由 build_portable.ps1 在冻结（EXE）构建后调用：
      - ``--portable-self-test``            拓扑内核自检
      - ``--portable-pipeline-self-test``   端到端多进程流水线自检
    源码模式下同样可手动调用以验证逻辑正确性。
    """
    # 1) 依赖探针（_FROZEN 模式下健康检查的子进程会用到）
    # 模式 I-3：仅允许白名单模块，拒绝任意字符串 —— 防止 import 任意模块触发其导入期
    # 副作用 / 代码执行。白名单覆盖 TopoQuant 声明依赖 + 项目包；可用环境变量
    # TOPOQUANT_DEPENDENCY_PROBE_ALLOW（逗号分隔）追加，无需改码即可扩展。
    if len(sys.argv) == 3 and sys.argv[1] == "--dependency-probe":
        import importlib

        _probe_module = sys.argv[2]
        _probe_allow = {
            "numpy",
            "pandas",
            "scipy",
            "rich",
            "ripser",
            "gudhi",
            "topoquant",
            "tools",
        }
        _extra = os.environ.get("TOPOQUANT_DEPENDENCY_PROBE_ALLOW", "")
        if _extra:
            _probe_allow |= {m.strip() for m in _extra.split(",") if m.strip()}
        if _probe_module.split(".")[0] not in _probe_allow:
            print(f"dependency-probe 拒绝未授权模块: {_probe_module}")
            return 2
        importlib.import_module(_probe_module)
        return 0

    # 2) 拓扑内核自检：单点云持久同调 + 一对持久图的瓶颈距离
    if len(sys.argv) == 2 and sys.argv[1] == "--portable-self-test":
        import numpy as np

        from topoquant.topology import bottleneck_distance, compute_persistence

        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        diagrams = compute_persistence(points, 2.0, 2)
        distance = bottleneck_distance(
            np.array([[0.0, 1.0]]),
            np.array([[0.0, 1.2]]),
        )
        if set(diagrams) != {0, 1, 2} or not 0.19 <= distance <= 0.21:
            raise RuntimeError(
                f"便携版拓扑后端自检结果异常: diagrams={set(diagrams)} distance={distance!r}"
            )
        print(f"TopoQuant portable self-test passed: {_PROJECT_ROOT}")
        return 0

    # 3) 端到端多进程流水线自检：合成 3 支股票 → 拓扑/匹配/预测全链路
    if len(sys.argv) == 2 and sys.argv[1] == "--portable-pipeline-self-test":
        import tempfile

        import numpy as np
        import pandas as pd

        from topoquant.config import PipelineConfig
        from topoquant.pipeline import run_all

        # 关键：确保主进程 root logger 有 handler。否则 _silence_worker_logging()
        # 会把真实 fd 1/2 重定向到 devnull，吞掉自检输出与任何异常（直接 `python
        # run.py` 运行时 root logger 无 handler；pytest 下因 pytest 已配置而不触发）。
        ensure_root_handler()

        with tempfile.TemporaryDirectory(
            prefix="topoquant-portable-", ignore_cleanup_errors=True
        ) as temp_text:
            temp_root = Path(temp_text)
            source_dir = temp_root / "stock"
            source_dir.mkdir()
            dates = pd.bdate_range("2024-01-01", periods=24)
            angle = np.arange(24, dtype=float) * (2 * np.pi / 8)
            for offset, code in enumerate(("000001.SZ", "000002.SZ", "000003.SZ")):
                close = 20.0 + offset + np.sin(angle)
                pd.DataFrame(
                    {
                        "EventDate": dates,
                        "money": np.cos(angle) * 100 + offset,
                        "volume": np.sin(angle) * 100 + offset,
                        "high": 21.0 + np.cos(angle),
                        "close": close,
                        "prev_close": np.roll(close, 1),
                    }
                ).to_csv(source_dir / f"{code}.csv", index=False)

            config = PipelineConfig(
                source_dir=source_dir,
                work_dir=temp_root / "work",
                as_of_date=dates[15].date(),
                window_size=8,
                lookback_trading_days=16,
                min_windows=2,
                distance_threshold_h0=0.1,
                distance_threshold_h1=0.1,
                top_k=1,
                forecast_horizon=3,
                # 便携 EXE（_FROZEN）强制走自研 C-DLL 内核，绕开 topp 包（§3#2）；
                # 源码模式沿用默认 ripser_topp，验证源码默认链路。
                topology_backend="native_c_dll" if _FROZEN else "ripser_topp",
                topology_workers=2,
                matching_workers=2,
                forecast_workers=2,
            )
            result = run_all(config)
            topo_complete = result["topology"]["complete"]
            matching_selected = result["matching"]["selected"]
            report_predictions = result["report"]["predictions"]
            if topo_complete != 6 or matching_selected != 3 or report_predictions != 9:
                raise RuntimeError(
                    f"便携版端到端自检结果异常: topo_complete={topo_complete}(期望6) "
                    f"selected={matching_selected}(期望3) predictions={report_predictions}(期望9)"
                )
        print("TopoQuant portable pipeline self-test passed")
        return 0

    return None


if __name__ == "__main__":
    import multiprocessing

    # Windows 冻结程序的 ProcessPoolExecutor 子进程必须先由此接管。
    multiprocessing.freeze_support()
    _special_exit_code = _run_special_mode()
    if _special_exit_code is not None:
        raise SystemExit(_special_exit_code)
    main()
