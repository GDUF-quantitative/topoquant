"""
TopoQuant 交互式运行脚本
在 VSCode 终端中交互输入参数 → 自动更新 config.json → 运行全流程并监控进度。

⚠ 重要：请在 VSCode 中右键 → "Run Python File in Terminal" 运行整个文件，
  不要逐行执行（流水线使用多进程 spawn，需要 if __name__ == "__main__" 保护）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 开发模式从源码目录读取配置；PyInstaller 便携版从 exe 所在目录读取。
_FROZEN = bool(getattr(sys, "frozen", False))
_PROJECT_ROOT = (
    Path(sys.executable).resolve().parent
    if _FROZEN
    else Path(__file__).resolve().parent
)
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

if (
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
from datetime import date, datetime
from typing import Any
import subprocess

# ── 颜色常量 ──────────────────────────────────────────
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
}


def _c(text: str, *codes: str) -> str:
    """包裹 ANSI 颜色码。"""
    prefix = "".join(codes)
    return f"{prefix}{text}{C['reset']}"


def _sep(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{_c('─' * 4 + ' ' + title + ' ' + '─' * (width - len(title) - 6), C['dim'], C['bold'])}")
    else:
        print(f"{_c('─' * width, C['dim'])}")


def _ok(text: str) -> str:
    return _c(f"✓ {text}", C["green"])


def _warn(text: str) -> str:
    return _c(f"⚠ {text}", C["yellow"])


def _err(text: str) -> str:
    return _c(f"✗ {text}", C["red"])


def _info(text: str) -> str:
    return _c(text, C["cyan"])


def _hint(text: str) -> str:
    return _c(text, C["dim"])


# ── 默认值 ────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "source_dir": "./data/stock",
    "as_of_date": "2024-06-28",
    "window_size": 60,
    "lookback_trading_days": 480,
    "min_windows": 4,
    "features": ["money", "volume", "high", "close"],
    "max_edge_length": 3.0,
    "max_homology_dimension": 2,
    "distance_dimensions": [0, 1],
    "distance_threshold": 0.1,
    "top_k": 5,
    "matching_pivots": 8,
    "forecast_horizon": 5,
    "topology_workers": 0,
    "matching_workers": 0,
    "forecast_workers": 0,
}

VALIDATORS: dict[str, str] = {
    "as_of_date": r"\d{4}-\d{2}-\d{2}",
    "features": r"^[a-z_]+(,[a-z_]+)*$",
    "distance_dimensions": r"^\d+,\d+$",
}


def load_defaults() -> dict[str, Any]:
    """优先加载上次协议，其次加载示例协议，最后使用内置默认值。"""
    saved_path = _PROJECT_ROOT / "config.json"
    if saved_path.is_file():
        try:
            return json.loads(saved_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"实验协议文件无效: {saved_path}\n{exc}") from exc

    example_path = _PROJECT_ROOT / "config.example.json"
    if example_path.is_file():
        try:
            return json.loads(example_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULTS)


# ── 交互式输入 ────────────────────────────────────────

def prompt_str(label: str, default: Any, hint_text: str = "") -> str:
    """字符串输入（Enter 使用默认值）。"""
    d_str = str(default)
    extra = f"  {_hint(hint_text)}" if hint_text else ""
    prompt = f"  {_c(label, C['bold'])} [{_c(d_str, C['cyan'])}]{extra}: "
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
            print(f"    {_err('请输入整数')}")


def prompt_float(label: str, default: float, hint_text: str = "") -> float:
    """浮点数输入，带校验。"""
    while True:
        raw = prompt_str(label, default, hint_text)
        if raw == str(default):
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"    {_err('请输入数字')}")


def prompt_date(label: str, default: str) -> str:
    """日期输入 YYYY-MM-DD。"""
    while True:
        raw = prompt_str(label, default, "格式 YYYY-MM-DD")
        try:
            date.fromisoformat(raw)
            return raw
        except ValueError:
            print(f"    {_err(f'无效日期: {raw}，请用 YYYY-MM-DD 格式')}")


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


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """是/否确认。"""
    yn = "Y/n" if default else "y/N"
    raw = input(f"\n  {_c(question, C['bold'])} [{_c(yn, C['cyan'])}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "是")


# ── 参数输入流程 ──────────────────────────────────────

def collect_params(defaults: dict[str, Any]) -> dict[str, Any]:
    """分组交互式收集全部参数。"""
    print(f"\n{_c('╔══════════════════════════════════════════════════╗', C['cyan'])}")
    print(f"{_c('║', C['cyan'])}     {_c('TopoQuant 交互式参数配置', C['bold'])}{' ' * 25}{_c('║', C['cyan'])}")
    print(f"{_c('║', C['cyan'])}     {_c('Enter 直接回车 = 使用方括号内默认值', C['dim'])}{' ' * 10}{_c('║', C['cyan'])}")
    print(f"{_c('╚══════════════════════════════════════════════════╝', C['cyan'])}")

    params: dict[str, Any] = {}

    # ── 第1组：数据与日期 ──
    _sep("数据与日期")
    params["source_dir"] = prompt_str("行情数据目录", defaults["source_dir"], "CSV 文件所在路径")
    params["as_of_date"] = prompt_date("预测基准日", str(defaults["as_of_date"]))
    params["work_dir"] = prompt_str(
        "实验结果目录",
        defaults.get("work_dir", f"./runs/{params['as_of_date']}"),
        "同一天更改实验协议时请使用新目录",
    )

    # ── 第2组：窗口参数 ──
    _sep("窗口参数")
    params["window_size"] = prompt_int("点云窗口大小(交易日)", int(defaults["window_size"]), "建议 40~80")
    params["lookback_trading_days"] = prompt_int("回溯交易日数", int(defaults["lookback_trading_days"]), f"需 ≥ {params['window_size']}")
    params["min_windows"] = prompt_int("最少窗口数", int(defaults["min_windows"]), "每支股票至少需几个完整窗口")

    # ── 第3组：特征与拓扑 ──
    _sep("特征与拓扑")
    params["features"] = prompt_list("特征列", list(defaults["features"]), "逗号分隔，如 money,volume,high,close")
    params["max_edge_length"] = prompt_float("VR复形最大边长", float(defaults["max_edge_length"]), "通常 2.0~5.0")
    params["max_homology_dimension"] = prompt_int("最大同调维度", int(defaults["max_homology_dimension"]), "2 = H0/H1/H2")

    # ── 第4组：匹配参数 ──
    _sep("匹配参数")
    params["distance_dimensions"] = prompt_list(
        "瓶颈距离维度", list(defaults["distance_dimensions"]), "当前实验定义固定为 0,1 (H0+H1)"
    )
    params["distance_threshold"] = prompt_float("距离阈值(r)", float(defaults["distance_threshold"]), "H0 和 H1 均需 < r")
    params["top_k"] = prompt_int("Top-K 相似点云数", int(defaults["top_k"]), "多数投票用，建议奇数")
    params["matching_pivots"] = prompt_int(
        "Pivot 数量", int(defaults.get("matching_pivots", 8)), "0=禁用，建议 8~16"
    )

    # ── 第5组：预测 ──
    _sep("预测参数")
    params["forecast_horizon"] = prompt_int("预测天数", int(defaults["forecast_horizon"]), "取未来 N 天涨跌")

    # ── 第6组：并发 ──
    _sep("并发控制（0=自动）")
    params["topology_workers"] = prompt_int("持续同调并发数", int(defaults["topology_workers"]), "进程数，0=自动(min(8,cpu-1))")
    params["matching_workers"] = prompt_int("瓶颈匹配并发数", int(defaults["matching_workers"]), "进程数，0=自动(min(16,cpu-1))；mmap 共享内存可放心调高")
    params["forecast_workers"] = prompt_int("行情预测并发数", int(defaults["forecast_workers"]), "线程数，0=自动(min(32,cpu×2))")

    # ── 第7组：重算控制 ──
    _sep("重算控制")
    params["reset"] = prompt_yes_no(
        "清空实验库与 mmap 后从头重算（换/改股票数据时用）？", default=False
    )
    params["force_rebuild"] = prompt_yes_no(
        "若行情/配置已变化，是否授权就地清空旧数据并重算？", default=False
    )

    return params


def show_summary(params: dict[str, Any]) -> None:
    """打印参数摘要供确认。"""
    _sep("参数摘要")

    groups = [
        ("数据与日期", [
            ("source_dir", "行情目录"),
            ("as_of_date", "基准日期"),
        ]),
        ("窗口", [
            ("window_size", "窗口大小"),
            ("lookback_trading_days", "回溯天数"),
            ("min_windows", "最少窗口"),
        ]),
        ("特征与拓扑", [
            ("features", "特征列"),
            ("max_edge_length", "最大边长"),
            ("max_homology_dimension", "同调维度"),
        ]),
        ("匹配", [
            ("distance_dimensions", "距离维度"),
            ("distance_threshold", "距离阈值"),
            ("top_k", "Top-K"),
            ("matching_pivots", "Pivot 数量"),
        ]),
        ("预测", [
            ("forecast_horizon", "预测天数"),
        ]),
        ("并发", [
            ("topology_workers", "同调并发"),
            ("matching_workers", "匹配并发"),
            ("forecast_workers", "预测并发"),
        ]),
    ]

    for group_name, fields in groups:
        print(f"\n  {_c(group_name, C['bold'], C['yellow'])}")
        for key, label in fields:
            value = params[key]
            if isinstance(value, list):
                value = ", ".join(str(x) for x in value)
            print(f"    {label:<12} {_c(str(value), C['cyan'])}")

    # 重算控制单独展示：reset 是破坏性动作，必须让用户在确认前看见。
    print(f"\n  {_c('重算控制', C['bold'], C['yellow'])}")
    _reset = bool(params.get("reset", False))
    _force = bool(params.get("force_rebuild", False))
    print(
        f"    {'清空重算':<12} "
        + (_c("是（将删除实验库与 mmap，从头计算）", C["red"], C["bold"]) if _reset else _c("否", C["cyan"]))
    )
    print(
        f"    {'授权就地清空':<12} "
        + (_c("是（签名不匹配时自动清空旧数据）", C["yellow"]) if _force else _c("否", C["cyan"]))
    )

    print(f"\n  {_hint('工作目录:')} {_c(params['work_dir'], C['cyan'])}")


# ── 主流程 ────────────────────────────────────────────

def _try_sqlite_access(db_path: Path) -> bool:
    """尝试打开 SQLite 数据库，返回是否成功。"""
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path), timeout=1)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


def _find_topoquant_processes() -> list[int]:
    """查找占用当前数据库的 run.py / topoquant 进程 PID。"""
    import subprocess as _sp
    pids: list[int] = []
    if sys.platform != "win32":
        return pids
    try:
        result = _sp.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.replace('"', "").split(",")
            if len(parts) >= 2:
                try:
                    pids.append(int(parts[1]))
                except ValueError:
                    pass
        # 排除当前进程
        pids = [p for p in pids if p != os.getpid()]
    except Exception:
        pass
    return pids


def _kill_processes(pids: list[int]) -> None:
    """强制终止指定 PID 的进程。"""
    import subprocess as _sp
    for pid in pids:
        try:
            _sp.run(["taskkill", "/F", "/PID", str(pid)],
                     capture_output=True, timeout=5)
        except Exception:
            pass


def main() -> None:
    print(_c("\n  TopoQuant · 持续同调股票点云实验流水线\n", C["bold"], C["cyan"]))

    # 1. 加载默认值
    try:
        defaults = load_defaults()
    except RuntimeError as exc:
        print(f"\n{_err(str(exc))}")
        print(f"  {_hint('请修正或重命名 config.json 后重试；程序没有改用其他协议。')}")
        return

    # 2. 交互式收集参数
    try:
        params = collect_params(defaults)
    except (KeyboardInterrupt, EOFError):
        print(f"\n{_warn('已取消')}")
        return

    # 3. 显示摘要并确认
    show_summary(params)
    if not prompt_yes_no("确认以上参数并开始运行？"):
        print(f"\n{_warn('已取消')}")
        return

    # 4. 写入 config.json
    #    reset / force_rebuild 是"本次运行的动作"，不是实验口径配置，
    #    落盘会误导下次运行（看到 reset=true 以为是常设项），故剔除。
    config_path = _PROJECT_ROOT / "config.json"
    persisted = {k: v for k, v in params.items() if k not in ("reset", "force_rebuild")}
    config_path.write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{_ok(f'配置已写入 {config_path}')}")
    print(f"  {_hint('如需恢复默认: 删除或修改此文件后重新运行本脚本')}")

    # 5. 导入 topoquant 并运行
    print(f"\n{_c('正在初始化 TopoQuant 流水线...', C['bold'])}")
    project_root = _PROJECT_ROOT
    sys.path.insert(0, str(project_root / "src"))

    start_time = datetime.now()

    # ── 前置健康检查：子进程导入 + 超时保护 ──
    # 原因: 持续同调后端包含本地扩展，导入异常有时无法由普通 try/except
    #       稳定捕获。用子进程+超时检测。
    _DEP_CHECKS = [
        ("ripser", "持续同调计算", "pip install ripser"),
        ("topp",   "瓶颈距离", "pip install topp==0.1.0"),
        ("numpy",  "数组与标准化",
         "pip install numpy"),
        ("pandas", "CSV 与数据处理",
         "pip install pandas"),
        ("rich",   "终端进度显示",
         "pip install rich"),
    ]

    _all_ok = True
    for _pkg, _desc, _fix in _DEP_CHECKS:
        print(f"  {_hint(f'检查 {_pkg} ({_desc}) ...')}", end="", flush=True)
        try:
            _probe_command = (
                [sys.executable, "--dependency-probe", _pkg]
                if _FROZEN
                else [sys.executable, "-c", f"import {_pkg}"]
            )
            _check = subprocess.run(
                _probe_command,
                capture_output=True, text=True, timeout=15,
            )
            if _check.returncode != 0:
                _stderr_tail = _check.stderr.strip()[-200:] or "(无错误输出)"
                print(f"\r  {_err(f'{_pkg} 导入失败')}")
                print(f"    {_hint(f'原因: {_stderr_tail}')}")
                print(f"    {_hint(f'解决: {_fix}')}")
                _all_ok = False
                break
            print(f"\r  {_ok(_pkg)}" + " " * 20)
        except subprocess.TimeoutExpired:
            print(f"\r  {_err(f'{_pkg} 导入超时（>15s，可能是 DLL 卡死）')}")
            print(f"    {_hint(f'解决: {_fix}')}")
            _all_ok = False
            break
        except Exception as _exc:
            print(f"\r  {_err(f'{_pkg} 检查异常: {_exc}')}")
            _all_ok = False
            break

    if not _all_ok:
        print(f"\n{_err('依赖检查未通过，无法继续。')}")
        print(f"  {_hint('也可运行 setup_env.py 自动修复环境')}")
        sys.exit(1)

    # ── 逐模块导入 topoquant（前置检查已确认依赖可用） ──
    _IMPORT_STEPS = [
        ("topoquant.config",    "配置管理"),
        ("topoquant.cli",       "终端界面"),
        ("topoquant.pipeline",  "流水线引擎（Ripser + Topp 距离）"),
        ("topoquant.preflight", "环境检测"),
        ("rich.console",        "Rich 控制台"),
        ("rich.progress",       "Rich 进度条"),
        ("rich.panel",          "Rich 面板"),
    ]

    for _mod, _desc in _IMPORT_STEPS:
        print(f"  {_hint(f'导入 {_mod} ({_desc}) ...')}", end="", flush=True)
        try:
            if _mod == "rich.console":
                from rich.console import Console                         # noqa: E402, F811
            elif _mod == "rich.progress":
                from rich.progress import BarColumn, Progress            # noqa: E402
                from rich.progress import SpinnerColumn                  # noqa: E402
                from rich.progress import TaskProgressColumn             # noqa: E402
                from rich.progress import TextColumn                     # noqa: E402
                from rich.progress import TimeElapsedColumn              # noqa: E402
            elif _mod == "rich.panel":
                from rich.panel import Panel                             # noqa: E402
            elif _mod == "topoquant.config":
                from topoquant.config import PipelineConfig              # noqa: E402
            elif _mod == "topoquant.cli":
                from topoquant.cli import render_preflight               # noqa: E402
                from topoquant.cli import render_action_result           # noqa: E402
                from topoquant.cli import render_status                  # noqa: E402
            elif _mod == "topoquant.pipeline":
                from topoquant.pipeline import run_all                   # noqa: E402
            elif _mod == "topoquant.preflight":
                from topoquant.preflight import inspect_environment      # noqa: E402
                from topoquant.preflight import ensure_ready             # noqa: E402
                from topoquant.preflight import inspect_results          # noqa: E402
            print(f"\r  {_ok(_mod + ' ✓')}" + " " * 30)
        except ImportError as _exc:
            print(f"\r  {_err(f'{_mod} 导入失败')}")
            print(f"    {_hint(str(_exc))}")
            print(f"\n{_err('导入失败，请运行 setup_env.py 修复环境')}")
            sys.exit(1)

    console = Console()

    # 5.5 数据库锁检测（防止上一个失败进程残留锁）
    config = PipelineConfig.from_json(config_path)
    _db_path = config.database_path
    _wal_path = Path(str(_db_path) + "-wal")
    _shm_path = Path(str(_db_path) + "-shm")

    if _wal_path.exists() or _shm_path.exists():
        print(f"\n{_warn('检测到上次运行残留的数据库锁文件（可能上次进程崩溃或仍在运行）：')}")
        print(f"  {_hint(str(_wal_path))}")
        print(f"  {_hint(str(_shm_path))}")
        print()
        print(f"  {_c('[1]', C['cyan'])} 清理残留 → 继续运行（丢弃上次未提交的数据）")
        print(f"  {_c('[q]', C['cyan'])} 退出 → 稍后手动处理")
        choice = input(f"\n  {_c('请选择', C['bold'])} [{_c('1', C['cyan'])}]: ").strip().lower()
        if choice in ("q", "quit", "退出"):
            print(f"\n{_warn('已取消')}")
            sys.exit(0)

        # 尝试检测是否有进程在持有锁
        _locked = False
        try:
            import sqlite3
            _locked = not _try_sqlite_access(_db_path)
        except Exception:
            _locked = False

        if _locked:
            _pids = _find_topoquant_processes()
            if _pids:
                _pid_str = ", ".join(str(p) for p in _pids)
                print(f"\n  {_warn(f'仍有 {len(_pids)} 个进程持有数据库锁 (PID: {_pid_str})')}")
                print(f"  {_c('[1]', C['cyan'])} 强制终止所有相关进程 + 清理锁文件")
                print(f"  {_c('[q]', C['cyan'])} 退出")
                choice2 = input(f"\n  {_c('请选择', C['bold'])} [{_c('1', C['cyan'])}]: ").strip().lower()
                if choice2 in ("q", "quit", "退出"):
                    print(f"\n{_warn('已取消')}")
                    sys.exit(0)
                _kill_processes(_pids)
                print(f"  {_ok('进程已终止')}")

        # 清理 WAL/SHM 文件
        for _f in (_wal_path, _shm_path):
            try:
                _f.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"  {_ok('锁文件已清理')}\n")

    # 6. 预启动检测
    config = PipelineConfig.from_json(config_path)
    report = inspect_environment(
        config, require_source=True, require_database=False, require_topology_backend=True
    )
    render_preflight(report)

    try:
        ensure_ready(report)
    except Exception as e:
        console.print(Panel(str(e), title="预检失败", border_style="red"))
        sys.exit(2)

    # 7. 运行全流程（带 Rich 进度条）
    _sep()
    print(f"\n{_c('开始运行全流程...', C['bold'], C['green'])}\n")

    stage_labels = {
        "topology": "持续同调",
        "matching": "瓶颈匹配",
        "forecast": "行情预测",
    }
    task_ids: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[summary]}"),
        console=console,
    ) as display:

        def on_progress(stage: str, current: int, total: int, stats) -> None:
            summary = ", ".join(f"{k}={v}" for k, v in stats.items())
            display_total = max(1, total)
            completed = current if total else 1
            if stage not in task_ids:
                task_ids[stage] = display.add_task(
                    stage_labels.get(stage, stage),
                    total=display_total,
                    summary=summary,
                )
            display.update(
                task_ids[stage],
                total=display_total,
                completed=completed,
                summary=summary,
            )

        try:
            result = run_all(
                config, on_progress,
                reset=bool(params.get("reset", False)),
                force_rebuild=bool(params.get("force_rebuild", False)),
            )
        except Exception as e:
            console.print(Panel(str(e), title="运行失败", border_style="red"))
            sys.exit(2)

    # 8. 结果展示
    elapsed = (datetime.now() - start_time).total_seconds()
    elapsed_str = (
        f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        if elapsed >= 60
        else f"{elapsed:.1f}s"
    )

    _sep("运行结果")
    render_action_result(result)
    try:
        render_status(inspect_results(config))
    except Exception:
        pass  # 如果数据库尚未生成 metrics，静默跳过

    print(f"\n{_ok(f'全流程完成 · 总耗时 {elapsed_str}')}")
    print(f"  {_hint('报告文件:')} {_c(config.output_dir, C['cyan'])}")
    print(f"  {_hint('实验库:')} {_c(config.database_path, C['cyan'])}\n")


def _run_special_mode() -> int | None:
    """处理便携版依赖探针和原生扩展自检，不进入交互界面。"""
    if len(sys.argv) == 3 and sys.argv[1] == "--dependency-probe":
        import importlib

        importlib.import_module(sys.argv[2])
        return 0

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
            raise RuntimeError("便携版拓扑后端自检结果异常")
        print(f"TopoQuant portable self-test passed: {_PROJECT_ROOT}")
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == "--portable-pipeline-self-test":
        import tempfile

        import numpy as np
        import pandas as pd

        from topoquant.config import PipelineConfig
        from topoquant.pipeline import run_all

        with tempfile.TemporaryDirectory(prefix="topoquant-portable-") as temp_text:
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
                distance_threshold=0.1,
                top_k=1,
                forecast_horizon=3,
                topology_workers=2,
                matching_workers=2,
                forecast_workers=2,
            )
            result = run_all(config)
            if (
                result["topology"]["complete"] != 6
                or result["matching"]["selected"] != 3
                or result["report"]["predictions"] != 9
            ):
                raise RuntimeError(f"便携版端到端自检结果异常: {result}")
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
