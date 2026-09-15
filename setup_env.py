"""
TopoQuant 环境自动检测与配置脚本
用法：VSCode 右键 → Run Python File in Terminal，或终端执行 python setup_env.py
功能：
  1. 检测当前 .venv 是否就绪 → 是则直接通过
  2. 否则通过 py launcher 发现 Python 版本 → 用户选择 → 创建 .venv → 安装依赖
  3. 完整校验五个核心包 + topoquant CLI
  4. 检查 data/stock/ 行情数据并提示
  5. 自动检测 uv → 有则优先用 uv 加速安装 → 无则回退标准 venv + pip
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import NamedTuple

# ── 常量 ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DATA_DIR = PROJECT_ROOT / "data" / "stock"
PYTHON_MIN = (3, 10)
PYTHON_MAX = (4, 0)   # 不设上限，由 pip 与科学计算 wheel 可用性自然约束

CORE_PACKAGES = ("ripser", "topp", "numpy", "pandas", "rich")
UV_AVAILABLE = False  # 运行时动态检测

# ── 颜色 ──────────────────────────────────────────────

C = {
    "R": "\033[0m",   "B": "\033[1m",   "D": "\033[2m",
    "G": "\033[32m",  "Y": "\033[33m",  "R_": "\033[31m",  "C": "\033[36m",
}


def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + C["R"]


def sep(title: str = "") -> None:
    w = 62
    if title:
        print(f"\n{c('── ' + title + ' ' + '─' * (w - len(title) - 5), C['D'])}")
    else:
        print(c("─" * w, C["D"]))


def ok(s: str) -> str:   return c(f"✓ {s}", C["G"])
def warn(s: str) -> str: return c(f"⚠ {s}", C["Y"])
def err(s: str) -> str:  return c(f"✗ {s}", C["R_"])
def info(s: str) -> str: return c(s, C["C"])
def hint(s: str) -> str: return c(s, C["D"])


# ── 数据结构 ──────────────────────────────────────────

class PythonVersion(NamedTuple):
    path: Path       # python.exe 绝对路径
    version: str     # "3.12.5"
    major: int
    minor: int
    arch: str        # "64-bit" / "32-bit"

    @property
    def label(self) -> str:
        return f"Python {self.version} ({self.arch}) — {self.path}"


class EnvStatus(NamedTuple):
    venv_ready: bool
    reason: str
    python_path: str | None


# ── 工具函数 ──────────────────────────────────────────

def get_venv_python() -> Path:
    """返回 .venv 中的 python.exe 路径（跨平台）。"""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """静默运行命令。"""
    return subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
        **kwargs,
    )


def run_venv(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """在 venv 中运行命令。"""
    py = get_venv_python()
    return subprocess.run(
        [str(py)] + cmd,
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
        **kwargs,
    )


def check_uv() -> str | None:
    """检测 uv 是否已安装，返回版本号或 None。"""
    uv = shutil.which("uv")
    if uv is None:
        uv = shutil.which("uv.exe")
    if uv is None:
        return None
    try:
        result = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip().split()[1]  # "uv 0.6.x"
    except Exception:
        pass
    return None


# ── 第1步：检测现有 venv ──────────────────────────────

def _get_package_version(pkg: str) -> str:
    """获取包版本（优先 importlib.metadata，回退 __version__）。"""
    code = (
        "import importlib.metadata as m;"
        f"print(m.version('{pkg}'))"
    )
    result = run_venv(["-c", code])
    if result.returncode == 0:
        return result.stdout.strip()
    # 回退 __version__
    result2 = run_venv(["-c", f"import {pkg}; print(getattr({pkg}, '__version__', 'unknown'))"])
    return result2.stdout.strip() if result2.returncode == 0 else "?"


def check_venv() -> EnvStatus:
    """检测 .venv 是否完整可用。"""
    py_path = get_venv_python()

    # 1a. venv 目录是否存在
    if not py_path.is_file():
        return EnvStatus(False, f".venv 不存在（缺少 {py_path.name}）", None)

    # 1b. pip 是否可用
    pip_check = run_venv(["-m", "pip", "--version"])
    if pip_check.returncode != 0:
        return EnvStatus(False, "pip 不可用", str(py_path))

    # 1c. 五个核心包是否可 import
    for pkg in CORE_PACKAGES:
        result = run_venv(["-c", f"import {pkg}"])
        if result.returncode != 0:
            return EnvStatus(False, f"缺少依赖包: {pkg}", str(py_path))

    # 1d. topoquant CLI 是否可运行
    cli_check = run_venv(["-m", "topoquant", "--help"])
    if cli_check.returncode != 0:
        return EnvStatus(False, "topoquant 未安装或损坏", str(py_path))

    # 1e. 额外：验证 pyproject.toml 中的 Python 版本约束
    try:
        constraint = _read_python_requires()
        venv_ver = run_venv(["-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"])
        venv_ver_str = venv_ver.stdout.strip()
        if not _version_in_range(venv_ver_str, constraint):
            return EnvStatus(
                False,
                f"venv Python {venv_ver_str} 不满足 pyproject.toml 要求 {constraint}",
                str(py_path),
            )
    except Exception:
        pass  # 版本检查失败不阻塞

    return EnvStatus(True, "环境就绪", str(py_path))


# ── 第2步：通过 py launcher 发现 Python 版本 ──────────

def _read_python_requires() -> str:
    """从 pyproject.toml 读取 requires-python 约束。"""
    if PYPROJECT.is_file():
        text = PYPROJECT.read_text(encoding="utf-8")
        m = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
        if m:
            return m.group(1)
    return ">=3.10,<3.14"


def _parse_version_constraint(constraint: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """解析 requires-python 如 '>=3.10,<3.14' → ((3,10), (3,14))"""
    lo, hi = (3, 10), (3, 14)
    for part in constraint.replace(" ", "").split(","):
        m = re.match(r"([><=!]+)\s*(\d+)\.(\d+)", part)
        if not m:
            continue
        op, maj, min_ = m.group(1), int(m.group(2)), int(m.group(3))
        ver = (maj, min_)
        if op in (">=", ">", "=="):
            lo = (ver[0], ver[1] + (1 if op == ">" else 0))
        if op in ("<=", "<", "=="):
            hi = (ver[0], ver[1] + (1 if op in ("<=", "==") else 0))
    return lo, hi


def _version_in_range(ver_str: str, constraint: str) -> bool:
    """检查版本字符串是否在约束范围内。"""
    m = re.match(r"(\d+)\.(\d+)", ver_str)
    if not m:
        return False
    v = (int(m.group(1)), int(m.group(2)))
    lo, hi = _parse_version_constraint(constraint)
    return lo <= v < hi


def discover_python_versions() -> list[PythonVersion]:
    """通过 py launcher 发现所有合格 Python 版本。"""
    constraint = _read_python_requires()

    # 尝试 py --list-paths
    result = subprocess.run(
        ["py", "--list-paths"],
        capture_output=True, text=True,
    )

    versions: list[PythonVersion] = []

    if result.returncode == 0:
        # 解析输出:  -V:3.12          C:\Python312\python.exe
        for line in result.stdout.splitlines():
            m = re.match(
                r"\s*-V:(\d+)\.(\d+)(?:\*)?\s+(.+?)(?:\s+\*(.*))?$",
                line.strip(),
            )
            if not m:
                continue
            maj, min_, path_str = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            arch = (m.group(4) or "").strip()
            if not arch:
                arch = "64-bit"  # 默认

            path = Path(path_str)
            if not path.is_file():
                continue

            ver = (maj, min_)
            if not _version_in_range(f"{maj}.{min_}", constraint):
                continue

            # 确认这个 Python 真实存在且可运行
            probe = subprocess.run(
                [str(path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
                capture_output=True, text=True,
            )
            if probe.returncode != 0:
                continue

            full_ver = probe.stdout.strip()
            versions.append(PythonVersion(
                path=path,
                version=full_ver,
                major=maj,
                minor=min_,
                arch=arch,
            ))

    # 如果没有 py launcher，检查当前 python
    if not versions and sys.version_info[:2] >= PYTHON_MIN:
        v = sys.version_info
        arch = "64-bit" if sys.maxsize > 2**32 else "32-bit"
        py_path = Path(sys.executable)
        if _version_in_range(f"{v.major}.{v.minor}", constraint):
            versions.append(PythonVersion(
                path=py_path,
                version=f"{v.major}.{v.minor}.{v.micro}",
                major=v.major,
                minor=v.minor,
                arch=arch,
            ))

    # 按版本降序排列
    versions.sort(key=lambda x: (x.major, x.minor), reverse=True)
    return versions


# ── 第3步：创建 venv 并安装 ───────────────────────────

def create_and_install(python: PythonVersion) -> bool:
    """用指定 Python 创建 .venv + 安装项目依赖（有 uv 则优先用 uv）。"""
    global UV_AVAILABLE
    uv_ver = check_uv()
    UV_AVAILABLE = uv_ver is not None

    sep("创建虚拟环境")

    # 如已有旧的 .venv，先删除
    if VENV_DIR.exists():
        print(f"  {hint('移除旧 .venv ...')}")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    print(f"  {info('Python:')} {python.label}")

    if UV_AVAILABLE:
        print(f"  {ok(f'检测到 uv {uv_ver}，使用加速安装')}")
        print(f"  {hint('uv venv + uv pip install ...')}")

        uv_create = subprocess.run(
            ["uv", "venv", str(VENV_DIR), "--python", str(python.path)],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        if uv_create.returncode != 0:
            print(f"\n{err('uv venv 失败: ' + uv_create.stderr.strip()[-300:])}")
            return False

        py = get_venv_python()
        if not py.is_file():
            print(f"\n{err('.venv 创建失败（python.exe 未找到）')}")
            return False
        print(f"  {ok('.venv 创建成功')}")

        # uv pip install 并行加速
        sep("安装依赖（uv 并行加速）")
        install = subprocess.run(
            ["uv", "pip", "install", "-e", "."],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,
            env={**__import__("os").environ, "VIRTUAL_ENV": str(VENV_DIR)},
        )
        installer_name = "uv pip"
    else:
        print(f"  {hint('未检测到 uv，使用标准 venv + pip')}")
        print(f"  {hint('建议安装: pip install uv  或  winget install astral-sh.uv')}")
        print(f"  {hint('创建 .venv ...')}")

        builder = venv.EnvBuilder(
            with_pip=True,
            upgrade_deps=True,
            clear=True,
        )
        builder.create(str(VENV_DIR))

        py = get_venv_python()
        if not py.is_file():
            print(f"\n{err('.venv 创建失败')}")
            return False
        print(f"  {ok('.venv 创建成功')}")

        sep("安装依赖")
        print(f"  {hint('pip install -e .  (可能需要几分钟)...')}")
        install = subprocess.run(
            [str(py), "-m", "pip", "install", "-e", "."],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,
        )
        installer_name = "pip"

    if install.returncode != 0:
        print(f"\n{err(f'{installer_name} install 失败')}")
        print(f"  {c(install.stderr.strip()[-500:], C['D'])}")
        if "ripser" in install.stderr.lower() or "topp" in install.stderr.lower():
            print(f"\n  {warn('Ripser/Topp 可能没有当前 Python 版本的可用 wheel')}")
            print(f"  {hint('建议换一个 Python 版本重试（如 3.11 或 3.12）')}")
            others = [v for v in discover_python_versions() if v.path != python.path]
            if others:
                print(f"\n  {c('其他可用 Python 版本:', C['B'])}")
                for v in others[:5]:
                    print(f"    · {v.label}")
        return False

    print(f"  {ok('依赖安装成功')}")
    return True


# ── 第4步：完整校验 ───────────────────────────────────

def full_validation() -> bool:
    """验证五个核心包 + topoquant CLI。"""
    sep("依赖校验")

    all_ok = True

    # 检查包版本
    for pkg in CORE_PACKAGES:
        result = run_venv(["-c", f"import {pkg}"])
        if result.returncode == 0:
            ver = _get_package_version(pkg)
            print(f"  {ok(pkg + ' ' + ver)}")
        else:
            print(f"  {err(pkg + ' 导入失败')}")
            all_ok = False

    # 检查 topoquant CLI
    cli = run_venv(["-m", "topoquant", "--help"])
    if cli.returncode == 0:
        print(f"  {ok('topoquant CLI 可用')}")
    else:
        print(f"  {err('topoquant CLI 不可用')}")
        all_ok = False

    return all_ok


# ── 第5步：数据目录检查 ───────────────────────────────

def check_data_dir() -> None:
    """检查 data/stock/ 下是否有行情 CSV。"""
    sep("数据检查")

    if not DATA_DIR.is_dir():
        print(f"  {warn('data/stock/ 目录不存在')}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  {hint('已自动创建，请将行情 CSV 放入该目录')}")
        return

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "README.md"]
    csv_count = len(csv_files)

    if csv_count == 0:
        print(f"  {warn('data/stock/ 下没有行情 CSV 文件')}")
        print(f"  {hint('请将沪深股票行情数据放入该目录，每支股票一个 CSV')}")
        print(f"  {hint('文件命名格式: 000001.SZ.csv / 600000.SH.csv')}")
        print(f"  {hint('必须包含列: EventDate, money, volume, high, close, prev_close')}")
    else:
        # 检测几个 CSV 的表头
        print(f"  {ok(f'发现 {csv_count} 个 CSV 文件')}")
        try:
            import csv as csv_module
            import io
            issues = 0
            required = {"EventDate", "money", "volume", "high", "close", "prev_close"}
            for f in csv_files[:5]:  # 抽查前5个
                raw = f.open("rb").read(65536)
                for enc in ("utf-8-sig", "utf-8", "gb18030"):
                    try:
                        header = {h.strip() for h in next(csv_module.reader(io.StringIO(raw.decode(enc))))}
                        missing = required - header
                        if missing:
                            print(f"  {warn(f'{f.name}: 缺少列 {missing}')}")
                            issues += 1
                        break
                    except (UnicodeDecodeError, StopIteration):
                        continue
            if issues == 0:
                print(f"  {ok('表头契约抽查通过')}")
            else:
                print(f"  {hint(f'共 {issues} 个文件表头有问题（仅抽查前 5 个）')}")
        except Exception:
            pass  # 表头检查失败不阻塞


# ── 用户选择 Python 版本 ──────────────────────────────

def choose_python(versions: list[PythonVersion]) -> PythonVersion | None:
    """交互式让用户选择 Python 版本。"""
    if not versions:
        print(f"\n{err('未发现任何可用的 Python 版本')}")
        print(f"  {hint('请安装 Python 3.10–3.13（64-bit）并确保 py launcher 可用')}")
        print(f"  {hint('下载: https://www.python.org/downloads/')}")
        return None

    if len(versions) == 1:
        v = versions[0]
        print(f"\n  {info('自动选择唯一可用版本:')}")
        print(f"    {v.label}")
        return v

    print(f"\n  {c('发现以下 Python 版本:', C['B'])}")
    for i, v in enumerate(versions):
        print(f"    {c(f'[{i+1}]', C['C'])} {v.label}")

    while True:
        try:
            choice = input(f"\n  {c('请选择版本序号', C['B'])} [{c('1', C['C'])}]: ").strip()
            if not choice:
                return versions[0]
            idx = int(choice) - 1
            if 0 <= idx < len(versions):
                return versions[idx]
            print(f"    {err(f'请输入 1-{len(versions)}')}")
        except (ValueError, EOFError, KeyboardInterrupt):
            return None


# ── 主流程 ────────────────────────────────────────────

def main() -> int:
    print(c("\n  TopoQuant · 环境检测与配置", C["B"], C["C"]))

    # ── 1. 检查现有 venv ──
    sep("检测现有环境")
    status = check_venv()
    if status.venv_ready:
        print(f"  {ok(status.reason)}")
        print(f"  {hint('Python:')} {status.python_path}")

        # 快速显示核心依赖版本
        for pkg in CORE_PACKAGES:
            ver = _get_package_version(pkg)
            print(f"    {pkg} {ver}")

        # 即使 venv 就绪，也检查数据
        check_data_dir()

        _print_done()
        return 0

    print(f"  {warn(status.reason)}")
    print(f"  {hint('将自动配置环境...')}")

    # ── 2. 发现 Python 版本 ──
    sep("发现 Python 版本")
    versions = discover_python_versions()
    chosen = choose_python(versions)
    if chosen is None:
        print(f"\n{err('用户取消')}")
        return 1

    # ── 3. 创建 venv + 安装 ──
    if not create_and_install(chosen):
        return 2

    # ── 4. 完整校验 ──
    if not full_validation():
        print(f"\n{err('依赖校验未通过，请检查上述错误')}")
        return 2

    # ── 5. 数据检查 ──
    check_data_dir()

    _print_done()
    return 0


def _print_done() -> None:
    print(f"\n{c('═' * 62, C['D'])}")
    print(f"  {ok('环境配置完成')}")
    print(f"  {hint('下一步:')}")
    print(f"  {c('  · 放入行情 CSV 到 data/stock/', C['C'])}")
    print(f"  {c('  · VSCode 右键运行 run.py 开始实验', C['C'])}")
    print(f"  {c('  · 或: .venv\\Scripts\\python -m topoquant --config config.example.json preflight', C['C'])}")
    print(f"  {hint('换了行情数据后:')}")
    print(f"  {c('  · run.py 交互界面最后一组「重算控制」选 y，或加 --reset 从头重算', C['C'])}")
    print(f"  {c('  · 匹配阶段已改为 mmap 零拷贝共享，matching_workers 可放心开到 CPU 核数', C['C'])}")
    print(f"{c('═' * 62, C['D'])}\n")


if __name__ == "__main__":
    sys.exit(main())
