"""一键对比不同参数组合预测结果。

扫描多个参数实验目录 → 读取各自 ``outputs/topk<N>/metrics.json``（兼容旧 ``outputs/metrics.json``）
→ 输出横向对比表
（终端 rich 表格 + 可选 CSV / Markdown 落盘），让用户一眼看出哪个参数组合更优。

用法
----
    python tools/compare_runs.py [ROOT] [--csv OUT.csv] [--md OUT.md]
    [--sort overall|name] [--layout full|legend]

- ROOT：实验根目录，默认 ``./runs``；传入 ``./runs/2024-06-28`` 则只对比该日期下的参数组合。
- --csv / --md：落盘路径，二者可同时给；落盘失败仅打印 warning，不中断终端输出。
- --sort：``overall``（默认，按 overall 准确率降序；无数据 run 恒排末尾）或 ``name``
  （按标签字典序）。
    - --layout：``full``（默认，标签列显示完整 4 段原文）或 ``legend``（标签列改为紧凑 run id，
      表格下方单独列出 run id → 完整标签 + 路径，多 run 横评时更紧凑）或 ``days``
      （转置布局：每行一个预测天 d1..dN、每列一个 run 的当日准确率；表宽只随 run 数增长、
      与预测天数无关，专治预测天数过多时逐日单列把表压扁的问题；**表末另附「总正确率」汇总行，
      每列一个 run（不同参数组合）的 overall 预测正确率，末列为所有 run 的均值**，便于对比各参数总体表现）。
      CSV/MD 落盘始终用完整标签（含 full 布局的 overall 列），days 汇总行仅终端展示。
    - M1 横评增强：``--filter KEY=VAL``（KEY∈data|topo|match|fc，可重复，子串匹配）、
      ``--min FLOAT``（overall≥阈值）、``--date YYYY-MM-DD``（按 ``as_of_date`` 精确筛选）、
      ``--top N``（保留前 N 行）、``--base auto|<run_id>``（追加 Δ 差异列，相对基准的准确率差）。
      表格首列新增 ``rank`` 名次；筛选条件在表格上方用 ``hint`` 回显。落盘 CSV/MD 不含 rank / Δ。

约束（来自工具规范）
--------------------
- 复用项目约定 ``from topoquant._term import C, c, sep, ok, warn, err, info, hint``；
  终端表格用 ``rich``；不修改/重写既有代码；不引入新第三方依赖。
- 手工解析 ``sys.argv``（不引 argparse），与 run.py 的 re-exec 风格保持一致。
- 单个 run 缺 ``metrics.json`` / JSON 损坏 / 缺 ``overall`` / ``by_horizon`` 非 dict
  → 记录 warning 并跳过，继续其余 run，整体不崩溃。
- 不同 run 的 ``forecast_horizon`` 不同时，列取所有 run 的 dN 并集；某 run 缺某 dN
  → 该格留空，不臆造。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# 调试日志：默认静默；设置环境变量 COMPARE_RUNS_DEBUG=1 可打开，不改变命令行接口。
_log = logging.getLogger("compare_runs")
if os.environ.get("COMPARE_RUNS_DEBUG"):
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

# ── 自动检测并切换到项目 .venv（与 run.py 风格一致）────────
_PROJECT_ROOT = Path(__file__).resolve().parents[0]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

if _VENV_PYTHON.is_file() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    import subprocess as _sp

    print("\033[33m检测到未使用项目虚拟环境，自动切换到 .venv ...\033[0m")
    _probe = _sp.run([str(_VENV_PYTHON), "-c", "import sys; sys.exit(0)"],
                     capture_output=True, timeout=20)
    if _probe.returncode != 0:
        print("\033[31m错误: .venv 解释器不可用，请先执行 setup_env.py\033[0m")
        sys.exit(1)
    result = _sp.run([str(_VENV_PYTHON), __file__, *sys.argv[1:]])
    sys.exit(result.returncode)

# ── Windows 终端默认 GBK 编码，无法输出 ⚠/✗ 等符号 → UnicodeEncodeError ──
# 强制标准流按 UTF-8 处理，命令行直接运行也生效（非 Windows 或无缓冲对象则跳过）。
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── 让 topoquant 可被导入（兼容未 editable 安装的情况）────
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table, box  # noqa: E402

from topoquant._term import err, hint, info, ok, sep, warn  # noqa: E402
from topoquant.tabular import write_table  # noqa: E402

_console = Console()

# run 标签列最大显示宽度（rich 以“显示列”计，CJK 记 2）。超长 token 会被折叠换行，
# 防止把列撑爆、破坏 HEAVY_HEAD 框线对齐。可按需调大/调小。
LABEL_COL_WIDTH = 30


# ── 参数解析（手工解析 sys.argv）─────────────────────────
def _parse_args(argv: list[str]) -> tuple[Path, str | None, str | None, str, str, dict]:
    """返回 (root, csv_path, md_path, sort_key, layout, filters)。

    filters 为聚焦筛选/基准字典：
        {"filter": list[str], "min": str|None, "date": str|None,
         "top": str|None, "base": str|None}
    其中 list[str] 项为原始 "KEY=VAL" 字符串，便于回显与解析。
    """
    root = _PROJECT_ROOT / "runs"
    csv_path: str | None = None
    md_path: str | None = None
    sort_key = "overall"
    layout = "full"
    filters: dict = {"filter": [], "min": None, "date": None, "top": None, "base": None}

    positional: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif a == "--csv":
            i += 1
            if i >= len(argv):
                print(err("--csv 需要一个路径参数"))
                sys.exit(2)
            csv_path = argv[i]
        elif a == "--md":
            i += 1
            if i >= len(argv):
                print(err("--md 需要一个路径参数"))
                sys.exit(2)
            md_path = argv[i]
        elif a == "--sort":
            i += 1
            if i >= len(argv):
                print(err("--sort 需要一个取值 (overall|name)"))
                sys.exit(2)
            sort_key = argv[i]
        elif a == "--layout":
            i += 1
            if i >= len(argv):
                print(err("--layout 需要一个取值 (full|legend)"))
                sys.exit(2)
            layout = argv[i]
        # ── M1：聚焦筛选 / 基准（可重复 / 可 --opt=VAL）──
        elif a == "--filter":
            i += 1
            if i >= len(argv):
                print(err("--filter 需要一个取值 (KEY=VAL)"))
                sys.exit(2)
            filters["filter"].append(argv[i])
        elif a == "--min":
            i += 1
            if i >= len(argv):
                print(err("--min 需要一个浮点数"))
                sys.exit(2)
            filters["min"] = argv[i]
        elif a == "--date":
            i += 1
            if i >= len(argv):
                print(err("--date 需要一个日期 (YYYY-MM-DD)"))
                sys.exit(2)
            filters["date"] = argv[i]
        elif a == "--top":
            i += 1
            if i >= len(argv):
                print(err("--top 需要一个正整数"))
                sys.exit(2)
            filters["top"] = argv[i]
        elif a == "--base":
            i += 1
            if i >= len(argv):
                print(err("--base 需要一个取值 (auto|<run_id>)"))
                sys.exit(2)
            filters["base"] = argv[i]
        elif a.startswith("--csv="):
            csv_path = a.split("=", 1)[1]
        elif a.startswith("--md="):
            md_path = a.split("=", 1)[1]
        elif a.startswith("--sort="):
            sort_key = a.split("=", 1)[1]
        elif a.startswith("--layout="):
            layout = a.split("=", 1)[1]
        elif a.startswith("--filter="):
            filters["filter"].append(a.split("=", 1)[1])
        elif a.startswith("--min="):
            filters["min"] = a.split("=", 1)[1]
        elif a.startswith("--date="):
            filters["date"] = a.split("=", 1)[1]
        elif a.startswith("--top="):
            filters["top"] = a.split("=", 1)[1]
        elif a.startswith("--base="):
            filters["base"] = a.split("=", 1)[1]
        elif a.startswith("-"):
            print(err(f"未知选项: {a}"))
            sys.exit(2)
        else:
            positional.append(a)
        i += 1

    if positional:
        root = Path(positional[0]).resolve()
    if sort_key not in ("overall", "name"):
        print(err(f"--sort 取值非法: {sort_key!r}（应为 overall|name），回退为 overall"))
        sort_key = "overall"
    if layout not in ("full", "legend", "days"):
        print(err(f"--layout 取值非法: {layout!r}（应为 full|legend|days），回退为 full"))
        layout = "full"
    return root, csv_path, md_path, sort_key, layout, filters


# ── 数据读取 ────────────────────────────────────────────
def _fmt_generated_at(raw: str) -> str:
    """把 ISO 时间压缩为 'YYYY-MM-DD HH:MM'，失败则原样返回。"""
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(raw)


def _load_run(metrics_path: Path) -> dict | None:
    """读取单个 metrics.json，返回归一化的 run 记录；异常返回 None（由调用方 warn+跳过）。

    兼容两种布局：新结构 ``<work_dir>/outputs/topk<N>/metrics.json``（按 top_k 命名空间）、
    旧结构 ``<work_dir>/outputs/metrics.json``。run 标签在 work_dir 4 段（data/topo/match/fc）
    基础上，新结构追加 topk 段，使同一 work_dir 下不同 top_k 可区分。
    """
    # 定位 work_dir（含参数 4 段的目录）与可选 topk 段：
    #   新：.../fc_xxx/outputs/topk10/metrics.json → topk=topk10, work_dir=fc_xxx
    #   旧：.../fc_xxx/outputs/metrics.json        → topk=None,   work_dir=fc_xxx
    parent = metrics_path.parent
    if parent.name.startswith("topk") and parent.parent.name == "outputs":
        topk_dir = parent
        work_dir = parent.parent.parent
    else:
        topk_dir = None
        work_dir = parent.parent
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(warn(f"跳过（读取/解析失败）{work_dir} : {e}"))
        return None

    # 边界校验：顶层必须是对象；合法 JSON 但为数组/标量时同样是损坏，
    # 否则后续 data.get(...) 会抛 AttributeError 导致整个工具崩溃。
    if not isinstance(data, dict):
        print(warn(f"跳过（JSON 顶层非对象）{work_dir}"))
        _log.debug("metrics.json 顶层类型异常: %s", type(data).__name__)
        return None

    overall = data.get("overall")
    if not isinstance(overall, dict) or "accuracy" not in overall:
        print(warn(f"跳过（缺 overall.accuracy）{work_dir}"))
        return None
    overall_acc = overall["accuracy"]
    if overall_acc is not None and not isinstance(overall_acc, (int, float)):
        # 非数值且非 None（如字符串/对象）→ 视为损坏，跳过该 run
        print(warn(f"跳过（overall.accuracy 非数值）{work_dir} : {overall_acc!r}"))
        return None

    by_horizon = data.get("by_horizon")
    if not isinstance(by_horizon, dict):
        print(warn(f"跳过（by_horizon 非字典）{work_dir}"))
        return None

    # run 标签 = work_dir 最后 4 段（data/topo/match/fc）拼接；新布局追加 topk 段。
    label = " / ".join(work_dir.parts[-4:])
    if topk_dir is not None:
        label = f"{label} / {topk_dir.name}"

    horizons: dict[str, float | None] = {}
    for k, v in by_horizon.items():
        acc = None
        if isinstance(v, dict):
            acc = v.get("accuracy")
        horizons[k] = acc

    return {
        "label": label,
        "work_dir": work_dir,
        "target_count": data.get("target_count"),
        "overall_acc": overall["accuracy"],
        "horizons": horizons,
        "generated_at_raw": data.get("generated_at", ""),
        "generated_at": _fmt_generated_at(data.get("generated_at", "")),
        "as_of_date": data.get("as_of_date", ""),
    }


def _collect_runs(root: Path) -> list[dict]:
    if not root.is_dir():
        print(warn(f"根目录不存在：{root}"))
        return []
    runs: list[dict] = []
    seen: set[Path] = set()
    # 新结构：<work_dir>/outputs/topk<N>/metrics.json；
    # 旧结构（兼容）：<work_dir>/outputs/metrics.json
    for pattern in ("outputs/*/metrics.json", "outputs/metrics.json"):
        for metrics_path in sorted(root.rglob(pattern)):
            if metrics_path in seen:
                continue
            seen.add(metrics_path)
            rec = _load_run(metrics_path)
            if rec is not None:
                runs.append(rec)
    _log.debug("已收集 %d 个有效 run（来自 %s）", len(runs), root)
    return runs


# ── 输出 ───────────────────────────────────────────────
def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _delta(d: float) -> str:
    """格式化 Δ 值（rich 标记语法）：显式正负号；正绿负红，零为中性默认色。

    用 rich 标记而非 topoquant._term.c() 的裸 ANSI，rich 才能正确测量列宽。
    """
    s = f"{d * 100:+.2f}%"
    if d > 0:
        return f"[green]{s}[/]"
    elif d < 0:
        return f"[red]{s}[/]"
    return s


def _overall_sort_key(r: dict) -> float:
    """排序键（降序安全）：有效准确率按原值；无数据(None)归为 -inf，恒排末尾。

    克服 Python sort 不支持 NoneType 与 float 比较的限制；
    不把 None 填 0（0% 是真实准确率，会污染横评），仅将"无数据"置底。
    """
    o = r.get("overall_acc")
    return o if isinstance(o, (int, float)) else float("-inf")


def _union_horizons(runs: list[dict]) -> list[str]:
    keys: set[str] = set()
    for r in runs:
        keys.update(r["horizons"].keys())
    # 按 d 后的整数升序：d1 < d2 < ... < d10
    def _n(k: str) -> int:
        digits = "".join(ch for ch in k if ch.isdigit())
        return int(digits) if digits else 0
    return sorted(keys, key=_n)


def _short_id(label: str) -> str:
    """由 4 段目录标签生成紧凑 run id：取每段尾部 8 位哈希的前 4 位，用 '/' 连接。

    例：``data_2b831c56 / ...__9a880b77 / ...__f8a46bdc / fc_h5__fc8aa44c``
    → ``2b83/9a88/f8a4/fc8a``。无哈希段时退化为末 4 字符，保证总有 id。
    """
    parts: list[str] = []
    for seg in label.split(" / "):
        m = re.findall(r"[0-9a-f]{8,}", seg)
        parts.append(m[-1][:4] if m else seg[-4:])
    return "/".join(parts)


# ── M1：聚焦筛选 / 基准辅助 ────────────────────────────
_FILTER_KEYS = {"data": 0, "topo": 1, "match": 2, "fc": 3}


def _apply_filters(runs: list[dict], filters: dict, sort_key: str) -> list[dict]:
    """按 date → min → filter（AND）→ 排序 → top 顺序过滤+排序，返回新列表。

    任一非法项仅 warn + 忽略该项，不中断；结果为空时原样返回空列表。
    """
    result = list(runs)

    # date：as_of_date 精确匹配（接受任意格式，仅做字符串相等比较）
    date = filters.get("date")
    if date:
        result = [r for r in result if r.get("as_of_date") == date]

    # min：overall_acc >= min
    min_raw = filters.get("min")
    if min_raw:
        try:
            min_v = float(min_raw)
        except (ValueError, TypeError):
            print(warn(f"--min 取值非法: {min_raw!r}（应为浮点数），已忽略"))
        else:
            # 无数据(None) run 不满足任何数值阈值（不能谎称 ≥X%）
            result = [r for r in result
                      if isinstance(r["overall_acc"], (int, float)) and r["overall_acc"] >= min_v]

    # filter：KEY=VAL（KEY ∈ data|topo|match|fc），对应 label 4 段索引；子串匹配
    for fraw in filters.get("filter") or []:
        if "=" not in fraw:
            print(warn(f"--filter 格式非法: {fraw!r}（应为 KEY=VAL），已忽略"))
            continue
        key, val = fraw.split("=", 1)
        key = key.strip().lower()
        if key not in _FILTER_KEYS:
            print(warn(f"--filter 未知 KEY: {key!r}（应为 data|topo|match|fc），已忽略"))
            continue
        idx = _FILTER_KEYS[key]
        val_l = val.lower()
        kept: list[dict] = []
        for r in result:
            segs = r["label"].split(" / ")
            seg = segs[idx] if idx < len(segs) else ""
            if val_l in seg.lower():
                kept.append(r)
        result = kept

    # 排序（复用既有规则）
    if sort_key == "name":
        result.sort(key=lambda r: r["label"])
    else:
        result.sort(key=_overall_sort_key, reverse=True)

    # top：保留前 N
    top_raw = filters.get("top")
    if top_raw:
        try:
            top_n = int(top_raw)
            if top_n < 1:
                raise ValueError
        except (ValueError, TypeError):
            print(warn(f"--top 取值非法: {top_raw!r}（应为正整数），已忽略"))
        else:
            result = result[:top_n]

    return result


def _echo_filters(filters: dict) -> None:
    """在表格上方用 hint() 回显用户实际生效的筛选/基准条件。"""
    cond: list[str] = []
    for fraw in filters.get("filter") or []:
        if "=" in fraw:
            k, v = fraw.split("=", 1)
            cond.append(f"{k.strip().lower()}~{v}")
        else:
            cond.append(fraw)
    if filters.get("min"):
        cond.append(f"min={filters['min']}")
    if filters.get("date"):
        cond.append(f"date={filters['date']}")
    if filters.get("top"):
        cond.append(f"top={filters['top']}")
    base = filters.get("base")
    if not cond and not base:
        return
    line = ("筛选: " + ", ".join(cond)) if cond else ""
    if base:
        line += (f" | base={base}" if line else f"base={base}")
    print(hint(line))


def _render_days_table(runs: list[dict], base_id: str | None = None) -> Table:
    """转置布局：每行一个预测天（d1..dN），每列一个 run 的当日准确率。

    默认布局（full/legend）对**每个预测天建一列**，预测天数多（如 60）时列数爆炸
    （表宽 ≈ 5+30+12+N*9+14+17，N=60 → 约 618 列），在 160 列终端里被 rich 折叠压扁变形。
    转置后：竖向维度是预测天（行，可滚动），横向维度是 run（列），表宽只随 run 数增长、
    与预测天数无关，无论 horizon 多大都不会横向挤压。

    run 列表头用紧凑 short id（见 _short_id），完整标签映射由调用方在表格下方
    用 _print_legend 列出。``base_id`` 在转置布局下忽略（Δ 列仅默认布局有意义），
    保持最小改动、不重写既有对比逻辑。CSV/MD 落盘始终全量逐日，不受影响。
    """
    horizons = _union_horizons(runs)
    table = Table(
        title="按预测天对比（每天 × 各 run 当日准确率）",
        show_lines=True,
        expand=False,
        box=box.HEAVY_HEAD,
    )
    # 预测天作首列；每 run 一列（表头用 short id，宽度限 12 防长 id 撑爆）；
    # 末列“均值”为该天所有 run 的准确率算术平均，便于一眼看整体走势。
    table.add_column("预测天", justify="left", width=10, overflow="fold")
    for r in runs:
        table.add_column(_short_id(r["label"]), justify="right", width=12, overflow="fold")
    table.add_column("均值", justify="right", width=10, overflow="fold")

    for h in horizons:
        row: list[str] = [h]
        vals: list[float] = []
        for r in runs:
            acc = r["horizons"].get(h)
            if isinstance(acc, (int, float)):
                row.append(_pct(acc))
                vals.append(acc)
            else:
                row.append("")
        mean_acc = (sum(vals) / len(vals)) if vals else None
        row.append(_pct(mean_acc) if mean_acc is not None else "")
        table.add_row(*row)

    # 汇总行：每个 run（不同参数组合）的 total（overall）预测正确率，
    # 末列“均值”为该行所有 run 的 overall 算术平均。置于表末，便于一眼对比各参数总体表现。
    total_row: list[str] = ["总正确率"]
    total_vals: list[float] = []
    for r in runs:
        o = r["overall_acc"]
        if isinstance(o, (int, float)):
            total_row.append(_pct(o))
            total_vals.append(o)
        else:
            total_row.append("无数据")
    total_mean = (sum(total_vals) / len(total_vals)) if total_vals else None
    total_row.append(_pct(total_mean) if total_mean is not None else "")
    table.add_row(*total_row)
    return table


def _render_table(runs: list[dict], layout: str = "full", base_id: str | None = None) -> Table:
    """runs 应已按筛选+排序后的最终顺序传入（rank 即其索引+1）。

    base_id：``auto`` 取集合中 overall 最高者为基准；``<run_id>`` 按 _short_id 匹配；
    为 None 则不显示 Δ 列。
    """
    # 转置布局：天作行、run 作列，专治预测天数过多把逐日单列表压扁的问题。
    if layout == "days":
        return _render_days_table(runs, base_id=base_id)
    horizons = _union_horizons(runs)

    # 仅在"有数据"的 run 中取最优/最差；无数据(None)不参与着色比较
    _valid_acc = [r["overall_acc"] for r in runs if isinstance(r["overall_acc"], (int, float))]
    best_overall = max(_valid_acc) if _valid_acc else None
    worst_overall = min(_valid_acc) if _valid_acc else None

    # 解析基准 run（失败不影响其余渲染）
    base_run: dict | None = None
    if base_id:
        if base_id == "auto":
            base_run = max(
                (r for r in runs if isinstance(r["overall_acc"], (int, float))),
                key=lambda r: r["overall_acc"], default=None,
            )
        else:
            for r in runs:
                if _short_id(r["label"]) == base_id:
                    base_run = r
                    break
            if base_run is None:
                print(warn(f"--base {base_id!r} 未匹配到任何 run，已忽略 Δ 列"))

    # 列宽约束：数值列右对齐、文本列左对齐；run 标签列限宽并折叠换行，
    # 防止超长 token（如 topo_w60_...）把列撑爆、破坏 HEAVY_HEAD 框线对齐。
    table = Table(
        title="不同参数组合预测结果对比",
        show_lines=True,
        expand=False,
        box=box.HEAVY_HEAD,
    )
    # overflow="fold"：窄终端下整表被 rich 压缩时，表头/长文本按列宽换行完整显示，
    # 而非默认 ellipsis 截断成「target_…」「准确…」丢失表头文字（见用户反馈）。
    # 数值单元格（100.00% 等）均短于列宽，不会被折叠；仅长表头/长标签换行。
    table.add_column("rank", justify="right", width=5, overflow="fold")
    if layout == "legend":
        table.add_column("run id", overflow="fold", width=20)
    else:
        table.add_column("run 标签", overflow="fold", width=LABEL_COL_WIDTH)
    table.add_column("target count", justify="right", width=12, overflow="fold")
    for h in horizons:
        table.add_column(f"{h} 准确率", justify="right", width=9, overflow="fold")
    # 着色单元格改用 rich 原生标记语法（而非 topoquant._term.c() 的裸 ANSI）：
    # 裸 ANSI 会被 rich 的 cell_len 误把 CSI 转义序列计入宽度，导致列被高估、在窄终端下
    # 触发 ellipsis 截断；rich 标记可被正确剥离测量，列宽精确、可设固定宽度、不截断。
    table.add_column("overall 准确率", justify="right", width=14, overflow="fold")
    table.add_column("generated at", justify="left", width=17, overflow="fold")
    if base_run is not None:
        for h in horizons:
            table.add_column(f"Δ{h}", justify="right", width=9, overflow="fold")
        table.add_column("Δoverall", justify="right", width=10, overflow="fold")

    for idx, r in enumerate(runs):
        if layout == "legend":
            row: list[str] = [str(idx + 1), _short_id(r["label"])]
        else:
            row = [str(idx + 1), r["label"]]
        row.append(str(r["target_count"]) if r["target_count"] is not None else "-")
        for h in horizons:
            acc = r["horizons"].get(h)
            row.append(_pct(acc) if isinstance(acc, (int, float)) else "")
        # overall 着色：有数据→最优绿加粗、最差红；无数据→"无数据"（不参与比较）。
        # 用 rich 标记语法（[green]/[red]/[bold]），终端配色与原 c() 一致，且列宽测量正确。
        o = r["overall_acc"]
        if isinstance(o, (int, float)):
            if best_overall is not None and o == best_overall:
                row.append(f"[bold green]{_pct(o)}[/]")
            elif worst_overall is not None and o == worst_overall:
                row.append(f"[red]{_pct(o)}[/]")
            else:
                row.append(_pct(o))
        else:
            row.append("无数据")
        row.append(r["generated_at"])
        if base_run is not None:
            for h in horizons:
                racc = r["horizons"].get(h)
                bacc = base_run["horizons"].get(h)
                if isinstance(racc, (int, float)) and isinstance(bacc, (int, float)):
                    row.append(_delta(racc - bacc))
                else:
                    row.append("")
            # 仅当本 run 与基准均为有效数值时才计算 Δ；否则留空
            # （避免 None 相减抛 TypeError，例如 --base 命中一个"无数据" run）。
            row.append(
                _delta(o - base_run["overall_acc"])
                if isinstance(o, (int, float)) and isinstance(base_run["overall_acc"], (int, float))
                else ""
            )
        table.add_row(*row)
    return table


def _print_legend(runs_sorted: list[dict]) -> None:
    """legend 布局：表格下方单独列出 run id → 完整标签 + 文件系统路径。"""
    sep("图例：run id → 完整标签 / 路径")
    for r in runs_sorted:
        sid = _short_id(r["label"])
        print(info(f"{sid:<22} {r['label']}"))
        print(hint(f"{'':<22} {r['work_dir']}"))


def _write_csv(path: str, runs: list[dict], sort_key: str | None = None) -> None:
    horizons = _union_horizons(runs)
    if sort_key == "name":
        runs = sorted(runs, key=lambda r: r["label"])
    elif sort_key == "overall":
        runs = sorted(runs, key=_overall_sort_key, reverse=True)
    header = ["run_label", "target_count", *horizons, "overall_accuracy", "generated_at"]
    rows = []
    for r in runs:
        row = [r["label"], r["target_count"]]
        for h in horizons:
            acc = r["horizons"].get(h)
            row.append(acc if isinstance(acc, (int, float)) else "")
        row.append(r["overall_acc"] if isinstance(r["overall_acc"], (int, float)) else "")
        row.append(r["generated_at_raw"])
        rows.append(row)
    try:
        # 经统一出口写出为 CSV（utf-8-sig）；若用户误传 .xls/.xlsx 会被强制改写为 .csv。
        write_table(path, header=header, rows=rows)
    except Exception as e:  # 落盘失败不中断终端输出（兼容 write_table 抛出的各类异常）
        print(warn(f"CSV 落盘失败（已跳过）：{e}"))
        _log.debug("CSV 落盘异常", exc_info=True)


def _write_md(path: str, runs: list[dict], sort_key: str | None = None) -> None:
    horizons = _union_horizons(runs)
    if sort_key == "name":
        runs = sorted(runs, key=lambda r: r["label"])
    elif sort_key == "overall":
        runs = sorted(runs, key=_overall_sort_key, reverse=True)
    try:
        lines = ["# 不同参数组合预测结果对比", ""]
        header = ["run 标签", "target_count"] + [f"{h} 准确率" for h in horizons] + [
            "overall 准确率", "generated_at"
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in runs:
            cells = [r["label"], str(r["target_count"]) if r["target_count"] is not None else "-"]
            for h in horizons:
                acc = r["horizons"].get(h)
                cells.append(_pct(acc) if isinstance(acc, (int, float)) else "")
            acc_val = r["overall_acc"]
            cells.append(
                _pct(acc_val) if isinstance(acc_val, (int, float)) else "无数据"
            )
            cells.append(r["generated_at"])
            lines.append("| " + " | ".join(cells) + " |")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(warn(f"Markdown 落盘失败（已跳过）：{e}"))
        _log.debug("Markdown 落盘异常", exc_info=True)


def main() -> int:
    root, csv_path, md_path, sort_key, layout, filters = _parse_args(sys.argv[1:])
    try:
        sep("对比不同参数组合预测结果")
        print(info(f"扫描根目录 : {root}"))
        print(info(f"排序方式   : {sort_key}"))
        print(info(f"布局       : {layout}"))

        runs = _collect_runs(root)

        if not runs:
            print(warn("未发现任何有效 run（均无可用的 outputs/(topk<N>/)metrics.json）"))
            return 0

        # M1：聚焦筛选（date → min → filter → 排序 → top）
        runs_sorted = _apply_filters(runs, filters, sort_key)

        # 4.4 生效条件回显（表格上方，便于核对）
        _echo_filters(filters)

        if not runs_sorted:
            print(warn("无满足条件的 run"))
            return 0

        print(ok(f"已加载 {len(runs)} 个 run，符合条件 {len(runs_sorted)} 个"))
        base_id = filters.get("base")
        _console.print(_render_table(runs_sorted, layout, base_id=base_id))

        if layout in ("legend", "days"):
            _print_legend(runs_sorted)

        if csv_path:
            _write_csv(csv_path, runs_sorted)
            print(ok(f"CSV 已写入 : {csv_path}"))
        if md_path:
            _write_md(md_path, runs_sorted)
            print(ok(f"Markdown 已写入 : {md_path}"))

        return 0
    except Exception as e:  # 顶层兜底：任何意外异常都转为友好错误，避免裸 traceback 退出
        print(err(f"compare_runs 意外错误: {e}"))
        _log.exception("compare_runs 顶层异常")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
