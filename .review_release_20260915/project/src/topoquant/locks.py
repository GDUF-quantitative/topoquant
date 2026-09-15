"""导出锁原语：从 god-module 抽取的文件锁心跳与进程存活判定（Phase 3）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中的导出锁相关函数与常量平移至此；
``pipeline.py`` 以 ``from .locks import (...)`` 重导出，外部 ``pipeline._export_lock_heartbeat``
等引用零破坏。函数体逐字节一致，无循环依赖（仅依赖 ``policy.POLICY`` 与标准库）。
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

from .policy import POLICY

# 持锁方心跳间隔（秒）：持有锁的进程每隔该间隔刷新一次锁文件 mtime，
# 使「锁年龄」=「最近一次心跳」而非「创建时刻」。这样长耗时但健康的导出不会被
# 误判为过期锁而遭抢占，避免两个进程并发写同一个 mmap 持久图导致数据损坏。
EXPORT_LOCK_HEARTBEAT_INTERVAL = POLICY.export_lock_heartbeat  # 兼容别名（R5）
# R7b：进程启动（本模块导入）时生成、全程不变的 boot_token；写入导出锁文件（pid:token 格式），
# 与 pid 共同锁定持有者身份。新进程不会写出旧 token，Windows PID 复用不再被误判为「持有者仍活」。
_EXPORT_LOCK_BOOT_TOKEN = uuid.uuid4().hex


def _export_lock_heartbeat(lock_path: Path, stop_event: threading.Event) -> None:
    """持锁方后台心跳：定期刷新锁文件 mtime，证明持有者仍在健康导出。

    仅做 ``os.utime``（不改动内容、不重新打开文件）。心跳线程自身若异常退出，
    也不应影响主导出流程；此时 mtime 停止刷新，等待方会在
    ``EXPORT_LOCK_STALE_SECONDS`` 后按失联抢断，符合预期。
    """
    while not stop_event.is_set():
        if stop_event.wait(EXPORT_LOCK_HEARTBEAT_INTERVAL):
            break
        try:
            os.utime(str(lock_path), None)
        except OSError:
            break


def _lock_file_age(lock_path: Path) -> float:
    """锁文件自最后修改起经过的秒数；无法 stat（已被删除等）时返回 inf，即按过期处理。"""
    try:
        return max(0.0, time.time() - lock_path.stat().st_mtime)
    except OSError:
        return float("inf")


def _read_lock_owner_and_token(lock_path: Path) -> tuple[int, str | None]:
    """解析导出锁文件：返回 ``(owner_pid, boot_token)``。

    R7b：锁文件格式从纯 ``pid`` 扩展为 ``pid:token``；旧格式无 token 时 token 返回 ``None``，
    保证新旧程序互读。解析失败返回 ``(0, None)``。
    """
    try:
        with open(str(lock_path)) as handle:
            raw = handle.read().strip()
    except OSError:
        return 0, None
    if not raw:
        return 0, None
    pid_part, _, token_part = raw.partition(":")
    try:
        pid = int(pid_part)
    except ValueError:
        return 0, None
    return pid, (token_part or None)


def _read_lock_owner(lock_path: Path) -> int:
    """兼容旧调用：仅返回 owner pid（忽略 token）。"""
    return _read_lock_owner_and_token(lock_path)[0]


def _lock_owner_is_valid(lock_path: Path) -> bool:
    """导出锁持有者是否仍应被视为有效（有效=继续等待；无效=可抢占）。

    R7b：有效 = pid 存活 且（旧纯 pid 格式无 token，或 token 与当前进程 boot_token 一致）。
    旧格式退回 ``_pid_alive`` 判定；新格式多一道 token 校验以根除 PID 复用误活。
    """
    owner, owner_token = _read_lock_owner_and_token(lock_path)
    if not owner or not _pid_alive(owner):
        return False
    if owner_token is None:
        return True
    return owner_token == _EXPORT_LOCK_BOOT_TOKEN


def _pid_alive(pid: int) -> bool:
    """跨平台判断进程是否存活。"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        ctypes = __import__("ctypes")
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_uint32()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True
