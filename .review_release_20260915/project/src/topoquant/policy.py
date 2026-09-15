"""流水线各阶段策略参数（R5）：从 god-module 抽取的集中配置对象。

机械搬迁（Phase 3）：原 ``pipeline.py`` 中的 ``StagePolicy`` 类与 ``POLICY`` 实例
整体平移到此模块；``pipeline.py`` 以 ``from .policy import POLICY, StagePolicy``
重导出，外部 ``pipeline.StagePolicy`` / ``pipeline.POLICY`` 引用零破坏。

函数体逐字节一致，无行为变更（INV-1/2/3/4 均不触）。本模块不依赖任何
``topoquant`` 内部子模块，故可被 ``pipeline`` 及其余解耦子模块安全导入，无循环依赖。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


# ── StagePolicy：策略常量集中（R5）────────────────────────────────────
# 把原先散落在下方的策略常量收敛为单一可测配置对象，并修正「环境变量 import 时求值 +
# spawn 父子不一致」的排查盲区：from_env() 把最终生效值打一条 DEBUG 日志，父子进程
# 取值不一致时即可被发现。旧常量名一律保留为兼容别名（见下方 = POLICY.x），现有代码
# 与测试的全部引用 / monkeypatch 继续有效，零破坏。
@dataclass(frozen=True)
class StagePolicy:
    """流水线各阶段的可调策略参数（全部带安全默认值，逐项对齐重构前的既有常量）。"""

    task_timeout: float = 600.0
    export_lock_stale: float = 300.0
    export_lock_poll: float = 0.5
    export_lock_heartbeat: float = 60.0
    commit_retry_attempts: int = 5
    commit_retry_base_delay: float = 1.0
    match_candidate_cap: int = 500
    commit_every_topology: int = 25
    commit_every_matching: int = 100
    commit_every_reselect: int = 200
    commit_every_forecast: int = 50  # R2-S3 统一 forecast 提交节奏（原魔法数 % 50）

    @classmethod
    def from_env(cls) -> StagePolicy:
        """集中读取环境变量，构造 StagePolicy，并把最终生效值打到 DEBUG 日志。

        仅 ``TOPO_TASK_TIMEOUT`` / ``TOPO_COMMIT_RETRY`` 由环境变量覆盖（与重构前一致），
        其余字段取安全默认值。DEBUG 日志让 spawn 父子进程取值不一致时得以核对。
        """
        task_timeout = float(os.environ.get("TOPO_TASK_TIMEOUT", "600"))
        commit_retry_attempts = int(os.environ.get("TOPO_COMMIT_RETRY", "5"))
        policy = cls(task_timeout=task_timeout, commit_retry_attempts=commit_retry_attempts)
        LOGGER.debug(
            "StagePolicy 生效：task_timeout=%.1f, export_lock_stale=%.1f, "
            "export_lock_poll=%.2f, export_lock_heartbeat=%.1f, commit_retry_attempts=%d, "
            "commit_retry_base_delay=%.1f, match_candidate_cap=%d, commit_every_topology=%d, "
            "commit_every_matching=%d, commit_every_reselect=%d",
            policy.task_timeout,
            policy.export_lock_stale,
            policy.export_lock_poll,
            policy.export_lock_heartbeat,
            policy.commit_retry_attempts,
            policy.commit_retry_base_delay,
            policy.match_candidate_cap,
            policy.commit_every_topology,
            policy.commit_every_matching,
            policy.commit_every_reselect,
        )
        return policy


POLICY = StagePolicy.from_env()
