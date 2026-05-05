"""
图谱记忆更新服务（本地版）
将模拟中的 Agent 行为增量写入本地图谱
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale
from .local_graph_store import LocalGraphStore

logger = get_logger('mirofish.graph_memory_updater')


@dataclass
class AgentActivity:
    platform: str
    agent_id: int
    agent_name: str
    action_type: str
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        content = self.action_args.get("content") or self.action_args.get("quote_content") or ""
        target = self.action_args.get("target_user_name") or self.action_args.get("post_author_name") or ""
        base = f"[{self.platform}] Round {self.round_num} {self.agent_name} {self.action_type}"
        if target:
            base += f" target={target}"
        if content:
            base += f" content={content}"
        return base


class ZepGraphMemoryUpdater:
    """兼容旧命名的本地图谱记忆更新器"""

    BATCH_SIZE = 5
    SEND_INTERVAL = 0.2

    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        self.graph_id = graph_id
        self.store = LocalGraphStore()
        self._activity_queue: Queue = Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0

    def start(self):
        if self._running:
            return
        current_locale = get_locale()
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_locale,),
            daemon=True,
            name=f"LocalGraphMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()

    def stop(self):
        self._running = False
        self._flush_remaining()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

    def add_activity(self, activity: AgentActivity):
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return
        self._activity_queue.put(activity)
        self._total_activities += 1

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        if "event_type" in data:
            return
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        self.add_activity(activity)

    def _worker_loop(self, locale: str = 'zh'):
        set_locale(locale)
        buffer: List[AgentActivity] = []

        while self._running or not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get(timeout=1)
                buffer.append(activity)
                if len(buffer) >= self.BATCH_SIZE:
                    self._send_batch_activities(buffer)
                    buffer = []
                    time.sleep(self.SEND_INTERVAL)
            except Empty:
                if buffer:
                    self._send_batch_activities(buffer)
                    buffer = []
            except Exception as e:
                logger.error(f"本地图谱记忆更新异常: {e}")
                self._failed_count += 1

    def _send_batch_activities(self, activities: List[AgentActivity]):
        if not activities:
            return
        try:
            combined_text = "\n".join([a.to_episode_text() for a in activities])
            self.store.add_episode(self.graph_id, combined_text)
            self._total_sent += 1
            self._total_items_sent += len(activities)
        except Exception as e:
            logger.error(f"写入本地图谱失败: {e}")
            self._failed_count += 1

    def _flush_remaining(self):
        leftovers: List[AgentActivity] = []
        while not self._activity_queue.empty():
            try:
                leftovers.append(self._activity_queue.get_nowait())
            except Empty:
                break
        if leftovers:
            self._send_batch_activities(leftovers)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """管理多个模拟的图谱记忆更新器"""

    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    _stop_all_done = False

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]

    @classmethod
    def stop_all(cls):
        if cls._stop_all_done:
            return
        cls._stop_all_done = True
        with cls._lock:
            for _, updater in list(cls._updaters.items()):
                try:
                    updater.stop()
                except Exception:
                    pass
            cls._updaters.clear()

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        return {sim_id: updater.get_stats() for sim_id, updater in cls._updaters.items()}
