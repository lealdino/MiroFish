"""
图谱构建服务（本地版）
移除云端图谱依赖，使用本地存储进行构建与查询
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

from ..models.task import TaskManager, TaskStatus
from ..utils.locale import t, get_locale, set_locale
from .text_processor import TextProcessor
from .local_graph_store import LocalGraphStore


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """本地图谱构建服务"""

    def __init__(self, api_key: Optional[str] = None):
        # 保留参数签名以兼容既有调用方
        self.task_manager = TaskManager()
        self.store = LocalGraphStore()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        current_locale = get_locale()
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale)
        )
        thread.daemon = True
        thread.start()
        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = 'zh'
    ):
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t('progress.startBuildingGraph')
            )

            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t('progress.graphCreated', graphId=graph_id)
            )

            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t('progress.ontologySet')
            )

            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t('progress.textSplit', count=total_chunks)
            )

            self.add_text_batches(
                graph_id,
                chunks,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.7),
                    message=msg,
                )
            )

            self.task_manager.update_task(
                task_id,
                progress=95,
                message=t('progress.fetchingGraphInfo')
            )
            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })
        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{str(e)}\n{traceback.format_exc()}")

    def create_graph(self, name: str) -> str:
        return self.store.create_graph(name=name, description="MiroFish Local Graph")

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        self.store.set_ontology(graph_id, ontology)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        episode_ids = []
        total_chunks = max(1, len(chunks))

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            for chunk in batch_chunks:
                episode_ids.append(self.store.add_episode(graph_id, chunk))

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    t('progress.sendingBatch', current=batch_num, total=total_batches, chunks=len(batch_chunks)),
                    progress,
                )

            # 保留轻微间隔，兼容既有进度节奏
            time.sleep(0.05)

        return episode_ids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600
    ):
        # 本地模式下 episodes 为同步处理
        if progress_callback:
            progress_callback(t('progress.processingComplete', completed=len(episode_uuids), total=len(episode_uuids)), 1.0)

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        data = self.store.get_graph_data(graph_id)
        entity_types = set()
        for node in data.get("nodes", []):
            for label in node.get("labels", []):
                if label not in ["Entity", "Node"]:
                    entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
            entity_types=sorted(list(entity_types)),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        return self.store.get_graph_data(graph_id)

    def delete_graph(self, graph_id: str):
        self.store.delete_graph(graph_id)
