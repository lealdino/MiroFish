"""
本地知识图谱存储与检索
替代云端图谱依赖，实现完全本地构建与查询
"""

import json
import os
import re
import threading
import uuid
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_graph')


class LocalGraphStore:
    """本地图谱存储服务（JSON 持久化）"""

    _locks: Dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self):
        self.base_dir = Config.LOCAL_GRAPH_DATA_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, graph_id: str) -> str:
        return os.path.join(self.base_dir, f"{graph_id}.json")

    def _get_lock(self, graph_id: str) -> threading.Lock:
        with self._locks_guard:
            if graph_id not in self._locks:
                self._locks[graph_id] = threading.Lock()
            return self._locks[graph_id]

    def create_graph(self, name: str, description: str = "") -> str:
        graph_id = f"mirofish_local_{uuid.uuid4().hex[:16]}"
        now = datetime.now().isoformat()
        data = {
            "graph_id": graph_id,
            "name": name,
            "description": description,
            "ontology": {"entity_types": [], "edge_types": []},
            "nodes": [],
            "edges": [],
            "episodes": [],
            "created_at": now,
            "updated_at": now,
        }
        self.save_graph(graph_id, data)
        return graph_id

    def load_graph(self, graph_id: str) -> Dict[str, Any]:
        path = self._path(graph_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph not found: {graph_id}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_graph(self, graph_id: str, data: Dict[str, Any]):
        path = self._path(graph_id)
        data["updated_at"] = datetime.now().isoformat()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def delete_graph(self, graph_id: str):
        path = self._path(graph_id)
        if os.path.exists(path):
            os.remove(path)

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        lock = self._get_lock(graph_id)
        with lock:
            graph = self.load_graph(graph_id)
            graph["ontology"] = {
                "entity_types": ontology.get("entity_types", []),
                "edge_types": ontology.get("edge_types", []),
            }
            self.save_graph(graph_id, graph)

    def add_episode(self, graph_id: str, text: str) -> str:
        lock = self._get_lock(graph_id)
        with lock:
            graph = self.load_graph(graph_id)
            episode_id = f"ep_{uuid.uuid4().hex[:12]}"
            graph["episodes"].append({
                "episode_id": episode_id,
                "text": text,
                "processed": True,
                "created_at": datetime.now().isoformat(),
            })

            self._ingest_text(graph, text, episode_id)
            self.save_graph(graph_id, graph)
            return episode_id

    def _ingest_text(self, graph: Dict[str, Any], text: str, episode_id: str):
        entities = self._extract_entities(text, graph.get("ontology", {}))
        node_ids = []
        for name, entity_type in entities:
            node_ids.append(self._upsert_node(graph, name, entity_type, text))

        # 构建轻量关系链，保障后续检索和画像流程有可用结构
        for idx in range(len(node_ids) - 1):
            source_id = node_ids[idx]
            target_id = node_ids[idx + 1]
            if source_id == target_id:
                continue
            self._append_edge(
                graph,
                source_id,
                target_id,
                edge_name="related_to",
                fact=self._build_fact(graph, source_id, target_id, text),
                episode_id=episode_id,
            )

    def _extract_entities(self, text: str, ontology: Dict[str, Any]) -> List[Tuple[str, str]]:
        entity_types = ontology.get("entity_types", [])
        preferred_types = [e.get("name", "Entity") for e in entity_types if e.get("name")]
        default_type = preferred_types[0] if preferred_types else "Entity"

        # 英文实体候选：首字母大写词组
        english_candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{2,}\b", text)
        # 中文实体候选：2-8字连续中文
        chinese_candidates = re.findall(r"[\u4e00-\u9fff]{2,8}", text)

        all_candidates = [c.strip() for c in (english_candidates + chinese_candidates)]
        all_candidates = [c for c in all_candidates if len(c) >= 2]
        if not all_candidates:
            return []

        counts = Counter(all_candidates)
        selected = [name for name, _ in counts.most_common(20)]

        return [(name, default_type) for name in selected]

    def _upsert_node(self, graph: Dict[str, Any], name: str, entity_type: str, text: str) -> str:
        nodes = graph.setdefault("nodes", [])
        existing = next((n for n in nodes if n.get("name") == name), None)
        if existing:
            if entity_type and entity_type not in existing.get("labels", []):
                existing.setdefault("labels", []).append(entity_type)
            if not existing.get("summary"):
                existing["summary"] = self._make_summary(name, text)
            return existing["uuid"]

        node_id = f"node_{uuid.uuid4().hex[:12]}"
        labels = ["Entity", "Node"]
        if entity_type and entity_type not in labels:
            labels.append(entity_type)

        nodes.append({
            "uuid": node_id,
            "name": name,
            "labels": labels,
            "summary": self._make_summary(name, text),
            "attributes": {},
            "created_at": datetime.now().isoformat(),
        })
        return node_id

    def _append_edge(
        self,
        graph: Dict[str, Any],
        source_node_uuid: str,
        target_node_uuid: str,
        edge_name: str,
        fact: str,
        episode_id: Optional[str] = None,
    ):
        edges = graph.setdefault("edges", [])
        edge_id = f"edge_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        edges.append({
            "uuid": edge_id,
            "name": edge_name,
            "fact": fact,
            "source_node_uuid": source_node_uuid,
            "target_node_uuid": target_node_uuid,
            "attributes": {},
            "created_at": now,
            "valid_at": now,
            "invalid_at": None,
            "expired_at": None,
            "episodes": [episode_id] if episode_id else [],
        })

    def _build_fact(self, graph: Dict[str, Any], source_id: str, target_id: str, text: str) -> str:
        source = self._node_name(graph, source_id)
        target = self._node_name(graph, target_id)
        snippet = text[:120].replace('\n', ' ').strip()
        return f"{source} related to {target}. Context: {snippet}"

    def _node_name(self, graph: Dict[str, Any], node_id: str) -> str:
        node = next((n for n in graph.get("nodes", []) if n.get("uuid") == node_id), None)
        return node.get("name", node_id) if node else node_id

    def _make_summary(self, entity_name: str, text: str) -> str:
        idx = text.find(entity_name)
        if idx < 0:
            return f"Entity extracted from source text: {entity_name}"
        start = max(0, idx - 60)
        end = min(len(text), idx + 120)
        return text[start:end].replace('\n', ' ').strip()

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        graph = self.load_graph(graph_id)
        nodes = graph.get("nodes", [])
        node_map = {n.get("uuid"): n.get("name", "") for n in nodes}

        edges = []
        for edge in graph.get("edges", []):
            edges.append({
                **edge,
                "source_node_name": node_map.get(edge.get("source_node_uuid"), ""),
                "target_node_name": node_map.get(edge.get("target_node_uuid"), ""),
                "fact_type": edge.get("name", ""),
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "ontology": graph.get("ontology", {}),
            "updated_at": graph.get("updated_at"),
        }

    def search(self, graph_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
        query_l = (query or "").strip().lower()
        data = self.get_graph_data(graph_id)

        if not query_l:
            return {
                "facts": [e.get("fact", "") for e in data["edges"][:limit]],
                "edges": data["edges"][:limit],
                "nodes": data["nodes"][:limit],
            }

        matched_nodes = [
            n for n in data["nodes"]
            if query_l in (n.get("name", "").lower() + " " + n.get("summary", "").lower())
        ]
        matched_edges = [
            e for e in data["edges"]
            if query_l in (e.get("fact", "").lower() + " " + e.get("name", "").lower())
        ]

        facts = [e.get("fact", "") for e in matched_edges[:limit]]
        return {
            "facts": facts,
            "edges": matched_edges[:limit],
            "nodes": matched_nodes[:limit],
        }
