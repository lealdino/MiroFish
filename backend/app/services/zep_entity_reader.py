"""
实体读取与过滤服务（本地版）
从本地图谱读取节点，筛选符合实体类型定义的节点
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from .local_graph_store import LocalGraphStore

logger = get_logger('mirofish.entity_reader')


@dataclass
class EntityNode:
    """实体节点数据结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """兼容旧命名的本地实体读取器"""

    def __init__(self, api_key: Optional[str] = None):
        self.store = LocalGraphStore()

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        data = self.store.get_graph_data(graph_id)
        return data.get("nodes", [])

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        data = self.store.get_graph_data(graph_id)
        return data.get("edges", [])

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[Dict[str, Any]]:
        edges = self.get_all_edges(graph_id)
        return [
            e for e in edges
            if e.get("source_node_uuid") == node_uuid or e.get("target_node_uuid") == node_uuid
        ]

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        all_nodes = self.get_all_nodes(graph_id)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n.get("uuid"): n for n in all_nodes}

        filtered_entities: List[EntityNode] = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            if not custom_labels:
                continue

            if defined_entity_types:
                matches = [l for l in custom_labels if l in defined_entity_types]
                if not matches:
                    continue
                entity_type = matches[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                for edge in all_edges:
                    if edge.get("source_node_uuid") == entity.uuid:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge.get("name", ""),
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge.get("target_node_uuid", ""),
                        })
                        related_node_uuids.add(edge.get("target_node_uuid"))
                    elif edge.get("target_node_uuid") == entity.uuid:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge.get("name", ""),
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge.get("source_node_uuid", ""),
                        })
                        related_node_uuids.add(edge.get("source_node_uuid"))

                entity.related_edges = related_edges
                entity.related_nodes = [
                    {
                        "uuid": node_map[nid].get("uuid", ""),
                        "name": node_map[nid].get("name", ""),
                        "labels": node_map[nid].get("labels", []),
                        "summary": node_map[nid].get("summary", ""),
                    }
                    for nid in related_node_uuids if nid in node_map
                ]

            filtered_entities.append(entity)

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=len(all_nodes),
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        all_nodes = self.get_all_nodes(graph_id)
        node = next((n for n in all_nodes if n.get("uuid") == entity_uuid), None)
        if not node:
            return None

        result = self.filter_defined_entities(graph_id, enrich_with_edges=True)
        for entity in result.entities:
            if entity.uuid == entity_uuid:
                return entity

        return EntityNode(
            uuid=node.get("uuid", ""),
            name=node.get("name", ""),
            labels=node.get("labels", []),
            summary=node.get("summary", ""),
            attributes=node.get("attributes", {}),
        )

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
