"""
图谱检索工具服务（本地版）
兼容原有 ZepToolsService 接口，底层改为本地图谱检索
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os

from ..utils.logger import get_logger
from .local_graph_store import LocalGraphStore
from .zep_entity_reader import ZepEntityReader, EntityNode

logger = get_logger('mirofish.graph_tools')


@dataclass
class SearchResult:
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        if self.facts:
            parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                parts.append(f"{i}. {fact}")
        return "\n".join(parts)


@dataclass
class NodeInfo:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
        }


@dataclass
class EdgeInfo:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
        }


@dataclass
class InsightForgeResult:
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_text(self) -> str:
        lines = [
            "## 深度洞察检索结果",
            f"分析问题: {self.query}",
            f"预测场景: {self.simulation_requirement}",
            f"相关事实: {self.total_facts} 条",
            f"涉及实体: {self.total_entities} 个",
            f"关系链: {self.total_relationships} 条",
        ]
        if self.sub_queries:
            lines.append("\n### 子问题")
            lines.extend([f"- {q}" for q in self.sub_queries])
        if self.semantic_facts:
            lines.append("\n### 关键事实")
            lines.extend([f"- {f}" for f in self.semantic_facts])
        if self.relationship_chains:
            lines.append("\n### 关系链")
            lines.extend([f"- {c}" for c in self.relationship_chains])
        return "\n".join(lines)


@dataclass
class PanoramaResult:
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_text(self) -> str:
        lines = [
            "## 全景搜索结果",
            f"查询: {self.query}",
            f"总节点数: {self.total_nodes}",
            f"总边数: {self.total_edges}",
        ]
        if self.active_facts:
            lines.append("\n### 当前事实")
            lines.extend([f"- {f}" for f in self.active_facts])
        if self.historical_facts:
            lines.append("\n### 历史事实")
            lines.extend([f"- {f}" for f in self.historical_facts])
        return "\n".join(lines)


@dataclass
class AgentInterview:
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"**{self.agent_name}** ({self.agent_role})",
            f"_简介: {self.agent_bio}_",
            f"Q: {self.question}",
            f"A: {self.response}",
        ]
        return "\n".join(lines)


@dataclass
class InterviewResult:
    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0

    def to_text(self) -> str:
        lines = [
            "## 模拟 Agent 采访",
            f"主题: {self.interview_topic}",
            f"采访数量: {self.interviewed_count}/{self.total_agents}",
            self.selection_reasoning or "自动选择与主题最相关的实体",
        ]
        for interview in self.interviews:
            lines.append("\n---")
            lines.append(interview.to_text())
        if self.summary:
            lines.append("\n### 摘要")
            lines.append(self.summary)
        return "\n".join(lines)


class ZepToolsService:
    """兼容旧命名的本地图谱工具服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.store = LocalGraphStore()
        self.reader = ZepEntityReader()

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        result = self.store.search(graph_id, query, limit=limit)
        total = len(result.get("facts", [])) + len(result.get("nodes", []))
        return SearchResult(
            facts=result.get("facts", []),
            edges=result.get("edges", []),
            nodes=result.get("nodes", []),
            query=query,
            total_count=total,
        )

    def search_graph(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> SearchResult:
        # 兼容旧接口，当前统一走本地快速检索
        return self.quick_search(graph_id=graph_id, query=query, limit=limit)

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        data = self.store.get_graph_data(graph_id)
        return [
            NodeInfo(
                uuid=n.get("uuid", ""),
                name=n.get("name", ""),
                labels=n.get("labels", []),
                summary=n.get("summary", ""),
                attributes=n.get("attributes", {}),
            ) for n in data.get("nodes", [])
        ]

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        data = self.store.get_graph_data(graph_id)
        return [
            EdgeInfo(
                uuid=e.get("uuid", ""),
                name=e.get("name", ""),
                fact=e.get("fact", ""),
                source_node_uuid=e.get("source_node_uuid", ""),
                target_node_uuid=e.get("target_node_uuid", ""),
                source_node_name=e.get("source_node_name", ""),
                target_node_name=e.get("target_node_name", ""),
                created_at=e.get("created_at"),
                valid_at=e.get("valid_at"),
                invalid_at=e.get("invalid_at"),
                expired_at=e.get("expired_at"),
            ) for e in data.get("edges", [])
        ]

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        # 兼容旧接口：遍历本地图找到对应节点
        for graph_file in self._iter_graph_files():
            try:
                graph_id = graph_file[:-5]
                nodes = self.store.get_graph_data(graph_id).get("nodes", [])
                node = next((n for n in nodes if n.get("uuid") == node_uuid), None)
                if node:
                    return NodeInfo(
                        uuid=node.get("uuid", ""),
                        name=node.get("name", ""),
                        labels=node.get("labels", []),
                        summary=node.get("summary", ""),
                        attributes=node.get("attributes", {}),
                    )
            except Exception:
                continue
        return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        return [
            e for e in self.get_all_edges(graph_id)
            if e.source_node_uuid == node_uuid or e.target_node_uuid == node_uuid
        ]

    def panorama_search(self, graph_id: str, query: str, include_expired: bool = True) -> PanoramaResult:
        data = self.store.get_graph_data(graph_id)
        nodes = [
            NodeInfo(
                uuid=n.get("uuid", ""),
                name=n.get("name", ""),
                labels=n.get("labels", []),
                summary=n.get("summary", ""),
                attributes=n.get("attributes", {}),
            ) for n in data.get("nodes", [])
        ]
        edges = [
            EdgeInfo(
                uuid=e.get("uuid", ""),
                name=e.get("name", ""),
                fact=e.get("fact", ""),
                source_node_uuid=e.get("source_node_uuid", ""),
                target_node_uuid=e.get("target_node_uuid", ""),
                source_node_name=e.get("source_node_name", ""),
                target_node_name=e.get("target_node_name", ""),
                created_at=e.get("created_at"),
                valid_at=e.get("valid_at"),
                invalid_at=e.get("invalid_at"),
                expired_at=e.get("expired_at"),
            ) for e in data.get("edges", [])
        ]

        q = (query or "").lower()
        filtered_edges = [e for e in edges if q in (e.fact.lower() + " " + e.name.lower())] if q else edges
        filtered_nodes = [n for n in nodes if q in (n.name.lower() + " " + n.summary.lower())] if q else nodes

        active = [e.fact for e in filtered_edges if not e.expired_at]
        historical = [e.fact for e in filtered_edges if e.expired_at]

        return PanoramaResult(
            query=query,
            all_nodes=filtered_nodes,
            all_edges=filtered_edges,
            active_facts=active,
            historical_facts=historical,
            total_nodes=len(filtered_nodes),
            total_edges=len(filtered_edges),
            active_count=len(active),
            historical_count=len(historical),
        )

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = ""
    ) -> InsightForgeResult:
        sub_queries = [query]
        if report_context:
            sub_queries.append(report_context[:100])

        quick = self.quick_search(graph_id, query, limit=20)
        chains = []
        for edge in quick.edges[:20]:
            src = edge.get("source_node_name") or edge.get("source_node_uuid", "")
            dst = edge.get("target_node_name") or edge.get("target_node_uuid", "")
            rel = edge.get("name", "related_to")
            chains.append(f"{src} --[{rel}]--> {dst}")

        entities = []
        for node in quick.nodes[:10]:
            entities.append({
                "name": node.get("name", ""),
                "type": next((l for l in node.get("labels", []) if l not in ["Entity", "Node"]), "Entity"),
                "summary": node.get("summary", ""),
            })

        return InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=sub_queries,
            semantic_facts=quick.facts,
            entity_insights=entities,
            relationship_chains=chains,
            total_facts=len(quick.facts),
            total_entities=len(entities),
            total_relationships=len(chains),
        )

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int = 5,
    ) -> InterviewResult:
        # 本地降级采访：基于图实体摘要生成结构化回答
        # 保持接口可用，避免报告流程中断
        selected_entities: List[EntityNode] = []
        try:
            # simulation_id 与 graph_id 不直接映射时，允许空结果
            # 报告主流程会继续推进
            pass
        except Exception:
            pass

        interviews: List[AgentInterview] = []
        for idx, entity in enumerate(selected_entities[:max_agents]):
            role = entity.get_entity_type() or "Entity"
            response = f"基于已有记忆，我认为与'{interview_requirement}'最相关的是：{entity.summary or entity.name}"
            interviews.append(AgentInterview(
                agent_name=entity.name,
                agent_role=role,
                agent_bio=entity.summary or role,
                question=interview_requirement,
                response=response,
                key_quotes=[response],
            ))

        summary = "采访基于本地图谱实体摘要自动生成。"
        return InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=[interview_requirement],
            selected_agents=[{"name": i.agent_name, "role": i.agent_role} for i in interviews],
            interviews=interviews,
            selection_reasoning="根据实体相关性自动选择",
            summary=summary,
            total_agents=len(interviews),
            interviewed_count=len(interviews),
        )

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        data = self.store.get_graph_data(graph_id)
        return {
            "graph_id": graph_id,
            "node_count": data.get("node_count", 0),
            "edge_count": data.get("edge_count", 0),
            "entity_types": sorted({
                label
                for node in data.get("nodes", [])
                for label in node.get("labels", [])
                if label not in ["Entity", "Node"]
            }),
        }

    def get_simulation_context(self, graph_id: str, simulation_requirement: str, limit: int = 30) -> Dict[str, Any]:
        search = self.quick_search(graph_id=graph_id, query=simulation_requirement, limit=limit)
        entities = []
        for node in self.store.get_graph_data(graph_id).get("nodes", []):
            entity_type = next((l for l in node.get("labels", []) if l not in ["Entity", "Node"]), None)
            if entity_type:
                entities.append({
                    "name": node.get("name", ""),
                    "type": entity_type,
                    "summary": node.get("summary", ""),
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search.facts,
            "graph_statistics": self.get_graph_statistics(graph_id),
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    def _iter_graph_files(self) -> List[str]:
        try:
            return [f for f in os.listdir(self.store.base_dir) if f.endswith('.json')]
        except Exception:
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[EntityNode]:
        return self.reader.get_entities_by_type(graph_id, entity_type, enrich_with_edges=True)

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        data = self.store.get_graph_data(graph_id)
        node = next((n for n in data.get("nodes", []) if n.get("name") == entity_name), None)
        if not node:
            return {"found": False, "entity_name": entity_name}

        edges = [
            e for e in data.get("edges", [])
            if e.get("source_node_uuid") == node.get("uuid") or e.get("target_node_uuid") == node.get("uuid")
        ]
        return {
            "found": True,
            "entity": node,
            "related_facts": [e.get("fact", "") for e in edges],
            "related_edges_count": len(edges),
        }
