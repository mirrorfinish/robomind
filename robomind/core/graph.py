"""
RoboMind System Graph - NetworkX-based dependency graph for ROS2 systems.

Builds a complete graph representation of:
- ROS2 nodes as graph nodes
- Topics as graph nodes (with edges connecting publishers/subscribers)
- Services as graph nodes
- Parameters as graph nodes
- Launch files as graph nodes

Supports:
- Cycle detection
- Critical path analysis
- Centrality calculations
- Connected component analysis
- Export to GraphML, GEXF, DOT formats
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult, TopicConnection

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components in the system graph."""
    ROS2_NODE = auto()
    TOPIC = auto()
    SERVICE = auto()
    ACTION = auto()
    PARAMETER = auto()
    LAUNCH_FILE = auto()
    HARDWARE_TARGET = auto()


class EdgeType(Enum):
    """Types of edges in the system graph."""
    PUBLISHES = auto()
    SUBSCRIBES = auto()
    PROVIDES_SERVICE = auto()
    CALLS_SERVICE = auto()
    PROVIDES_ACTION = auto()
    CALLS_ACTION = auto()
    DECLARES_PARAMETER = auto()
    LAUNCHES = auto()
    RUNS_ON = auto()
    DEPENDS_ON = auto()


@dataclass
class GraphNode:
    """A node in the system graph."""
    id: str
    name: str
    component_type: ComponentType
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    hardware_target: Optional[str] = None
    package: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.component_type.name,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_number": self.line_number,
            "hardware_target": self.hardware_target,
            "package": self.package,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """An edge in the system graph."""
    source: str  # Node ID
    target: str  # Node ID
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.name,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class SystemGraph:
    """
    NetworkX-based system graph for ROS2 robotics systems.

    Provides:
    - Node and edge management
    - Graph analysis (cycles, centrality, components)
    - Export to various formats
    - Hardware target grouping
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for SystemGraph. Install with: pip install networkx")

        self._graph = nx.DiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self._nodes[node.id] = node

        # Build attributes, filtering out None values for export compatibility
        attrs = {
            "name": node.name,
            "component_type": node.component_type.name,
        }
        if node.file_path:
            attrs["file_path"] = str(node.file_path)
        if node.line_number is not None:
            attrs["line_number"] = node.line_number
        if node.hardware_target:
            attrs["hardware_target"] = node.hardware_target
        if node.package:
            attrs["package"] = node.package

        # Add non-None metadata
        for k, v in node.metadata.items():
            if v is not None:
                attrs[k] = v

        self._graph.add_node(node.id, **attrs)

    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        self._edges.append(edge)

        # Build attributes, filtering out None values for export compatibility
        attrs = {
            "edge_type": edge.edge_type.name,
            "weight": edge.weight,
        }
        # Add non-None metadata
        for k, v in edge.metadata.items():
            if v is not None:
                attrs[k] = v

        self._graph.add_edge(edge.source, edge.target, **attrs)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[GraphNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_edges(self) -> List[GraphEdge]:
        """Get all edges."""
        return list(self._edges)

    def get_nodes_by_type(self, component_type: ComponentType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.component_type == component_type]

    def get_nodes_by_hardware(self, hardware: str) -> List[GraphNode]:
        """Get all nodes running on specific hardware."""
        return [n for n in self._nodes.values() if n.hardware_target == hardware]

    def get_hardware_targets(self) -> Set[str]:
        """Get all unique hardware targets."""
        targets = set()
        for node in self._nodes.values():
            if node.hardware_target:
                targets.add(node.hardware_target)
        return targets

    def get_packages(self) -> Set[str]:
        """Get all unique packages."""
        packages = set()
        for node in self._nodes.values():
            if node.package:
                packages.add(node.package)
        return packages

    # Graph Analysis Methods

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        try:
            return list(nx.simple_cycles(self._graph))
        except nx.NetworkXError:
            return []

    def get_strongly_connected_components(self) -> List[Set[str]]:
        """Get strongly connected components (nodes reachable from each other)."""
        return [set(c) for c in nx.strongly_connected_components(self._graph)]

    def get_weakly_connected_components(self) -> List[Set[str]]:
        """Get weakly connected components (ignoring edge direction)."""
        return [set(c) for c in nx.weakly_connected_components(self._graph)]

    def calculate_centrality(self, method: str = 'betweenness') -> Dict[str, float]:
        """
        Calculate node centrality using various methods.

        Args:
            method: 'betweenness', 'degree', 'closeness', or 'eigenvector'

        Returns:
            Dict mapping node ID to centrality score
        """
        if len(self._graph) == 0:
            return {}

        if method == 'betweenness':
            return nx.betweenness_centrality(self._graph)
        elif method == 'degree':
            return nx.degree_centrality(self._graph)
        elif method == 'closeness':
            return nx.closeness_centrality(self._graph)
        elif method == 'eigenvector':
            try:
                return nx.eigenvector_centrality(self._graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge")
                return {}
        else:
            raise ValueError(f"Unknown centrality method: {method}")

    def get_critical_nodes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most critical nodes by betweenness centrality.

        Args:
            top_n: Number of nodes to return

        Returns:
            List of (node_id, centrality_score) tuples
        """
        centrality = self.calculate_centrality('betweenness')
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def get_node_dependencies(self, node_id: str) -> Dict[str, List[str]]:
        """
        Get dependencies for a specific node.

        Returns:
            Dict with 'upstream' (nodes this node depends on) and
            'downstream' (nodes that depend on this node)
        """
        if node_id not in self._graph:
            return {"upstream": [], "downstream": []}

        return {
            "upstream": list(self._graph.predecessors(node_id)),
            "downstream": list(self._graph.successors(node_id)),
        }

    def get_topological_order(self) -> Optional[List[str]]:
        """
        Get topological ordering of nodes (if graph is acyclic).

        Returns:
            List of node IDs in topological order, or None if cycles exist
        """
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            return None

    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Get shortest path between two nodes.

        Returns:
            List of node IDs in path, or None if no path exists
        """
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_paths(self, source: str, target: str, cutoff: int = 10) -> List[List[str]]:
        """
        Get all simple paths between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            cutoff: Maximum path length

        Returns:
            List of paths, where each path is a list of node IDs
        """
        try:
            return list(nx.all_simple_paths(self._graph, source, target, cutoff=cutoff))
        except nx.NodeNotFound:
            return []

    # Export Methods

    def export_graphml(self, output_path: Path):
        """Export graph to GraphML format."""
        nx.write_graphml(self._graph, str(output_path))
        logger.info(f"Exported GraphML to {output_path}")

    def export_gexf(self, output_path: Path):
        """Export graph to GEXF format (for Gephi)."""
        nx.write_gexf(self._graph, str(output_path))
        logger.info(f"Exported GEXF to {output_path}")

    def export_dot(self, output_path: Path):
        """Export graph to DOT format (for Graphviz)."""
        try:
            from networkx.drawing.nx_pydot import write_dot
            write_dot(self._graph, str(output_path))
            logger.info(f"Exported DOT to {output_path}")
        except ImportError:
            logger.warning("pydot not installed, cannot export DOT format")

    def export_adjacency_list(self) -> Dict[str, List[str]]:
        """Export as adjacency list."""
        return {node: list(self._graph.successors(node)) for node in self._graph.nodes()}

    # Statistics

    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_counts = {}
        for ctype in ComponentType:
            count = len(self.get_nodes_by_type(ctype))
            if count > 0:
                node_counts[ctype.name] = count

        edge_counts = {}
        for edge in self._edges:
            etype = edge.edge_type.name
            edge_counts[etype] = edge_counts.get(etype, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": node_counts,
            "edge_types": edge_counts,
            "hardware_targets": list(self.get_hardware_targets()),
            "packages": list(self.get_packages()),
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
            "strongly_connected_components": len(list(nx.strongly_connected_components(self._graph))),
            "weakly_connected_components": len(list(nx.weakly_connected_components(self._graph))),
        }

    def to_dict(self) -> Dict:
        """Convert entire graph to dictionary."""
        return {
            "stats": self.stats(),
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def __len__(self) -> int:
        return len(self._nodes)


class GraphBuilder:
    """
    Build a SystemGraph from ROS2 node and topic information.

    Combines:
    - ROS2NodeInfo objects
    - TopicGraphResult
    - Launch file information
    - Parameter configurations
    """

    def __init__(self):
        self.graph = SystemGraph()
        self._added_topics: Set[str] = set()
        self._added_services: Set[str] = set()

    def add_ros2_node(self, node: ROS2NodeInfo):
        """Add a ROS2 node and its connections to the graph."""
        # Create node ID
        node_id = f"node:{node.name}"

        # Add the node
        graph_node = GraphNode(
            id=node_id,
            name=node.name,
            component_type=ComponentType.ROS2_NODE,
            file_path=node.file_path,
            line_number=node.line_number,
            package=node.package_name,
            metadata={
                "class_name": node.class_name,
                "publishers": len(node.publishers),
                "subscribers": len(node.subscribers),
                "timers": len(node.timers),
                "services": len(node.services),
                "parameters": len(node.parameters),
                "has_tf_broadcaster": node.has_tf_broadcaster,
                "has_tf_listener": node.has_tf_listener,
            },
        )
        self.graph.add_node(graph_node)

        # Add topic connections
        for pub in node.publishers:
            topic_id = f"topic:{pub.topic}"
            self._ensure_topic_node(topic_id, pub.topic, pub.msg_type)

            self.graph.add_edge(GraphEdge(
                source=node_id,
                target=topic_id,
                edge_type=EdgeType.PUBLISHES,
                metadata={"msg_type": pub.msg_type, "qos": pub.qos},
            ))

        for sub in node.subscribers:
            topic_id = f"topic:{sub.topic}"
            self._ensure_topic_node(topic_id, sub.topic, sub.msg_type)

            self.graph.add_edge(GraphEdge(
                source=topic_id,
                target=node_id,
                edge_type=EdgeType.SUBSCRIBES,
                metadata={"msg_type": sub.msg_type, "qos": sub.qos, "callback": sub.callback},
            ))

        # Add service connections
        for srv in node.services:
            srv_id = f"service:{srv.name}"
            self._ensure_service_node(srv_id, srv.name, srv.srv_type)

            self.graph.add_edge(GraphEdge(
                source=node_id,
                target=srv_id,
                edge_type=EdgeType.PROVIDES_SERVICE,
                metadata={"srv_type": srv.srv_type, "callback": srv.callback},
            ))

        for client in node.service_clients:
            srv_id = f"service:{client.name}"
            self._ensure_service_node(srv_id, client.name, client.srv_type)

            self.graph.add_edge(GraphEdge(
                source=node_id,
                target=srv_id,
                edge_type=EdgeType.CALLS_SERVICE,
                metadata={"srv_type": client.srv_type},
            ))

        # Add action connections
        for action in node.action_servers:
            action_id = f"action:{action.name}"
            self._ensure_action_node(action_id, action.name, action.action_type)

            self.graph.add_edge(GraphEdge(
                source=node_id,
                target=action_id,
                edge_type=EdgeType.PROVIDES_ACTION,
                metadata={"action_type": action.action_type},
            ))

        for action in node.action_clients:
            action_id = f"action:{action.name}"
            self._ensure_action_node(action_id, action.name, action.action_type)

            self.graph.add_edge(GraphEdge(
                source=node_id,
                target=action_id,
                edge_type=EdgeType.CALLS_ACTION,
                metadata={"action_type": action.action_type},
            ))

        # Add parameter declarations
        for param in node.parameters:
            param_id = f"param:{node.name}.{param.name}"
            param_node = GraphNode(
                id=param_id,
                name=param.name,
                component_type=ComponentType.PARAMETER,
                metadata={
                    "default_value": param.default_value,
                    "param_type": param.param_type,
                    "node": node.name,
                },
            )
            self.graph.add_node(param_node)

            self.graph.add_edge(GraphEdge(
                source=node_id,
                target=param_id,
                edge_type=EdgeType.DECLARES_PARAMETER,
            ))

    def _ensure_topic_node(self, topic_id: str, topic_name: str, msg_type: str):
        """Ensure a topic node exists in the graph."""
        if topic_id not in self._added_topics:
            self._added_topics.add(topic_id)
            topic_node = GraphNode(
                id=topic_id,
                name=topic_name,
                component_type=ComponentType.TOPIC,
                metadata={"msg_type": msg_type},
            )
            self.graph.add_node(topic_node)

    def _ensure_service_node(self, srv_id: str, srv_name: str, srv_type: str):
        """Ensure a service node exists in the graph."""
        if srv_id not in self._added_services:
            self._added_services.add(srv_id)
            srv_node = GraphNode(
                id=srv_id,
                name=srv_name,
                component_type=ComponentType.SERVICE,
                metadata={"srv_type": srv_type},
            )
            self.graph.add_node(srv_node)

    def _ensure_action_node(self, action_id: str, action_name: str, action_type: str):
        """Ensure an action node exists in the graph."""
        if action_id not in self._added_services:  # Reuse set for simplicity
            self._added_services.add(action_id)
            action_node = GraphNode(
                id=action_id,
                name=action_name,
                component_type=ComponentType.ACTION,
                metadata={"action_type": action_type},
            )
            self.graph.add_node(action_node)

    def add_ros2_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add multiple ROS2 nodes to the graph."""
        for node in nodes:
            self.add_ros2_node(node)

    def add_topic_graph(self, topic_graph: TopicGraphResult):
        """
        Add topic graph information (supplements node-based extraction).

        This can fill in topics that weren't directly associated with nodes.
        """
        for topic_name, topic in topic_graph.topics.items():
            topic_id = f"topic:{topic_name}"
            if topic_id not in self._added_topics:
                self._ensure_topic_node(topic_id, topic_name, topic.msg_type)

    def add_hardware_target(self, node_name: str, hardware: str):
        """Associate a node with a hardware target."""
        node_id = f"node:{node_name}"
        node = self.graph.get_node(node_id)
        if node:
            node.hardware_target = hardware
            # Update in NetworkX graph too
            if node_id in self.graph._graph:
                self.graph._graph.nodes[node_id]['hardware_target'] = hardware

    def build(self) -> SystemGraph:
        """Return the built graph."""
        return self.graph


def build_system_graph(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
) -> SystemGraph:
    """
    Convenience function to build a system graph from nodes and topics.

    Args:
        nodes: List of ROS2NodeInfo objects
        topic_graph: Optional TopicGraphResult for additional topic info

    Returns:
        Built SystemGraph
    """
    builder = GraphBuilder()
    builder.add_ros2_nodes(nodes)

    if topic_graph:
        builder.add_topic_graph(topic_graph)

    return builder.build()


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python graph.py <project_path>")
        sys.exit(1)

    project_path = Path(sys.argv[1])

    # Extract nodes
    node_extractor = ROS2NodeExtractor()
    topic_extractor = TopicExtractor()
    all_nodes = []

    for py_file in project_path.rglob("*.py"):
        if "__pycache__" in str(py_file) or "/build/" in str(py_file):
            continue

        nodes = node_extractor.extract_from_file(py_file)
        all_nodes.extend(nodes)
        topic_extractor.add_nodes(nodes)

    topic_graph = topic_extractor.build()

    # Build system graph
    system_graph = build_system_graph(all_nodes, topic_graph)

    print("\n" + "=" * 60)
    print("SYSTEM GRAPH")
    print("=" * 60)
    print(json.dumps(system_graph.stats(), indent=2))

    print("\n" + "=" * 60)
    print("CRITICAL NODES (by betweenness centrality)")
    print("=" * 60)
    for node_id, score in system_graph.get_critical_nodes(10):
        node = system_graph.get_node(node_id)
        if node:
            print(f"  {node.name} ({node.component_type.name}): {score:.4f}")

    cycles = system_graph.find_cycles()
    if cycles:
        print("\n" + "=" * 60)
        print(f"CYCLES FOUND: {len(cycles)}")
        print("=" * 60)
        for cycle in cycles[:5]:
            print(f"  {' -> '.join(cycle)}")
