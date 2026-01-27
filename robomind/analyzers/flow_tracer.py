"""
RoboMind Flow Tracer - Trace data flow paths through ROS2 systems.

This module traces how data flows from one component to another:
- Find all paths between two nodes
- Trace a topic from publishers to final consumers
- Identify potential bottlenecks and single points of failure
- Generate flow diagrams
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)


@dataclass
class FlowPath:
    """A single flow path through the system."""
    nodes: List[str]  # Sequence of node names
    topics: List[str]  # Topics connecting the nodes
    length: int = 0
    bottlenecks: List[str] = field(default_factory=list)  # Single points of failure

    def __post_init__(self):
        self.length = len(self.nodes)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "nodes": self.nodes,
            "topics": self.topics,
            "length": self.length,
            "bottlenecks": self.bottlenecks,
        }

    def to_mermaid(self) -> str:
        """Convert to Mermaid diagram syntax."""
        if not self.nodes:
            return ""

        lines = ["graph LR"]
        for i in range(len(self.nodes) - 1):
            node_a = self.nodes[i].replace("/", "_").replace("-", "_")
            node_b = self.nodes[i + 1].replace("/", "_").replace("-", "_")
            topic = self.topics[i] if i < len(self.topics) else ""
            topic_short = topic.split("/")[-1] if topic else ""

            if self.nodes[i] in self.bottlenecks:
                lines.append(f"    {node_a}[[\"{self.nodes[i]}\"]]")
            else:
                lines.append(f"    {node_a}[\"{self.nodes[i]}\"]")

            lines.append(f"    {node_a} -->|{topic_short}| {node_b}")

        # Last node
        last = self.nodes[-1].replace("/", "_").replace("-", "_")
        if self.nodes[-1] in self.bottlenecks:
            lines.append(f"    {last}[[\"{self.nodes[-1]}\"]]")
        else:
            lines.append(f"    {last}[\"{self.nodes[-1]}\"]")

        return "\n".join(lines)


@dataclass
class FlowTraceResult:
    """Result of a flow trace operation."""
    paths: List[FlowPath] = field(default_factory=list)
    source: str = ""
    target: str = ""
    bottlenecks: List[str] = field(default_factory=list)  # Nodes appearing in all paths
    success: bool = False
    error: Optional[str] = None

    def summary(self) -> Dict:
        """Generate summary statistics."""
        if not self.paths:
            return {
                "paths_found": 0,
                "source": self.source,
                "target": self.target,
                "bottlenecks": [],
            }

        return {
            "paths_found": len(self.paths),
            "source": self.source,
            "target": self.target,
            "shortest_path": min(p.length for p in self.paths),
            "longest_path": max(p.length for p in self.paths),
            "bottlenecks": self.bottlenecks,
            "total_nodes": len(set(n for p in self.paths for n in p.nodes)),
            "total_topics": len(set(t for p in self.paths for t in p.topics)),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "summary": self.summary(),
            "paths": [p.to_dict() for p in self.paths],
            "error": self.error,
        }


class FlowTracer:
    """
    Trace data flow paths through ROS2 systems.

    Usage:
        tracer = FlowTracer(nodes, topic_graph)
        result = tracer.trace("sensor_node", "controller_node")
        print(result.summary())

        # Or trace all consumers of a topic
        result = tracer.trace_topic("/cmd_vel")
    """

    def __init__(
        self,
        nodes: List[ROS2NodeInfo],
        topic_graph: Optional[TopicGraphResult] = None,
    ):
        self.nodes = nodes
        self.topic_graph = topic_graph
        self._node_map: Dict[str, ROS2NodeInfo] = {n.name: n for n in nodes}

        # Build adjacency maps
        self._forward_adj: Dict[str, Dict[str, str]] = {}  # node -> {target_node: connecting_topic}
        self._reverse_adj: Dict[str, Dict[str, str]] = {}  # node -> {source_node: connecting_topic}

        self._build_adjacency()

    def _build_adjacency(self):
        """Build forward and reverse adjacency maps."""
        if self.topic_graph:
            for topic_name, topic in self.topic_graph.topics.items():
                for pub_node in topic.publishers:
                    if pub_node not in self._forward_adj:
                        self._forward_adj[pub_node] = {}

                    for sub_node in topic.subscribers:
                        if pub_node != sub_node:
                            self._forward_adj[pub_node][sub_node] = topic_name

                            if sub_node not in self._reverse_adj:
                                self._reverse_adj[sub_node] = {}
                            self._reverse_adj[sub_node][pub_node] = topic_name
        else:
            # Build from nodes directly
            topic_publishers: Dict[str, List[str]] = {}
            topic_subscribers: Dict[str, List[str]] = {}

            for node in self.nodes:
                for pub in node.publishers:
                    if pub.topic not in topic_publishers:
                        topic_publishers[pub.topic] = []
                    topic_publishers[pub.topic].append(node.name)

                for sub in node.subscribers:
                    if sub.topic not in topic_subscribers:
                        topic_subscribers[sub.topic] = []
                    topic_subscribers[sub.topic].append(node.name)

            # Match publishers to subscribers
            all_topics = set(topic_publishers.keys()) | set(topic_subscribers.keys())
            for topic in all_topics:
                for pub_node in topic_publishers.get(topic, []):
                    if pub_node not in self._forward_adj:
                        self._forward_adj[pub_node] = {}

                    for sub_node in topic_subscribers.get(topic, []):
                        if pub_node != sub_node:
                            self._forward_adj[pub_node][sub_node] = topic

                            if sub_node not in self._reverse_adj:
                                self._reverse_adj[sub_node] = {}
                            self._reverse_adj[sub_node][pub_node] = topic

    def trace(
        self,
        source: str,
        target: str,
        max_paths: int = 10,
        max_depth: int = 15,
    ) -> FlowTraceResult:
        """
        Find all paths from source to target.

        Args:
            source: Source node name
            target: Target node name
            max_paths: Maximum number of paths to find
            max_depth: Maximum path length

        Returns:
            FlowTraceResult with all paths found
        """
        result = FlowTraceResult(source=source, target=target)

        # Validate inputs
        if source not in self._forward_adj and source not in self._node_map:
            result.error = f"Source node '{source}' not found"
            return result

        if target not in self._reverse_adj and target not in self._node_map:
            result.error = f"Target node '{target}' not found"
            return result

        # BFS to find all paths
        paths = self._find_all_paths(source, target, max_paths, max_depth)

        for node_path, topic_path in paths:
            result.paths.append(FlowPath(
                nodes=node_path,
                topics=topic_path,
            ))

        result.success = len(result.paths) > 0

        # Find bottlenecks (nodes in ALL paths)
        if len(result.paths) > 1:
            all_nodes = [set(p.nodes[1:-1]) for p in result.paths]  # Exclude source/target
            if all_nodes:
                common = all_nodes[0]
                for nodes in all_nodes[1:]:
                    common &= nodes
                result.bottlenecks = list(common)

                # Mark bottlenecks on paths
                for path in result.paths:
                    path.bottlenecks = [n for n in path.nodes if n in common]

        return result

    def _find_all_paths(
        self,
        source: str,
        target: str,
        max_paths: int,
        max_depth: int,
    ) -> List[Tuple[List[str], List[str]]]:
        """Find all paths using BFS."""
        paths = []

        # Queue: (current_node, path_so_far, topics_so_far)
        queue = deque([(source, [source], [])])
        visited_paths: Set[tuple] = set()

        while queue and len(paths) < max_paths:
            current, path, topics = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == target:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)
                    paths.append((path, topics))
                continue

            # Explore neighbors
            for neighbor, topic in self._forward_adj.get(current, {}).items():
                if neighbor not in path:  # Avoid cycles
                    queue.append((
                        neighbor,
                        path + [neighbor],
                        topics + [topic],
                    ))

        return paths

    def trace_topic(self, topic_name: str) -> FlowTraceResult:
        """
        Trace all data flows through a specific topic.

        Args:
            topic_name: Topic name to trace

        Returns:
            FlowTraceResult showing publishers and all downstream consumers
        """
        result = FlowTraceResult(source=topic_name, target="(all consumers)")

        if not self.topic_graph:
            result.error = "Topic graph not available"
            return result

        # Normalize topic name
        normalized = topic_name
        if not topic_name.startswith("/"):
            normalized = f"/{topic_name}"
            if normalized not in self.topic_graph.topics:
                normalized = topic_name

        topic = self.topic_graph.topics.get(normalized) or self.topic_graph.topics.get(topic_name)
        if not topic:
            result.error = f"Topic '{topic_name}' not found"
            return result

        # For each publisher, trace to all reachable consumers
        for publisher in topic.publishers:
            for subscriber in topic.subscribers:
                # Direct connection
                path = FlowPath(
                    nodes=[publisher, subscriber],
                    topics=[topic_name],
                )
                result.paths.append(path)

                # Continue tracing from subscriber
                downstream = self._trace_downstream(subscriber, max_depth=10)
                for node_path, topic_path in downstream:
                    full_path = [publisher] + node_path
                    full_topics = [topic_name] + topic_path
                    result.paths.append(FlowPath(
                        nodes=full_path,
                        topics=full_topics,
                    ))

        result.success = len(result.paths) > 0
        return result

    def _trace_downstream(
        self,
        start: str,
        max_depth: int = 10,
    ) -> List[Tuple[List[str], List[str]]]:
        """Trace all downstream consumers from a node."""
        paths = []
        visited = {start}

        queue = deque([([start], [])])

        while queue:
            path, topics = queue.popleft()
            current = path[-1]

            if len(path) > max_depth:
                continue

            # Get downstream neighbors
            neighbors = self._forward_adj.get(current, {})
            if not neighbors and len(path) > 1:
                # End of chain
                paths.append((path, topics))

            for neighbor, topic in neighbors.items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        path + [neighbor],
                        topics + [topic],
                    ))

        return paths

    def get_flow_summary(self) -> Dict[str, Any]:
        """Get overall flow summary for the system."""
        # Find entry points (nodes with no upstream)
        entry_points = []
        for node in self._node_map.keys():
            if node not in self._reverse_adj or not self._reverse_adj[node]:
                # Check if it publishes anything
                if node in self._forward_adj and self._forward_adj[node]:
                    entry_points.append(node)

        # Find exit points (nodes with no downstream)
        exit_points = []
        for node in self._node_map.keys():
            if node not in self._forward_adj or not self._forward_adj[node]:
                # Check if it subscribes to anything
                if node in self._reverse_adj and self._reverse_adj[node]:
                    exit_points.append(node)

        # Find high-traffic nodes (many connections)
        traffic = {}
        for node in self._node_map.keys():
            incoming = len(self._reverse_adj.get(node, {}))
            outgoing = len(self._forward_adj.get(node, {}))
            traffic[node] = incoming + outgoing

        high_traffic = sorted(traffic.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "entry_points": entry_points,
            "exit_points": exit_points,
            "high_traffic_nodes": [{"node": n, "connections": c} for n, c in high_traffic if c > 0],
            "total_connections": sum(len(adj) for adj in self._forward_adj.values()),
        }


def trace_flow(
    source: str,
    target: str,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
) -> FlowTraceResult:
    """
    Convenience function to trace flow between two nodes.

    Args:
        source: Source node name
        target: Target node name
        nodes: List of ROS2NodeInfo
        topic_graph: Optional TopicGraphResult

    Returns:
        FlowTraceResult with all paths found
    """
    tracer = FlowTracer(nodes, topic_graph)
    return tracer.trace(source, target)


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python flow_tracer.py <project_path> <source_node> <target_node>")
        print("       python flow_tracer.py <project_path> --topic <topic_name>")
        sys.exit(1)

    project_path = Path(sys.argv[1])

    # Extract nodes
    node_extractor = ROS2NodeExtractor()
    topic_extractor = TopicExtractor()
    all_nodes = []

    print(f"Analyzing {project_path}...")
    for py_file in project_path.rglob("*.py"):
        if "__pycache__" in str(py_file) or "/build/" in str(py_file):
            continue
        nodes = node_extractor.extract_from_file(py_file)
        all_nodes.extend(nodes)
        topic_extractor.add_nodes(nodes)

    topic_graph = topic_extractor.build()
    print(f"Found {len(all_nodes)} nodes, {len(topic_graph.topics)} topics")

    # Create tracer
    tracer = FlowTracer(all_nodes, topic_graph)

    if sys.argv[2] == "--topic":
        topic_name = sys.argv[3]
        print(f"\nTracing topic: {topic_name}")
        result = tracer.trace_topic(topic_name)
    else:
        source = sys.argv[2]
        target = sys.argv[3]
        print(f"\nTracing from {source} to {target}")
        result = tracer.trace(source, target)

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    print(f"\nFound {len(result.paths)} paths")
    print(f"Summary: {json.dumps(result.summary(), indent=2)}")

    for i, path in enumerate(result.paths[:5], 1):
        print(f"\nPath {i}: {' -> '.join(path.nodes)}")
        print(f"  Topics: {' -> '.join(path.topics)}")
        if path.bottlenecks:
            print(f"  Bottlenecks: {', '.join(path.bottlenecks)}")

    # Print flow summary
    print("\n" + "=" * 60)
    print("FLOW SUMMARY")
    print("=" * 60)
    summary = tracer.get_flow_summary()
    print(f"Entry points: {', '.join(summary['entry_points'][:5])}")
    print(f"Exit points: {', '.join(summary['exit_points'][:5])}")
    print(f"High traffic nodes:")
    for item in summary['high_traffic_nodes'][:5]:
        print(f"  {item['node']}: {item['connections']} connections")
