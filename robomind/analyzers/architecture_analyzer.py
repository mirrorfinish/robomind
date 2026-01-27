"""
RoboMind Architecture Analyzer

Analyzes ROS2 system architecture for:
- Circular dependencies
- Namespace collisions
- Architecture anti-patterns
- Property-based queries
- Dead code detection
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult
from robomind.ros2.launch_analyzer import LaunchTopology


@dataclass
class ArchitectureFinding:
    """An architecture-related finding."""
    finding_type: str
    severity: str
    title: str
    description: str
    affected_nodes: List[str]
    affected_topics: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class PropertyQuery:
    """A property query to check against the architecture."""
    name: str
    description: str
    check: Callable  # Function that returns (passed: bool, violations: List[str])


class ArchitectureAnalyzer:
    """
    Analyzes ROS2 system architecture for patterns and anti-patterns.

    Capabilities:
    - Detect circular dependencies between nodes
    - Find namespace/topic collisions
    - Check architectural properties (queries)
    - Identify dead code (unreachable nodes)
    """

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.launch_topology: Optional[LaunchTopology] = None
        self.launched_nodes: Set[str] = set()

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def add_topic_graph(self, topic_graph: TopicGraphResult):
        """Add topic graph."""
        self.topic_graph = topic_graph

    def add_launch_topology(self, topology: LaunchTopology, launched_nodes: Set[str] = None):
        """Add launch topology for dead code detection."""
        self.launch_topology = topology
        if launched_nodes:
            self.launched_nodes = launched_nodes

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build a graph of node dependencies based on topic connections."""
        # Node A depends on Node B if A subscribes to a topic that B publishes
        dependencies: Dict[str, Set[str]] = {n.name: set() for n in self.nodes}

        if not self.topic_graph:
            return dependencies

        # Build publisher map: topic -> [node names]
        pub_map: Dict[str, List[str]] = {}
        for node in self.nodes:
            for pub in node.publishers:
                if pub.topic:
                    if pub.topic not in pub_map:
                        pub_map[pub.topic] = []
                    pub_map[pub.topic].append(node.name)

        # For each subscriber, add dependency on publishers
        for node in self.nodes:
            for sub in node.subscribers:
                if sub.topic and sub.topic in pub_map:
                    for pub_node in pub_map[sub.topic]:
                        if pub_node != node.name:
                            dependencies[node.name].add(pub_node)

        return dependencies

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find all cycles in a dependency graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def detect_circular_dependencies(self) -> List[ArchitectureFinding]:
        """Detect circular dependencies between nodes."""
        findings = []

        graph = self._build_dependency_graph()
        cycles = self._find_cycles(graph)

        for cycle in cycles:
            # Determine severity based on cycle content
            severity = 'medium'
            if any('control' in n.lower() or 'motor' in n.lower() for n in cycle):
                severity = 'high'

            cycle_str = ' → '.join(cycle)
            findings.append(ArchitectureFinding(
                finding_type='circular_dependency',
                severity=severity,
                title=f'Circular dependency detected',
                description=f'Nodes form a dependency cycle: {cycle_str}',
                affected_nodes=cycle[:-1],  # Remove duplicate end node
                recommendation='Break the cycle by using a message broker pattern or restructuring node responsibilities',
            ))

        return findings

    def detect_namespace_collisions(self) -> List[ArchitectureFinding]:
        """Detect potential namespace/topic naming collisions."""
        findings = []

        # Check for same topic with different message types
        topic_types: Dict[str, Set[str]] = {}
        topic_nodes: Dict[str, Set[str]] = {}

        for node in self.nodes:
            for pub in node.publishers:
                if pub.topic:
                    if pub.topic not in topic_types:
                        topic_types[pub.topic] = set()
                        topic_nodes[pub.topic] = set()
                    if pub.msg_type:
                        topic_types[pub.topic].add(pub.msg_type)
                    topic_nodes[pub.topic].add(node.name)

            for sub in node.subscribers:
                if sub.topic:
                    if sub.topic not in topic_types:
                        topic_types[sub.topic] = set()
                        topic_nodes[sub.topic] = set()
                    if sub.msg_type:
                        topic_types[sub.topic].add(sub.msg_type)
                    topic_nodes[sub.topic].add(node.name)

        # Find topics with multiple message types
        for topic, types in topic_types.items():
            if len(types) > 1:
                findings.append(ArchitectureFinding(
                    finding_type='type_mismatch',
                    severity='high',
                    title=f'Topic type mismatch: {topic}',
                    description=f'Topic "{topic}" uses multiple message types: {", ".join(types)}',
                    affected_nodes=list(topic_nodes[topic]),
                    affected_topics=[topic],
                    recommendation='Ensure all publishers and subscribers use the same message type',
                ))

        # Check for similar topic names (potential typos)
        topics = list(topic_types.keys())
        for i, t1 in enumerate(topics):
            for t2 in topics[i+1:]:
                # Check if topics are very similar (potential typo)
                t1_base = t1.replace('/', '').replace('_', '').lower()
                t2_base = t2.replace('/', '').replace('_', '').lower()

                if t1_base == t2_base and t1 != t2:
                    findings.append(ArchitectureFinding(
                        finding_type='similar_topic_names',
                        severity='low',
                        title=f'Similar topic names',
                        description=f'Topics "{t1}" and "{t2}" are very similar (potential inconsistency)',
                        affected_nodes=list(topic_nodes[t1] | topic_nodes[t2]),
                        affected_topics=[t1, t2],
                        recommendation='Standardize topic naming to avoid confusion',
                    ))

        return findings

    def detect_dead_code(self) -> List[ArchitectureFinding]:
        """Detect nodes that are defined but never launched."""
        findings = []

        if not self.launched_nodes:
            return findings

        # Find nodes that exist in code but are not launched
        launched_normalized = {n.lower().replace('_', '').replace('-', '') for n in self.launched_nodes}

        dead_nodes = []
        for node in self.nodes:
            node_normalized = node.name.lower().replace('_', '').replace('-', '')
            class_normalized = (node.class_name or '').lower().replace('_', '').replace('-', '')
            file_normalized = Path(node.file_path).stem.lower().replace('_', '').replace('-', '') if node.file_path else ''

            is_launched = any(
                launched in [node_normalized, class_normalized, file_normalized]
                or node_normalized in launched
                or launched in node_normalized
                for launched in launched_normalized
            )

            if not is_launched:
                dead_nodes.append(node)

        if dead_nodes:
            findings.append(ArchitectureFinding(
                finding_type='dead_code',
                severity='low',
                title=f'{len(dead_nodes)} nodes defined but not launched',
                description=f'These nodes exist in source but are not in the launch file: {", ".join(n.name for n in dead_nodes[:10])}{"..." if len(dead_nodes) > 10 else ""}',
                affected_nodes=[n.name for n in dead_nodes],
                recommendation='Remove dead code or add nodes to launch configuration',
            ))

        return findings

    def check_property_queries(self) -> List[ArchitectureFinding]:
        """Check architectural property queries."""
        findings = []

        # Property 1: All cmd_vel publishers should have emergency stop capability
        cmd_vel_publishers = set()
        estop_subscribers = set()

        for node in self.nodes:
            for pub in node.publishers:
                if pub.topic and 'cmd_vel' in pub.topic.lower():
                    cmd_vel_publishers.add(node.name)
            for sub in node.subscribers:
                if sub.topic and ('emergency' in sub.topic.lower() or 'stop' in sub.topic.lower()):
                    estop_subscribers.add(node.name)

        nodes_without_estop = cmd_vel_publishers - estop_subscribers
        if nodes_without_estop:
            findings.append(ArchitectureFinding(
                finding_type='property_violation',
                severity='high',
                title='cmd_vel publishers without emergency stop',
                description=f'Nodes that publish movement commands but do not subscribe to emergency stop: {", ".join(nodes_without_estop)}',
                affected_nodes=list(nodes_without_estop),
                recommendation='All nodes that can move the robot should respond to emergency stop signals',
            ))

        # Property 2: No node should publish and subscribe to the same topic (potential feedback loop)
        for node in self.nodes:
            pub_topics = {p.topic for p in node.publishers if p.topic}
            sub_topics = {s.topic for s in node.subscribers if s.topic}
            overlap = pub_topics & sub_topics

            if overlap:
                findings.append(ArchitectureFinding(
                    finding_type='feedback_loop',
                    severity='medium',
                    title=f'Potential feedback loop in {node.name}',
                    description=f'Node publishes and subscribes to same topics: {", ".join(overlap)}',
                    affected_nodes=[node.name],
                    affected_topics=list(overlap),
                    recommendation='Review if self-subscription is intentional; may cause infinite loops',
                ))

        # Property 3: Critical topics should have multiple subscribers (redundancy)
        if self.topic_graph:
            critical_topics = ['cmd_vel', 'emergency_stop', 'estop']
            for topic_name, topic_conn in self.topic_graph.topics.items():
                is_critical = any(ct in topic_name.lower() for ct in critical_topics)
                if is_critical and len(topic_conn.subscribers) < 2:
                    findings.append(ArchitectureFinding(
                        finding_type='no_redundancy',
                        severity='low',
                        title=f'Critical topic lacks redundancy: {topic_name}',
                        description=f'Topic "{topic_name}" has only {len(topic_conn.subscribers)} subscriber(s)',
                        affected_nodes=topic_conn.subscribers,
                        affected_topics=[topic_name],
                        recommendation='Consider adding monitoring/logging subscriber for critical topics',
                    ))

        return findings

    def analyze(self) -> List[ArchitectureFinding]:
        """Run all architecture analyses."""
        findings = []

        findings.extend(self.detect_circular_dependencies())
        findings.extend(self.detect_namespace_collisions())
        findings.extend(self.detect_dead_code())
        findings.extend(self.check_property_queries())

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 4))

        return findings


def analyze_architecture(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    launched_nodes: Optional[Set[str]] = None,
) -> List[ArchitectureFinding]:
    """Convenience function for architecture analysis."""
    analyzer = ArchitectureAnalyzer()
    analyzer.add_nodes(nodes)
    if topic_graph:
        analyzer.add_topic_graph(topic_graph)
    if launched_nodes:
        analyzer.launched_nodes = launched_nodes
    return analyzer.analyze()
