"""
RoboMind Coupling Analyzer - Calculate coupling strength between ROS2 components.

Coupling factors (weighted sum → 0.0-1.0):
1. Topic connections (0.0-0.4): Multiple topics, high-frequency topics
2. Shared parameters (0.0-0.3): More shared params = higher coupling
3. Data dependencies (0.0-0.2): Message type complexity, data flow
4. Temporal coupling (0.0-0.1): Timer synchronization, delays

High coupling indicates components that are tightly integrated and may be
harder to modify independently.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)


# Weight constants for coupling factors
WEIGHT_TOPIC_CONNECTION = 0.40
WEIGHT_SHARED_PARAMETERS = 0.30
WEIGHT_DATA_DEPENDENCY = 0.20
WEIGHT_TEMPORAL_COUPLING = 0.10


@dataclass
class CouplingScore:
    """Coupling score between two components."""
    source: str
    target: str
    score: float  # 0.0 to 1.0
    strength: str  # LOW, MEDIUM, HIGH, CRITICAL
    factors: Dict[str, float] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)  # Topics connecting them
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_strength_label(score: float) -> str:
        """Convert score to human-readable strength label."""
        if score >= 0.7:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "score": round(self.score, 4),
            "strength": self.strength,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "connecting_topics": self.topics,
            "metadata": self.metadata,
        }


@dataclass
class CouplingMatrix:
    """Matrix of coupling scores between all node pairs."""
    scores: Dict[Tuple[str, str], CouplingScore] = field(default_factory=dict)
    nodes: List[str] = field(default_factory=list)

    def get_score(self, source: str, target: str) -> Optional[CouplingScore]:
        """Get coupling score between two nodes."""
        return self.scores.get((source, target))

    def get_top_coupled_pairs(self, n: int = 10) -> List[CouplingScore]:
        """Get the n most coupled pairs."""
        sorted_scores = sorted(self.scores.values(), key=lambda x: x.score, reverse=True)
        return sorted_scores[:n]

    def get_coupling_for_node(self, node: str) -> List[CouplingScore]:
        """Get all coupling scores involving a specific node."""
        result = []
        for (source, target), score in self.scores.items():
            if source == node or target == node:
                result.append(score)
        return sorted(result, key=lambda x: x.score, reverse=True)

    def get_critical_pairs(self) -> List[CouplingScore]:
        """Get all pairs with CRITICAL coupling."""
        return [s for s in self.scores.values() if s.strength == "CRITICAL"]

    def get_high_coupled_pairs(self) -> List[CouplingScore]:
        """Get all pairs with HIGH or CRITICAL coupling."""
        return [s for s in self.scores.values() if s.strength in ("HIGH", "CRITICAL")]

    def summary(self) -> Dict:
        """Generate summary statistics."""
        if not self.scores:
            return {
                "total_pairs": 0,
                "nodes_analyzed": len(self.nodes),
                "critical_pairs": 0,
                "high_pairs": 0,
                "medium_pairs": 0,
                "low_pairs": 0,
                "average_coupling": 0.0,
            }

        strength_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        total_score = 0.0

        for score in self.scores.values():
            strength_counts[score.strength] = strength_counts.get(score.strength, 0) + 1
            total_score += score.score

        return {
            "total_pairs": len(self.scores),
            "nodes_analyzed": len(self.nodes),
            "critical_pairs": strength_counts["CRITICAL"],
            "high_pairs": strength_counts["HIGH"],
            "medium_pairs": strength_counts["MEDIUM"],
            "low_pairs": strength_counts["LOW"],
            "average_coupling": round(total_score / len(self.scores), 4) if self.scores else 0.0,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.summary(),
            "nodes": self.nodes,
            "pairs": [s.to_dict() for s in sorted(self.scores.values(), key=lambda x: x.score, reverse=True)],
        }


class CouplingAnalyzer:
    """
    Analyze coupling strength between ROS2 nodes.

    Coupling is calculated based on:
    - Topic connections (pub/sub relationships)
    - Shared parameters
    - Data dependencies (message complexity)
    - Temporal coupling (timer frequencies)

    Usage:
        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()
        print(matrix.summary())
    """

    def __init__(
        self,
        nodes: List[ROS2NodeInfo],
        topic_graph: Optional[TopicGraphResult] = None,
    ):
        self.nodes = nodes
        self.topic_graph = topic_graph
        self._node_map: Dict[str, ROS2NodeInfo] = {n.name: n for n in nodes}

        # Build topic connection map
        self._topic_connections: Dict[str, Dict[str, List[str]]] = self._build_topic_connections()

    def _build_topic_connections(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build a map of which nodes are connected via which topics.

        Returns:
            Dict mapping (publisher_node, subscriber_node) to list of connecting topics
        """
        connections: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        if self.topic_graph:
            for topic_name, topic in self.topic_graph.topics.items():
                for pub_node in topic.publishers:
                    for sub_node in topic.subscribers:
                        if pub_node != sub_node:
                            connections[pub_node][sub_node].append(topic_name)
        else:
            # Build from nodes directly
            topic_publishers: Dict[str, List[str]] = defaultdict(list)
            topic_subscribers: Dict[str, List[str]] = defaultdict(list)

            for node in self.nodes:
                for pub in node.publishers:
                    topic_publishers[pub.topic].append(node.name)
                for sub in node.subscribers:
                    topic_subscribers[sub.topic].append(node.name)

            # Match publishers to subscribers
            all_topics = set(topic_publishers.keys()) | set(topic_subscribers.keys())
            for topic in all_topics:
                for pub_node in topic_publishers.get(topic, []):
                    for sub_node in topic_subscribers.get(topic, []):
                        if pub_node != sub_node:
                            connections[pub_node][sub_node].append(topic)

        return connections

    def analyze(self) -> CouplingMatrix:
        """
        Analyze coupling between all node pairs.

        Returns:
            CouplingMatrix with all coupling scores
        """
        matrix = CouplingMatrix(nodes=[n.name for n in self.nodes])

        # Calculate coupling for each pair with connections
        processed_pairs: Set[Tuple[str, str]] = set()

        for source_node in self.nodes:
            for target_name, topics in self._topic_connections.get(source_node.name, {}).items():
                # Skip if already processed (we do bidirectional)
                pair_key = tuple(sorted([source_node.name, target_name]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                target_node = self._node_map.get(target_name)
                if not target_node:
                    continue

                score = self._calculate_coupling(source_node, target_node, topics)
                if score.score > 0:
                    matrix.scores[(source_node.name, target_node.name)] = score

        return matrix

    def _calculate_coupling(
        self,
        source: ROS2NodeInfo,
        target: ROS2NodeInfo,
        connecting_topics: List[str],
    ) -> CouplingScore:
        """Calculate coupling score between two nodes."""
        factors = {}

        # Factor 1: Topic connections (0.0 - 0.4)
        topic_score = self._calculate_topic_coupling(source, target, connecting_topics)
        factors["topic_connections"] = topic_score

        # Factor 2: Shared parameters (0.0 - 0.3)
        param_score = self._calculate_parameter_coupling(source, target)
        factors["shared_parameters"] = param_score

        # Factor 3: Data dependencies (0.0 - 0.2)
        data_score = self._calculate_data_coupling(source, target, connecting_topics)
        factors["data_dependencies"] = data_score

        # Factor 4: Temporal coupling (0.0 - 0.1)
        temporal_score = self._calculate_temporal_coupling(source, target)
        factors["temporal_coupling"] = temporal_score

        # Weighted sum
        total_score = (
            topic_score * WEIGHT_TOPIC_CONNECTION +
            param_score * WEIGHT_SHARED_PARAMETERS +
            data_score * WEIGHT_DATA_DEPENDENCY +
            temporal_score * WEIGHT_TEMPORAL_COUPLING
        )

        # Normalize to 0-1
        total_score = min(1.0, total_score)

        return CouplingScore(
            source=source.name,
            target=target.name,
            score=total_score,
            strength=CouplingScore.get_strength_label(total_score),
            factors=factors,
            topics=connecting_topics,
            metadata={
                "source_package": source.package_name,
                "target_package": target.package_name,
            },
        )

    def _calculate_topic_coupling(
        self,
        source: ROS2NodeInfo,
        target: ROS2NodeInfo,
        connecting_topics: List[str],
    ) -> float:
        """
        Calculate topic-based coupling.

        Factors:
        - Number of connecting topics
        - QoS profile alignment
        - Topic direction (bidirectional = higher coupling)
        """
        if not connecting_topics:
            return 0.0

        # Base score from number of topics (normalized)
        # 1 topic = 0.3, 2 topics = 0.5, 3+ topics = 0.7+
        num_topics = len(connecting_topics)
        base_score = min(0.7, 0.2 + (num_topics * 0.15))

        # Check for bidirectional communication
        reverse_topics = self._topic_connections.get(target.name, {}).get(source.name, [])
        if reverse_topics:
            # Bidirectional communication adds coupling
            base_score += 0.15

        # Check for high-frequency topics (QoS > 50)
        high_freq_count = 0
        for pub in source.publishers:
            if pub.topic in connecting_topics and pub.qos >= 50:
                high_freq_count += 1
        if high_freq_count > 0:
            base_score += 0.1

        return min(1.0, base_score)

    def _calculate_parameter_coupling(
        self,
        source: ROS2NodeInfo,
        target: ROS2NodeInfo,
    ) -> float:
        """
        Calculate parameter-based coupling.

        Nodes sharing similar parameter names suggest shared configuration.
        """
        source_params = {p.name for p in source.parameters}
        target_params = {p.name for p in target.parameters}

        if not source_params or not target_params:
            return 0.0

        # Find common parameter names
        common_params = source_params & target_params

        # Also check for similar parameter prefixes
        source_prefixes = {p.split('.')[0] if '.' in p else p for p in source_params}
        target_prefixes = {p.split('.')[0] if '.' in p else p for p in target_params}
        common_prefixes = source_prefixes & target_prefixes

        # Score based on common params and prefixes
        if common_params:
            return min(1.0, 0.3 + (len(common_params) * 0.15))
        elif common_prefixes:
            return min(0.5, len(common_prefixes) * 0.1)

        return 0.0

    def _calculate_data_coupling(
        self,
        source: ROS2NodeInfo,
        target: ROS2NodeInfo,
        connecting_topics: List[str],
    ) -> float:
        """
        Calculate data dependency coupling.

        Based on message type complexity and data flow patterns.
        """
        if not connecting_topics:
            return 0.0

        # Get message types for connecting topics
        msg_types = set()
        for pub in source.publishers:
            if pub.topic in connecting_topics:
                msg_types.add(pub.msg_type)

        # Complex message types indicate tighter coupling
        complex_types = {"Image", "PointCloud2", "LaserScan", "JointState", "Odometry"}
        simple_types = {"Bool", "Float32", "Int32", "String", "Empty"}

        complex_count = sum(1 for mt in msg_types if any(ct in mt for ct in complex_types))
        simple_count = sum(1 for mt in msg_types if any(st in mt for st in simple_types))

        if complex_count > 0:
            return min(1.0, 0.4 + (complex_count * 0.2))
        elif simple_count > 0:
            return 0.2
        else:
            return 0.3  # Unknown/custom types

    def _calculate_temporal_coupling(
        self,
        source: ROS2NodeInfo,
        target: ROS2NodeInfo,
    ) -> float:
        """
        Calculate temporal coupling based on timer frequencies.

        Nodes with similar timer frequencies may be temporally coupled.
        """
        source_timers = [t.frequency_hz for t in source.timers if t.frequency_hz > 0]
        target_timers = [t.frequency_hz for t in target.timers if t.frequency_hz > 0]

        if not source_timers or not target_timers:
            return 0.0

        # Check for frequency alignment (within 10%)
        aligned_count = 0
        for sf in source_timers:
            for tf in target_timers:
                ratio = sf / tf if tf > 0 else 0
                if 0.9 <= ratio <= 1.1:
                    aligned_count += 1
                elif ratio in (0.5, 2.0):  # Harmonic frequencies
                    aligned_count += 0.5

        if aligned_count > 0:
            return min(1.0, 0.3 + (aligned_count * 0.2))

        return 0.0

    def get_coupling_summary(self) -> str:
        """Generate a human-readable coupling summary."""
        matrix = self.analyze()
        summary = matrix.summary()

        lines = [
            "=" * 60,
            "COUPLING ANALYSIS SUMMARY",
            "=" * 60,
            f"Nodes analyzed: {summary['nodes_analyzed']}",
            f"Connected pairs: {summary['total_pairs']}",
            f"Average coupling: {summary['average_coupling']:.3f}",
            "",
            "Coupling Distribution:",
            f"  CRITICAL: {summary['critical_pairs']}",
            f"  HIGH:     {summary['high_pairs']}",
            f"  MEDIUM:   {summary['medium_pairs']}",
            f"  LOW:      {summary['low_pairs']}",
        ]

        critical = matrix.get_critical_pairs()
        if critical:
            lines.extend([
                "",
                "Critical Coupling Pairs:",
            ])
            for score in critical[:10]:
                lines.append(f"  {score.source} <-> {score.target}: {score.score:.3f}")
                lines.append(f"    Topics: {', '.join(score.topics)}")

        return "\n".join(lines)


def analyze_coupling(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
) -> CouplingMatrix:
    """
    Convenience function to analyze coupling.

    Args:
        nodes: List of ROS2NodeInfo objects
        topic_graph: Optional TopicGraphResult

    Returns:
        CouplingMatrix with all scores
    """
    analyzer = CouplingAnalyzer(nodes, topic_graph)
    return analyzer.analyze()


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python coupling.py <project_path>")
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

    # Analyze coupling
    analyzer = CouplingAnalyzer(all_nodes, topic_graph)
    print(analyzer.get_coupling_summary())

    print("\n" + "=" * 60)
    print("DETAILED COUPLING DATA")
    print("=" * 60)

    matrix = analyzer.analyze()
    top_pairs = matrix.get_top_coupled_pairs(20)

    for score in top_pairs:
        print(f"\n{score.source} <-> {score.target}")
        print(f"  Score: {score.score:.4f} ({score.strength})")
        print(f"  Factors: {json.dumps({k: round(v, 3) for k, v in score.factors.items()})}")
        print(f"  Topics: {', '.join(score.topics)}")
