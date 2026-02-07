"""
RoboMind Impact Analyzer - Analyze what's affected by changes.

Answers questions like:
- "If I rename /cmd_vel, what breaks?"
- "If motor_controller_node goes down, what's affected?"
- "What uses sensor_msgs/msg/LaserScan?"
- "If I change this file, what's impacted?"

Uses the existing topic graph to trace change propagation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)

# Safety-critical topics get elevated severity
CRITICAL_TOPICS = frozenset({
    "/cmd_vel", "/emergency_stop", "/motor/command",
    "/betaray/motors/cmd_vel", "/robot/cmd_vel",
    "/betaray/emergency_stop",
})

# Monitoring/logging topics get reduced severity
LOW_PRIORITY_PREFIXES = (
    "/betaray/debug/", "/betaray/log/", "/diagnostics",
    "/rosout", "/parameter_events", "/betaray/guardian/",
)


@dataclass
class ImpactItem:
    """A single affected entity."""
    name: str
    kind: str  # "node", "topic", "service"
    impact_type: str  # "broken_subscriber", "lost_publisher", "cascade", "type_user"
    severity: str  # "critical", "high", "medium", "low"
    file_path: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "impact_type": self.impact_type,
            "severity": self.severity,
            "file_path": self.file_path,
            "description": self.description,
        }


@dataclass
class ImpactResult:
    """Result of an impact analysis query."""
    query: str
    query_type: str  # "topic_change", "node_removal", "message_type_change", "file_change"
    directly_affected: List[ImpactItem] = field(default_factory=list)
    cascade_affected: List[ImpactItem] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        all_items = self.directly_affected + self.cascade_affected
        by_severity = {}
        for item in all_items:
            by_severity[item.severity] = by_severity.get(item.severity, 0) + 1
        return {
            "query": self.query,
            "query_type": self.query_type,
            "directly_affected": len(self.directly_affected),
            "cascade_affected": len(self.cascade_affected),
            "total_affected": len(all_items),
            "by_severity": by_severity,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "directly_affected": [i.to_dict() for i in self.directly_affected],
            "cascade_affected": [i.to_dict() for i in self.cascade_affected],
        }


class ImpactAnalyzer:
    """
    Analyze the impact of changes to a ROS2 system.

    Usage:
        analyzer = ImpactAnalyzer(nodes, topic_graph)
        result = analyzer.analyze_topic_change("/cmd_vel")
        print(result.summary())
    """

    def __init__(
        self,
        nodes: List[ROS2NodeInfo],
        topic_graph: Optional[TopicGraphResult] = None,
    ):
        self.nodes = nodes
        self.topic_graph = topic_graph

        # Build indices
        self._node_map: Dict[str, ROS2NodeInfo] = {}
        self._file_to_nodes: Dict[str, List[str]] = {}
        self._topic_publishers: Dict[str, List[str]] = {}  # topic -> [node_names]
        self._topic_subscribers: Dict[str, List[str]] = {}
        self._topic_types: Dict[str, str] = {}  # topic -> msg_type
        self._node_pub_topics: Dict[str, List[str]] = {}  # node -> [topics it publishes]
        self._node_sub_topics: Dict[str, List[str]] = {}  # node -> [topics it subscribes to]

        self._build_indices()

    def _build_indices(self):
        """Build lookup indices from nodes and topic graph."""
        for node in self.nodes:
            self._node_map[node.name] = node

            # File index
            if node.file_path:
                fp = str(node.file_path)
                if fp not in self._file_to_nodes:
                    self._file_to_nodes[fp] = []
                self._file_to_nodes[fp].append(node.name)

            # Publisher index
            pub_topics = []
            for pub in node.publishers:
                topic = pub.topic
                if topic not in self._topic_publishers:
                    self._topic_publishers[topic] = []
                self._topic_publishers[topic].append(node.name)
                pub_topics.append(topic)
                if pub.msg_type:
                    self._topic_types[topic] = pub.msg_type
            self._node_pub_topics[node.name] = pub_topics

            # Subscriber index
            sub_topics = []
            for sub in node.subscribers:
                topic = sub.topic
                if topic not in self._topic_subscribers:
                    self._topic_subscribers[topic] = []
                self._topic_subscribers[topic].append(node.name)
                sub_topics.append(topic)
                if sub.msg_type and topic not in self._topic_types:
                    self._topic_types[topic] = sub.msg_type
            self._node_sub_topics[node.name] = sub_topics

    def _get_topic_severity(self, topic: str) -> str:
        """Determine severity based on topic name."""
        if topic in CRITICAL_TOPICS:
            return "critical"
        for prefix in LOW_PRIORITY_PREFIXES:
            if topic.startswith(prefix):
                return "low"
        return "medium"

    def _get_node_file(self, node_name: str) -> str:
        """Get file path for a node."""
        node = self._node_map.get(node_name)
        return str(node.file_path) if node and node.file_path else ""

    def analyze_topic_change(self, topic_name: str) -> ImpactResult:
        """
        Analyze what's affected if a topic is renamed, removed, or its type changes.

        Returns all publishers and subscribers, plus cascade effects.
        """
        result = ImpactResult(query=topic_name, query_type="topic_change")
        base_severity = self._get_topic_severity(topic_name)

        # All publishers of this topic
        for pub_node in self._topic_publishers.get(topic_name, []):
            severity = "critical" if base_severity == "critical" else "high"
            result.directly_affected.append(ImpactItem(
                name=pub_node,
                kind="node",
                impact_type="lost_publisher",
                severity=severity,
                file_path=self._get_node_file(pub_node),
                description=f"Publishes to {topic_name} — must update topic name",
            ))

        # All subscribers of this topic
        for sub_node in self._topic_subscribers.get(topic_name, []):
            severity = "critical" if base_severity == "critical" else "high"
            result.directly_affected.append(ImpactItem(
                name=sub_node,
                kind="node",
                impact_type="broken_subscriber",
                severity=severity,
                file_path=self._get_node_file(sub_node),
                description=f"Subscribes to {topic_name} — will stop receiving data",
            ))

        # Cascade: for each subscriber that loses this topic,
        # check if any of its downstream subscribers are affected
        affected_nodes = set()
        for item in result.directly_affected:
            affected_nodes.add(item.name)

        for sub_node in self._topic_subscribers.get(topic_name, []):
            # What topics does this subscriber publish?
            for downstream_topic in self._node_pub_topics.get(sub_node, []):
                if downstream_topic == topic_name:
                    continue
                for downstream_sub in self._topic_subscribers.get(downstream_topic, []):
                    if downstream_sub not in affected_nodes:
                        affected_nodes.add(downstream_sub)
                        result.cascade_affected.append(ImpactItem(
                            name=downstream_sub,
                            kind="node",
                            impact_type="cascade",
                            severity="medium",
                            file_path=self._get_node_file(downstream_sub),
                            description=f"Downstream of {sub_node} via {downstream_topic}",
                        ))

        return result

    def analyze_node_removal(self, node_name: str) -> ImpactResult:
        """
        Analyze what's affected if a node goes down or is removed.

        Finds all topics that lose a publisher and their subscribers.
        """
        result = ImpactResult(query=node_name, query_type="node_removal")
        node = self._node_map.get(node_name)
        if not node:
            return result

        affected_nodes: Set[str] = {node_name}

        # For each topic this node publishes, find subscribers that lose data
        for topic in self._node_pub_topics.get(node_name, []):
            # How many other publishers exist for this topic?
            other_pubs = [n for n in self._topic_publishers.get(topic, []) if n != node_name]

            for sub_node in self._topic_subscribers.get(topic, []):
                if sub_node == node_name or sub_node in affected_nodes:
                    continue

                if not other_pubs:
                    # This was the only publisher — subscriber loses all data
                    severity = self._get_topic_severity(topic)
                    if severity == "medium":
                        # Check if subscriber has other inputs
                        other_inputs = [t for t in self._node_sub_topics.get(sub_node, [])
                                       if t != topic]
                        if not other_inputs:
                            severity = "critical"  # Node goes blind
                        else:
                            severity = "high"  # Loses primary input
                    result.directly_affected.append(ImpactItem(
                        name=sub_node,
                        kind="node",
                        impact_type="broken_subscriber",
                        severity=severity,
                        file_path=self._get_node_file(sub_node),
                        description=f"Loses {topic} (no remaining publishers)",
                    ))
                    affected_nodes.add(sub_node)
                else:
                    result.directly_affected.append(ImpactItem(
                        name=sub_node,
                        kind="node",
                        impact_type="lost_publisher",
                        severity="low",
                        file_path=self._get_node_file(sub_node),
                        description=f"Loses one publisher on {topic} ({len(other_pubs)} remaining)",
                    ))

            # The topic itself is affected
            result.directly_affected.append(ImpactItem(
                name=topic,
                kind="topic",
                impact_type="lost_publisher",
                severity=self._get_topic_severity(topic),
                description=f"Loses publisher {node_name}" + (
                    " (no publishers remaining)" if not other_pubs else
                    f" ({len(other_pubs)} remaining)"
                ),
            ))

        # Cascade: check downstream of directly affected nodes
        direct_node_names = {i.name for i in result.directly_affected if i.kind == "node"}
        for affected_node in list(direct_node_names):
            for downstream_topic in self._node_pub_topics.get(affected_node, []):
                for downstream_sub in self._topic_subscribers.get(downstream_topic, []):
                    if downstream_sub not in affected_nodes and downstream_sub not in direct_node_names:
                        result.cascade_affected.append(ImpactItem(
                            name=downstream_sub,
                            kind="node",
                            impact_type="cascade",
                            severity="medium",
                            file_path=self._get_node_file(downstream_sub),
                            description=f"Downstream of {affected_node} via {downstream_topic}",
                        ))

        return result

    def analyze_message_type_change(self, msg_type: str) -> ImpactResult:
        """
        Analyze what's affected if a message type changes.

        Finds all topics using this type and their publishers/subscribers.
        """
        result = ImpactResult(query=msg_type, query_type="message_type_change")

        # Normalize type name for matching
        msg_type_short = msg_type.split("/")[-1]

        # Find all topics using this message type
        affected_topics = []
        for topic, topic_type in self._topic_types.items():
            if (topic_type == msg_type or
                topic_type.endswith(f"/{msg_type_short}") or
                topic_type == msg_type_short):
                affected_topics.append(topic)

        seen_nodes: Set[str] = set()
        for topic in affected_topics:
            # Add the topic
            result.directly_affected.append(ImpactItem(
                name=topic,
                kind="topic",
                impact_type="type_user",
                severity="high",
                description=f"Uses message type {msg_type}",
            ))

            # Add all publishers and subscribers
            for pub_node in self._topic_publishers.get(topic, []):
                if pub_node not in seen_nodes:
                    seen_nodes.add(pub_node)
                    result.directly_affected.append(ImpactItem(
                        name=pub_node,
                        kind="node",
                        impact_type="type_user",
                        severity="high",
                        file_path=self._get_node_file(pub_node),
                        description=f"Publishes {msg_type} on {topic}",
                    ))

            for sub_node in self._topic_subscribers.get(topic, []):
                if sub_node not in seen_nodes:
                    seen_nodes.add(sub_node)
                    result.directly_affected.append(ImpactItem(
                        name=sub_node,
                        kind="node",
                        impact_type="type_user",
                        severity="high",
                        file_path=self._get_node_file(sub_node),
                        description=f"Subscribes to {msg_type} on {topic}",
                    ))

        return result

    def analyze_file_change(self, file_path: str) -> ImpactResult:
        """
        Analyze what's affected if a source file changes.

        Finds all nodes defined in the file and runs node removal analysis.
        """
        result = ImpactResult(query=file_path, query_type="file_change")

        # Normalize path for matching
        file_path_normalized = str(Path(file_path))
        matching_nodes = []

        for fp, node_names in self._file_to_nodes.items():
            if fp.endswith(file_path_normalized) or file_path_normalized.endswith(fp) or file_path in fp:
                matching_nodes.extend(node_names)

        if not matching_nodes:
            # Try matching by filename only
            file_name = Path(file_path).name
            for fp, node_names in self._file_to_nodes.items():
                if fp.endswith(file_name):
                    matching_nodes.extend(node_names)

        seen: Set[str] = set()
        for node_name in matching_nodes:
            sub_result = self.analyze_node_removal(node_name)
            for item in sub_result.directly_affected:
                if item.name not in seen:
                    seen.add(item.name)
                    item.description = f"[via {node_name}] " + item.description
                    result.directly_affected.append(item)
            for item in sub_result.cascade_affected:
                if item.name not in seen:
                    seen.add(item.name)
                    item.description = f"[via {node_name}] " + item.description
                    result.cascade_affected.append(item)

        return result


def analyze_impact(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    target: str = "",
    target_type: str = "topic",
) -> ImpactResult:
    """
    Convenience function for impact analysis.

    Args:
        nodes: List of ROS2NodeInfo
        topic_graph: Optional TopicGraphResult
        target: What to analyze (topic name, node name, file path, or message type)
        target_type: "topic", "node", "file", or "message_type"

    Returns:
        ImpactResult
    """
    analyzer = ImpactAnalyzer(nodes, topic_graph)

    if target_type == "topic":
        return analyzer.analyze_topic_change(target)
    elif target_type == "node":
        return analyzer.analyze_node_removal(target)
    elif target_type == "message_type":
        return analyzer.analyze_message_type_change(target)
    elif target_type == "file":
        return analyzer.analyze_file_change(target)
    else:
        return ImpactResult(query=target, query_type=target_type)
