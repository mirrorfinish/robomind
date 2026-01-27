"""
RoboMind Message Type Analyzer

Analyzes ROS2 message types for:
- Type mismatches between publishers/subscribers
- Deprecated message types
- Custom vs standard message usage
- Message compatibility across versions
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult


@dataclass
class MessageTypeFinding:
    """A message type-related finding."""
    finding_type: str
    severity: str
    topic: str
    description: str
    types_found: List[str]
    affected_nodes: List[str]
    recommendation: str


# Standard ROS2 message packages
STANDARD_PACKAGES = {
    'std_msgs', 'geometry_msgs', 'sensor_msgs', 'nav_msgs',
    'visualization_msgs', 'diagnostic_msgs', 'actionlib_msgs',
    'trajectory_msgs', 'shape_msgs', 'stereo_msgs', 'tf2_msgs',
    'builtin_interfaces', 'rcl_interfaces', 'rosgraph_msgs',
    'std_srvs', 'unique_identifier_msgs', 'action_msgs',
}

# Deprecated or discouraged message types
DEPRECATED_TYPES = {
    'std_msgs/msg/Header': 'Use builtin_interfaces/msg/Time for timestamps',
    'std_msgs/msg/Time': 'Use builtin_interfaces/msg/Time',
    'std_msgs/msg/Duration': 'Use builtin_interfaces/msg/Duration',
}

# Common type misuse patterns
TYPE_MISUSE = {
    ('std_msgs/msg/String', '/cmd_vel'): 'cmd_vel should use geometry_msgs/msg/Twist',
    ('std_msgs/msg/String', '/odom'): 'odom should use nav_msgs/msg/Odometry',
    ('std_msgs/msg/Float32', '/cmd_vel'): 'cmd_vel should use geometry_msgs/msg/Twist',
}


class MessageTypeAnalyzer:
    """
    Analyzes message types used in ROS2 topics.
    """

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def add_topic_graph(self, topic_graph: TopicGraphResult):
        """Add topic graph."""
        self.topic_graph = topic_graph

    def _normalize_type(self, msg_type: str) -> str:
        """Normalize message type string."""
        if not msg_type:
            return ''
        # Handle common variations
        # "Twist" -> "geometry_msgs/msg/Twist"
        # "geometry_msgs.msg.Twist" -> "geometry_msgs/msg/Twist"
        msg_type = msg_type.replace('.', '/')
        return msg_type

    def _extract_package(self, msg_type: str) -> Optional[str]:
        """Extract package name from message type."""
        parts = msg_type.split('/')
        if len(parts) >= 1:
            return parts[0]
        return None

    def analyze(self) -> List[MessageTypeFinding]:
        """Analyze message types for issues."""
        findings = []

        # Build topic -> types map
        topic_types: Dict[str, Dict[str, List[str]]] = {}  # topic -> {type -> [nodes]}

        for node in self.nodes:
            for pub in node.publishers:
                if pub.topic and pub.msg_type:
                    normalized = self._normalize_type(pub.msg_type)
                    if pub.topic not in topic_types:
                        topic_types[pub.topic] = {}
                    if normalized not in topic_types[pub.topic]:
                        topic_types[pub.topic][normalized] = []
                    topic_types[pub.topic][normalized].append(f"{node.name} (pub)")

            for sub in node.subscribers:
                if sub.topic and sub.msg_type:
                    normalized = self._normalize_type(sub.msg_type)
                    if sub.topic not in topic_types:
                        topic_types[sub.topic] = {}
                    if normalized not in topic_types[sub.topic]:
                        topic_types[sub.topic][normalized] = []
                    topic_types[sub.topic][normalized].append(f"{node.name} (sub)")

        # Check for type mismatches on same topic
        for topic, types in topic_types.items():
            if len(types) > 1:
                all_nodes = []
                for t, nodes in types.items():
                    all_nodes.extend(nodes)

                # Determine severity
                severity = 'high'
                if 'cmd' in topic.lower() or 'control' in topic.lower():
                    severity = 'critical'

                findings.append(MessageTypeFinding(
                    finding_type='type_mismatch',
                    severity=severity,
                    topic=topic,
                    description=f'Multiple message types used for topic "{topic}"',
                    types_found=list(types.keys()),
                    affected_nodes=all_nodes,
                    recommendation='Ensure all publishers/subscribers use the same message type',
                ))

            # Check for deprecated types
            for msg_type in types.keys():
                if msg_type in DEPRECATED_TYPES:
                    findings.append(MessageTypeFinding(
                        finding_type='deprecated_type',
                        severity='low',
                        topic=topic,
                        description=f'Deprecated message type: {msg_type}',
                        types_found=[msg_type],
                        affected_nodes=types[msg_type],
                        recommendation=DEPRECATED_TYPES[msg_type],
                    ))

            # Check for type misuse
            for msg_type in types.keys():
                for (misused_type, pattern), recommendation in TYPE_MISUSE.items():
                    if misused_type in msg_type and pattern in topic.lower():
                        findings.append(MessageTypeFinding(
                            finding_type='type_misuse',
                            severity='medium',
                            topic=topic,
                            description=f'Inappropriate message type for topic',
                            types_found=[msg_type],
                            affected_nodes=types[msg_type],
                            recommendation=recommendation,
                        ))

        # Check for custom messages (might indicate missing dependencies)
        custom_types = set()
        for topic, types in topic_types.items():
            for msg_type in types.keys():
                pkg = self._extract_package(msg_type)
                if pkg and pkg not in STANDARD_PACKAGES:
                    custom_types.add(msg_type)

        if len(custom_types) > 5:  # Many custom types might indicate dependency issues
            findings.append(MessageTypeFinding(
                finding_type='many_custom_types',
                severity='low',
                topic='(multiple)',
                description=f'{len(custom_types)} custom message types used',
                types_found=list(custom_types)[:10],
                affected_nodes=[],
                recommendation='Ensure all custom message packages are properly declared as dependencies',
            ))

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 4))

        return findings


def analyze_message_types(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
) -> List[MessageTypeFinding]:
    """Convenience function for message type analysis."""
    analyzer = MessageTypeAnalyzer()
    analyzer.add_nodes(nodes)
    if topic_graph:
        analyzer.add_topic_graph(topic_graph)
    return analyzer.analyze()
