"""
RoboMind QoS Mismatch Analyzer

Detects Quality of Service (QoS) incompatibilities between publishers and subscribers.
QoS mismatches can cause silent communication failures.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robomind.ros2.node_extractor import ROS2NodeInfo


@dataclass
class QoSProfile:
    """Extracted QoS settings."""
    reliability: Optional[str] = None  # RELIABLE, BEST_EFFORT
    durability: Optional[str] = None   # VOLATILE, TRANSIENT_LOCAL
    history: Optional[str] = None      # KEEP_LAST, KEEP_ALL
    depth: Optional[int] = None        # Queue depth
    deadline: Optional[float] = None   # Deadline in seconds
    lifespan: Optional[float] = None   # Lifespan in seconds
    liveliness: Optional[str] = None   # AUTOMATIC, MANUAL_BY_NODE, MANUAL_BY_TOPIC

    def is_compatible_with(self, other: 'QoSProfile') -> Tuple[bool, List[str]]:
        """Check if this publisher QoS is compatible with subscriber QoS."""
        issues = []

        # Reliability compatibility
        # RELIABLE pub can work with any sub
        # BEST_EFFORT pub only works with BEST_EFFORT sub
        if self.reliability == "BEST_EFFORT" and other.reliability == "RELIABLE":
            issues.append("Publisher uses BEST_EFFORT but subscriber expects RELIABLE - messages may be lost")

        # Durability compatibility
        # TRANSIENT_LOCAL pub can work with any sub
        # VOLATILE pub won't satisfy TRANSIENT_LOCAL sub (late joiners miss messages)
        if self.durability == "VOLATILE" and other.durability == "TRANSIENT_LOCAL":
            issues.append("Publisher uses VOLATILE but subscriber expects TRANSIENT_LOCAL - late-joining subscribers will miss messages")

        # Deadline compatibility
        if self.deadline and other.deadline:
            if self.deadline > other.deadline:
                issues.append(f"Publisher deadline ({self.deadline}s) exceeds subscriber deadline ({other.deadline}s)")

        return len(issues) == 0, issues


@dataclass
class QoSFinding:
    """A QoS-related finding."""
    topic: str
    publisher_node: str
    subscriber_node: str
    publisher_file: str
    subscriber_file: str
    pub_qos: QoSProfile
    sub_qos: QoSProfile
    issues: List[str]
    severity: str  # "high", "medium", "low"


class QoSAnalyzer:
    """
    Analyzes QoS settings for compatibility issues.

    Common QoS mismatches:
    - RELIABLE vs BEST_EFFORT reliability
    - TRANSIENT_LOCAL vs VOLATILE durability
    - Incompatible deadline/lifespan settings
    """

    # Patterns to extract QoS settings from code
    QOS_PATTERNS = {
        'reliability': [
            r'ReliabilityPolicy\.(RELIABLE|BEST_EFFORT)',
            r'reliability\s*=\s*["\']?(RELIABLE|BEST_EFFORT)["\']?',
            r'qos_profile_sensor_data',  # implies BEST_EFFORT
            r'qos_profile_system_default',
        ],
        'durability': [
            r'DurabilityPolicy\.(VOLATILE|TRANSIENT_LOCAL)',
            r'durability\s*=\s*["\']?(VOLATILE|TRANSIENT_LOCAL)["\']?',
        ],
        'depth': [
            r'depth\s*=\s*(\d+)',
            r'QoSProfile\([^)]*depth\s*=\s*(\d+)',
        ],
    }

    # Known QoS profile defaults
    KNOWN_PROFILES = {
        'qos_profile_sensor_data': QoSProfile(reliability='BEST_EFFORT', durability='VOLATILE', depth=5),
        'qos_profile_system_default': QoSProfile(reliability='RELIABLE', durability='VOLATILE', depth=10),
        'qos_profile_services_default': QoSProfile(reliability='RELIABLE', durability='VOLATILE'),
        'qos_profile_parameters': QoSProfile(reliability='RELIABLE', durability='VOLATILE'),
        'qos_profile_parameter_events': QoSProfile(reliability='RELIABLE', durability='VOLATILE'),
    }

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.file_contents: Dict[str, str] = {}

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content with caching."""
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r') as f:
                    self.file_contents[file_path] = f.read()
            except Exception:
                return None
        return self.file_contents.get(file_path)

    def _extract_qos_from_context(self, file_path: str, line_num: int) -> QoSProfile:
        """Extract QoS settings from code context around a pub/sub declaration."""
        content = self._get_file_content(file_path)
        if not content:
            return QoSProfile()

        lines = content.split('\n')
        # Look at surrounding context (5 lines before and after)
        start = max(0, line_num - 6)
        end = min(len(lines), line_num + 5)
        context = '\n'.join(lines[start:end])

        qos = QoSProfile()

        # Check for known profiles
        for profile_name, profile_qos in self.KNOWN_PROFILES.items():
            if profile_name in context:
                return profile_qos

        # Extract individual settings
        for match in re.finditer(r'ReliabilityPolicy\.(RELIABLE|BEST_EFFORT)', context):
            qos.reliability = match.group(1)

        for match in re.finditer(r'DurabilityPolicy\.(VOLATILE|TRANSIENT_LOCAL)', context):
            qos.durability = match.group(1)

        for match in re.finditer(r'depth\s*=\s*(\d+)', context):
            qos.depth = int(match.group(1))

        # Check for common patterns that imply QoS
        if 'sensor' in context.lower() or 'camera' in context.lower() or 'lidar' in context.lower():
            if qos.reliability is None:
                qos.reliability = 'BEST_EFFORT'  # Common for sensor data

        return qos

    def analyze(self) -> List[QoSFinding]:
        """Analyze all nodes for QoS mismatches."""
        findings = []

        # Build topic -> publishers/subscribers map
        topic_pubs: Dict[str, List[Tuple[ROS2NodeInfo, QoSProfile, int]]] = {}
        topic_subs: Dict[str, List[Tuple[ROS2NodeInfo, QoSProfile, int]]] = {}

        for node in self.nodes:
            for pub in node.publishers:
                topic = pub.topic
                if not topic:
                    continue
                qos = self._extract_qos_from_context(node.file_path, pub.line_number) if node.file_path else QoSProfile()
                if topic not in topic_pubs:
                    topic_pubs[topic] = []
                topic_pubs[topic].append((node, qos, pub.line_number))

            for sub in node.subscribers:
                topic = sub.topic
                if not topic:
                    continue
                qos = self._extract_qos_from_context(node.file_path, sub.line_number) if node.file_path else QoSProfile()
                if topic not in topic_subs:
                    topic_subs[topic] = []
                topic_subs[topic].append((node, qos, sub.line_number))

        # Check compatibility for each topic
        for topic in set(topic_pubs.keys()) & set(topic_subs.keys()):
            for pub_node, pub_qos, pub_line in topic_pubs[topic]:
                for sub_node, sub_qos, sub_line in topic_subs[topic]:
                    compatible, issues = pub_qos.is_compatible_with(sub_qos)
                    if not compatible:
                        # Determine severity
                        if 'cmd_vel' in topic.lower() or 'emergency' in topic.lower():
                            severity = 'high'
                        elif 'sensor' in topic.lower() or 'camera' in topic.lower():
                            severity = 'medium'
                        else:
                            severity = 'low'

                        findings.append(QoSFinding(
                            topic=topic,
                            publisher_node=pub_node.name,
                            subscriber_node=sub_node.name,
                            publisher_file=pub_node.file_path or '',
                            subscriber_file=sub_node.file_path or '',
                            pub_qos=pub_qos,
                            sub_qos=sub_qos,
                            issues=issues,
                            severity=severity,
                        ))

        return findings


def analyze_qos(nodes: List[ROS2NodeInfo]) -> List[QoSFinding]:
    """Convenience function to analyze QoS compatibility."""
    analyzer = QoSAnalyzer()
    analyzer.add_nodes(nodes)
    return analyzer.analyze()
