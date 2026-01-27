"""
RoboMind Recommendations Engine

Analyzes ROS2 code and generates actionable improvement suggestions
with specific file locations and code fixes.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"  # Safety issues, will cause failures
    HIGH = "high"          # Bugs that will cause problems
    MEDIUM = "medium"      # Code quality issues
    LOW = "low"            # Style/consistency suggestions


class Category(Enum):
    """Issue categories."""
    SAFETY = "safety"
    CONNECTIVITY = "connectivity"
    NAMING = "naming"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    CODE_QUALITY = "code_quality"


@dataclass
class CodeFix:
    """A suggested code fix."""
    file_path: str
    line_number: int
    original: str
    replacement: str
    description: str


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    id: str
    title: str
    severity: Severity
    category: Category
    description: str
    impact: str
    affected_nodes: List[str]
    affected_files: List[str]
    fixes: List[CodeFix] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "impact": self.impact,
            "affected_nodes": self.affected_nodes,
            "affected_files": self.affected_files,
            "fixes": [
                {
                    "file": f.file_path,
                    "line": f.line_number,
                    "original": f.original,
                    "replacement": f.replacement,
                    "description": f.description,
                }
                for f in self.fixes
            ],
        }


@dataclass
class RecommendationReport:
    """Full recommendations report."""
    recommendations: List[Recommendation] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        by_severity = {s.value: 0 for s in Severity}
        by_category = {c.value: 0 for c in Category}

        for rec in self.recommendations:
            by_severity[rec.severity.value] += 1
            by_category[rec.category.value] += 1

        return {
            "summary": {
                "total": len(self.recommendations),
                "by_severity": by_severity,
                "by_category": by_category,
                "fixable": sum(1 for r in self.recommendations if r.fixes),
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


class RecommendationEngine:
    """
    Analyzes ROS2 nodes and generates actionable recommendations.

    Usage:
        engine = RecommendationEngine()
        engine.add_nodes(ros2_nodes)
        engine.add_topic_graph(topic_graph)
        report = engine.analyze()
    """

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.file_contents: Dict[str, str] = {}  # Cache file contents

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add ROS2 nodes to analyze."""
        self.nodes = nodes

    def add_topic_graph(self, topic_graph: TopicGraphResult):
        """Add topic graph for connectivity analysis."""
        self.topic_graph = topic_graph

    def analyze(self) -> RecommendationReport:
        """Run all analyzers and generate recommendations."""
        report = RecommendationReport()

        # Run all analysis passes
        report.recommendations.extend(self._check_relative_topics())
        report.recommendations.extend(self._check_emergency_stop())
        report.recommendations.extend(self._check_cmd_vel_consistency())
        report.recommendations.extend(self._check_orphaned_connections())
        report.recommendations.extend(self._check_duplicate_publishers())
        report.recommendations.extend(self._check_high_frequency_timers())
        report.recommendations.extend(self._check_missing_error_handling())

        # Sort by severity
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        report.recommendations.sort(key=lambda r: severity_order[r.severity])

        return report

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content, with caching."""
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r') as f:
                    self.file_contents[file_path] = f.read()
            except Exception:
                return None
        return self.file_contents.get(file_path)

    def _get_line(self, file_path: str, line_num: int) -> Optional[str]:
        """Get a specific line from a file."""
        content = self._get_file_content(file_path)
        if content:
            lines = content.split('\n')
            if 0 < line_num <= len(lines):
                return lines[line_num - 1]
        return None

    def _check_relative_topics(self) -> List[Recommendation]:
        """Check for relative topic names that will break in namespaces."""
        recommendations = []

        # Group by topic pattern
        relative_topics: Dict[str, List[Tuple[ROS2NodeInfo, str, str, int]]] = {}

        for node in self.nodes:
            for pub in node.publishers:
                if pub.topic and not pub.topic.startswith('/'):
                    key = pub.topic
                    if key not in relative_topics:
                        relative_topics[key] = []
                    relative_topics[key].append((node, 'publisher', pub.topic, pub.line_number))

            for sub in node.subscribers:
                if sub.topic and not sub.topic.startswith('/'):
                    key = sub.topic
                    if key not in relative_topics:
                        relative_topics[key] = []
                    relative_topics[key].append((node, 'subscriber', sub.topic, sub.line_number))

        # Create recommendations for each relative topic
        for topic, usages in relative_topics.items():
            # Skip if it's a parameter-based topic (starts with $)
            if topic.startswith('$') or topic.startswith('{'):
                continue

            fixes = []
            affected_nodes = set()
            affected_files = set()

            for node, direction, topic_name, line_num in usages:
                affected_nodes.add(node.name)
                affected_files.add(node.file_path)

                # Try to generate a fix
                if node.file_path and line_num > 0:
                    line = self._get_line(node.file_path, line_num)
                    if line and f"'{topic_name}'" in line:
                        fixes.append(CodeFix(
                            file_path=node.file_path,
                            line_number=line_num,
                            original=f"'{topic_name}'",
                            replacement=f"'/{topic_name}'",
                            description=f"Add leading slash to make topic absolute",
                        ))
                    elif line and f'"{topic_name}"' in line:
                        fixes.append(CodeFix(
                            file_path=node.file_path,
                            line_number=line_num,
                            original=f'"{topic_name}"',
                            replacement=f'"/{topic_name}"',
                            description=f"Add leading slash to make topic absolute",
                        ))

            # Determine severity based on topic criticality
            if 'cmd_vel' in topic.lower() or 'emergency' in topic.lower() or 'stop' in topic.lower():
                severity = Severity.HIGH
                impact = "Movement/safety commands may fail when node runs in a namespace"
            elif 'odom' in topic.lower() or 'scan' in topic.lower() or 'tf' in topic.lower():
                severity = Severity.MEDIUM
                impact = "Sensor data may not be received when node runs in a namespace"
            else:
                severity = Severity.LOW
                impact = "Topic may not connect when node runs in a namespace"

            recommendations.append(Recommendation(
                id=f"REL-{topic.replace('/', '_').upper()}",
                title=f"Relative topic name: {topic}",
                severity=severity,
                category=Category.NAMING,
                description=f"Topic '{topic}' uses relative naming. When nodes run in a namespace, "
                           f"this becomes '/namespace/{topic}' instead of '/{topic}'.",
                impact=impact,
                affected_nodes=list(affected_nodes),
                affected_files=list(affected_files),
                fixes=fixes,
            ))

        return recommendations

    def _check_emergency_stop(self) -> List[Recommendation]:
        """Check for emergency stop topic inconsistencies."""
        recommendations = []

        estop_patterns: Dict[str, List[Tuple[str, str, str, int]]] = {}

        for node in self.nodes:
            for pub in node.publishers:
                topic = pub.topic or ''
                if 'emergency' in topic.lower() or ('stop' in topic.lower() and 'motor' in topic.lower()):
                    if topic not in estop_patterns:
                        estop_patterns[topic] = []
                    estop_patterns[topic].append((node.name, node.file_path, 'pub', pub.line_number))

            for sub in node.subscribers:
                topic = sub.topic or ''
                if 'emergency' in topic.lower() or ('stop' in topic.lower() and 'motor' in topic.lower()):
                    if topic not in estop_patterns:
                        estop_patterns[topic] = []
                    estop_patterns[topic].append((node.name, node.file_path, 'sub', sub.line_number))

        # Check for multiple e-stop topic patterns
        if len(estop_patterns) > 1:
            all_nodes = set()
            all_files = set()
            fixes = []

            # Find the most common absolute topic
            absolute_topics = [t for t in estop_patterns.keys() if t.startswith('/')]
            relative_topics = [t for t in estop_patterns.keys() if not t.startswith('/')]

            for topic, usages in estop_patterns.items():
                for node_name, file_path, direction, line_num in usages:
                    all_nodes.add(node_name)
                    if file_path:
                        all_files.add(file_path)

                    # Generate fix for relative topics
                    if not topic.startswith('/') and file_path and line_num > 0:
                        suggested = '/betaray/motors/emergency_stop'
                        line = self._get_line(file_path, line_num)
                        if line:
                            if f"'{topic}'" in line:
                                fixes.append(CodeFix(
                                    file_path=file_path,
                                    line_number=line_num,
                                    original=f"'{topic}'",
                                    replacement=f"'{suggested}'",
                                    description="Standardize emergency stop topic",
                                ))
                            elif f'"{topic}"' in line:
                                fixes.append(CodeFix(
                                    file_path=file_path,
                                    line_number=line_num,
                                    original=f'"{topic}"',
                                    replacement=f'"{suggested}"',
                                    description="Standardize emergency stop topic",
                                ))

            topic_list = ', '.join(sorted(estop_patterns.keys()))
            recommendations.append(Recommendation(
                id="ESTOP-INCONSISTENT",
                title="Emergency stop topic inconsistency",
                severity=Severity.CRITICAL,
                category=Category.SAFETY,
                description=f"Multiple emergency stop topic patterns found: {topic_list}. "
                           f"This can cause e-stop commands to be missed.",
                impact="Emergency stop may fail to reach all nodes, creating a safety hazard",
                affected_nodes=list(all_nodes),
                affected_files=list(all_files),
                fixes=fixes,
            ))

        return recommendations

    def _check_cmd_vel_consistency(self) -> List[Recommendation]:
        """Check for cmd_vel topic naming consistency."""
        recommendations = []

        cmd_vel_topics: Dict[str, List[Tuple[str, str, str]]] = {}

        for node in self.nodes:
            for pub in node.publishers:
                if pub.topic and 'cmd_vel' in pub.topic.lower():
                    if pub.topic not in cmd_vel_topics:
                        cmd_vel_topics[pub.topic] = []
                    cmd_vel_topics[pub.topic].append((node.name, node.file_path, 'pub'))

            for sub in node.subscribers:
                if sub.topic and 'cmd_vel' in sub.topic.lower():
                    if sub.topic not in cmd_vel_topics:
                        cmd_vel_topics[sub.topic] = []
                    cmd_vel_topics[sub.topic].append((node.name, node.file_path, 'sub'))

        if len(cmd_vel_topics) > 1:
            all_nodes = set()
            all_files = set()

            for topic, usages in cmd_vel_topics.items():
                for node_name, file_path, _ in usages:
                    all_nodes.add(node_name)
                    if file_path:
                        all_files.add(file_path)

            topic_list = ', '.join(sorted(cmd_vel_topics.keys()))
            recommendations.append(Recommendation(
                id="CMDVEL-INCONSISTENT",
                title="cmd_vel topic naming inconsistency",
                severity=Severity.HIGH,
                category=Category.NAMING,
                description=f"Multiple cmd_vel topic patterns: {topic_list}. "
                           f"Nodes may not receive velocity commands.",
                impact="Robot may not respond to movement commands from all sources",
                affected_nodes=list(all_nodes),
                affected_files=list(all_files),
                fixes=[],  # Complex fix - needs manual review
            ))

        return recommendations

    def _check_orphaned_connections(self) -> List[Recommendation]:
        """Check for orphaned publishers/subscribers."""
        recommendations = []

        if not self.topic_graph:
            return recommendations

        # Find topics with only publishers (no subscribers listening)
        orphaned_pubs = []
        orphaned_subs = []

        for topic_name, topic_conn in self.topic_graph.topics.items():
            # Skip parameter-based topics
            if topic_name.startswith('$') or topic_name.startswith('{'):
                continue
            # Skip common external topics
            if topic_name in ['/tf', '/tf_static', '/clock', '/parameter_events', '/rosout']:
                continue

            if topic_conn.publishers and not topic_conn.subscribers:
                orphaned_pubs.append((topic_name, topic_conn.publishers))
            elif topic_conn.subscribers and not topic_conn.publishers:
                # Check if this is likely an external source
                external_sources = ['map', 'scan', 'odom', 'tf', 'image', 'camera', 'imu']
                is_external = any(ext in topic_name.lower() for ext in external_sources)
                if not is_external:
                    orphaned_subs.append((topic_name, topic_conn.subscribers))

        # Only report if there are significant orphaned connections
        if len(orphaned_subs) > 3:
            topics_str = ', '.join([t[0] for t in orphaned_subs[:5]])
            if len(orphaned_subs) > 5:
                topics_str += f" (+{len(orphaned_subs) - 5} more)"

            recommendations.append(Recommendation(
                id="ORPHAN-SUBS",
                title=f"{len(orphaned_subs)} subscribers with no publishers",
                severity=Severity.MEDIUM,
                category=Category.CONNECTIVITY,
                description=f"Topics with subscribers but no publishers: {topics_str}. "
                           f"These may be waiting for data that never arrives.",
                impact="Nodes may hang waiting for data, or functionality may be disabled",
                affected_nodes=[],
                affected_files=[],
                fixes=[],
            ))

        return recommendations

    def _check_duplicate_publishers(self) -> List[Recommendation]:
        """Check for multiple publishers to the same topic (potential race condition)."""
        recommendations = []

        if not self.topic_graph:
            return recommendations

        for topic_name, topic_conn in self.topic_graph.topics.items():
            if len(topic_conn.publishers) > 1:
                # Multiple publishers - might be intentional, might be a bug
                # Only flag if it's a command topic
                if 'cmd' in topic_name.lower() or 'control' in topic_name.lower():
                    recommendations.append(Recommendation(
                        id=f"MULTI-PUB-{topic_name.replace('/', '_').upper()[:20]}",
                        title=f"Multiple publishers to command topic: {topic_name}",
                        severity=Severity.MEDIUM,
                        category=Category.ARCHITECTURE,
                        description=f"Topic '{topic_name}' has {len(topic_conn.publishers)} publishers: "
                                   f"{', '.join(topic_conn.publishers)}. Command topics with multiple "
                                   f"publishers can cause race conditions.",
                        impact="Commands may conflict or override each other unpredictably",
                        affected_nodes=topic_conn.publishers,
                        affected_files=[],
                        fixes=[],
                    ))

        return recommendations

    def _check_high_frequency_timers(self) -> List[Recommendation]:
        """Check for potentially problematic timer frequencies."""
        recommendations = []

        for node in self.nodes:
            for timer in node.timers:
                if timer.period and timer.period < 0.001:  # >1000 Hz
                    recommendations.append(Recommendation(
                        id=f"TIMER-FREQ-{node.name.upper()[:15]}",
                        title=f"Very high frequency timer in {node.name}",
                        severity=Severity.LOW,
                        category=Category.PERFORMANCE,
                        description=f"Timer with period {timer.period}s ({1/timer.period:.0f} Hz) "
                                   f"may cause high CPU usage.",
                        impact="May cause CPU overload on resource-constrained devices",
                        affected_nodes=[node.name],
                        affected_files=[node.file_path] if node.file_path else [],
                        fixes=[],
                    ))

        return recommendations

    def _check_missing_error_handling(self) -> List[Recommendation]:
        """Check for potential missing error handling patterns."""
        # This is a placeholder - would need deeper AST analysis
        return []


def generate_recommendations(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
) -> RecommendationReport:
    """
    Convenience function to generate recommendations.

    Args:
        nodes: List of ROS2 nodes to analyze
        topic_graph: Optional topic connection graph

    Returns:
        RecommendationReport with all findings
    """
    engine = RecommendationEngine()
    engine.add_nodes(nodes)
    if topic_graph:
        engine.add_topic_graph(topic_graph)
    return engine.analyze()
