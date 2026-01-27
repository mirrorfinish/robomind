"""
RoboMind Markdown Reporter - Generate comprehensive markdown reports.

This module generates structured markdown reports containing:
- Executive summary
- Critical issues with severity levels
- Namespace analysis
- Coupling hotspots
- Topic flow diagrams (mermaid)
- Recommendations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult
from robomind.core.graph import SystemGraph
from robomind.analyzers.coupling import CouplingMatrix

logger = logging.getLogger(__name__)


@dataclass
class ReportResult:
    """Result of report generation."""
    success: bool = False
    output_path: Optional[Path] = None
    content: str = ""
    error: Optional[str] = None
    stats: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "error": self.error,
            "stats": self.stats,
        }


class MarkdownReporter:
    """
    Generate comprehensive markdown reports from RoboMind analysis.

    Usage:
        reporter = MarkdownReporter(
            nodes=nodes,
            topic_graph=topic_graph,
            system_graph=system_graph,
            coupling=coupling_matrix,
            project_name="MyRobot",
        )
        result = reporter.generate(output_path)
    """

    def __init__(
        self,
        nodes: List[ROS2NodeInfo],
        topic_graph: Optional[TopicGraphResult] = None,
        system_graph: Optional[SystemGraph] = None,
        coupling: Optional[CouplingMatrix] = None,
        project_name: str = "ROS2 Project",
        project_path: Optional[str] = None,
    ):
        self.nodes = nodes
        self.topic_graph = topic_graph
        self.system_graph = system_graph
        self.coupling = coupling
        self.project_name = project_name
        self.project_path = project_path

        # Pre-calculate analysis
        self._orphaned_pubs = []
        self._orphaned_subs = []
        self._namespace_issues = []
        self._missing_leading_slash = []

        if topic_graph:
            self._analyze_topics()

    def _analyze_topics(self):
        """Analyze topics for issues."""
        for topic_name, topic in self.topic_graph.topics.items():
            # Check for orphans
            if topic.publishers and not topic.subscribers:
                self._orphaned_pubs.append({
                    "topic": topic_name,
                    "publishers": list(topic.publishers),
                    "type": topic.msg_type,
                })
            elif topic.subscribers and not topic.publishers:
                self._orphaned_subs.append({
                    "topic": topic_name,
                    "subscribers": list(topic.subscribers),
                    "type": topic.msg_type,
                })

            # Check for missing leading slash
            if not topic_name.startswith("/"):
                self._missing_leading_slash.append(topic_name)

        # Detect namespace mismatches (e.g., /betaray/ vs /nexus/)
        namespaces = set()
        for topic_name in self.topic_graph.topics.keys():
            if topic_name.startswith("/"):
                parts = topic_name.split("/")
                if len(parts) > 2:
                    namespaces.add(parts[1])

        if len(namespaces) > 1:
            self._namespace_issues = list(namespaces)

    def generate(self, output_path: Optional[Path] = None) -> ReportResult:
        """
        Generate the markdown report.

        Args:
            output_path: Optional path to write report file

        Returns:
            ReportResult with success status and content
        """
        result = ReportResult()

        try:
            sections = []

            # Header
            sections.append(self._generate_header())

            # Executive Summary
            sections.append(self._generate_executive_summary())

            # Critical Issues
            sections.append(self._generate_critical_issues())

            # Topic Analysis
            sections.append(self._generate_topic_analysis())

            # Coupling Analysis
            if self.coupling:
                sections.append(self._generate_coupling_analysis())

            # Node Details
            sections.append(self._generate_node_summary())

            # Recommendations
            sections.append(self._generate_recommendations())

            # Footer
            sections.append(self._generate_footer())

            # Combine sections
            result.content = "\n\n".join(sections)
            result.success = True
            result.stats = {
                "sections": len(sections),
                "nodes": len(self.nodes),
                "topics": len(self.topic_graph.topics) if self.topic_graph else 0,
                "critical_issues": len(self._orphaned_subs) + len(self._namespace_issues),
            }

            # Write to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(result.content)
                result.output_path = output_path
                result.stats["file_size"] = len(result.content)

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            result.success = False
            result.error = str(e)

        return result

    def _generate_header(self) -> str:
        """Generate report header."""
        lines = [
            f"# {self.project_name} Architecture Report",
            "",
            f"**Generated by**: RoboMind",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        if self.project_path:
            lines.append(f"**Project Path**: `{self.project_path}`")
        lines.append("")
        lines.append("---")
        return "\n".join(lines)

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        # Calculate stats
        total_nodes = len(self.nodes)
        total_topics = len(self.topic_graph.topics) if self.topic_graph else 0
        connected_topics = len(self.topic_graph.get_connected_topics()) if self.topic_graph else 0
        orphaned_topics = total_topics - connected_topics

        total_publishers = sum(len(n.publishers) for n in self.nodes)
        total_subscribers = sum(len(n.subscribers) for n in self.nodes)

        # Determine health status
        critical_count = len(self._orphaned_subs)  # Subscribers waiting for data
        namespace_count = len(self._namespace_issues)

        if critical_count > 10 or namespace_count > 3:
            health = "Needs Attention"
            health_icon = "Warning"
        elif critical_count > 0 or namespace_count > 0:
            health = "Minor Issues"
            health_icon = "Note"
        else:
            health = "Healthy"
            health_icon = "Good"

        lines = [
            "## Executive Summary",
            "",
            f"**System Health**: {health}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| ROS2 Nodes | {total_nodes} |",
            f"| Topics | {total_topics} ({connected_topics} connected) |",
            f"| Publishers | {total_publishers} |",
            f"| Subscribers | {total_subscribers} |",
            f"| Orphaned Publishers | {len(self._orphaned_pubs)} |",
            f"| Orphaned Subscribers | {len(self._orphaned_subs)} |",
            f"| Missing Leading Slash | {len(self._missing_leading_slash)} |",
        ]

        if self.coupling:
            summary = self.coupling.summary()
            lines.extend([
                f"| Coupling Pairs | {summary['total_pairs']} |",
                f"| Critical Coupling | {summary['critical_pairs']} |",
            ])

        return "\n".join(lines)

    def _generate_critical_issues(self) -> str:
        """Generate critical issues section."""
        lines = [
            "## Critical Issues",
            "",
        ]

        issue_count = 0

        # Orphaned subscribers (nodes waiting for data)
        if self._orphaned_subs:
            lines.append("### Orphaned Subscribers (Nodes Waiting for Data)")
            lines.append("")
            lines.append("These subscribers have NO publishers - they will never receive data:")
            lines.append("")
            lines.append("| Topic | Subscribers | Type |")
            lines.append("|-------|-------------|------|")

            for item in sorted(self._orphaned_subs, key=lambda x: len(x["subscribers"]), reverse=True)[:20]:
                subs = ", ".join(item["subscribers"][:3])
                if len(item["subscribers"]) > 3:
                    subs += f" (+{len(item['subscribers']) - 3})"
                msg_type = item.get("type", "unknown")
                lines.append(f"| `{item['topic']}` | {subs} | {msg_type} |")

            if len(self._orphaned_subs) > 20:
                lines.append(f"| ... | {len(self._orphaned_subs) - 20} more | |")

            issue_count += len(self._orphaned_subs)
            lines.append("")

        # Namespace mismatches
        if self._namespace_issues:
            lines.append("### Namespace Inconsistencies")
            lines.append("")
            lines.append("Multiple namespace prefixes detected - this often causes connectivity issues:")
            lines.append("")
            lines.append("```")
            for ns in sorted(self._namespace_issues):
                lines.append(f"/{ns}/...")
            lines.append("```")
            lines.append("")
            lines.append("**Recommendation**: Standardize on a single namespace prefix.")
            issue_count += 1
            lines.append("")

        # Missing leading slashes
        if self._missing_leading_slash:
            lines.append("### Topics Missing Leading Slash")
            lines.append("")
            lines.append("These topics violate ROS2 naming convention (may cause namespace issues):")
            lines.append("")
            lines.append("```")
            for topic in sorted(self._missing_leading_slash)[:30]:
                lines.append(topic)
            if len(self._missing_leading_slash) > 30:
                lines.append(f"... and {len(self._missing_leading_slash) - 30} more")
            lines.append("```")
            issue_count += 1
            lines.append("")

        if issue_count == 0:
            lines.append("No critical issues detected.")

        return "\n".join(lines)

    def _generate_topic_analysis(self) -> str:
        """Generate topic analysis section."""
        if not self.topic_graph:
            return "## Topic Analysis\n\nNo topic graph available."

        lines = [
            "## Topic Analysis",
            "",
        ]

        # Topic flow diagram (mermaid)
        connected = self.topic_graph.get_connected_topics()
        if connected and len(connected) <= 30:
            lines.append("### Topic Flow Diagram")
            lines.append("")
            lines.append("```mermaid")
            lines.append("graph LR")

            for topic in list(connected)[:30]:
                topic_name = topic.name
                safe_topic = topic_name.replace("/", "_").replace("-", "_")

                for pub in list(topic.publishers)[:5]:
                    safe_pub = pub.replace("/", "_").replace("-", "_")
                    lines.append(f"    {safe_pub}[{pub}] --> {safe_topic}(({topic_name[-20:]}))")

                for sub in list(topic.subscribers)[:5]:
                    safe_sub = sub.replace("/", "_").replace("-", "_")
                    lines.append(f"    {safe_topic}(({topic_name[-20:]})) --> {safe_sub}[{sub}]")

            lines.append("```")
            lines.append("")

        # Orphaned publishers table
        if self._orphaned_pubs:
            lines.append("### Orphaned Publishers (No Subscribers)")
            lines.append("")
            lines.append("| Topic | Publishers | Type |")
            lines.append("|-------|------------|------|")

            for item in sorted(self._orphaned_pubs, key=lambda x: len(x["publishers"]), reverse=True)[:15]:
                pubs = ", ".join(item["publishers"][:3])
                if len(item["publishers"]) > 3:
                    pubs += f" (+{len(item['publishers']) - 3})"
                msg_type = item.get("type", "unknown")
                lines.append(f"| `{item['topic']}` | {pubs} | {msg_type} |")

            lines.append("")

        return "\n".join(lines)

    def _generate_coupling_analysis(self) -> str:
        """Generate coupling analysis section."""
        if not self.coupling:
            return ""

        summary = self.coupling.summary()
        lines = [
            "## Coupling Analysis",
            "",
            f"Analyzed **{summary['nodes_analyzed']}** nodes with **{summary['total_pairs']}** connected pairs.",
            "",
            "| Coupling Level | Count |",
            "|----------------|-------|",
            f"| Critical (>0.7) | {summary['critical_pairs']} |",
            f"| High (0.5-0.7) | {summary['high_pairs']} |",
            f"| Medium (0.3-0.5) | {summary['medium_pairs']} |",
            f"| Low (<0.3) | {summary['low_pairs']} |",
            "",
        ]

        # Top coupled pairs
        top_pairs = self.coupling.get_top_coupled_pairs(10)
        if top_pairs:
            lines.append("### Tightly Coupled Pairs")
            lines.append("")
            lines.append("These node pairs have high coupling and should be considered together during refactoring:")
            lines.append("")
            lines.append("| Node A | Node B | Score | Level | Topics |")
            lines.append("|--------|--------|-------|-------|--------|")

            for score in top_pairs:
                topics_str = ", ".join(score.topics[:3])
                if len(score.topics) > 3:
                    topics_str += "..."
                lines.append(
                    f"| {score.source} | {score.target} | {score.score:.3f} | {score.strength} | {topics_str} |"
                )

            lines.append("")

        return "\n".join(lines)

    def _generate_node_summary(self) -> str:
        """Generate node summary section."""
        lines = [
            "## Node Summary",
            "",
        ]

        # Group by package
        packages: Dict[str, List[ROS2NodeInfo]] = {}
        for node in self.nodes:
            pkg = node.package_name or "unknown"
            if pkg not in packages:
                packages[pkg] = []
            packages[pkg].append(node)

        lines.append(f"**{len(self.nodes)}** ROS2 nodes across **{len(packages)}** packages:")
        lines.append("")

        for pkg_name in sorted(packages.keys()):
            pkg_nodes = packages[pkg_name]
            lines.append(f"### {pkg_name} ({len(pkg_nodes)} nodes)")
            lines.append("")
            lines.append("| Node | Publishers | Subscribers | Timers | File |")
            lines.append("|------|------------|-------------|--------|------|")

            for node in sorted(pkg_nodes, key=lambda n: n.name):
                file_name = Path(node.file_path).name if node.file_path else "-"
                lines.append(
                    f"| {node.name} | {len(node.publishers)} | {len(node.subscribers)} | "
                    f"{len(node.timers)} | {file_name} |"
                )

            lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(self) -> str:
        """Generate recommendations section."""
        lines = [
            "## Recommendations",
            "",
        ]

        recommendations = []

        # Based on orphaned subscribers
        if len(self._orphaned_subs) > 5:
            recommendations.append({
                "priority": 1,
                "title": "Fix Orphaned Subscribers",
                "description": f"{len(self._orphaned_subs)} subscribers have no publishers. "
                              "These nodes are waiting for data that will never arrive.",
                "action": "Review topic names for typos, namespace mismatches, or missing nodes.",
            })

        # Based on namespace issues
        if self._namespace_issues:
            recommendations.append({
                "priority": 1,
                "title": "Standardize Namespace",
                "description": f"Multiple namespaces detected: {', '.join(self._namespace_issues)}",
                "action": "Pick ONE namespace prefix and update all topics to use it consistently.",
            })

        # Based on missing leading slashes
        if len(self._missing_leading_slash) > 10:
            recommendations.append({
                "priority": 2,
                "title": "Add Leading Slashes to Topics",
                "description": f"{len(self._missing_leading_slash)} topics missing leading slash (violates ROS2 convention).",
                "action": "Change topic names from 'topic' to '/topic' for consistency.",
            })

        # Based on coupling
        if self.coupling:
            critical = self.coupling.get_critical_pairs()
            if len(critical) > 5:
                recommendations.append({
                    "priority": 2,
                    "title": "Review Critical Coupling",
                    "description": f"{len(critical)} node pairs have critical coupling (>0.7).",
                    "action": "Consider refactoring tightly coupled components or documenting dependencies.",
                })

        # Output recommendations
        if recommendations:
            lines.append("### Priority Actions")
            lines.append("")

            for i, rec in enumerate(sorted(recommendations, key=lambda r: r["priority"]), 1):
                lines.append(f"**{i}. {rec['title']}** (Priority {rec['priority']})")
                lines.append("")
                lines.append(f"{rec['description']}")
                lines.append("")
                lines.append(f"*Action*: {rec['action']}")
                lines.append("")
        else:
            lines.append("No critical recommendations at this time. System appears well-structured.")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return "\n".join([
            "---",
            "",
            "*Report generated by [RoboMind](https://github.com/mirrorfinish/robomind)*",
        ])


def generate_report(
    output_path: Path,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    system_graph: Optional[SystemGraph] = None,
    coupling: Optional[CouplingMatrix] = None,
    project_name: str = "ROS2 Project",
    project_path: Optional[str] = None,
) -> ReportResult:
    """
    Convenience function to generate a markdown report.

    Args:
        output_path: Path to write the report
        nodes: List of ROS2NodeInfo from analysis
        topic_graph: Optional TopicGraphResult
        system_graph: Optional SystemGraph
        coupling: Optional CouplingMatrix
        project_name: Name of the project
        project_path: Path to the project

    Returns:
        ReportResult with success status
    """
    reporter = MarkdownReporter(
        nodes=nodes,
        topic_graph=topic_graph,
        system_graph=system_graph,
        coupling=coupling,
        project_name=project_name,
        project_path=project_path,
    )
    return reporter.generate(output_path)


# CLI for testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.analyzers.coupling import analyze_coupling

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python markdown_reporter.py <project_path> [output.md]")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("REPORT.md")

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
    coupling = analyze_coupling(all_nodes, topic_graph)

    print(f"Found {len(all_nodes)} nodes, {len(topic_graph.topics)} topics")

    # Generate report
    result = generate_report(
        output_path=output_path,
        nodes=all_nodes,
        topic_graph=topic_graph,
        coupling=coupling,
        project_name=project_path.name,
        project_path=str(project_path),
    )

    if result.success:
        print(f"Report generated: {result.output_path}")
        print(f"Size: {result.stats.get('file_size', 0):,} bytes")
    else:
        print(f"Report failed: {result.error}")
