"""
RoboMind SARIF Exporter - Export in Static Analysis Results Interchange Format.

SARIF is a standard format for static analysis tools, supported by:
- GitHub Code Scanning
- VS Code SARIF Viewer
- Azure DevOps
- Many CI/CD platforms

This exporter converts RoboMind findings to SARIF 2.1.0 format.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)

# SARIF 2.1.0 schema
SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"

# Rule definitions for RoboMind findings
RULES = {
    "ROBOMIND001": {
        "id": "ROBOMIND001",
        "name": "OrphanedSubscriber",
        "shortDescription": {"text": "Topic has subscribers but no publishers"},
        "fullDescription": {"text": "A ROS2 topic has subscriber nodes but no publisher nodes were found in the codebase. This may indicate a missing publisher or an external data source."},
        "helpUri": "https://github.com/mirrorfinish/robomind/wiki/Findings#ROBOMIND001",
        "defaultConfiguration": {"level": "warning"},
    },
    "ROBOMIND002": {
        "id": "ROBOMIND002",
        "name": "OrphanedPublisher",
        "shortDescription": {"text": "Topic has publishers but no subscribers"},
        "fullDescription": {"text": "A ROS2 topic has publisher nodes but no subscriber nodes were found in the codebase. The data may be consumed by external tools or go unused."},
        "helpUri": "https://github.com/mirrorfinish/robomind/wiki/Findings#ROBOMIND002",
        "defaultConfiguration": {"level": "note"},
    },
    "ROBOMIND003": {
        "id": "ROBOMIND003",
        "name": "RelativeTopicName",
        "shortDescription": {"text": "Topic uses relative name without leading slash"},
        "fullDescription": {"text": "A ROS2 topic is declared with a relative name (no leading slash). This can cause namespace-related issues when the node is launched in different namespaces."},
        "helpUri": "https://github.com/mirrorfinish/robomind/wiki/Findings#ROBOMIND003",
        "defaultConfiguration": {"level": "warning"},
    },
    "ROBOMIND004": {
        "id": "ROBOMIND004",
        "name": "TopicTypeMismatch",
        "shortDescription": {"text": "Topic type mismatch between publisher and subscriber"},
        "fullDescription": {"text": "A ROS2 topic has different message types declared by publishers and subscribers. This will cause runtime communication failures."},
        "helpUri": "https://github.com/mirrorfinish/robomind/wiki/Findings#ROBOMIND004",
        "defaultConfiguration": {"level": "error"},
    },
    "ROBOMIND005": {
        "id": "ROBOMIND005",
        "name": "LowConfidenceNode",
        "shortDescription": {"text": "Node has low confidence score (likely dead code)"},
        "fullDescription": {"text": "A ROS2 node has a low confidence score based on analysis factors like absence from launch files, location in archive directories, etc."},
        "helpUri": "https://github.com/mirrorfinish/robomind/wiki/Findings#ROBOMIND005",
        "defaultConfiguration": {"level": "note"},
    },
    "ROBOMIND006": {
        "id": "ROBOMIND006",
        "name": "HighCouplingPair",
        "shortDescription": {"text": "High coupling between nodes"},
        "fullDescription": {"text": "Two ROS2 nodes have a high coupling score, indicating they are tightly connected. Consider if this coupling is intentional and maintainable."},
        "helpUri": "https://github.com/mirrorfinish/robomind/wiki/Findings#ROBOMIND006",
        "defaultConfiguration": {"level": "note"},
    },
}


@dataclass
class SARIFResult:
    """A single SARIF result (finding)."""
    rule_id: str
    message: str
    file_path: Optional[str] = None
    line_number: int = 1
    column: int = 1
    level: str = "warning"  # error, warning, note
    confidence: float = 0.5

    def to_sarif(self, base_path: str = "") -> Dict:
        """Convert to SARIF result object."""
        result = {
            "ruleId": self.rule_id,
            "level": self.level,
            "message": {"text": self.message},
            "properties": {
                "confidence": self.confidence,
            },
        }

        if self.file_path:
            # Make path relative if base_path provided
            file_path = self.file_path
            if base_path and file_path.startswith(base_path):
                file_path = file_path[len(base_path):].lstrip("/")

            result["locations"] = [{
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": file_path,
                        "uriBaseId": "%SRCROOT%",
                    },
                    "region": {
                        "startLine": self.line_number,
                        "startColumn": self.column,
                    },
                },
            }]

        return result


class SARIFExporter:
    """
    Export RoboMind findings in SARIF 2.1.0 format.

    Creates output compatible with GitHub Code Scanning, VS Code, and
    other SARIF-compatible tools.

    Usage:
        exporter = SARIFExporter()
        exporter.add_result(SARIFResult(...))
        exporter.export(Path("results.sarif"))
    """

    def __init__(self, project_name: str = "robomind", project_path: str = ""):
        self.project_name = project_name
        self.project_path = project_path
        self.results: List[SARIFResult] = []

    def add_result(self, result: SARIFResult):
        """Add a SARIF result."""
        self.results.append(result)

    def build(self) -> Dict:
        """Build complete SARIF structure."""
        # Get rules used in results
        used_rules = set(r.rule_id for r in self.results)
        rules = [RULES[rid] for rid in used_rules if rid in RULES]

        sarif = {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "RoboMind",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/mirrorfinish/robomind",
                        "rules": rules,
                    },
                },
                "results": [r.to_sarif(self.project_path) for r in self.results],
                "invocations": [{
                    "executionSuccessful": True,
                    "endTimeUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }],
            }],
        }

        return sarif

    def export(self, output_path: Path) -> bool:
        """Export to SARIF JSON file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data = self.build()

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported SARIF to {output_path} ({len(self.results)} results)")
            return True

        except Exception as e:
            logger.error(f"Failed to export SARIF: {e}")
            return False


def export_sarif(
    output_path: Path,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    coupling=None,
    confidence_scores: Dict[str, float] = None,
    project_name: str = "robomind",
    project_path: str = "",
) -> bool:
    """
    Convenience function to export analysis as SARIF.

    Args:
        output_path: Path to output file
        nodes: List of ROS2NodeInfo
        topic_graph: Optional TopicGraphResult
        coupling: Optional CouplingMatrix
        confidence_scores: Optional dict of node confidence scores
        project_name: Project name
        project_path: Project root path for relative paths

    Returns:
        True if export succeeded
    """
    exporter = SARIFExporter(project_name, project_path)

    # Generate findings from analysis
    _generate_sarif_results(exporter, nodes, topic_graph, coupling, confidence_scores)

    return exporter.export(output_path)


def _generate_sarif_results(
    exporter: SARIFExporter,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult],
    coupling,
    confidence_scores: Optional[Dict[str, float]],
):
    """Generate SARIF results from analysis."""
    # Check topics
    if topic_graph:
        for topic_name, topic in topic_graph.topics.items():
            # Orphaned subscriber
            if topic.subscribers and not topic.publishers:
                for sub_node in topic.subscribers:
                    # Find the node info
                    node_info = next((n for n in nodes if n.name == sub_node), None)
                    file_path = str(node_info.file_path) if node_info else None
                    line = 1
                    if node_info:
                        for sub in node_info.subscribers:
                            if sub.topic == topic_name:
                                line = sub.line_number
                                break

                    exporter.add_result(SARIFResult(
                        rule_id="ROBOMIND001",
                        message=f"Subscriber for topic '{topic_name}' has no matching publisher in codebase",
                        file_path=file_path,
                        line_number=line,
                        level="warning",
                        confidence=0.7,
                    ))

            # Orphaned publisher (note level)
            elif topic.publishers and not topic.subscribers:
                for pub_node in topic.publishers:
                    node_info = next((n for n in nodes if n.name == pub_node), None)
                    file_path = str(node_info.file_path) if node_info else None
                    line = 1
                    if node_info:
                        for pub in node_info.publishers:
                            if pub.topic == topic_name:
                                line = pub.line_number
                                break

                    exporter.add_result(SARIFResult(
                        rule_id="ROBOMIND002",
                        message=f"Publisher for topic '{topic_name}' has no matching subscriber in codebase",
                        file_path=file_path,
                        line_number=line,
                        level="note",
                        confidence=0.5,
                    ))

        # Relative topic names
        for topic_name in topic_graph.topics.keys():
            if not topic_name.startswith("/"):
                topic = topic_graph.topics[topic_name]
                # Find a node that uses this topic
                node_name = (topic.publishers + topic.subscribers)[0] if (topic.publishers or topic.subscribers) else None
                node_info = next((n for n in nodes if n.name == node_name), None) if node_name else None

                exporter.add_result(SARIFResult(
                    rule_id="ROBOMIND003",
                    message=f"Topic '{topic_name}' uses relative name (should start with '/')",
                    file_path=str(node_info.file_path) if node_info else None,
                    line_number=1,
                    level="warning",
                    confidence=0.8,
                ))

    # Low confidence nodes
    if confidence_scores:
        for name, score in confidence_scores.items():
            if score < 0.3:
                node_info = next((n for n in nodes if n.name == name), None)
                exporter.add_result(SARIFResult(
                    rule_id="ROBOMIND005",
                    message=f"Node '{name}' has low confidence score ({score:.2f}) - likely dead code",
                    file_path=str(node_info.file_path) if node_info else None,
                    line_number=node_info.line_number if node_info else 1,
                    level="note",
                    confidence=score,
                ))

    # High coupling pairs
    if coupling:
        critical = coupling.get_critical_pairs()
        for score in critical[:10]:  # Limit
            exporter.add_result(SARIFResult(
                rule_id="ROBOMIND006",
                message=f"High coupling ({score.score:.2f}) between '{score.source}' and '{score.target}'",
                level="note",
                confidence=score.score,
            ))
