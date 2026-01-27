"""
RoboMind AI Context Exporter - Generate LLM-optimized output.

Creates output specifically designed for AI assistants:
- Structured summary with deployment status
- Actionable vs non-actionable findings clearly separated
- Confidence scores with explanations
- Suggested fixes where applicable
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult
from robomind.analyzers.coupling import CouplingMatrix

logger = logging.getLogger(__name__)


@dataclass
class AIFinding:
    """A finding structured for AI consumption."""
    id: str
    summary: str
    file_path: Optional[str] = None
    line_number: int = 0
    confidence: float = 0.5
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    effort_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    current_code: Optional[str] = None
    suggested_fix: Optional[str] = None
    runtime_confirmed: bool = False
    actionable: bool = True
    non_actionable_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "summary": self.summary,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "effort_level": self.effort_level,
            "actionable": self.actionable,
        }
        if self.file_path:
            result["file"] = f"{self.file_path}:{self.line_number}" if self.line_number else self.file_path
        if self.current_code:
            result["current_code"] = self.current_code
        if self.suggested_fix:
            result["suggested_fix"] = self.suggested_fix
        if self.runtime_confirmed:
            result["runtime_confirmed"] = True
        if not self.actionable and self.non_actionable_reason:
            result["reason_non_actionable"] = self.non_actionable_reason
        return result


class AIContextExporter:
    """
    Export RoboMind analysis in AI-optimized format.

    Creates a YAML structure designed for LLM consumption with:
    - System summary (architecture, protocol, domains)
    - Deployment status (running vs likely dead)
    - Actionable findings with fixes
    - Non-actionable findings with reasons

    Usage:
        exporter = AIContextExporter()
        exporter.set_project_info("BetaRay")
        exporter.set_nodes(nodes)
        exporter.add_actionable_finding(...)
        exporter.export(Path("ai_context.yaml"))
    """

    def __init__(self):
        self.project_name: str = "Unknown"
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.coupling: Optional[CouplingMatrix] = None
        self.http_comm_map = None
        self.validation_result = None
        self.confidence_scores: Dict[str, float] = {}

        self.actionable_findings: List[AIFinding] = []
        self.non_actionable_findings: List[AIFinding] = []

        # Deployment status
        self.confirmed_running: List[str] = []
        self.likely_dead: List[str] = []

    def set_project_info(self, name: str):
        """Set project name."""
        self.project_name = name

    def set_nodes(self, nodes: List[ROS2NodeInfo]):
        """Set ROS2 nodes."""
        self.nodes = nodes

    def set_topic_graph(self, topic_graph: TopicGraphResult):
        """Set topic graph."""
        self.topic_graph = topic_graph

    def set_coupling(self, coupling: CouplingMatrix):
        """Set coupling analysis."""
        self.coupling = coupling

    def set_http_communication(self, http_comm_map):
        """Set HTTP communication map."""
        self.http_comm_map = http_comm_map

    def set_validation_result(self, validation_result):
        """Set validation result."""
        self.validation_result = validation_result

    def set_confidence_scores(self, scores: Dict[str, float]):
        """Set node confidence scores."""
        self.confidence_scores = scores

    def set_deployment_status(
        self,
        confirmed_running: List[str],
        likely_dead: List[str],
    ):
        """Set deployment status."""
        self.confirmed_running = confirmed_running
        self.likely_dead = likely_dead

    def add_actionable_finding(self, finding: AIFinding):
        """Add an actionable finding."""
        finding.actionable = True
        self.actionable_findings.append(finding)

    def add_non_actionable_finding(self, finding: AIFinding):
        """Add a non-actionable finding."""
        finding.actionable = False
        self.non_actionable_findings.append(finding)

    def _build_system_summary(self) -> Dict:
        """Build system summary section."""
        summary = {}

        # Include project name
        if self.project_name and self.project_name != "Unknown":
            summary["project"] = self.project_name

        # Determine architecture type
        if self.http_comm_map:
            http_summary = self.http_comm_map.summary()
            if http_summary.get("http_clients", 0) > 0:
                summary["architecture"] = "HTTP-based distributed system"
                summary["cross_system_protocol"] = "http"
            else:
                summary["architecture"] = "ROS2-based system"
                summary["cross_system_protocol"] = "ros2"
        else:
            summary["architecture"] = "ROS2-based system"
            summary["cross_system_protocol"] = "ros2"

        # ROS2 domain info
        if self.nodes:
            packages = set(n.package_name for n in self.nodes if n.package_name)
            summary["ros2_packages"] = len(packages)
            summary["ros2_nodes"] = len(self.nodes)

        # Topic info
        if self.topic_graph:
            connected = self.topic_graph.get_connected_topics()
            summary["topics_total"] = len(self.topic_graph.topics)
            summary["topics_connected"] = len(connected)

        # HTTP info
        if self.http_comm_map:
            http_summary = self.http_comm_map.summary()
            if http_summary.get("http_endpoints", 0) > 0:
                summary["http_endpoints"] = http_summary["http_endpoints"]
                summary["http_target_hosts"] = http_summary.get("http_target_hosts", [])

        return summary

    def _build_deployment_status(self) -> Dict:
        """Build deployment status section."""
        status = {}

        if self.confirmed_running:
            status["confirmed_running"] = self.confirmed_running

        if self.likely_dead:
            status["likely_dead"] = self.likely_dead

        # Derive from validation if available
        if self.validation_result and self.validation_result.live_info:
            live_nodes = set(self.validation_result.live_info.nodes)
            code_nodes = {n.name for n in self.nodes}

            running = code_nodes & live_nodes
            not_running = code_nodes - live_nodes

            if running and "confirmed_running" not in status:
                status["confirmed_running"] = sorted(list(running))[:20]
            if not_running and "likely_dead" not in status:
                status["not_running"] = sorted(list(not_running))[:20]

        # Use confidence scores for "likely dead"
        if self.confidence_scores and "likely_dead" not in status:
            low_conf = [
                name for name, score in self.confidence_scores.items()
                if score < 0.3
            ]
            if low_conf:
                status["low_confidence_nodes"] = low_conf[:20]

        return status

    def build(self) -> Dict:
        """Build complete AI context structure."""
        context = {}

        # Header
        context["_comment"] = "AI-optimized context generated by RoboMind"
        context["_generated_at"] = datetime.now().isoformat()

        # System summary
        context["system_summary"] = self._build_system_summary()

        # Deployment status
        deployment = self._build_deployment_status()
        if deployment:
            context["deployment_status"] = deployment

        # Actionable findings
        if self.actionable_findings:
            context["actionable_findings"] = [
                f.to_dict() for f in sorted(
                    self.actionable_findings,
                    key=lambda f: (f.risk_level != "CRITICAL", f.confidence),
                    reverse=False,
                )
            ]

        # Non-actionable findings
        if self.non_actionable_findings:
            context["non_actionable_findings"] = [
                f.to_dict() for f in self.non_actionable_findings
            ]

        # Quick reference for common tasks
        context["ai_hints"] = {
            "to_understand_architecture": "See system_summary section",
            "to_find_issues": "Check actionable_findings, sorted by risk",
            "to_understand_dead_code": "See deployment_status.likely_dead or low_confidence_nodes",
            "for_fixes": "Each actionable finding includes suggested_fix when available",
        }

        return context

    def export(self, output_path: Path) -> bool:
        """Export to YAML file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data = self.build()

            # Custom YAML formatting
            header = "# AI-Optimized Context - Generated by RoboMind\n"
            header += "# Feed this to an AI assistant for architecture understanding\n\n"

            with open(output_path, "w") as f:
                f.write(header)
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            logger.info(f"Exported AI context to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export AI context: {e}")
            return False


def export_ai_context(
    output_path: Path,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    coupling: Optional[CouplingMatrix] = None,
    http_comm_map=None,
    validation_result=None,
    confidence_scores: Dict[str, float] = None,
    project_name: str = "Unknown",
) -> bool:
    """
    Convenience function to export AI-optimized context.

    Args:
        output_path: Path to output file
        nodes: List of ROS2NodeInfo
        topic_graph: Optional TopicGraphResult
        coupling: Optional CouplingMatrix
        http_comm_map: Optional HTTP CommunicationMap
        validation_result: Optional ValidationResult
        confidence_scores: Optional dict of node confidence scores
        project_name: Project name

    Returns:
        True if export succeeded
    """
    exporter = AIContextExporter()
    exporter.set_project_info(project_name)
    exporter.set_nodes(nodes)

    if topic_graph:
        exporter.set_topic_graph(topic_graph)
    if coupling:
        exporter.set_coupling(coupling)
    if http_comm_map:
        exporter.set_http_communication(http_comm_map)
    if validation_result:
        exporter.set_validation_result(validation_result)
    if confidence_scores:
        exporter.set_confidence_scores(confidence_scores)

    # Auto-generate findings from analysis
    _auto_generate_findings(exporter, nodes, topic_graph, confidence_scores)

    return exporter.export(output_path)


def _auto_generate_findings(
    exporter: AIContextExporter,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult],
    confidence_scores: Optional[Dict[str, float]],
):
    """Auto-generate findings from analysis results."""
    finding_id = 0

    # Find orphaned topics (subscribers without publishers)
    if topic_graph:
        for topic_name, topic in topic_graph.topics.items():
            # Subscriber without publisher
            if topic.subscribers and not topic.publishers:
                finding_id += 1
                exporter.add_actionable_finding(AIFinding(
                    id=f"TOPIC-{finding_id:03d}",
                    summary=f"Topic '{topic_name}' has subscribers but no publishers",
                    confidence=0.7,
                    risk_level="MEDIUM",
                    effort_level="LOW",
                    suggested_fix=f"Add a publisher for topic '{topic_name}' or check if external node publishes it",
                ))

            # Publisher without subscriber
            elif topic.publishers and not topic.subscribers:
                finding_id += 1
                exporter.add_non_actionable_finding(AIFinding(
                    id=f"TOPIC-{finding_id:03d}",
                    summary=f"Topic '{topic_name}' has publishers but no subscribers in this codebase",
                    confidence=0.5,
                    risk_level="LOW",
                    non_actionable_reason="May be consumed by external tools (rviz, rosbag, etc.)",
                ))

    # Find low-confidence nodes
    if confidence_scores:
        for name, score in confidence_scores.items():
            if score < 0.3:
                finding_id += 1
                exporter.add_non_actionable_finding(AIFinding(
                    id=f"CONF-{finding_id:03d}",
                    summary=f"Node '{name}' has low confidence ({score:.2f})",
                    confidence=score,
                    risk_level="LOW",
                    non_actionable_reason="Likely dead code - not in launch files or archive directory",
                ))

    # Check for topic naming issues
    if topic_graph:
        for topic_name in topic_graph.topics.keys():
            if not topic_name.startswith("/"):
                finding_id += 1
                exporter.add_actionable_finding(AIFinding(
                    id=f"NAMING-{finding_id:03d}",
                    summary=f"Topic '{topic_name}' uses relative name (no leading slash)",
                    confidence=0.8,
                    risk_level="MEDIUM",
                    effort_level="LOW",
                    suggested_fix=f"Change to absolute topic name: '/{topic_name}'",
                ))
