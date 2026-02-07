"""
RoboMind JSON Exporter - Export system analysis to structured JSON.

Generates comprehensive JSON output including:
- Project metadata
- System graph (nodes, edges, statistics)
- Coupling analysis
- Launch topology
- Parameter configurations

Output is designed for machine consumption and further processing.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from robomind.core.graph import SystemGraph
from robomind.analyzers.coupling import CouplingMatrix
from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult
from robomind.ros2.launch_analyzer import LaunchTopology
from robomind.ros2.param_extractor import ParameterCollection

# Optional HTTP communication support
try:
    from robomind.http.communication_map import CommunicationMap
except ImportError:
    CommunicationMap = None

logger = logging.getLogger(__name__)


@dataclass
class ProjectMetadata:
    """Metadata about the analyzed project."""
    name: str
    path: str
    analyzed_at: str
    robomind_version: str = "0.1.0"
    python_files: int = 0
    ros2_packages: int = 0
    total_lines: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnalysisSummary:
    """Summary statistics of the analysis."""
    ros2_nodes: int = 0
    topics: int = 0
    connected_topics: int = 0
    services: int = 0
    actions: int = 0
    parameters: int = 0
    publishers: int = 0
    subscribers: int = 0
    timers: int = 0
    launch_files: int = 0
    config_files: int = 0
    hardware_targets: List[str] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExportResult:
    """Result of a JSON export operation."""
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class JSONExporter:
    """
    Export RoboMind analysis results to JSON format.

    Creates a comprehensive JSON file with:
    - Project metadata
    - Analysis summary
    - Full system graph
    - Coupling analysis (optional)
    - Launch topology (optional)
    - Parameter configurations (optional)

    Usage:
        exporter = JSONExporter()
        exporter.set_metadata(name="BetaRay", path="/home/thor/betaray")
        exporter.set_nodes(ros2_nodes)
        exporter.set_graph(system_graph)
        exporter.set_coupling(coupling_matrix)
        result = exporter.export(Path("output.json"))
    """

    def __init__(self):
        self.metadata: Optional[ProjectMetadata] = None
        self.summary: Optional[AnalysisSummary] = None
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.system_graph: Optional[SystemGraph] = None
        self.coupling: Optional[CouplingMatrix] = None
        self.launch_topology: Optional[LaunchTopology] = None
        self.parameters: Optional[ParameterCollection] = None
        self.http_comm_map = None  # Optional HTTP communication map
        self.external_dependencies: List[Dict] = []  # External ROS2 packages
        self.ai_services = None  # Optional AIServiceAnalysisResult
        self.message_definitions = None  # Optional Dict from MessageDatabase

    def set_metadata(
        self,
        name: str,
        path: str,
        python_files: int = 0,
        ros2_packages: int = 0,
        total_lines: int = 0,
    ):
        """Set project metadata."""
        self.metadata = ProjectMetadata(
            name=name,
            path=path,
            analyzed_at=datetime.now().isoformat(),
            python_files=python_files,
            ros2_packages=ros2_packages,
            total_lines=total_lines,
        )

    def set_nodes(self, nodes: List[ROS2NodeInfo]):
        """Set ROS2 nodes."""
        self.nodes = nodes
        self._update_summary_from_nodes()

    def set_topic_graph(self, topic_graph: TopicGraphResult):
        """Set topic graph."""
        self.topic_graph = topic_graph

    def set_graph(self, graph: SystemGraph):
        """Set system graph."""
        self.system_graph = graph

    def set_coupling(self, coupling: CouplingMatrix):
        """Set coupling analysis."""
        self.coupling = coupling

    def set_launch_topology(self, topology: LaunchTopology):
        """Set launch topology."""
        self.launch_topology = topology

    def set_parameters(self, parameters: ParameterCollection):
        """Set parameter collection."""
        self.parameters = parameters

    def set_http_communication(self, http_comm_map):
        """Set HTTP communication map."""
        self.http_comm_map = http_comm_map

    def set_external_dependencies(self, external_deps: List[Dict]):
        """Set external ROS2 package dependencies (from launch files)."""
        self.external_dependencies = external_deps

    def set_ai_services(self, ai_services):
        """Set AI service analysis result."""
        self.ai_services = ai_services

    def set_message_definitions(self, message_defs: dict):
        """Set message definitions dict (from MessageDatabase.to_dict() filtered to used types)."""
        self.message_definitions = message_defs

    def _update_summary_from_nodes(self):
        """Update summary statistics from nodes."""
        if self.summary is None:
            self.summary = AnalysisSummary()

        self.summary.ros2_nodes = len(self.nodes)
        self.summary.publishers = sum(len(n.publishers) for n in self.nodes)
        self.summary.subscribers = sum(len(n.subscribers) for n in self.nodes)
        self.summary.timers = sum(len(n.timers) for n in self.nodes)
        self.summary.services = sum(len(n.services) for n in self.nodes)
        self.summary.parameters = sum(len(n.parameters) for n in self.nodes)

        # Collect packages
        packages = set()
        for node in self.nodes:
            if node.package_name:
                packages.add(node.package_name)
        self.summary.packages = sorted(packages)

    def build(self) -> Dict[str, Any]:
        """
        Build the complete JSON structure.

        Returns:
            Dictionary ready for JSON serialization
        """
        result = {}

        # Metadata
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()

        # Summary
        if self.summary:
            # Update from other sources
            if self.topic_graph:
                self.summary.topics = len(self.topic_graph.topics)
                self.summary.connected_topics = len(self.topic_graph.get_connected_topics())

            if self.system_graph:
                self.summary.hardware_targets = list(self.system_graph.get_hardware_targets())

            if self.launch_topology:
                self.summary.launch_files = len(self.launch_topology.launch_files)

            if self.parameters:
                self.summary.config_files = len(self.parameters.files)

            result["summary"] = self.summary.to_dict()

        # Nodes
        if self.nodes:
            result["nodes"] = [node.to_dict() for node in self.nodes]

        # Topic graph
        if self.topic_graph:
            result["topics"] = self.topic_graph.to_dict()

        # System graph
        if self.system_graph:
            result["graph"] = self.system_graph.to_dict()

        # Coupling
        if self.coupling:
            result["coupling"] = self.coupling.to_dict()

        # Launch topology
        if self.launch_topology:
            result["launch"] = self.launch_topology.to_dict()

        # Parameters
        if self.parameters:
            result["parameters"] = self.parameters.to_dict()

        # HTTP Communication
        if self.http_comm_map:
            result["http_communication"] = self.http_comm_map.to_dict()

        # External dependencies (from launch files, not in project)
        if self.external_dependencies:
            result["external_dependencies"] = self.external_dependencies

        # AI services
        if self.ai_services:
            result["ai_services"] = self.ai_services.to_dict()

        # Message definitions (only types used by detected nodes)
        if self.message_definitions:
            result["message_definitions"] = self.message_definitions

        return result

    def export(self, output_path: Path, indent: int = 2) -> ExportResult:
        """
        Export to JSON file.

        Args:
            output_path: Path to output file
            indent: JSON indentation level

        Returns:
            ExportResult with success status
        """
        try:
            data = self.build()

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(data, f, indent=indent, default=str)

            stats = {
                "file_size": output_path.stat().st_size,
                "nodes": len(self.nodes),
                "has_graph": self.system_graph is not None,
                "has_coupling": self.coupling is not None,
            }

            logger.info(f"Exported JSON to {output_path} ({stats['file_size']} bytes)")

            return ExportResult(
                success=True,
                output_path=output_path,
                stats=stats,
            )

        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            return ExportResult(
                success=False,
                error=str(e),
            )

    def export_string(self, indent: int = 2) -> str:
        """Export to JSON string."""
        data = self.build()
        return json.dumps(data, indent=indent, default=str)


def export_analysis_json(
    output_path: Path,
    nodes: List[ROS2NodeInfo],
    system_graph: Optional[SystemGraph] = None,
    coupling: Optional[CouplingMatrix] = None,
    topic_graph: Optional[TopicGraphResult] = None,
    launch_topology: Optional[LaunchTopology] = None,
    parameters: Optional[ParameterCollection] = None,
    http_comm_map=None,
    external_dependencies: Optional[List[Dict]] = None,
    ai_services=None,
    message_definitions: Optional[dict] = None,
    project_name: str = "Unknown",
    project_path: str = "",
) -> ExportResult:
    """
    Convenience function to export analysis to JSON.

    Args:
        output_path: Path to output file
        nodes: List of ROS2NodeInfo
        system_graph: Optional SystemGraph
        coupling: Optional CouplingMatrix
        topic_graph: Optional TopicGraphResult
        launch_topology: Optional LaunchTopology
        parameters: Optional ParameterCollection
        http_comm_map: Optional HTTP CommunicationMap
        external_dependencies: Optional list of external ROS2 packages
        ai_services: Optional AIServiceAnalysisResult
        message_definitions: Optional dict of message definitions
        project_name: Name of the project
        project_path: Path to the project

    Returns:
        ExportResult
    """
    exporter = JSONExporter()
    exporter.set_metadata(name=project_name, path=project_path)
    exporter.set_nodes(nodes)

    if topic_graph:
        exporter.set_topic_graph(topic_graph)
    if system_graph:
        exporter.set_graph(system_graph)
    if coupling:
        exporter.set_coupling(coupling)
    if launch_topology:
        exporter.set_launch_topology(launch_topology)
    if parameters:
        exporter.set_parameters(parameters)
    if http_comm_map:
        exporter.set_http_communication(http_comm_map)
    if external_dependencies:
        exporter.set_external_dependencies(external_dependencies)
    if ai_services:
        exporter.set_ai_services(ai_services)
    if message_definitions:
        exporter.set_message_definitions(message_definitions)

    return exporter.export(output_path)


# CLI for testing
if __name__ == "__main__":
    import sys
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.core.graph import build_system_graph
    from robomind.analyzers.coupling import analyze_coupling

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python json_exporter.py <project_path> [output.json]")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis.json")

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
    system_graph = build_system_graph(all_nodes, topic_graph)
    coupling = analyze_coupling(all_nodes, topic_graph)

    # Export
    result = export_analysis_json(
        output_path=output_path,
        nodes=all_nodes,
        system_graph=system_graph,
        coupling=coupling,
        topic_graph=topic_graph,
        project_name=project_path.name,
        project_path=str(project_path),
    )

    if result.success:
        print(f"Exported to {result.output_path}")
        print(f"Stats: {result.stats}")
    else:
        print(f"Export failed: {result.error}")
