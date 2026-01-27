"""
RoboMind YAML Exporter - Export AI-optimized context in NEXUS schema format.

Generates token-efficient YAML designed for LLM consumption:
- CONTEXT_SUMMARY.yaml (~150-200 tokens) - Quick reference
- system_context.yaml - Full architecture context

Output is optimized for AI assistants to understand system architecture quickly.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import yaml

from robomind.core.graph import SystemGraph, ComponentType
from robomind.analyzers.coupling import CouplingMatrix
from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult
from robomind.ros2.launch_analyzer import LaunchTopology
from robomind.ros2.param_extractor import ParameterCollection

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of a YAML export operation."""
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    token_estimate: int = 0


class YAMLExporter:
    """
    Export RoboMind analysis to AI-optimized YAML format.

    Generates two files:
    1. CONTEXT_SUMMARY.yaml - Quick reference (~150-200 tokens)
    2. system_context.yaml - Full architecture context

    The format is designed to be token-efficient for LLM context windows
    while providing enough detail for AI-assisted development.

    Usage:
        exporter = YAMLExporter()
        exporter.set_project_info(name="BetaRay", version="1.0.0")
        exporter.set_nodes(ros2_nodes)
        exporter.set_graph(system_graph)
        exporter.export_all(output_dir)
    """

    def __init__(self):
        self.project_name: str = "Unknown"
        self.project_version: str = "auto"
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.system_graph: Optional[SystemGraph] = None
        self.coupling: Optional[CouplingMatrix] = None
        self.launch_topology: Optional[LaunchTopology] = None
        self.parameters: Optional[ParameterCollection] = None
        self.hardware_targets: Dict[str, List[str]] = {}  # hardware -> node names
        self.http_comm_map = None  # Optional HTTP communication map

    def set_project_info(self, name: str, version: str = "auto"):
        """Set project metadata."""
        self.project_name = name
        if version == "auto":
            self.project_version = f"auto-{datetime.now().strftime('%Y-%m-%d')}"
        else:
            self.project_version = version

    def set_nodes(self, nodes: List[ROS2NodeInfo]):
        """Set ROS2 nodes."""
        self.nodes = nodes

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

    def set_hardware_mapping(self, mapping: Dict[str, List[str]]):
        """Set hardware target to node mapping."""
        self.hardware_targets = mapping

    def set_http_communication(self, http_comm_map):
        """Set HTTP communication map."""
        self.http_comm_map = http_comm_map

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token average)."""
        return len(text) // 4

    # ========== CONTEXT_SUMMARY.yaml ==========

    def build_context_summary(self) -> Dict[str, Any]:
        """
        Build quick reference context (~150-200 tokens).

        This is the minimal context needed to understand the system.
        """
        summary = {
            "system": self.project_name.lower().replace(" ", "_"),
            "version": self.project_version,
        }

        # Node count
        summary["nodes"] = len(self.nodes)

        # Package list
        packages = set()
        for node in self.nodes:
            if node.package_name:
                packages.add(node.package_name)
        if packages:
            summary["packages"] = sorted(packages)

        # Hardware targets
        if self.hardware_targets:
            summary["hardware"] = list(self.hardware_targets.keys())

        # Key topics (connected ones only)
        if self.topic_graph:
            connected = self.topic_graph.get_connected_topics()
            if connected:
                # Show top 10 most-connected topics
                sorted_topics = sorted(
                    connected,
                    key=lambda t: len(t.publishers) + len(t.subscribers),
                    reverse=True
                )[:10]
                summary["key_topics"] = [t.name for t in sorted_topics]

        # Critical coupling pairs
        if self.coupling:
            critical = self.coupling.get_critical_pairs()
            if critical:
                summary["critical_coupling"] = [
                    f"{s.source} <-> {s.target}"
                    for s in critical[:5]
                ]

        # Launch file count
        if self.launch_topology:
            summary["launch_files"] = len(self.launch_topology.launch_files)

        return summary

    def export_context_summary(self, output_path: Path) -> ExportResult:
        """Export CONTEXT_SUMMARY.yaml."""
        try:
            data = self.build_context_summary()

            # Add header comment
            header = "# Quick Reference - Auto-generated by RoboMind\n"
            header += f"# ~{self._estimate_tokens(yaml.dump(data))} tokens\n\n"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(header)
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            content = output_path.read_text()
            token_estimate = self._estimate_tokens(content)

            logger.info(f"Exported CONTEXT_SUMMARY to {output_path} (~{token_estimate} tokens)")

            return ExportResult(
                success=True,
                output_path=output_path,
                token_estimate=token_estimate,
            )

        except Exception as e:
            logger.error(f"Failed to export CONTEXT_SUMMARY: {e}")
            return ExportResult(success=False, error=str(e))

    # ========== system_context.yaml ==========

    def build_system_context(self) -> Dict[str, Any]:
        """
        Build full system context for AI consumption.

        Structured hierarchically:
        - metadata
        - architecture (by subsystem)
        - topics
        - services
        - parameters
        - coupling
        """
        context = {}

        # Metadata
        context["metadata"] = {
            "name": self.project_name,
            "version": self.project_version,
            "generated_at": datetime.now().isoformat(),
            "tool": "RoboMind",
            "nodes": len(self.nodes),
        }

        # Hardware targets
        if self.hardware_targets:
            context["metadata"]["distributed_hosts"] = list(self.hardware_targets.keys())

        # Architecture - group nodes by package
        architecture = {}
        nodes_by_package: Dict[str, List[ROS2NodeInfo]] = {}

        for node in self.nodes:
            pkg = node.package_name or "unknown"
            if pkg not in nodes_by_package:
                nodes_by_package[pkg] = []
            nodes_by_package[pkg].append(node)

        for pkg, pkg_nodes in sorted(nodes_by_package.items()):
            pkg_info = {
                "components": [n.name for n in pkg_nodes],
                "node_count": len(pkg_nodes),
            }

            # Add hardware if known
            hardware = set()
            for node in pkg_nodes:
                for hw, hw_nodes in self.hardware_targets.items():
                    if node.name in hw_nodes:
                        hardware.add(hw)
            if hardware:
                pkg_info["hardware"] = list(hardware)

            architecture[pkg] = pkg_info

        if architecture:
            context["architecture"] = architecture

        # Nodes detail
        nodes_detail = {}
        for node in self.nodes:
            node_info = {
                "class": node.class_name,
                "file": str(node.file_path.name) if node.file_path else None,
            }

            if node.publishers:
                node_info["publishes"] = [
                    {"topic": p.topic, "type": p.msg_type}
                    for p in node.publishers
                ]

            if node.subscribers:
                node_info["subscribes"] = [
                    {"topic": s.topic, "type": s.msg_type}
                    for s in node.subscribers
                ]

            if node.services:
                node_info["services"] = [s.name for s in node.services]

            if node.service_clients:
                node_info["service_clients"] = [c.name for c in node.service_clients]

            if node.timers:
                node_info["timers"] = [
                    {"period": t.period, "hz": round(t.frequency_hz, 1)}
                    for t in node.timers
                ]

            if node.parameters:
                node_info["parameters"] = [p.name for p in node.parameters]

            nodes_detail[node.name] = node_info

        if nodes_detail:
            context["nodes"] = nodes_detail

        # Topics
        if self.topic_graph:
            topics = {}
            for topic_name, topic in self.topic_graph.topics.items():
                topic_info = {
                    "msg_type": topic.msg_type,
                }
                if topic.publishers:
                    topic_info["publishers"] = topic.publishers
                if topic.subscribers:
                    topic_info["subscribers"] = topic.subscribers
                topic_info["connected"] = topic.is_connected

                topics[topic_name] = topic_info

            if topics:
                context["topics"] = topics

        # Coupling matrix (high/critical only)
        if self.coupling:
            high_coupled = self.coupling.get_high_coupled_pairs()
            if high_coupled:
                coupling_info = {}
                for score in high_coupled[:20]:  # Top 20
                    key = f"{score.source} -> {score.target}"
                    coupling_info[key] = {
                        "score": round(score.score, 3),
                        "strength": score.strength,
                        "topics": score.topics[:5],  # Limit topics shown
                    }
                context["coupling_matrix"] = coupling_info

        # Parameters summary
        if self.parameters:
            params_summary = {}
            for file_info in self.parameters.files:
                for node_params in file_info.nodes:
                    if node_params.parameters:
                        params_summary[node_params.node_name] = {
                            p.name: p.value for p in node_params.parameters[:10]
                        }
            if params_summary:
                context["parameters"] = params_summary

        # Launch topology summary
        if self.launch_topology:
            launch_info = {
                "files": len(self.launch_topology.launch_files),
                "total_nodes": sum(lf.total_nodes for lf in self.launch_topology.launch_files),
            }

            # Launch arguments
            all_args = []
            for lf in self.launch_topology.launch_files:
                all_args.extend(lf.arguments)

            if all_args:
                launch_info["arguments"] = {
                    arg.name: arg.default_value
                    for arg in all_args[:15]  # Limit
                }

            context["launch"] = launch_info

        # HTTP Communication
        if self.http_comm_map:
            http_info = {}

            # HTTP endpoints (servers)
            if self.http_comm_map.http_endpoints:
                endpoints = {}
                for ep in self.http_comm_map.http_endpoints[:30]:  # Limit
                    key = f"{ep.method} {ep.path}"
                    endpoints[key] = {
                        "framework": ep.framework,
                        "file": str(Path(ep.file_path).name) if ep.file_path else None,
                    }
                    if ep.handler_name:
                        endpoints[key]["handler"] = ep.handler_name
                http_info["endpoints"] = endpoints

            # HTTP clients (outbound calls)
            if self.http_comm_map.http_clients:
                clients = {}
                for client in self.http_comm_map.http_clients[:30]:  # Limit
                    url = client.target_url or client.target_variable or "dynamic"
                    key = f"{client.method} {url}"
                    if key not in clients:
                        clients[key] = {
                            "library": client.library,
                            "file": str(Path(client.file_path).name) if client.file_path else None,
                        }
                        if client.context:
                            clients[key]["context"] = client.context
                http_info["clients"] = clients

            # Communication summary
            summary = self.http_comm_map.summary()
            http_info["summary"] = {
                "endpoints": summary["http_endpoints"],
                "clients": summary["http_clients"],
                "target_hosts": summary["http_target_hosts"][:10],  # Limit
                "cross_system_protocol": summary["cross_system_protocol"],
            }

            context["http_communication"] = http_info

        return context

    def export_system_context(self, output_path: Path) -> ExportResult:
        """Export system_context.yaml."""
        try:
            data = self.build_system_context()

            # Add header comment
            header = "# System Context - Auto-generated by RoboMind\n"
            header += "# Full architecture context for AI-assisted development\n\n"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(header)
                yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                         allow_unicode=True, width=120)

            content = output_path.read_text()
            token_estimate = self._estimate_tokens(content)

            logger.info(f"Exported system_context to {output_path} (~{token_estimate} tokens)")

            return ExportResult(
                success=True,
                output_path=output_path,
                token_estimate=token_estimate,
            )

        except Exception as e:
            logger.error(f"Failed to export system_context: {e}")
            return ExportResult(success=False, error=str(e))

    # ========== Combined Export ==========

    def export_all(self, output_dir: Path) -> Dict[str, ExportResult]:
        """
        Export all YAML files to a directory.

        Creates:
        - CONTEXT_SUMMARY.yaml
        - system_context.yaml
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Context summary
        results["context_summary"] = self.export_context_summary(
            output_dir / "CONTEXT_SUMMARY.yaml"
        )

        # Full context
        results["system_context"] = self.export_system_context(
            output_dir / "system_context.yaml"
        )

        return results


def export_yaml_context(
    output_dir: Path,
    nodes: List[ROS2NodeInfo],
    system_graph: Optional[SystemGraph] = None,
    coupling: Optional[CouplingMatrix] = None,
    topic_graph: Optional[TopicGraphResult] = None,
    launch_topology: Optional[LaunchTopology] = None,
    parameters: Optional[ParameterCollection] = None,
    http_comm_map=None,
    project_name: str = "Unknown",
    project_version: str = "auto",
    hardware_mapping: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, ExportResult]:
    """
    Convenience function to export YAML context files.

    Args:
        output_dir: Directory for output files
        nodes: List of ROS2NodeInfo
        system_graph: Optional SystemGraph
        coupling: Optional CouplingMatrix
        topic_graph: Optional TopicGraphResult
        launch_topology: Optional LaunchTopology
        parameters: Optional ParameterCollection
        http_comm_map: Optional HTTP CommunicationMap
        project_name: Name of the project
        project_version: Version string
        hardware_mapping: Dict mapping hardware targets to node names

    Returns:
        Dict of export results keyed by file type
    """
    exporter = YAMLExporter()
    exporter.set_project_info(name=project_name, version=project_version)
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
    if hardware_mapping:
        exporter.set_hardware_mapping(hardware_mapping)

    return exporter.export_all(output_dir)


# CLI for testing
if __name__ == "__main__":
    import sys
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.core.graph import build_system_graph
    from robomind.analyzers.coupling import analyze_coupling

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python yaml_exporter.py <project_path> [output_dir]")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("robomind_context")

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
    results = export_yaml_context(
        output_dir=output_dir,
        nodes=all_nodes,
        system_graph=system_graph,
        coupling=coupling,
        topic_graph=topic_graph,
        project_name=project_path.name,
    )

    print("\nExport Results:")
    for name, result in results.items():
        if result.success:
            print(f"  {name}: {result.output_path} (~{result.token_estimate} tokens)")
        else:
            print(f"  {name}: FAILED - {result.error}")
