"""
RoboMind HTML Exporter - Generate interactive D3.js visualizations.

Creates standalone HTML files with:
- D3.js force-directed graph layout
- Color-coded nodes by component type
- Interactive zoom/pan
- Click-to-select for node details
- Search and filter functionality
- No server required - opens directly in browser
"""

import json
import logging
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from robomind.core.graph import SystemGraph, ComponentType
from robomind.analyzers.coupling import CouplingMatrix
from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)

# Template path
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
TEMPLATE_FILE = TEMPLATE_DIR / "visualization.html"


@dataclass
class ExportResult:
    """Result of an HTML export operation."""
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphStats:
    """Statistics for the visualization."""
    ros2_nodes: int = 0
    topics: int = 0
    services: int = 0
    parameters: int = 0
    edges: int = 0
    packages: int = 0


class HTMLExporter:
    """
    Export RoboMind analysis to interactive HTML visualization.

    Creates a standalone HTML file using D3.js that can be opened
    directly in a browser without any server.

    Features:
    - Force-directed graph layout
    - Color-coded nodes by type (ROS2_NODE, TOPIC, SERVICE, etc.)
    - Interactive zoom and pan
    - Node selection with details panel
    - Search filtering
    - Type filtering

    Usage:
        exporter = HTMLExporter()
        exporter.set_project_name("BetaRay")
        exporter.set_graph(system_graph)
        result = exporter.export(Path("visualization.html"))
    """

    def __init__(self):
        self.project_name: str = "RoboMind Project"
        self.system_graph: Optional[SystemGraph] = None
        self.coupling: Optional[CouplingMatrix] = None
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None

    def set_project_name(self, name: str):
        """Set project name for display."""
        self.project_name = name

    def set_graph(self, graph: SystemGraph):
        """Set the system graph to visualize."""
        self.system_graph = graph

    def set_coupling(self, coupling: CouplingMatrix):
        """Set coupling analysis for edge weights."""
        self.coupling = coupling

    def set_nodes(self, nodes: List[ROS2NodeInfo]):
        """Set ROS2 nodes for additional metadata."""
        self.nodes = nodes

    def set_topic_graph(self, topic_graph: TopicGraphResult):
        """Set topic graph for topic information."""
        self.topic_graph = topic_graph

    def set_http_communication(self, http_comm_map):
        """Set HTTP communication map (currently stored for future use)."""
        # HTTP communication visualization can be extended in future
        self.http_comm_map = http_comm_map

    def _build_graph_data(self) -> Dict[str, Any]:
        """Build graph data structure for D3.js."""
        if not self.system_graph:
            return {"nodes": [], "edges": []}

        # Build nodes list
        nodes = []
        for graph_node in self.system_graph.get_nodes():
            node_data = {
                "id": graph_node.id,
                "name": graph_node.name,
                "type": graph_node.component_type.name,
                "file_path": str(graph_node.file_path.name) if graph_node.file_path else None,
                "line_number": graph_node.line_number,
                "package": graph_node.package,
                "hardware_target": graph_node.hardware_target,
                "metadata": {k: v for k, v in graph_node.metadata.items() if v is not None},
            }
            nodes.append(node_data)

        # Build edges list
        edges = []
        for edge in self.system_graph.get_edges():
            edge_data = {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.name,
                "weight": edge.weight,
            }
            edges.append(edge_data)

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def _calculate_stats(self) -> GraphStats:
        """Calculate statistics for the sidebar."""
        stats = GraphStats()

        if self.system_graph:
            for node in self.system_graph.get_nodes():
                if node.component_type == ComponentType.ROS2_NODE:
                    stats.ros2_nodes += 1
                elif node.component_type == ComponentType.TOPIC:
                    stats.topics += 1
                elif node.component_type == ComponentType.SERVICE:
                    stats.services += 1
                elif node.component_type == ComponentType.PARAMETER:
                    stats.parameters += 1

            stats.edges = len(self.system_graph.get_edges())
            stats.packages = len(self.system_graph.get_packages())

        return stats

    def _render_template(self, graph_data: Dict, stats: GraphStats) -> str:
        """Render the HTML template with data."""
        # Read template
        if not TEMPLATE_FILE.exists():
            raise FileNotFoundError(f"Template not found: {TEMPLATE_FILE}")

        template_content = TEMPLATE_FILE.read_text()

        # Simple template substitution (avoiding Jinja2 dependency)
        html = template_content

        # Replace template variables
        html = html.replace("{{ project_name }}", self.project_name)
        html = html.replace("{{ graph_data | safe }}", json.dumps(graph_data))
        html = html.replace("{{ stats.ros2_nodes }}", str(stats.ros2_nodes))
        html = html.replace("{{ stats.topics }}", str(stats.topics))
        html = html.replace("{{ stats.edges }}", str(stats.edges))
        html = html.replace("{{ stats.packages }}", str(stats.packages))

        return html

    def build(self) -> str:
        """
        Build the complete HTML content.

        Returns:
            HTML string ready to be written to file
        """
        graph_data = self._build_graph_data()
        stats = self._calculate_stats()
        return self._render_template(graph_data, stats)

    def export(self, output_path: Path, open_browser: bool = False) -> ExportResult:
        """
        Export visualization to HTML file.

        Args:
            output_path: Path to output HTML file
            open_browser: Whether to open the file in default browser

        Returns:
            ExportResult with success status
        """
        try:
            html_content = self.build()

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            output_path.write_text(html_content)

            stats = {
                "file_size": output_path.stat().st_size,
                "nodes": len(self.system_graph.get_nodes()) if self.system_graph else 0,
                "edges": len(self.system_graph.get_edges()) if self.system_graph else 0,
            }

            logger.info(f"Exported HTML visualization to {output_path}")

            # Open in browser if requested
            if open_browser:
                webbrowser.open(f"file://{output_path.absolute()}")

            return ExportResult(
                success=True,
                output_path=output_path,
                stats=stats,
            )

        except Exception as e:
            logger.error(f"Failed to export HTML: {e}")
            return ExportResult(
                success=False,
                error=str(e),
            )


def export_html_visualization(
    output_path: Path,
    system_graph: SystemGraph,
    project_name: str = "RoboMind Project",
    coupling: Optional[CouplingMatrix] = None,
    nodes: Optional[List[ROS2NodeInfo]] = None,
    topic_graph: Optional[TopicGraphResult] = None,
    http_comm_map=None,
    open_browser: bool = False,
) -> ExportResult:
    """
    Convenience function to export HTML visualization.

    Args:
        output_path: Path to output HTML file
        system_graph: SystemGraph to visualize
        project_name: Name for display
        coupling: Optional coupling analysis
        nodes: Optional ROS2 nodes for metadata
        topic_graph: Optional topic graph
        http_comm_map: Optional HTTP communication map
        open_browser: Whether to open in browser

    Returns:
        ExportResult
    """
    exporter = HTMLExporter()
    exporter.set_project_name(project_name)
    exporter.set_graph(system_graph)

    if coupling:
        exporter.set_coupling(coupling)
    if nodes:
        exporter.set_nodes(nodes)
    if topic_graph:
        exporter.set_topic_graph(topic_graph)
    if http_comm_map:
        exporter.set_http_communication(http_comm_map)

    return exporter.export(output_path, open_browser=open_browser)


# CLI for testing
if __name__ == "__main__":
    import sys
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor
    from robomind.core.graph import build_system_graph

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python html_exporter.py <project_path> [output.html]")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization.html")

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

    # Export
    result = export_html_visualization(
        output_path=output_path,
        system_graph=system_graph,
        project_name=project_path.name,
        nodes=all_nodes,
        topic_graph=topic_graph,
        open_browser=True,
    )

    if result.success:
        print(f"Exported to {result.output_path}")
        print(f"Stats: {result.stats}")
    else:
        print(f"Export failed: {result.error}")
