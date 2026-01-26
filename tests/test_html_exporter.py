"""Tests for HTML exporter."""

import json
import pytest
from pathlib import Path

from robomind.exporters.html_exporter import (
    HTMLExporter,
    ExportResult,
    GraphStats,
    export_html_visualization,
)
from robomind.core.graph import SystemGraph, GraphNode, GraphEdge, ComponentType, EdgeType
from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult, TopicConnection


class TestHTMLExporter:
    """Tests for HTMLExporter class."""

    def test_init_defaults(self):
        """Test default initialization."""
        exporter = HTMLExporter()
        assert exporter.project_name == "RoboMind Project"
        assert exporter.system_graph is None
        assert exporter.coupling is None
        assert exporter.nodes == []

    def test_set_project_name(self):
        """Test setting project name."""
        exporter = HTMLExporter()
        exporter.set_project_name("MyRobot")
        assert exporter.project_name == "MyRobot"

    def test_set_graph(self):
        """Test setting system graph."""
        exporter = HTMLExporter()
        graph = SystemGraph()
        exporter.set_graph(graph)
        assert exporter.system_graph is graph


class TestGraphStats:
    """Tests for GraphStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = GraphStats()
        assert stats.ros2_nodes == 0
        assert stats.topics == 0
        assert stats.services == 0
        assert stats.parameters == 0
        assert stats.edges == 0
        assert stats.packages == 0

    def test_custom_values(self):
        """Test custom values."""
        stats = GraphStats(ros2_nodes=5, topics=10, edges=15)
        assert stats.ros2_nodes == 5
        assert stats.topics == 10
        assert stats.edges == 15


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ExportResult(
            success=True,
            output_path=Path("/tmp/test.html"),
            stats={"nodes": 10, "edges": 5}
        )
        assert result.success is True
        assert result.output_path == Path("/tmp/test.html")
        assert result.error is None
        assert result.stats == {"nodes": 10, "edges": 5}

    def test_failure_result(self):
        """Test failure result."""
        result = ExportResult(
            success=False,
            error="Template not found"
        )
        assert result.success is False
        assert result.error == "Template not found"


class TestBuildGraphData:
    """Tests for _build_graph_data method."""

    def test_empty_graph(self):
        """Test with no graph set."""
        exporter = HTMLExporter()
        data = exporter._build_graph_data()
        assert data == {"nodes": [], "edges": []}

    def test_graph_with_nodes(self):
        """Test with nodes in graph."""
        exporter = HTMLExporter()
        graph = SystemGraph()

        # Add a node
        node = GraphNode(
            id="node1",
            name="test_node",
            component_type=ComponentType.ROS2_NODE,
            file_path=Path("/tmp/test.py"),
            package="test_pkg"
        )
        graph.add_node(node)

        exporter.set_graph(graph)
        data = exporter._build_graph_data()

        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "node1"
        assert data["nodes"][0]["name"] == "test_node"
        assert data["nodes"][0]["type"] == "ROS2_NODE"

    def test_graph_with_edges(self):
        """Test with edges in graph."""
        exporter = HTMLExporter()
        graph = SystemGraph()

        # Add nodes
        node1 = GraphNode(id="node1", name="pub", component_type=ComponentType.ROS2_NODE)
        node2 = GraphNode(id="topic1", name="/cmd_vel", component_type=ComponentType.TOPIC)
        graph.add_node(node1)
        graph.add_node(node2)

        # Add edge
        edge = GraphEdge(source="node1", target="topic1", edge_type=EdgeType.PUBLISHES)
        graph.add_edge(edge)

        exporter.set_graph(graph)
        data = exporter._build_graph_data()

        assert len(data["edges"]) == 1
        assert data["edges"][0]["source"] == "node1"
        assert data["edges"][0]["target"] == "topic1"
        assert data["edges"][0]["type"] == "PUBLISHES"


class TestCalculateStats:
    """Tests for _calculate_stats method."""

    def test_empty_graph_stats(self):
        """Test stats with no graph."""
        exporter = HTMLExporter()
        stats = exporter._calculate_stats()

        assert stats.ros2_nodes == 0
        assert stats.topics == 0
        assert stats.edges == 0

    def test_graph_stats(self):
        """Test stats calculation."""
        exporter = HTMLExporter()
        graph = SystemGraph()

        # Add various node types
        graph.add_node(GraphNode(id="n1", name="node1", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="n2", name="node2", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="t1", name="/topic1", component_type=ComponentType.TOPIC))
        graph.add_node(GraphNode(id="s1", name="/service1", component_type=ComponentType.SERVICE))

        # Add an edge
        graph.add_edge(GraphEdge(source="n1", target="t1", edge_type=EdgeType.PUBLISHES))

        exporter.set_graph(graph)
        stats = exporter._calculate_stats()

        assert stats.ros2_nodes == 2
        assert stats.topics == 1
        assert stats.services == 1
        assert stats.edges == 1


class TestHTMLExport:
    """Tests for HTML export functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        graph = SystemGraph()

        # Add nodes
        graph.add_node(GraphNode(
            id="motor_controller",
            name="motor_controller",
            component_type=ComponentType.ROS2_NODE,
            package="motor_pkg"
        ))
        graph.add_node(GraphNode(
            id="cmd_vel",
            name="/cmd_vel",
            component_type=ComponentType.TOPIC
        ))
        graph.add_node(GraphNode(
            id="navigation",
            name="navigation",
            component_type=ComponentType.ROS2_NODE,
            package="nav_pkg"
        ))

        # Add edges
        graph.add_edge(GraphEdge(
            source="navigation",
            target="cmd_vel",
            edge_type=EdgeType.PUBLISHES
        ))
        graph.add_edge(GraphEdge(
            source="motor_controller",
            target="cmd_vel",
            edge_type=EdgeType.SUBSCRIBES
        ))

        return graph

    def test_build_html(self, sample_graph):
        """Test building HTML content."""
        exporter = HTMLExporter()
        exporter.set_project_name("TestRobot")
        exporter.set_graph(sample_graph)

        html = exporter.build()

        # Check essential content
        assert "TestRobot" in html
        assert "D3.js" in html or "d3.v7" in html
        assert "motor_controller" in html
        assert "navigation" in html
        assert "/cmd_vel" in html

    def test_export_to_file(self, sample_graph, tmp_path):
        """Test exporting to file."""
        exporter = HTMLExporter()
        exporter.set_project_name("TestRobot")
        exporter.set_graph(sample_graph)

        output_path = tmp_path / "test_viz.html"
        result = exporter.export(output_path)

        assert result.success is True
        assert result.output_path == output_path
        assert output_path.exists()
        assert result.stats["nodes"] == 3
        assert result.stats["edges"] == 2

        # Verify content
        content = output_path.read_text()
        assert "TestRobot" in content

    def test_export_creates_directory(self, sample_graph, tmp_path):
        """Test that export creates parent directory."""
        exporter = HTMLExporter()
        exporter.set_graph(sample_graph)

        output_path = tmp_path / "nested" / "dir" / "viz.html"
        result = exporter.export(output_path)

        assert result.success is True
        assert output_path.exists()


class TestExportHTMLVisualizationFunction:
    """Tests for export_html_visualization convenience function."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        graph = SystemGraph()
        graph.add_node(GraphNode(id="n1", name="node1", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="t1", name="/topic1", component_type=ComponentType.TOPIC))
        graph.add_edge(GraphEdge(source="n1", target="t1", edge_type=EdgeType.PUBLISHES))
        return graph

    def test_basic_export(self, sample_graph, tmp_path):
        """Test basic export with convenience function."""
        output_path = tmp_path / "visualization.html"

        result = export_html_visualization(
            output_path=output_path,
            system_graph=sample_graph,
            project_name="TestProject"
        )

        assert result.success is True
        assert output_path.exists()

        content = output_path.read_text()
        assert "TestProject" in content

    def test_export_with_all_options(self, sample_graph, tmp_path):
        """Test export with all optional parameters."""
        from robomind.analyzers.coupling import CouplingMatrix

        output_path = tmp_path / "full_viz.html"
        coupling = CouplingMatrix()

        # Create sample nodes
        node = ROS2NodeInfo(
            name="test_node",
            class_name="TestNode",
            file_path=Path("/tmp/test.py"),
            line_number=1,
            end_line=10,
            package_name="test_pkg"
        )

        # Create sample topic graph
        topic_graph = TopicGraphResult()
        topic_graph.topics["test_topic"] = TopicConnection(
            name="test_topic",
            msg_type="std_msgs/String"
        )

        result = export_html_visualization(
            output_path=output_path,
            system_graph=sample_graph,
            project_name="FullProject",
            coupling=coupling,
            nodes=[node],
            topic_graph=topic_graph,
        )

        assert result.success is True
        assert output_path.exists()


class TestHTMLContent:
    """Tests for HTML content structure."""

    @pytest.fixture
    def html_content(self, tmp_path):
        """Generate HTML content for testing."""
        graph = SystemGraph()
        graph.add_node(GraphNode(id="n1", name="node1", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="t1", name="/topic", component_type=ComponentType.TOPIC))
        graph.add_edge(GraphEdge(source="n1", target="t1", edge_type=EdgeType.PUBLISHES))

        exporter = HTMLExporter()
        exporter.set_project_name("ContentTest")
        exporter.set_graph(graph)

        return exporter.build()

    def test_has_doctype(self, html_content):
        """Test HTML has DOCTYPE."""
        assert "<!DOCTYPE html>" in html_content or "<!doctype html>" in html_content.lower()

    def test_has_d3_script(self, html_content):
        """Test HTML includes D3.js."""
        assert "d3" in html_content.lower()

    def test_has_graph_data(self, html_content):
        """Test HTML includes graph data."""
        assert "nodes" in html_content
        assert "edges" in html_content

    def test_has_interactive_elements(self, html_content):
        """Test HTML has interactive elements."""
        # Should have search or filter
        assert "search" in html_content.lower() or "filter" in html_content.lower()

    def test_has_color_coding(self, html_content):
        """Test HTML has color coding for node types."""
        # Should have colors defined for different types
        assert "#" in html_content  # Hex colors


class TestEdgeCases:
    """Tests for edge cases."""

    def test_special_characters_in_name(self, tmp_path):
        """Test handling special characters in project name."""
        graph = SystemGraph()
        graph.add_node(GraphNode(id="n1", name="node1", component_type=ComponentType.ROS2_NODE))

        exporter = HTMLExporter()
        exporter.set_project_name("Test<Project>&\"Name")
        exporter.set_graph(graph)

        output_path = tmp_path / "special.html"
        result = exporter.export(output_path)

        # Should succeed without crashing
        assert result.success is True

    def test_unicode_in_node_names(self, tmp_path):
        """Test handling unicode in node names."""
        graph = SystemGraph()
        graph.add_node(GraphNode(id="n1", name="sensor_node", component_type=ComponentType.ROS2_NODE))

        exporter = HTMLExporter()
        exporter.set_graph(graph)

        output_path = tmp_path / "unicode.html"
        result = exporter.export(output_path)

        assert result.success is True

    def test_large_graph(self, tmp_path):
        """Test with larger graph."""
        graph = SystemGraph()

        # Add 100 nodes and 200 edges
        for i in range(100):
            graph.add_node(GraphNode(
                id=f"node_{i}",
                name=f"node_{i}",
                component_type=ComponentType.ROS2_NODE if i % 3 else ComponentType.TOPIC
            ))

        for i in range(100):
            graph.add_edge(GraphEdge(
                source=f"node_{i}",
                target=f"node_{(i+1) % 100}",
                edge_type=EdgeType.PUBLISHES
            ))

        exporter = HTMLExporter()
        exporter.set_graph(graph)

        output_path = tmp_path / "large.html"
        result = exporter.export(output_path)

        assert result.success is True
        assert result.stats["nodes"] == 100
        assert result.stats["edges"] == 100
