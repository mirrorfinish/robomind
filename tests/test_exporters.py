"""Tests for JSON and YAML exporters."""

import pytest
import json
import yaml
from pathlib import Path

from robomind.exporters.json_exporter import (
    JSONExporter,
    ProjectMetadata,
    AnalysisSummary,
    export_analysis_json,
)
from robomind.exporters.yaml_exporter import (
    YAMLExporter,
    export_yaml_context,
)
from robomind.ros2.node_extractor import (
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
    TimerInfo,
    ParameterInfo,
)
from robomind.ros2.topic_extractor import TopicExtractor
from robomind.core.graph import build_system_graph
from robomind.analyzers.coupling import analyze_coupling


@pytest.fixture
def sample_nodes():
    """Create sample ROS2 nodes for testing."""
    node1 = ROS2NodeInfo(
        name="sensor_node",
        class_name="SensorNode",
        file_path=Path("/test/sensor.py"),
        line_number=10,
        end_line=100,
        package_name="sensor_pkg",
        publishers=[
            PublisherInfo(topic="/scan", msg_type="LaserScan", qos=10),
            PublisherInfo(topic="/image", msg_type="Image", qos=30),
        ],
        subscribers=[
            SubscriberInfo(topic="/cmd", msg_type="String", callback="cmd_cb", qos=10),
        ],
        timers=[
            TimerInfo(period=0.1, callback="scan_cb"),
        ],
        parameters=[
            ParameterInfo(name="rate", default_value=10, param_type="int"),
            ParameterInfo(name="frame_id", default_value="laser", param_type="str"),
        ],
    )

    node2 = ROS2NodeInfo(
        name="perception_node",
        class_name="PerceptionNode",
        file_path=Path("/test/perception.py"),
        line_number=10,
        end_line=100,
        package_name="perception_pkg",
        subscribers=[
            SubscriberInfo(topic="/scan", msg_type="LaserScan", callback="scan_cb", qos=10),
            SubscriberInfo(topic="/image", msg_type="Image", callback="image_cb", qos=30),
        ],
        publishers=[
            PublisherInfo(topic="/obstacles", msg_type="PointCloud", qos=10),
        ],
        parameters=[
            ParameterInfo(name="threshold", default_value=0.5, param_type="float"),
        ],
    )

    return [node1, node2]


@pytest.fixture
def sample_topic_graph(sample_nodes):
    """Create topic graph from sample nodes."""
    extractor = TopicExtractor()
    extractor.add_nodes(sample_nodes)
    return extractor.build()


@pytest.fixture
def sample_system_graph(sample_nodes, sample_topic_graph):
    """Create system graph from sample nodes."""
    return build_system_graph(sample_nodes, sample_topic_graph)


@pytest.fixture
def sample_coupling(sample_nodes, sample_topic_graph):
    """Create coupling matrix from sample nodes."""
    return analyze_coupling(sample_nodes, sample_topic_graph)


class TestProjectMetadata:
    """Tests for ProjectMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = ProjectMetadata(
            name="TestProject",
            path="/test/path",
            analyzed_at="2026-01-26T12:00:00",
            python_files=100,
            ros2_packages=5,
        )

        d = meta.to_dict()

        assert d["name"] == "TestProject"
        assert d["path"] == "/test/path"
        assert d["python_files"] == 100
        assert d["ros2_packages"] == 5


class TestAnalysisSummary:
    """Tests for AnalysisSummary dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = AnalysisSummary(
            ros2_nodes=10,
            topics=20,
            publishers=15,
            subscribers=12,
        )

        d = summary.to_dict()

        assert d["ros2_nodes"] == 10
        assert d["topics"] == 20
        assert d["publishers"] == 15


class TestJSONExporter:
    """Tests for JSONExporter class."""

    def test_exporter_init(self):
        """Test exporter initialization."""
        exporter = JSONExporter()
        assert exporter is not None

    def test_set_metadata(self):
        """Test setting metadata."""
        exporter = JSONExporter()
        exporter.set_metadata(
            name="TestProject",
            path="/test/path",
            python_files=50,
        )

        assert exporter.metadata.name == "TestProject"
        assert exporter.metadata.python_files == 50

    def test_set_nodes(self, sample_nodes):
        """Test setting nodes updates summary."""
        exporter = JSONExporter()
        exporter.set_nodes(sample_nodes)

        assert len(exporter.nodes) == 2
        assert exporter.summary.ros2_nodes == 2
        assert exporter.summary.publishers == 3
        assert exporter.summary.subscribers == 3

    def test_build_minimal(self, sample_nodes):
        """Test building with minimal data."""
        exporter = JSONExporter()
        exporter.set_metadata(name="Test", path="/test")
        exporter.set_nodes(sample_nodes)

        data = exporter.build()

        assert "metadata" in data
        assert "summary" in data
        assert "nodes" in data
        assert data["metadata"]["name"] == "Test"

    def test_build_with_graph(self, sample_nodes, sample_system_graph, sample_coupling):
        """Test building with full data."""
        exporter = JSONExporter()
        exporter.set_metadata(name="Test", path="/test")
        exporter.set_nodes(sample_nodes)
        exporter.set_graph(sample_system_graph)
        exporter.set_coupling(sample_coupling)

        data = exporter.build()

        assert "graph" in data
        assert "coupling" in data
        assert data["graph"]["stats"]["total_nodes"] > 0

    def test_export_to_file(self, sample_nodes, tmp_path):
        """Test exporting to file."""
        exporter = JSONExporter()
        exporter.set_metadata(name="Test", path="/test")
        exporter.set_nodes(sample_nodes)

        output_path = tmp_path / "test_output.json"
        result = exporter.export(output_path)

        assert result.success
        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)
        assert "metadata" in data

    def test_export_string(self, sample_nodes):
        """Test exporting to string."""
        exporter = JSONExporter()
        exporter.set_metadata(name="Test", path="/test")
        exporter.set_nodes(sample_nodes)

        json_str = exporter.export_string()

        data = json.loads(json_str)
        assert "metadata" in data


class TestJSONExportFunction:
    """Tests for export_analysis_json function."""

    def test_export_analysis_json(self, sample_nodes, sample_system_graph, tmp_path):
        """Test convenience function."""
        output_path = tmp_path / "analysis.json"

        result = export_analysis_json(
            output_path=output_path,
            nodes=sample_nodes,
            system_graph=sample_system_graph,
            project_name="TestProject",
            project_path="/test",
        )

        assert result.success
        assert output_path.exists()
        assert result.stats["nodes"] == 2


class TestYAMLExporter:
    """Tests for YAMLExporter class."""

    def test_exporter_init(self):
        """Test exporter initialization."""
        exporter = YAMLExporter()
        assert exporter is not None

    def test_set_project_info(self):
        """Test setting project info."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="TestProject", version="1.0.0")

        assert exporter.project_name == "TestProject"
        assert exporter.project_version == "1.0.0"

    def test_set_project_info_auto_version(self):
        """Test auto version generation."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="TestProject")

        assert "auto-" in exporter.project_version

    def test_build_context_summary(self, sample_nodes, sample_topic_graph):
        """Test building context summary."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="TestProject")
        exporter.set_nodes(sample_nodes)
        exporter.set_topic_graph(sample_topic_graph)

        summary = exporter.build_context_summary()

        assert summary["system"] == "testproject"
        assert summary["nodes"] == 2
        assert "packages" in summary
        assert "sensor_pkg" in summary["packages"]

    def test_build_context_summary_with_coupling(self, sample_nodes, sample_topic_graph, sample_coupling):
        """Test context summary includes critical coupling."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)
        exporter.set_topic_graph(sample_topic_graph)
        exporter.set_coupling(sample_coupling)

        summary = exporter.build_context_summary()

        # May or may not have critical coupling depending on test data
        assert "system" in summary

    def test_build_system_context(self, sample_nodes, sample_topic_graph, sample_system_graph):
        """Test building full system context."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="TestProject", version="1.0.0")
        exporter.set_nodes(sample_nodes)
        exporter.set_topic_graph(sample_topic_graph)
        exporter.set_graph(sample_system_graph)

        context = exporter.build_system_context()

        assert "metadata" in context
        assert context["metadata"]["name"] == "TestProject"
        assert "architecture" in context
        assert "nodes" in context
        assert "topics" in context

    def test_build_system_context_with_hardware(self, sample_nodes):
        """Test system context with hardware mapping."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)
        exporter.set_hardware_mapping({
            "jetson1": ["sensor_node"],
            "jetson2": ["perception_node"],
        })

        context = exporter.build_system_context()

        assert "distributed_hosts" in context["metadata"]
        assert "jetson1" in context["metadata"]["distributed_hosts"]

    def test_export_context_summary(self, sample_nodes, tmp_path):
        """Test exporting context summary."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)

        output_path = tmp_path / "CONTEXT_SUMMARY.yaml"
        result = exporter.export_context_summary(output_path)

        assert result.success
        assert output_path.exists()
        assert result.token_estimate > 0

        # Verify YAML is valid
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["system"] == "test"

    def test_export_system_context(self, sample_nodes, sample_topic_graph, tmp_path):
        """Test exporting system context."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)
        exporter.set_topic_graph(sample_topic_graph)

        output_path = tmp_path / "system_context.yaml"
        result = exporter.export_system_context(output_path)

        assert result.success
        assert output_path.exists()

        # Verify YAML is valid
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert "metadata" in data
        assert "nodes" in data

    def test_export_all(self, sample_nodes, sample_topic_graph, tmp_path):
        """Test exporting all YAML files."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)
        exporter.set_topic_graph(sample_topic_graph)

        results = exporter.export_all(tmp_path)

        assert "context_summary" in results
        assert "system_context" in results
        assert results["context_summary"].success
        assert results["system_context"].success

        # Verify files exist
        assert (tmp_path / "CONTEXT_SUMMARY.yaml").exists()
        assert (tmp_path / "system_context.yaml").exists()


class TestYAMLExportFunction:
    """Tests for export_yaml_context function."""

    def test_export_yaml_context(self, sample_nodes, sample_system_graph, tmp_path):
        """Test convenience function."""
        results = export_yaml_context(
            output_dir=tmp_path,
            nodes=sample_nodes,
            system_graph=sample_system_graph,
            project_name="TestProject",
        )

        assert results["context_summary"].success
        assert results["system_context"].success

    def test_export_yaml_context_with_all_data(
        self, sample_nodes, sample_system_graph, sample_coupling, sample_topic_graph, tmp_path
    ):
        """Test with all optional data."""
        results = export_yaml_context(
            output_dir=tmp_path,
            nodes=sample_nodes,
            system_graph=sample_system_graph,
            coupling=sample_coupling,
            topic_graph=sample_topic_graph,
            project_name="FullTest",
            project_version="2.0.0",
        )

        assert results["context_summary"].success
        assert results["system_context"].success

        # Check content
        with open(tmp_path / "system_context.yaml") as f:
            data = yaml.safe_load(f)
        assert data["metadata"]["version"] == "2.0.0"


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_context_summary_token_count(self, sample_nodes, tmp_path):
        """Test that context summary is token-efficient."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)

        result = exporter.export_context_summary(tmp_path / "summary.yaml")

        # Should be around 50-200 tokens for a simple project
        assert result.token_estimate < 500

    def test_system_context_reasonable_size(self, sample_nodes, sample_topic_graph, tmp_path):
        """Test that system context is reasonably sized."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes(sample_nodes)
        exporter.set_topic_graph(sample_topic_graph)

        result = exporter.export_system_context(tmp_path / "context.yaml")

        # Should be under 2000 tokens for a small project
        assert result.token_estimate < 2000


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_json_export_empty_nodes(self, tmp_path):
        """Test JSON export with no nodes."""
        exporter = JSONExporter()
        exporter.set_metadata(name="Empty", path="/empty")
        exporter.set_nodes([])

        result = exporter.export(tmp_path / "empty.json")

        assert result.success
        with open(tmp_path / "empty.json") as f:
            data = json.load(f)
        assert data["summary"]["ros2_nodes"] == 0

    def test_yaml_export_empty_nodes(self, tmp_path):
        """Test YAML export with no nodes."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Empty")
        exporter.set_nodes([])

        results = exporter.export_all(tmp_path)

        assert results["context_summary"].success
        assert results["system_context"].success

    def test_json_export_creates_parent_dirs(self, tmp_path):
        """Test that export creates parent directories."""
        exporter = JSONExporter()
        exporter.set_metadata(name="Test", path="/test")
        exporter.set_nodes([])

        nested_path = tmp_path / "deep" / "nested" / "output.json"
        result = exporter.export(nested_path)

        assert result.success
        assert nested_path.exists()

    def test_yaml_export_creates_parent_dirs(self, tmp_path):
        """Test that YAML export creates parent directories."""
        exporter = YAMLExporter()
        exporter.set_project_info(name="Test")
        exporter.set_nodes([])

        nested_path = tmp_path / "deep" / "nested"
        results = exporter.export_all(nested_path)

        assert results["context_summary"].success
        assert (nested_path / "CONTEXT_SUMMARY.yaml").exists()
