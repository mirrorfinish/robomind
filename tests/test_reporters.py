"""Tests for the Reporters module."""

import pytest
from pathlib import Path
import tempfile

from robomind.reporters.markdown_reporter import (
    MarkdownReporter,
    ReportResult,
    generate_report,
)
from robomind.ros2.node_extractor import (
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
    TimerInfo,
)
from robomind.ros2.topic_extractor import TopicExtractor
from robomind.analyzers.coupling import analyze_coupling


@pytest.fixture
def sample_nodes():
    """Create sample ROS2 nodes for testing."""
    return [
        ROS2NodeInfo(
            name="sensor_node",
            class_name="SensorNode",
            file_path=Path("/test/sensor.py"),
            line_number=1,
            end_line=50,
            package_name="sensors",
            publishers=[
                PublisherInfo(topic="/sensor/data", msg_type="SensorData", qos=10),
                PublisherInfo(topic="raw_data", msg_type="RawData", qos=10),  # No leading slash
            ],
        ),
        ROS2NodeInfo(
            name="controller_node",
            class_name="ControllerNode",
            file_path=Path("/test/controller.py"),
            line_number=1,
            end_line=50,
            package_name="control",
            subscribers=[
                SubscriberInfo(topic="/sensor/data", msg_type="SensorData", callback="cb", qos=10),
            ],
            publishers=[
                PublisherInfo(topic="/betaray/cmd_vel", msg_type="Twist", qos=10),
            ],
        ),
        ROS2NodeInfo(
            name="voice_node",
            class_name="VoiceNode",
            file_path=Path("/test/voice.py"),
            line_number=1,
            end_line=50,
            package_name="voice",
            subscribers=[
                SubscriberInfo(topic="/nexus/ai/response", msg_type="String", callback="cb", qos=10),
            ],
        ),
    ]


@pytest.fixture
def topic_graph(sample_nodes):
    """Create topic graph from sample nodes."""
    extractor = TopicExtractor()
    extractor.add_nodes(sample_nodes)
    return extractor.build()


@pytest.fixture
def coupling_matrix(sample_nodes, topic_graph):
    """Create coupling matrix."""
    return analyze_coupling(sample_nodes, topic_graph)


class TestReportResult:
    """Tests for ReportResult dataclass."""

    def test_report_result_creation(self):
        """Test creating a report result."""
        result = ReportResult(
            success=True,
            content="# Report",
            output_path=Path("/test/report.md"),
            stats={"sections": 5},
        )

        assert result.success is True
        assert result.content == "# Report"

    def test_report_result_to_dict(self):
        """Test converting to dictionary."""
        result = ReportResult(
            success=True,
            output_path=Path("/test/report.md"),
            stats={"sections": 5},
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["output_path"] == "/test/report.md"


class TestMarkdownReporter:
    """Tests for MarkdownReporter class."""

    def test_reporter_init(self, sample_nodes, topic_graph):
        """Test reporter initialization."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
            project_name="TestProject",
        )

        assert len(reporter.nodes) == 3
        assert reporter.project_name == "TestProject"

    def test_analyze_topics(self, sample_nodes, topic_graph):
        """Test topic analysis."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        # Should detect orphans and missing slashes
        assert len(reporter._orphaned_pubs) > 0 or len(reporter._orphaned_subs) > 0
        assert len(reporter._missing_leading_slash) > 0  # raw_data has no leading slash

    def test_namespace_detection(self, sample_nodes, topic_graph):
        """Test namespace inconsistency detection."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        # Should detect betaray and nexus namespaces
        assert len(reporter._namespace_issues) >= 2
        assert "betaray" in reporter._namespace_issues
        assert "nexus" in reporter._namespace_issues

    def test_generate_report_content(self, sample_nodes, topic_graph, coupling_matrix):
        """Test report generation without file output."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
            coupling=coupling_matrix,
            project_name="TestProject",
        )

        result = reporter.generate()

        assert result.success is True
        assert result.content != ""
        assert "# TestProject Architecture Report" in result.content
        assert "Executive Summary" in result.content
        assert "Critical Issues" in result.content

    def test_generate_report_to_file(self, sample_nodes, topic_graph):
        """Test report generation with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            reporter = MarkdownReporter(
                nodes=sample_nodes,
                topic_graph=topic_graph,
                project_name="TestProject",
            )

            result = reporter.generate(output_path)

            assert result.success is True
            assert result.output_path == output_path
            assert output_path.exists()

            content = output_path.read_text()
            assert "TestProject" in content

    def test_generate_header(self, sample_nodes, topic_graph):
        """Test header generation."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
            project_name="MyRobot",
            project_path="/home/user/myrobot",
        )

        header = reporter._generate_header()
        assert "# MyRobot Architecture Report" in header
        assert "/home/user/myrobot" in header

    def test_generate_executive_summary(self, sample_nodes, topic_graph):
        """Test executive summary generation."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        summary = reporter._generate_executive_summary()
        assert "Executive Summary" in summary
        assert "ROS2 Nodes" in summary
        assert "Topics" in summary

    def test_generate_critical_issues(self, sample_nodes, topic_graph):
        """Test critical issues section."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        issues = reporter._generate_critical_issues()
        assert "Critical Issues" in issues
        # Should report namespace inconsistencies
        assert "Namespace" in issues or "namespace" in issues.lower()

    def test_generate_topic_analysis(self, sample_nodes, topic_graph):
        """Test topic analysis section."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        analysis = reporter._generate_topic_analysis()
        assert "Topic Analysis" in analysis

    def test_generate_coupling_analysis(self, sample_nodes, topic_graph, coupling_matrix):
        """Test coupling analysis section."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
            coupling=coupling_matrix,
        )

        coupling_section = reporter._generate_coupling_analysis()
        assert "Coupling Analysis" in coupling_section

    def test_generate_node_summary(self, sample_nodes, topic_graph):
        """Test node summary section."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        summary = reporter._generate_node_summary()
        assert "Node Summary" in summary
        assert "sensor_node" in summary
        assert "controller_node" in summary

    def test_generate_recommendations(self, sample_nodes, topic_graph):
        """Test recommendations section."""
        reporter = MarkdownReporter(
            nodes=sample_nodes,
            topic_graph=topic_graph,
        )

        recs = reporter._generate_recommendations()
        assert "Recommendations" in recs


class TestConvenienceFunction:
    """Test convenience function."""

    def test_generate_report_function(self, sample_nodes, topic_graph):
        """Test the generate_report convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"

            result = generate_report(
                output_path=output_path,
                nodes=sample_nodes,
                topic_graph=topic_graph,
                project_name="TestProject",
            )

            assert result.success is True
            assert output_path.exists()
