"""Tests for the Impact Analyzer module."""

import pytest
from pathlib import Path

from robomind.analyzers.impact_analyzer import (
    ImpactAnalyzer,
    ImpactItem,
    ImpactResult,
    analyze_impact,
    CRITICAL_TOPICS,
    LOW_PRIORITY_PREFIXES,
)
from robomind.ros2.node_extractor import (
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
    ServiceInfo,
)
from robomind.ros2.topic_extractor import TopicExtractor


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
            package_name="test_pkg",
            publishers=[
                PublisherInfo(topic="/sensor/data", msg_type="sensor_msgs/msg/LaserScan", qos=10),
            ],
        ),
        ROS2NodeInfo(
            name="processor_node",
            class_name="ProcessorNode",
            file_path=Path("/test/processor.py"),
            line_number=1,
            end_line=50,
            package_name="test_pkg",
            subscribers=[
                SubscriberInfo(topic="/sensor/data", msg_type="sensor_msgs/msg/LaserScan", callback="cb", qos=10),
            ],
            publishers=[
                PublisherInfo(topic="/cmd_vel", msg_type="geometry_msgs/msg/Twist", qos=10),
            ],
        ),
        ROS2NodeInfo(
            name="motor_node",
            class_name="MotorNode",
            file_path=Path("/test/motor.py"),
            line_number=1,
            end_line=50,
            package_name="test_pkg",
            subscribers=[
                SubscriberInfo(topic="/cmd_vel", msg_type="geometry_msgs/msg/Twist", callback="cb", qos=10),
            ],
            publishers=[
                PublisherInfo(topic="/motor/status", msg_type="std_msgs/msg/String", qos=10),
            ],
        ),
        ROS2NodeInfo(
            name="logger_node",
            class_name="LoggerNode",
            file_path=Path("/test/logger.py"),
            line_number=1,
            end_line=50,
            package_name="test_pkg",
            subscribers=[
                SubscriberInfo(topic="/motor/status", msg_type="std_msgs/msg/String", callback="cb", qos=10),
                SubscriberInfo(topic="/sensor/data", msg_type="sensor_msgs/msg/LaserScan", callback="cb2", qos=10),
            ],
        ),
        ROS2NodeInfo(
            name="debug_node",
            class_name="DebugNode",
            file_path=Path("/test/debug.py"),
            line_number=1,
            end_line=50,
            package_name="test_pkg",
            publishers=[
                PublisherInfo(topic="/diagnostics", msg_type="diagnostic_msgs/msg/DiagnosticArray", qos=10),
            ],
        ),
    ]


@pytest.fixture
def topic_graph(sample_nodes):
    """Build topic graph from sample nodes."""
    extractor = TopicExtractor()
    extractor.add_nodes(sample_nodes)
    return extractor.build()


@pytest.fixture
def analyzer(sample_nodes, topic_graph):
    """Create an ImpactAnalyzer instance."""
    return ImpactAnalyzer(sample_nodes, topic_graph)


class TestImpactItem:
    """Tests for ImpactItem dataclass."""

    def test_to_dict(self):
        item = ImpactItem(
            name="test_node",
            kind="node",
            impact_type="broken_subscriber",
            severity="high",
            file_path="/test/file.py",
            description="Test description",
        )
        d = item.to_dict()
        assert d["name"] == "test_node"
        assert d["kind"] == "node"
        assert d["impact_type"] == "broken_subscriber"
        assert d["severity"] == "high"
        assert d["file_path"] == "/test/file.py"
        assert d["description"] == "Test description"


class TestImpactResult:
    """Tests for ImpactResult dataclass."""

    def test_empty_result(self):
        result = ImpactResult(query="/test", query_type="topic_change")
        summary = result.summary()
        assert summary["total_affected"] == 0
        assert summary["directly_affected"] == 0
        assert summary["cascade_affected"] == 0

    def test_summary_counts(self):
        result = ImpactResult(
            query="/test",
            query_type="topic_change",
            directly_affected=[
                ImpactItem(name="a", kind="node", impact_type="broken_subscriber", severity="high"),
                ImpactItem(name="b", kind="node", impact_type="lost_publisher", severity="critical"),
            ],
            cascade_affected=[
                ImpactItem(name="c", kind="node", impact_type="cascade", severity="medium"),
            ],
        )
        summary = result.summary()
        assert summary["total_affected"] == 3
        assert summary["directly_affected"] == 2
        assert summary["cascade_affected"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["medium"] == 1

    def test_to_dict(self):
        result = ImpactResult(
            query="/test",
            query_type="topic_change",
            directly_affected=[
                ImpactItem(name="a", kind="node", impact_type="broken_subscriber", severity="high"),
            ],
        )
        d = result.to_dict()
        assert "summary" in d
        assert "directly_affected" in d
        assert "cascade_affected" in d
        assert len(d["directly_affected"]) == 1


class TestImpactAnalyzerTopicChange:
    """Tests for analyze_topic_change."""

    def test_topic_change_finds_publishers_and_subscribers(self, analyzer):
        result = analyzer.analyze_topic_change("/sensor/data")
        names = [item.name for item in result.directly_affected]
        assert "sensor_node" in names  # publisher
        assert "processor_node" in names  # subscriber
        assert "logger_node" in names  # subscriber

    def test_critical_topic_gets_critical_severity(self, analyzer):
        result = analyzer.analyze_topic_change("/cmd_vel")
        for item in result.directly_affected:
            assert item.severity == "critical"

    def test_cascade_detection(self, analyzer):
        # Changing /sensor/data should cascade to motor_node via processor_node -> /cmd_vel
        result = analyzer.analyze_topic_change("/sensor/data")
        cascade_names = [item.name for item in result.cascade_affected]
        assert "motor_node" in cascade_names

    def test_unknown_topic_returns_empty(self, analyzer):
        result = analyzer.analyze_topic_change("/nonexistent")
        assert len(result.directly_affected) == 0
        assert len(result.cascade_affected) == 0

    def test_publisher_impact_type(self, analyzer):
        result = analyzer.analyze_topic_change("/sensor/data")
        pub_items = [i for i in result.directly_affected if i.impact_type == "lost_publisher"]
        assert any(i.name == "sensor_node" for i in pub_items)

    def test_subscriber_impact_type(self, analyzer):
        result = analyzer.analyze_topic_change("/sensor/data")
        sub_items = [i for i in result.directly_affected if i.impact_type == "broken_subscriber"]
        assert any(i.name == "processor_node" for i in sub_items)


class TestImpactAnalyzerNodeRemoval:
    """Tests for analyze_node_removal."""

    def test_node_removal_finds_affected_subscribers(self, analyzer):
        result = analyzer.analyze_node_removal("sensor_node")
        node_names = [i.name for i in result.directly_affected if i.kind == "node"]
        assert "processor_node" in node_names
        assert "logger_node" in node_names

    def test_node_removal_includes_topic(self, analyzer):
        result = analyzer.analyze_node_removal("sensor_node")
        topic_items = [i for i in result.directly_affected if i.kind == "topic"]
        topic_names = [i.name for i in topic_items]
        assert "/sensor/data" in topic_names

    def test_unknown_node_returns_empty(self, analyzer):
        result = analyzer.analyze_node_removal("nonexistent_node")
        assert len(result.directly_affected) == 0
        assert len(result.cascade_affected) == 0

    def test_node_removal_cascade(self, analyzer):
        # Removing sensor_node -> processor_node loses /sensor/data -> motor_node loses /cmd_vel (cascade)
        result = analyzer.analyze_node_removal("sensor_node")
        cascade_names = [i.name for i in result.cascade_affected]
        assert "motor_node" in cascade_names

    def test_sole_publisher_gives_high_severity(self, analyzer):
        result = analyzer.analyze_node_removal("sensor_node")
        broken_subs = [i for i in result.directly_affected
                       if i.kind == "node" and i.impact_type == "broken_subscriber"]
        # processor_node and logger_node lose their only source of /sensor/data
        for item in broken_subs:
            assert item.severity in ("critical", "high")


class TestImpactAnalyzerMessageType:
    """Tests for analyze_message_type_change."""

    def test_finds_all_users_of_type(self, analyzer):
        result = analyzer.analyze_message_type_change("sensor_msgs/msg/LaserScan")
        names = [i.name for i in result.directly_affected if i.kind == "node"]
        assert "sensor_node" in names
        assert "processor_node" in names
        assert "logger_node" in names

    def test_short_name_matching(self, analyzer):
        result = analyzer.analyze_message_type_change("LaserScan")
        names = [i.name for i in result.directly_affected if i.kind == "node"]
        assert "sensor_node" in names

    def test_includes_topic_items(self, analyzer):
        result = analyzer.analyze_message_type_change("geometry_msgs/msg/Twist")
        topic_items = [i for i in result.directly_affected if i.kind == "topic"]
        topic_names = [i.name for i in topic_items]
        assert "/cmd_vel" in topic_names

    def test_unknown_type_returns_empty(self, analyzer):
        result = analyzer.analyze_message_type_change("nonexistent_msgs/msg/FakeMsg")
        assert len(result.directly_affected) == 0


class TestImpactAnalyzerFileChange:
    """Tests for analyze_file_change."""

    def test_file_change_finds_nodes_in_file(self, analyzer):
        result = analyzer.analyze_file_change("/test/sensor.py")
        # Should find effects of removing sensor_node
        assert len(result.directly_affected) > 0

    def test_file_change_by_filename_only(self, analyzer):
        result = analyzer.analyze_file_change("sensor.py")
        assert len(result.directly_affected) > 0

    def test_unknown_file_returns_empty(self, analyzer):
        result = analyzer.analyze_file_change("/nonexistent/file.py")
        assert len(result.directly_affected) == 0


class TestSeverityRules:
    """Tests for severity determination."""

    def test_critical_topics(self, analyzer):
        for topic in CRITICAL_TOPICS:
            severity = analyzer._get_topic_severity(topic)
            assert severity == "critical", f"{topic} should be critical"

    def test_low_priority_topics(self, analyzer):
        for prefix in LOW_PRIORITY_PREFIXES:
            topic = prefix + "test"
            severity = analyzer._get_topic_severity(topic)
            assert severity == "low", f"{topic} should be low"

    def test_normal_topic_is_medium(self, analyzer):
        severity = analyzer._get_topic_severity("/some/regular/topic")
        assert severity == "medium"


class TestConvenienceFunction:
    """Tests for the analyze_impact convenience function."""

    def test_topic_analysis(self, sample_nodes, topic_graph):
        result = analyze_impact(sample_nodes, topic_graph, target="/cmd_vel", target_type="topic")
        assert result.query_type == "topic_change"
        assert len(result.directly_affected) > 0

    def test_node_analysis(self, sample_nodes, topic_graph):
        result = analyze_impact(sample_nodes, topic_graph, target="sensor_node", target_type="node")
        assert result.query_type == "node_removal"

    def test_message_type_analysis(self, sample_nodes, topic_graph):
        result = analyze_impact(sample_nodes, topic_graph, target="Twist", target_type="message_type")
        assert result.query_type == "message_type_change"

    def test_file_analysis(self, sample_nodes, topic_graph):
        result = analyze_impact(sample_nodes, topic_graph, target="sensor.py", target_type="file")
        assert result.query_type == "file_change"

    def test_unknown_type(self, sample_nodes, topic_graph):
        result = analyze_impact(sample_nodes, topic_graph, target="test", target_type="unknown")
        assert result.query_type == "unknown"
        assert len(result.directly_affected) == 0


class TestIndexBuilding:
    """Tests for internal index building."""

    def test_node_map(self, analyzer):
        assert "sensor_node" in analyzer._node_map
        assert "processor_node" in analyzer._node_map

    def test_topic_publishers(self, analyzer):
        assert "sensor_node" in analyzer._topic_publishers["/sensor/data"]

    def test_topic_subscribers(self, analyzer):
        assert "processor_node" in analyzer._topic_subscribers["/sensor/data"]

    def test_file_to_nodes(self, analyzer):
        assert "sensor_node" in analyzer._file_to_nodes[str(Path("/test/sensor.py"))]

    def test_topic_types(self, analyzer):
        assert analyzer._topic_types["/sensor/data"] == "sensor_msgs/msg/LaserScan"

    def test_node_pub_topics(self, analyzer):
        assert "/sensor/data" in analyzer._node_pub_topics["sensor_node"]

    def test_node_sub_topics(self, analyzer):
        assert "/sensor/data" in analyzer._node_sub_topics["processor_node"]
