"""Tests for the Validators module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from robomind.validators.live_validator import (
    LiveValidator,
    ValidationResult,
    ValidationDiff,
    LiveSystemInfo,
    DiffType,
    Severity,
    validate_against_live,
)
from robomind.ros2.node_extractor import (
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
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
                PublisherInfo(topic="/sensor/data", msg_type="SensorData", qos=10),
            ],
        ),
        ROS2NodeInfo(
            name="controller_node",
            class_name="ControllerNode",
            file_path=Path("/test/controller.py"),
            line_number=1,
            end_line=50,
            package_name="test_pkg",
            subscribers=[
                SubscriberInfo(topic="/sensor/data", msg_type="SensorData", callback="cb", qos=10),
            ],
            publishers=[
                PublisherInfo(topic="/cmd_vel", msg_type="Twist", qos=10),
            ],
        ),
    ]


@pytest.fixture
def topic_graph(sample_nodes):
    """Create topic graph from sample nodes."""
    extractor = TopicExtractor()
    extractor.add_nodes(sample_nodes)
    return extractor.build()


class TestValidationDiff:
    """Tests for ValidationDiff dataclass."""

    def test_diff_creation(self):
        """Test creating a validation diff."""
        diff = ValidationDiff(
            diff_type=DiffType.TOPIC_IN_CODE_NOT_LIVE,
            severity=Severity.WARNING,
            name="/test/topic",
            message="Topic not active",
            recommendation="Check if node is running",
        )

        assert diff.diff_type == DiffType.TOPIC_IN_CODE_NOT_LIVE
        assert diff.severity == Severity.WARNING
        assert diff.name == "/test/topic"

    def test_diff_to_dict(self):
        """Test converting diff to dictionary."""
        diff = ValidationDiff(
            diff_type=DiffType.TYPE_MISMATCH,
            severity=Severity.ERROR,
            name="/topic",
            message="Type mismatch",
            code_value="Twist",
            live_value="String",
        )

        d = diff.to_dict()
        assert d["type"] == "type_mismatch"
        assert d["severity"] == "error"
        assert d["code_value"] == "Twist"
        assert d["live_value"] == "String"


class TestLiveSystemInfo:
    """Tests for LiveSystemInfo dataclass."""

    def test_live_info_creation(self):
        """Test creating live system info."""
        info = LiveSystemInfo(
            nodes=["/node1", "/node2"],
            topics={"/topic1": {"type": "std_msgs/String"}},
            available=True,
        )

        assert len(info.nodes) == 2
        assert len(info.topics) == 1
        assert info.available is True

    def test_live_info_to_dict(self):
        """Test converting to dictionary."""
        info = LiveSystemInfo(nodes=["a", "b"], available=True)
        d = info.to_dict()
        assert d["available"] is True
        assert len(d["nodes"]) == 2


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_empty_result(self):
        """Test empty validation result."""
        result = ValidationResult()
        assert result.has_critical is False
        assert result.has_errors is False
        assert len(result.diffs) == 0

    def test_result_with_diffs(self):
        """Test result with various diffs."""
        result = ValidationResult(
            diffs=[
                ValidationDiff(DiffType.TOPIC_IN_CODE_NOT_LIVE, Severity.WARNING, "t1", "m1"),
                ValidationDiff(DiffType.TYPE_MISMATCH, Severity.ERROR, "t2", "m2"),
                ValidationDiff(DiffType.NODE_IN_CODE_NOT_LIVE, Severity.CRITICAL, "n1", "m3"),
            ],
            validated=True,
        )

        assert result.has_critical is True
        assert result.has_errors is True
        assert len(result.get_by_severity(Severity.WARNING)) == 1
        assert len(result.get_by_severity(Severity.CRITICAL)) == 1

    def test_result_summary(self):
        """Test summary generation."""
        result = ValidationResult(
            diffs=[
                ValidationDiff(DiffType.TOPIC_IN_CODE_NOT_LIVE, Severity.WARNING, "t1", "m1"),
                ValidationDiff(DiffType.TOPIC_IN_CODE_NOT_LIVE, Severity.WARNING, "t2", "m2"),
            ],
            validated=True,
            code_stats={"nodes": 5, "topics": 10},
        )

        summary = result.summary()
        assert summary["total_diffs"] == 2
        assert summary["by_severity"]["warning"] == 2
        assert summary["validated"] is True


class TestLiveValidator:
    """Tests for LiveValidator class."""

    def test_validator_init(self, sample_nodes, topic_graph):
        """Test validator initialization."""
        validator = LiveValidator(sample_nodes, topic_graph)
        assert len(validator._code_topics) == 2  # /sensor/data and /cmd_vel
        assert len(validator._code_nodes) == 2

    def test_build_code_topics(self, sample_nodes, topic_graph):
        """Test building code topics map."""
        validator = LiveValidator(sample_nodes, topic_graph)

        assert "/sensor/data" in validator._code_topics
        assert "/cmd_vel" in validator._code_topics

        sensor_topic = validator._code_topics["/sensor/data"]
        assert "sensor_node" in sensor_topic["publishers"]
        assert "controller_node" in sensor_topic["subscribers"]

    @patch.object(LiveValidator, "_run_command")
    def test_get_live_system_info_success(self, mock_run, sample_nodes, topic_graph):
        """Test getting live system info with mock commands."""
        mock_run.side_effect = [
            # ros2 node list
            "/sensor_node\n/controller_node\n/other_node\n",
            # ros2 topic list -t
            "/sensor/data [sensor_msgs/msg/SensorData]\n/cmd_vel [geometry_msgs/msg/Twist]\n",
            # ros2 topic info for first topic
            "Publisher count: 1\nSubscription count: 1\n",
            # ros2 topic info for second topic
            "Publisher count: 1\nSubscription count: 0\n",
            # ros2 service list
            "/get_state\n",
        ]

        validator = LiveValidator(sample_nodes, topic_graph)
        info = validator.get_live_system_info()

        assert info.available is True
        assert len(info.nodes) == 3
        assert len(info.topics) == 2

    @patch.object(LiveValidator, "_run_command")
    def test_get_live_system_info_failure(self, mock_run, sample_nodes, topic_graph):
        """Test handling ROS2 not running."""
        mock_run.return_value = None  # Command failed

        validator = LiveValidator(sample_nodes, topic_graph)
        info = validator.get_live_system_info()

        assert info.available is False
        assert info.error is not None

    @patch.object(LiveValidator, "get_live_system_info")
    def test_validate_finds_mismatches(self, mock_live, sample_nodes, topic_graph):
        """Test validation finding mismatches."""
        # Mock live system with different topics
        mock_live.return_value = LiveSystemInfo(
            nodes=["sensor_node", "other_node"],
            topics={
                "/different/topic": {"type": "std_msgs/msg/String"},
                "/rosout": {"type": "rcl_interfaces/msg/Log"},
            },
            available=True,
        )

        validator = LiveValidator(sample_nodes, topic_graph)
        result = validator.validate()

        assert result.validated is True
        assert len(result.diffs) > 0

        # Should find topics in code not live
        code_not_live = result.get_by_type(DiffType.TOPIC_IN_CODE_NOT_LIVE)
        assert len(code_not_live) > 0

    @patch.object(LiveValidator, "get_live_system_info")
    def test_validate_type_mismatch(self, mock_live, sample_nodes, topic_graph):
        """Test detection of type mismatches."""
        mock_live.return_value = LiveSystemInfo(
            nodes=["sensor_node", "controller_node"],
            topics={
                "/sensor/data": {"type": "wrong_msgs/msg/WrongType"},  # Wrong type
                "/cmd_vel": {"type": "geometry_msgs/msg/Twist"},
            },
            available=True,
        )

        validator = LiveValidator(sample_nodes, topic_graph)
        result = validator.validate()

        type_mismatches = result.get_by_type(DiffType.TYPE_MISMATCH)
        assert len(type_mismatches) == 1
        assert type_mismatches[0].severity == Severity.ERROR


class TestConvenienceFunction:
    """Test convenience function."""

    @patch.object(LiveValidator, "get_live_system_info")
    def test_validate_against_live(self, mock_live, sample_nodes, topic_graph):
        """Test the convenience function."""
        mock_live.return_value = LiveSystemInfo(
            nodes=["sensor_node"],
            topics={"/sensor/data": {"type": "sensor_msgs/msg/SensorData"}},
            available=True,
        )

        result = validate_against_live(sample_nodes, topic_graph)

        assert result.validated is True
        assert isinstance(result, ValidationResult)
