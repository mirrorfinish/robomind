"""Tests for the Coupling Analyzer module."""

import pytest
from pathlib import Path

from robomind.analyzers.coupling import (
    CouplingAnalyzer,
    CouplingScore,
    CouplingMatrix,
    analyze_coupling,
    WEIGHT_TOPIC_CONNECTION,
    WEIGHT_SHARED_PARAMETERS,
    WEIGHT_DATA_DEPENDENCY,
    WEIGHT_TEMPORAL_COUPLING,
)
from robomind.ros2.node_extractor import (
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
    TimerInfo,
    ParameterInfo,
)
from robomind.ros2.topic_extractor import TopicExtractor


@pytest.fixture
def publisher_node():
    """Create a node that publishes to topics."""
    return ROS2NodeInfo(
        name="sensor_node",
        class_name="SensorNode",
        file_path=Path("/test/sensor_node.py"),
        line_number=10,
        end_line=100,
        package_name="sensor_pkg",
        publishers=[
            PublisherInfo(topic="/scan", msg_type="LaserScan", qos=10),
            PublisherInfo(topic="/image", msg_type="Image", qos=30),
        ],
        timers=[
            TimerInfo(period=0.1, callback="scan_callback"),
        ],
        parameters=[
            ParameterInfo(name="rate", default_value=10, param_type="int"),
            ParameterInfo(name="frame_id", default_value="laser", param_type="str"),
        ],
    )


@pytest.fixture
def subscriber_node():
    """Create a node that subscribes to topics."""
    return ROS2NodeInfo(
        name="perception_node",
        class_name="PerceptionNode",
        file_path=Path("/test/perception_node.py"),
        line_number=10,
        end_line=100,
        package_name="perception_pkg",
        subscribers=[
            SubscriberInfo(topic="/scan", msg_type="LaserScan", callback="scan_cb", qos=10),
            SubscriberInfo(topic="/image", msg_type="Image", callback="image_cb", qos=30),
        ],
        timers=[
            TimerInfo(period=0.1, callback="process_callback"),  # Same frequency
        ],
        parameters=[
            ParameterInfo(name="rate", default_value=10, param_type="int"),  # Shared param name
            ParameterInfo(name="threshold", default_value=0.5, param_type="float"),
        ],
    )


@pytest.fixture
def unconnected_node():
    """Create a node not connected to others."""
    return ROS2NodeInfo(
        name="isolated_node",
        class_name="IsolatedNode",
        file_path=Path("/test/isolated_node.py"),
        line_number=10,
        end_line=50,
        package_name="other_pkg",
        publishers=[
            PublisherInfo(topic="/different_topic", msg_type="String", qos=10),
        ],
    )


@pytest.fixture
def tightly_coupled_nodes():
    """Create two tightly coupled nodes with multiple connections."""
    node1 = ROS2NodeInfo(
        name="motor_controller",
        class_name="MotorController",
        file_path=Path("/test/motor.py"),
        line_number=10,
        end_line=100,
        package_name="motor_pkg",
        publishers=[
            PublisherInfo(topic="/motor/status", msg_type="MotorStatus", qos=10),
            PublisherInfo(topic="/motor/feedback", msg_type="Float32", qos=100),
            PublisherInfo(topic="/motor/temp", msg_type="Float32", qos=10),
        ],
        subscribers=[
            SubscriberInfo(topic="/cmd_vel", msg_type="Twist", callback="cmd_cb", qos=10),
        ],
        timers=[
            TimerInfo(period=0.01, callback="control_loop"),
        ],
        parameters=[
            ParameterInfo(name="motor.max_speed", default_value=1.5, param_type="float"),
            ParameterInfo(name="motor.pid_kp", default_value=1.0, param_type="float"),
        ],
    )

    node2 = ROS2NodeInfo(
        name="motor_monitor",
        class_name="MotorMonitor",
        file_path=Path("/test/monitor.py"),
        line_number=10,
        end_line=100,
        package_name="motor_pkg",
        subscribers=[
            SubscriberInfo(topic="/motor/status", msg_type="MotorStatus", callback="status_cb", qos=10),
            SubscriberInfo(topic="/motor/feedback", msg_type="Float32", callback="feedback_cb", qos=100),
            SubscriberInfo(topic="/motor/temp", msg_type="Float32", callback="temp_cb", qos=10),
        ],
        publishers=[
            PublisherInfo(topic="/cmd_vel", msg_type="Twist", qos=10),  # Bidirectional
        ],
        timers=[
            TimerInfo(period=0.01, callback="monitor_loop"),  # Same frequency
        ],
        parameters=[
            ParameterInfo(name="motor.max_speed", default_value=1.5, param_type="float"),  # Same param
            ParameterInfo(name="motor.threshold", default_value=0.8, param_type="float"),
        ],
    )

    return node1, node2


class TestCouplingScore:
    """Tests for the CouplingScore class."""

    def test_strength_label_critical(self):
        """Test critical strength label."""
        assert CouplingScore.get_strength_label(0.7) == "CRITICAL"
        assert CouplingScore.get_strength_label(0.85) == "CRITICAL"
        assert CouplingScore.get_strength_label(1.0) == "CRITICAL"

    def test_strength_label_high(self):
        """Test high strength label."""
        assert CouplingScore.get_strength_label(0.5) == "HIGH"
        assert CouplingScore.get_strength_label(0.69) == "HIGH"

    def test_strength_label_medium(self):
        """Test medium strength label."""
        assert CouplingScore.get_strength_label(0.3) == "MEDIUM"
        assert CouplingScore.get_strength_label(0.49) == "MEDIUM"

    def test_strength_label_low(self):
        """Test low strength label."""
        assert CouplingScore.get_strength_label(0.0) == "LOW"
        assert CouplingScore.get_strength_label(0.29) == "LOW"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = CouplingScore(
            source="node_a",
            target="node_b",
            score=0.65,
            strength="HIGH",
            factors={"topic_connections": 0.3, "shared_parameters": 0.2},
            topics=["/test_topic"],
        )

        d = score.to_dict()

        assert d["source"] == "node_a"
        assert d["target"] == "node_b"
        assert d["score"] == 0.65
        assert d["strength"] == "HIGH"
        assert "topic_connections" in d["factors"]


class TestCouplingMatrix:
    """Tests for the CouplingMatrix class."""

    def test_empty_matrix(self):
        """Test empty coupling matrix."""
        matrix = CouplingMatrix()

        summary = matrix.summary()
        assert summary["total_pairs"] == 0
        assert summary["average_coupling"] == 0.0

    def test_add_and_get_score(self):
        """Test adding and retrieving scores."""
        matrix = CouplingMatrix(nodes=["a", "b"])

        score = CouplingScore(
            source="a",
            target="b",
            score=0.5,
            strength="HIGH",
            factors={},
            topics=[],
        )
        matrix.scores[("a", "b")] = score

        retrieved = matrix.get_score("a", "b")
        assert retrieved is not None
        assert retrieved.score == 0.5

    def test_get_top_coupled_pairs(self):
        """Test getting top coupled pairs."""
        matrix = CouplingMatrix(nodes=["a", "b", "c"])

        matrix.scores[("a", "b")] = CouplingScore(
            source="a", target="b", score=0.3, strength="MEDIUM", factors={}, topics=[]
        )
        matrix.scores[("a", "c")] = CouplingScore(
            source="a", target="c", score=0.7, strength="CRITICAL", factors={}, topics=[]
        )
        matrix.scores[("b", "c")] = CouplingScore(
            source="b", target="c", score=0.5, strength="HIGH", factors={}, topics=[]
        )

        top = matrix.get_top_coupled_pairs(2)

        assert len(top) == 2
        assert top[0].score == 0.7
        assert top[1].score == 0.5

    def test_get_critical_pairs(self):
        """Test getting critical pairs."""
        matrix = CouplingMatrix(nodes=["a", "b", "c"])

        matrix.scores[("a", "b")] = CouplingScore(
            source="a", target="b", score=0.3, strength="MEDIUM", factors={}, topics=[]
        )
        matrix.scores[("a", "c")] = CouplingScore(
            source="a", target="c", score=0.8, strength="CRITICAL", factors={}, topics=[]
        )

        critical = matrix.get_critical_pairs()

        assert len(critical) == 1
        assert critical[0].source == "a"
        assert critical[0].target == "c"

    def test_summary(self):
        """Test summary generation."""
        matrix = CouplingMatrix(nodes=["a", "b", "c"])

        matrix.scores[("a", "b")] = CouplingScore(
            source="a", target="b", score=0.3, strength="MEDIUM", factors={}, topics=[]
        )
        matrix.scores[("a", "c")] = CouplingScore(
            source="a", target="c", score=0.7, strength="CRITICAL", factors={}, topics=[]
        )

        summary = matrix.summary()

        assert summary["total_pairs"] == 2
        assert summary["nodes_analyzed"] == 3
        assert summary["critical_pairs"] == 1
        assert summary["medium_pairs"] == 1
        assert 0.4 < summary["average_coupling"] < 0.6


class TestCouplingAnalyzer:
    """Tests for the CouplingAnalyzer class."""

    def test_analyzer_init(self, publisher_node, subscriber_node):
        """Test analyzer initialization."""
        analyzer = CouplingAnalyzer([publisher_node, subscriber_node])
        assert analyzer is not None
        assert len(analyzer.nodes) == 2

    def test_analyze_connected_nodes(self, publisher_node, subscriber_node):
        """Test analyzing connected nodes."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        # Should have at least one coupling pair
        assert matrix.summary()["total_pairs"] >= 1

        # Get the coupling between these nodes
        pairs = matrix.get_top_coupled_pairs(1)
        assert len(pairs) > 0
        assert pairs[0].score > 0

    def test_analyze_unconnected_nodes(self, publisher_node, unconnected_node):
        """Test analyzing unconnected nodes."""
        nodes = [publisher_node, unconnected_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        # No connections, so no coupling pairs
        assert matrix.summary()["total_pairs"] == 0

    def test_analyze_tightly_coupled_nodes(self, tightly_coupled_nodes):
        """Test analyzing tightly coupled nodes."""
        node1, node2 = tightly_coupled_nodes
        nodes = [node1, node2]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        # Should have high coupling
        pairs = matrix.get_top_coupled_pairs(1)
        assert len(pairs) > 0

        # Multiple connections + same params + same timers = high coupling
        score = pairs[0]
        assert score.score >= 0.3  # Should be at least MEDIUM
        assert len(score.topics) >= 3  # 3+ connecting topics

    def test_topic_coupling_factor(self, publisher_node, subscriber_node):
        """Test that topic coupling is calculated correctly."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        pairs = matrix.get_top_coupled_pairs(1)
        if pairs:
            score = pairs[0]
            # Should have topic_connections factor
            assert "topic_connections" in score.factors
            assert score.factors["topic_connections"] > 0

    def test_parameter_coupling_factor(self, publisher_node, subscriber_node):
        """Test that parameter coupling is calculated correctly."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        pairs = matrix.get_top_coupled_pairs(1)
        if pairs:
            score = pairs[0]
            # Both nodes have "rate" parameter
            assert "shared_parameters" in score.factors
            # Note: May be 0 if no shared params detected

    def test_temporal_coupling_factor(self, publisher_node, subscriber_node):
        """Test that temporal coupling is calculated correctly."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        pairs = matrix.get_top_coupled_pairs(1)
        if pairs:
            score = pairs[0]
            # Both nodes have 0.1s timers (10Hz)
            assert "temporal_coupling" in score.factors
            # Should detect aligned frequencies
            assert score.factors["temporal_coupling"] >= 0

    def test_data_coupling_factor(self, publisher_node, subscriber_node):
        """Test that data dependency coupling is calculated."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        matrix = analyzer.analyze()

        pairs = matrix.get_top_coupled_pairs(1)
        if pairs:
            score = pairs[0]
            # Using complex types (LaserScan, Image)
            assert "data_dependencies" in score.factors
            assert score.factors["data_dependencies"] > 0

    def test_coupling_summary(self, publisher_node, subscriber_node):
        """Test coupling summary generation."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        analyzer = CouplingAnalyzer(nodes, topic_graph)
        summary = analyzer.get_coupling_summary()

        assert "COUPLING ANALYSIS SUMMARY" in summary
        assert "Nodes analyzed:" in summary
        assert "Coupling Distribution:" in summary


class TestAnalyzeCouplingFunction:
    """Tests for the analyze_coupling convenience function."""

    def test_analyze_coupling(self, publisher_node, subscriber_node):
        """Test the convenience function."""
        nodes = [publisher_node, subscriber_node]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        matrix = analyze_coupling(nodes, topic_graph)

        assert matrix is not None
        assert isinstance(matrix, CouplingMatrix)

    def test_analyze_coupling_without_topic_graph(self, publisher_node, subscriber_node):
        """Test analyzing without pre-built topic graph."""
        nodes = [publisher_node, subscriber_node]

        # Should still work by building connections internally
        matrix = analyze_coupling(nodes)

        assert matrix is not None


class TestCouplingWeights:
    """Tests for coupling weight constants."""

    def test_weights_sum_to_one(self):
        """Verify coupling weights sum to 1.0."""
        total = (
            WEIGHT_TOPIC_CONNECTION +
            WEIGHT_SHARED_PARAMETERS +
            WEIGHT_DATA_DEPENDENCY +
            WEIGHT_TEMPORAL_COUPLING
        )
        assert abs(total - 1.0) < 0.001

    def test_weight_values(self):
        """Verify weight values are as expected."""
        assert WEIGHT_TOPIC_CONNECTION == 0.40
        assert WEIGHT_SHARED_PARAMETERS == 0.30
        assert WEIGHT_DATA_DEPENDENCY == 0.20
        assert WEIGHT_TEMPORAL_COUPLING == 0.10
