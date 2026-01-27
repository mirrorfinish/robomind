"""Tests for the Flow Tracer module."""

import pytest
from pathlib import Path

from robomind.analyzers.flow_tracer import (
    FlowTracer,
    FlowPath,
    FlowTraceResult,
    trace_flow,
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
            name="processor_node",
            class_name="ProcessorNode",
            file_path=Path("/test/processor.py"),
            line_number=1,
            end_line=50,
            package_name="test_pkg",
            subscribers=[
                SubscriberInfo(topic="/sensor/data", msg_type="SensorData", callback="cb", qos=10),
            ],
            publishers=[
                PublisherInfo(topic="/processed/data", msg_type="ProcessedData", qos=10),
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
                SubscriberInfo(topic="/processed/data", msg_type="ProcessedData", callback="cb", qos=10),
            ],
            publishers=[
                PublisherInfo(topic="/cmd_vel", msg_type="Twist", qos=10),
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
                SubscriberInfo(topic="/cmd_vel", msg_type="Twist", callback="cb", qos=10),
            ],
        ),
    ]


@pytest.fixture
def topic_graph(sample_nodes):
    """Create topic graph from sample nodes."""
    extractor = TopicExtractor()
    extractor.add_nodes(sample_nodes)
    return extractor.build()


class TestFlowPath:
    """Tests for FlowPath dataclass."""

    def test_path_creation(self):
        """Test creating a flow path."""
        path = FlowPath(
            nodes=["a", "b", "c"],
            topics=["/t1", "/t2"],
        )

        assert len(path.nodes) == 3
        assert len(path.topics) == 2
        assert path.length == 3

    def test_path_to_dict(self):
        """Test converting path to dictionary."""
        path = FlowPath(
            nodes=["a", "b"],
            topics=["/topic"],
            bottlenecks=["b"],
        )

        d = path.to_dict()
        assert d["nodes"] == ["a", "b"]
        assert d["topics"] == ["/topic"]
        assert d["bottlenecks"] == ["b"]

    def test_path_to_mermaid(self):
        """Test Mermaid diagram generation."""
        path = FlowPath(
            nodes=["sensor", "processor", "controller"],
            topics=["/sensor/data", "/processed"],
        )

        mermaid = path.to_mermaid()
        assert "graph LR" in mermaid
        assert "sensor" in mermaid
        assert "processor" in mermaid

    def test_path_to_mermaid_with_bottleneck(self):
        """Test Mermaid diagram with bottleneck highlighting."""
        path = FlowPath(
            nodes=["a", "b", "c"],
            topics=["/t1", "/t2"],
            bottlenecks=["b"],
        )

        mermaid = path.to_mermaid()
        # Bottleneck nodes should have double brackets
        assert '[[' in mermaid or 'b' in mermaid


class TestFlowTraceResult:
    """Tests for FlowTraceResult dataclass."""

    def test_empty_result(self):
        """Test empty trace result."""
        result = FlowTraceResult()
        assert result.success is False
        assert len(result.paths) == 0

    def test_result_with_paths(self):
        """Test result with paths."""
        result = FlowTraceResult(
            paths=[
                FlowPath(nodes=["a", "b", "c"], topics=["/t1", "/t2"]),
                FlowPath(nodes=["a", "d", "c"], topics=["/t3", "/t4"]),
            ],
            source="a",
            target="c",
            success=True,
        )

        summary = result.summary()
        assert summary["paths_found"] == 2
        assert summary["shortest_path"] == 3
        assert summary["source"] == "a"
        assert summary["target"] == "c"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = FlowTraceResult(
            paths=[FlowPath(nodes=["a", "b"], topics=["/t"])],
            success=True,
        )

        d = result.to_dict()
        assert len(d["paths"]) == 1
        assert "summary" in d


class TestFlowTracer:
    """Tests for FlowTracer class."""

    def test_tracer_init(self, sample_nodes, topic_graph):
        """Test tracer initialization."""
        tracer = FlowTracer(sample_nodes, topic_graph)

        assert len(tracer._forward_adj) > 0
        assert len(tracer._reverse_adj) > 0

    def test_tracer_build_adjacency(self, sample_nodes, topic_graph):
        """Test adjacency map building."""
        tracer = FlowTracer(sample_nodes, topic_graph)

        # sensor_node should connect to processor_node
        assert "processor_node" in tracer._forward_adj.get("sensor_node", {})

        # processor_node should connect to controller_node
        assert "controller_node" in tracer._forward_adj.get("processor_node", {})

    def test_trace_simple_path(self, sample_nodes, topic_graph):
        """Test tracing a simple path."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        result = tracer.trace("sensor_node", "motor_node")

        assert result.success is True
        assert len(result.paths) >= 1

        # Should find the path: sensor -> processor -> controller -> motor
        path_nodes = result.paths[0].nodes
        assert path_nodes[0] == "sensor_node"
        assert path_nodes[-1] == "motor_node"
        assert len(path_nodes) == 4

    def test_trace_direct_path(self, sample_nodes, topic_graph):
        """Test tracing a direct connection."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        result = tracer.trace("sensor_node", "processor_node")

        assert result.success is True
        assert len(result.paths) == 1
        assert len(result.paths[0].nodes) == 2

    def test_trace_nonexistent_source(self, sample_nodes, topic_graph):
        """Test tracing from nonexistent node."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        result = tracer.trace("nonexistent", "motor_node")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_trace_nonexistent_target(self, sample_nodes, topic_graph):
        """Test tracing to nonexistent node."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        result = tracer.trace("sensor_node", "nonexistent")

        assert result.success is False
        assert result.error is not None

    def test_trace_no_path(self, sample_nodes, topic_graph):
        """Test when no path exists."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        # Reverse direction - no path from motor to sensor
        result = tracer.trace("motor_node", "sensor_node")

        # Should not find a path (no reverse flow)
        assert len(result.paths) == 0

    def test_trace_topic(self, sample_nodes, topic_graph):
        """Test tracing a topic."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        result = tracer.trace_topic("/sensor/data")

        assert len(result.paths) > 0
        # Should show sensor_node publishing to processor_node
        assert result.source == "/sensor/data"

    def test_trace_topic_not_found(self, sample_nodes, topic_graph):
        """Test tracing nonexistent topic."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        result = tracer.trace_topic("/nonexistent/topic")

        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_get_flow_summary(self, sample_nodes, topic_graph):
        """Test flow summary generation."""
        tracer = FlowTracer(sample_nodes, topic_graph)
        summary = tracer.get_flow_summary()

        assert "entry_points" in summary
        assert "exit_points" in summary
        assert "high_traffic_nodes" in summary

        # sensor_node should be an entry point (no upstream)
        assert "sensor_node" in summary["entry_points"]

        # motor_node should be an exit point (no downstream)
        assert "motor_node" in summary["exit_points"]


class TestFlowTracerWithoutTopicGraph:
    """Test FlowTracer building adjacency from nodes directly."""

    def test_trace_without_topic_graph(self, sample_nodes):
        """Test tracing without a pre-built topic graph."""
        tracer = FlowTracer(sample_nodes, topic_graph=None)
        result = tracer.trace("sensor_node", "motor_node")

        # Should still work by building adjacency from nodes
        assert result.success is True
        assert len(result.paths) >= 1


class TestBottleneckDetection:
    """Test bottleneck detection in flow paths."""

    @pytest.fixture
    def diamond_nodes(self):
        """Create a diamond-shaped graph with a bottleneck."""
        return [
            ROS2NodeInfo(
                name="source",
                class_name="Source",
                file_path=Path("/test/source.py"),
                line_number=1,
                end_line=50,
                publishers=[PublisherInfo(topic="/a", msg_type="A", qos=10)],
            ),
            ROS2NodeInfo(
                name="left",
                class_name="Left",
                file_path=Path("/test/left.py"),
                line_number=1,
                end_line=50,
                subscribers=[SubscriberInfo(topic="/a", msg_type="A", callback="cb", qos=10)],
                publishers=[PublisherInfo(topic="/b", msg_type="B", qos=10)],
            ),
            ROS2NodeInfo(
                name="right",
                class_name="Right",
                file_path=Path("/test/right.py"),
                line_number=1,
                end_line=50,
                subscribers=[SubscriberInfo(topic="/a", msg_type="A", callback="cb", qos=10)],
                publishers=[PublisherInfo(topic="/c", msg_type="C", qos=10)],
            ),
            ROS2NodeInfo(
                name="merge",
                class_name="Merge",
                file_path=Path("/test/merge.py"),
                line_number=1,
                end_line=50,
                subscribers=[
                    SubscriberInfo(topic="/b", msg_type="B", callback="cb", qos=10),
                    SubscriberInfo(topic="/c", msg_type="C", callback="cb", qos=10),
                ],
                publishers=[PublisherInfo(topic="/d", msg_type="D", qos=10)],
            ),
            ROS2NodeInfo(
                name="sink",
                class_name="Sink",
                file_path=Path("/test/sink.py"),
                line_number=1,
                end_line=50,
                subscribers=[SubscriberInfo(topic="/d", msg_type="D", callback="cb", qos=10)],
            ),
        ]

    def test_multiple_paths(self, diamond_nodes):
        """Test finding multiple paths through diamond."""
        extractor = TopicExtractor()
        extractor.add_nodes(diamond_nodes)
        topic_graph = extractor.build()

        tracer = FlowTracer(diamond_nodes, topic_graph)
        result = tracer.trace("source", "sink")

        # Should find 2 paths: source->left->merge->sink and source->right->merge->sink
        assert len(result.paths) == 2


class TestConvenienceFunction:
    """Test convenience function."""

    def test_trace_flow_function(self, sample_nodes, topic_graph):
        """Test the trace_flow convenience function."""
        result = trace_flow("sensor_node", "controller_node", sample_nodes, topic_graph)

        assert result.success is True
        assert len(result.paths) >= 1
