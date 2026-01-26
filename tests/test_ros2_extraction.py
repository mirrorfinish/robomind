"""Tests for ROS2 Node and Topic Extraction."""

import pytest
from pathlib import Path
import tempfile

from robomind.ros2.node_extractor import (
    ROS2NodeExtractor,
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
)
from robomind.ros2.topic_extractor import (
    TopicExtractor,
    TopicConnection,
    TopicGraphResult,
)


@pytest.fixture
def sample_node_path():
    """Path to sample ROS2 node fixture."""
    return Path(__file__).parent / "fixtures" / "sample_ros2_node.py"


@pytest.fixture
def minimal_node_code():
    """Minimal ROS2 node code for testing."""
    return '''
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalNode(Node):
    """A minimal ROS2 node."""

    def __init__(self):
        super().__init__('minimal_node')

        self.publisher = self.create_publisher(String, '/output', 10)
        self.subscription = self.create_subscription(
            String, '/input', self.callback, 10
        )
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.declare_parameter('my_param', 'default_value')
        self.declare_parameter('rate', 10.0)

    def callback(self, msg):
        pass

    def timer_callback(self):
        pass
'''


@pytest.fixture
def advanced_node_code():
    """Advanced ROS2 node with services and actions."""
    return '''
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav2_msgs.action import NavigateToPose

class AdvancedNode(Node):
    """Node with services and actions."""

    def __init__(self):
        super().__init__('advanced_node')

        # Publishers and subscribers
        self.pub1 = self.create_publisher(String, '/topic_a', 10)
        self.pub2 = self.create_publisher(String, '/topic_b', 5)
        self.sub1 = self.create_subscription(String, '/topic_c', self.cb, 10)

        # Services
        self.service = self.create_service(Trigger, '/my_service', self.service_cb)
        self.client = self.create_client(Trigger, '/other_service')

        # Actions
        self.action_client = ActionClient(self, NavigateToPose, '/navigate')

        # Multiple parameters
        self.declare_parameter('speed', 1.0)
        self.declare_parameter('enabled', True)
        self.declare_parameter('name', 'robot')

    def cb(self, msg):
        pass

    def service_cb(self, request, response):
        return response
'''


class TestROS2NodeExtractor:
    """Tests for ROS2NodeExtractor."""

    def test_extractor_init(self):
        """Test extractor initialization."""
        extractor = ROS2NodeExtractor()
        assert extractor is not None

    def test_extract_sample_node(self, sample_node_path):
        """Test extracting from sample ROS2 node."""
        extractor = ROS2NodeExtractor()
        nodes = extractor.extract_from_file(sample_node_path)

        assert len(nodes) == 1

        node = nodes[0]
        assert node.name == "sample_node"
        assert node.class_name == "SampleNode"

        # Check publishers
        assert len(node.publishers) >= 1
        pub_topics = [p.topic for p in node.publishers]
        # The topic is from get_parameter, so might be captured or not
        # At minimum we should have extracted the publisher

        # Check subscribers
        assert len(node.subscribers) >= 1
        sub_topics = [s.topic for s in node.subscribers]
        assert "command" in sub_topics or "/command" in sub_topics

        # Check timers
        assert len(node.timers) >= 1

        # Check parameters
        assert len(node.parameters) >= 3

    def test_extract_minimal_node(self, minimal_node_code):
        """Test extracting from minimal node."""
        extractor = ROS2NodeExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(minimal_node_code)
            temp_path = Path(f.name)

        try:
            nodes = extractor.extract_from_file(temp_path)

            assert len(nodes) == 1
            node = nodes[0]

            assert node.name == "minimal_node"
            assert node.class_name == "MinimalNode"

            # Check publisher
            assert len(node.publishers) == 1
            assert node.publishers[0].topic == "/output"
            assert node.publishers[0].msg_type == "String"
            assert node.publishers[0].qos == 10

            # Check subscriber
            assert len(node.subscribers) == 1
            assert node.subscribers[0].topic == "/input"
            assert node.subscribers[0].callback == "callback"

            # Check timer
            assert len(node.timers) == 1
            assert node.timers[0].period == 0.5
            assert node.timers[0].frequency_hz == 2.0

            # Check parameters
            assert len(node.parameters) == 2
            param_names = [p.name for p in node.parameters]
            assert "my_param" in param_names
            assert "rate" in param_names

        finally:
            temp_path.unlink()

    def test_extract_advanced_node(self, advanced_node_code):
        """Test extracting services, actions, and multiple constructs."""
        extractor = ROS2NodeExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(advanced_node_code)
            temp_path = Path(f.name)

        try:
            nodes = extractor.extract_from_file(temp_path)

            assert len(nodes) == 1
            node = nodes[0]

            # Check multiple publishers
            assert len(node.publishers) == 2
            pub_topics = {p.topic for p in node.publishers}
            assert "/topic_a" in pub_topics
            assert "/topic_b" in pub_topics

            # Check subscriber
            assert len(node.subscribers) == 1
            assert node.subscribers[0].topic == "/topic_c"

            # Check service server
            assert len(node.services) == 1
            assert node.services[0].name == "/my_service"
            assert node.services[0].srv_type == "Trigger"

            # Check service client
            assert len(node.service_clients) == 1
            assert node.service_clients[0].name == "/other_service"

            # Check action client
            assert len(node.action_clients) == 1
            assert node.action_clients[0].name == "/navigate"
            assert "NavigateToPose" in node.action_clients[0].action_type

            # Check parameters with types
            assert len(node.parameters) == 3
            params = {p.name: p for p in node.parameters}
            assert params["speed"].default_value == 1.0
            assert params["enabled"].default_value == True
            assert params["name"].default_value == "robot"

        finally:
            temp_path.unlink()

    def test_node_to_dict(self, minimal_node_code):
        """Test converting node to dictionary."""
        extractor = ROS2NodeExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(minimal_node_code)
            temp_path = Path(f.name)

        try:
            nodes = extractor.extract_from_file(temp_path)
            node = nodes[0]

            d = node.to_dict()

            assert d["name"] == "minimal_node"
            assert d["class_name"] == "MinimalNode"
            assert len(d["publishers"]) == 1
            assert len(d["subscribers"]) == 1
            assert len(d["timers"]) == 1
            assert len(d["parameters"]) == 2

        finally:
            temp_path.unlink()

    def test_non_node_class(self):
        """Test that non-Node classes are not extracted."""
        code = '''
class NotANode:
    def __init__(self):
        pass

class HelperClass:
    pass
'''
        extractor = ROS2NodeExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            nodes = extractor.extract_from_file(temp_path)
            assert len(nodes) == 0

        finally:
            temp_path.unlink()


class TestTopicExtractor:
    """Tests for TopicExtractor."""

    def test_extractor_init(self):
        """Test topic extractor initialization."""
        extractor = TopicExtractor()
        assert extractor is not None
        assert len(extractor.topics) == 0

    def test_add_single_node(self):
        """Test adding a single node."""
        extractor = TopicExtractor()

        node = ROS2NodeInfo(
            name="test_node",
            class_name="TestNode",
            file_path=Path("/test.py"),
            line_number=1,
            end_line=100,
        )
        node.publishers.append(PublisherInfo(topic="/output", msg_type="String", qos=10))
        node.subscribers.append(SubscriberInfo(topic="/input", msg_type="String",
                                               callback="cb", qos=10))

        extractor.add_node(node)
        result = extractor.build()

        assert result.nodes_processed == 1
        assert len(result.topics) == 2

        output_topic = result.topics["/output"]
        assert output_topic.has_publisher
        assert not output_topic.has_subscriber
        assert "test_node" in output_topic.publishers

        input_topic = result.topics["/input"]
        assert not input_topic.has_publisher
        assert input_topic.has_subscriber
        assert "test_node" in input_topic.subscribers

    def test_connected_topics(self):
        """Test detecting connected topics."""
        extractor = TopicExtractor()

        # Node 1 publishes
        node1 = ROS2NodeInfo(
            name="publisher_node",
            class_name="PubNode",
            file_path=Path("/pub.py"),
            line_number=1,
            end_line=50,
        )
        node1.publishers.append(PublisherInfo(topic="/shared", msg_type="String", qos=10))

        # Node 2 subscribes
        node2 = ROS2NodeInfo(
            name="subscriber_node",
            class_name="SubNode",
            file_path=Path("/sub.py"),
            line_number=1,
            end_line=50,
        )
        node2.subscribers.append(SubscriberInfo(topic="/shared", msg_type="String",
                                                callback="cb", qos=10))

        extractor.add_nodes([node1, node2])
        result = extractor.build()

        connected = result.get_connected_topics()
        assert len(connected) == 1
        assert connected[0].name == "/shared"
        assert "publisher_node" in connected[0].publishers
        assert "subscriber_node" in connected[0].subscribers

    def test_unconnected_topics(self):
        """Test detecting unconnected topics."""
        extractor = TopicExtractor()

        node = ROS2NodeInfo(
            name="lonely_node",
            class_name="LonelyNode",
            file_path=Path("/lonely.py"),
            line_number=1,
            end_line=50,
        )
        node.publishers.append(PublisherInfo(topic="/pub_only", msg_type="String", qos=10))
        node.subscribers.append(SubscriberInfo(topic="/sub_only", msg_type="String",
                                               callback="cb", qos=10))

        extractor.add_node(node)
        result = extractor.build()

        pub_only = result.get_publish_only_topics()
        sub_only = result.get_subscribe_only_topics()

        assert len(pub_only) == 1
        assert pub_only[0].name == "/pub_only"

        assert len(sub_only) == 1
        assert sub_only[0].name == "/sub_only"

    def test_adjacency_list(self):
        """Test building adjacency list."""
        extractor = TopicExtractor()

        # Create a chain: node1 -> topic -> node2
        node1 = ROS2NodeInfo(
            name="source",
            class_name="Source",
            file_path=Path("/source.py"),
            line_number=1,
            end_line=50,
        )
        node1.publishers.append(PublisherInfo(topic="/data", msg_type="String", qos=10))

        node2 = ROS2NodeInfo(
            name="sink",
            class_name="Sink",
            file_path=Path("/sink.py"),
            line_number=1,
            end_line=50,
        )
        node2.subscribers.append(SubscriberInfo(topic="/data", msg_type="String",
                                                callback="cb", qos=10))

        extractor.add_nodes([node1, node2])

        adjacency = extractor.build_adjacency_list()

        assert "source" in adjacency
        assert "sink" in adjacency["source"]

    def test_summary(self):
        """Test summary generation."""
        extractor = TopicExtractor()

        node = ROS2NodeInfo(
            name="test",
            class_name="Test",
            file_path=Path("/test.py"),
            line_number=1,
            end_line=50,
        )
        node.publishers.append(PublisherInfo(topic="/a", msg_type="String", qos=10))
        node.subscribers.append(SubscriberInfo(topic="/b", msg_type="String",
                                               callback="cb", qos=10))

        extractor.add_node(node)
        result = extractor.build()

        summary = result.summary()

        assert summary["total_topics"] == 2
        assert summary["connected_topics"] == 0
        assert summary["publish_only"] == 1
        assert summary["subscribe_only"] == 1
        assert summary["nodes_processed"] == 1
