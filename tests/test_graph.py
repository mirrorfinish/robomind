"""Tests for the System Graph module."""

import pytest
from pathlib import Path

from robomind.core.graph import (
    SystemGraph,
    GraphBuilder,
    GraphNode,
    GraphEdge,
    ComponentType,
    EdgeType,
    build_system_graph,
)
from robomind.ros2.node_extractor import (
    ROS2NodeInfo,
    PublisherInfo,
    SubscriberInfo,
    TimerInfo,
    ServiceInfo,
    ServiceClientInfo,
    ParameterInfo,
)
from robomind.ros2.topic_extractor import TopicExtractor


@pytest.fixture
def sample_ros2_node():
    """Create a sample ROS2 node for testing."""
    return ROS2NodeInfo(
        name="test_node",
        class_name="TestNode",
        file_path=Path("/test/test_node.py"),
        line_number=10,
        end_line=100,
        package_name="test_pkg",
        publishers=[
            PublisherInfo(topic="/cmd_vel", msg_type="Twist", qos=10),
            PublisherInfo(topic="/status", msg_type="String", qos=10),
        ],
        subscribers=[
            SubscriberInfo(topic="/scan", msg_type="LaserScan", callback="scan_callback", qos=10),
        ],
        timers=[
            TimerInfo(period=0.1, callback="control_loop"),
        ],
        parameters=[
            ParameterInfo(name="max_speed", default_value=1.5, param_type="float"),
            ParameterInfo(name="enabled", default_value=True, param_type="bool"),
        ],
    )


@pytest.fixture
def sample_ros2_node_subscriber():
    """Create another ROS2 node that subscribes to the first node's topics."""
    return ROS2NodeInfo(
        name="motor_node",
        class_name="MotorNode",
        file_path=Path("/test/motor_node.py"),
        line_number=10,
        end_line=50,
        package_name="motor_pkg",
        subscribers=[
            SubscriberInfo(topic="/cmd_vel", msg_type="Twist", callback="cmd_callback", qos=10),
        ],
        publishers=[
            PublisherInfo(topic="/odom", msg_type="Odometry", qos=10),
        ],
        services=[
            ServiceInfo(name="/reset_motor", srv_type="Empty", callback="reset_callback"),
        ],
    )


@pytest.fixture
def sample_ros2_node_with_service_client():
    """Create a node with service client."""
    return ROS2NodeInfo(
        name="controller_node",
        class_name="ControllerNode",
        file_path=Path("/test/controller_node.py"),
        line_number=10,
        end_line=50,
        package_name="controller_pkg",
        service_clients=[
            ServiceClientInfo(name="/reset_motor", srv_type="Empty"),
        ],
        subscribers=[
            SubscriberInfo(topic="/odom", msg_type="Odometry", callback="odom_callback", qos=10),
        ],
    )


class TestSystemGraph:
    """Tests for the SystemGraph class."""

    def test_graph_init(self):
        """Test graph initialization."""
        graph = SystemGraph()
        assert graph is not None
        assert len(graph) == 0

    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = SystemGraph()

        node = GraphNode(
            id="node:test",
            name="test",
            component_type=ComponentType.ROS2_NODE,
            file_path=Path("/test.py"),
        )
        graph.add_node(node)

        assert len(graph) == 1
        assert graph.get_node("node:test") is not None
        assert graph.get_node("node:test").name == "test"

    def test_add_edge(self):
        """Test adding an edge to the graph."""
        graph = SystemGraph()

        node1 = GraphNode(id="node:a", name="a", component_type=ComponentType.ROS2_NODE)
        node2 = GraphNode(id="node:b", name="b", component_type=ComponentType.ROS2_NODE)
        graph.add_node(node1)
        graph.add_node(node2)

        edge = GraphEdge(
            source="node:a",
            target="node:b",
            edge_type=EdgeType.PUBLISHES,
        )
        graph.add_edge(edge)

        assert len(graph.get_edges()) == 1

    def test_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        graph = SystemGraph()

        graph.add_node(GraphNode(id="n1", name="n1", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="t1", name="t1", component_type=ComponentType.TOPIC))
        graph.add_node(GraphNode(id="n2", name="n2", component_type=ComponentType.ROS2_NODE))

        ros2_nodes = graph.get_nodes_by_type(ComponentType.ROS2_NODE)
        topics = graph.get_nodes_by_type(ComponentType.TOPIC)

        assert len(ros2_nodes) == 2
        assert len(topics) == 1

    def test_get_nodes_by_hardware(self):
        """Test filtering nodes by hardware target."""
        graph = SystemGraph()

        n1 = GraphNode(id="n1", name="n1", component_type=ComponentType.ROS2_NODE, hardware_target="jetson1")
        n2 = GraphNode(id="n2", name="n2", component_type=ComponentType.ROS2_NODE, hardware_target="jetson2")
        n3 = GraphNode(id="n3", name="n3", component_type=ComponentType.ROS2_NODE, hardware_target="jetson1")

        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        jetson1_nodes = graph.get_nodes_by_hardware("jetson1")
        assert len(jetson1_nodes) == 2

    def test_find_cycles(self):
        """Test cycle detection."""
        graph = SystemGraph()

        # Create a cycle: a -> b -> c -> a
        for name in ["a", "b", "c"]:
            graph.add_node(GraphNode(id=name, name=name, component_type=ComponentType.ROS2_NODE))

        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="c", target="a", edge_type=EdgeType.PUBLISHES))

        cycles = graph.find_cycles()
        assert len(cycles) >= 1

    def test_no_cycles_dag(self):
        """Test that a DAG has no cycles."""
        graph = SystemGraph()

        for name in ["a", "b", "c"]:
            graph.add_node(GraphNode(id=name, name=name, component_type=ComponentType.ROS2_NODE))

        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.PUBLISHES))

        cycles = graph.find_cycles()
        assert len(cycles) == 0

        stats = graph.stats()
        assert stats["is_dag"] is True

    def test_calculate_centrality(self):
        """Test centrality calculation."""
        graph = SystemGraph()

        # Hub and spoke: central node connected to all others
        graph.add_node(GraphNode(id="hub", name="hub", component_type=ComponentType.ROS2_NODE))
        for i in range(4):
            graph.add_node(GraphNode(id=f"spoke{i}", name=f"spoke{i}", component_type=ComponentType.ROS2_NODE))
            graph.add_edge(GraphEdge(source="hub", target=f"spoke{i}", edge_type=EdgeType.PUBLISHES))

        centrality = graph.calculate_centrality("degree")
        assert "hub" in centrality
        assert centrality["hub"] > 0

    def test_get_node_dependencies(self):
        """Test getting node dependencies."""
        graph = SystemGraph()

        for name in ["a", "b", "c"]:
            graph.add_node(GraphNode(id=name, name=name, component_type=ComponentType.ROS2_NODE))

        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.PUBLISHES))

        deps = graph.get_node_dependencies("b")
        assert "a" in deps["upstream"]
        assert "c" in deps["downstream"]

    def test_topological_order(self):
        """Test topological ordering."""
        graph = SystemGraph()

        for name in ["a", "b", "c"]:
            graph.add_node(GraphNode(id=name, name=name, component_type=ComponentType.ROS2_NODE))

        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.PUBLISHES))

        order = graph.get_topological_order()
        assert order is not None
        assert order.index("a") < order.index("b") < order.index("c")

    def test_stats(self):
        """Test graph statistics."""
        graph = SystemGraph()

        graph.add_node(GraphNode(id="n1", name="n1", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="t1", name="t1", component_type=ComponentType.TOPIC))
        graph.add_edge(GraphEdge(source="n1", target="t1", edge_type=EdgeType.PUBLISHES))

        stats = graph.stats()

        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert ComponentType.ROS2_NODE.name in stats["node_types"]
        assert ComponentType.TOPIC.name in stats["node_types"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        graph = SystemGraph()

        graph.add_node(GraphNode(id="n1", name="n1", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="t1", name="t1", component_type=ComponentType.TOPIC))
        graph.add_edge(GraphEdge(source="n1", target="t1", edge_type=EdgeType.PUBLISHES))

        d = graph.to_dict()

        assert "stats" in d
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1


class TestGraphBuilder:
    """Tests for the GraphBuilder class."""

    def test_builder_init(self):
        """Test builder initialization."""
        builder = GraphBuilder()
        assert builder is not None
        assert builder.graph is not None

    def test_add_ros2_node(self, sample_ros2_node):
        """Test adding a ROS2 node."""
        import hashlib
        builder = GraphBuilder()
        builder.add_ros2_node(sample_ros2_node)

        graph = builder.build()

        # Should have: 1 node + 3 topics + 2 parameters = 6 nodes
        assert len(graph) >= 4  # At minimum: node + 3 topics

        # Check node exists (ID format is node:name:path_hash)
        path_hash = hashlib.md5(str(sample_ros2_node.file_path).encode()).hexdigest()[:8]
        node_id = f"node:test_node:{path_hash}"
        node = graph.get_node(node_id)
        assert node is not None
        assert node.name == "test_node"
        assert node.component_type == ComponentType.ROS2_NODE

        # Check topics exist
        topic = graph.get_node("topic:/cmd_vel")
        assert topic is not None
        assert topic.component_type == ComponentType.TOPIC

    def test_add_multiple_nodes(self, sample_ros2_node, sample_ros2_node_subscriber):
        """Test adding multiple connected nodes."""
        import hashlib
        builder = GraphBuilder()
        builder.add_ros2_node(sample_ros2_node)
        builder.add_ros2_node(sample_ros2_node_subscriber)

        graph = builder.build()

        # Both nodes should exist (ID format is node:name:path_hash)
        hash1 = hashlib.md5(str(sample_ros2_node.file_path).encode()).hexdigest()[:8]
        hash2 = hashlib.md5(str(sample_ros2_node_subscriber.file_path).encode()).hexdigest()[:8]
        assert graph.get_node(f"node:test_node:{hash1}") is not None
        assert graph.get_node(f"node:motor_node:{hash2}") is not None

        # /cmd_vel topic should have both publisher and subscriber edges
        edges = graph.get_edges()
        cmd_vel_edges = [e for e in edges if "cmd_vel" in e.source or "cmd_vel" in e.target]
        assert len(cmd_vel_edges) >= 2  # Publish and subscribe

    def test_add_service_connections(self, sample_ros2_node_subscriber, sample_ros2_node_with_service_client):
        """Test service connections."""
        builder = GraphBuilder()
        builder.add_ros2_node(sample_ros2_node_subscriber)
        builder.add_ros2_node(sample_ros2_node_with_service_client)

        graph = builder.build()

        # Service should exist
        service = graph.get_node("service:/reset_motor")
        assert service is not None
        assert service.component_type == ComponentType.SERVICE

    def test_build_system_graph_function(self, sample_ros2_node, sample_ros2_node_subscriber):
        """Test the convenience build_system_graph function."""
        nodes = [sample_ros2_node, sample_ros2_node_subscriber]

        topic_extractor = TopicExtractor()
        topic_extractor.add_nodes(nodes)
        topic_graph = topic_extractor.build()

        graph = build_system_graph(nodes, topic_graph)

        assert len(graph) > 0
        assert len(graph.get_nodes_by_type(ComponentType.ROS2_NODE)) == 2


class TestGraphExport:
    """Tests for graph export functionality."""

    def test_export_adjacency_list(self, sample_ros2_node, sample_ros2_node_subscriber):
        """Test adjacency list export."""
        builder = GraphBuilder()
        builder.add_ros2_node(sample_ros2_node)
        builder.add_ros2_node(sample_ros2_node_subscriber)

        graph = builder.build()
        adj_list = graph.export_adjacency_list()

        assert isinstance(adj_list, dict)
        # Nodes should be in adjacency list
        assert "node:test_node" in adj_list or len(adj_list) > 0

    def test_export_graphml(self, sample_ros2_node, tmp_path):
        """Test GraphML export."""
        builder = GraphBuilder()
        builder.add_ros2_node(sample_ros2_node)

        graph = builder.build()
        output_path = tmp_path / "test_graph.graphml"

        graph.export_graphml(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_gexf(self, sample_ros2_node, tmp_path):
        """Test GEXF export."""
        builder = GraphBuilder()
        builder.add_ros2_node(sample_ros2_node)

        graph = builder.build()
        output_path = tmp_path / "test_graph.gexf"

        graph.export_gexf(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestGraphAnalysis:
    """Tests for advanced graph analysis."""

    def test_get_shortest_path(self):
        """Test shortest path finding."""
        graph = SystemGraph()

        for name in ["a", "b", "c", "d"]:
            graph.add_node(GraphNode(id=name, name=name, component_type=ComponentType.ROS2_NODE))

        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="b", target="c", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="c", target="d", edge_type=EdgeType.PUBLISHES))

        path = graph.get_shortest_path("a", "d")
        assert path is not None
        assert path == ["a", "b", "c", "d"]

    def test_get_all_paths(self):
        """Test finding all paths."""
        graph = SystemGraph()

        for name in ["a", "b", "c", "d"]:
            graph.add_node(GraphNode(id=name, name=name, component_type=ComponentType.ROS2_NODE))

        # Create two paths from a to d
        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="b", target="d", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="a", target="c", edge_type=EdgeType.PUBLISHES))
        graph.add_edge(GraphEdge(source="c", target="d", edge_type=EdgeType.PUBLISHES))

        paths = graph.get_all_paths("a", "d")
        assert len(paths) == 2

    def test_connected_components(self):
        """Test connected component detection."""
        graph = SystemGraph()

        # Two disconnected components
        graph.add_node(GraphNode(id="a", name="a", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="b", name="b", component_type=ComponentType.ROS2_NODE))
        graph.add_edge(GraphEdge(source="a", target="b", edge_type=EdgeType.PUBLISHES))

        graph.add_node(GraphNode(id="c", name="c", component_type=ComponentType.ROS2_NODE))
        graph.add_node(GraphNode(id="d", name="d", component_type=ComponentType.ROS2_NODE))
        graph.add_edge(GraphEdge(source="c", target="d", edge_type=EdgeType.PUBLISHES))

        components = graph.get_weakly_connected_components()
        assert len(components) == 2
