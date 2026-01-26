"""Tests for ROS2 Launch File and Parameter Analysis."""

import pytest
from pathlib import Path
import tempfile

from robomind.ros2.launch_analyzer import (
    LaunchFileAnalyzer,
    LaunchFileInfo,
    LaunchNode,
    LaunchArgument,
    LaunchTopology,
)
from robomind.ros2.param_extractor import (
    ParameterExtractor,
    ParameterFileInfo,
    ParameterValue,
    NodeParameters,
    ConfigScanner,
)


@pytest.fixture
def sample_launch_path():
    """Path to sample launch file fixture."""
    return Path(__file__).parent / "fixtures" / "sample_launch.py"


@pytest.fixture
def sample_config_path():
    """Path to sample config file fixture."""
    return Path(__file__).parent / "fixtures" / "sample_config.yaml"


@pytest.fixture
def minimal_launch_code():
    """Minimal launch file code for testing."""
    return '''
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim', default_value='false', description='Use sim'),
        Node(
            package='test_pkg',
            executable='test_node',
            name='my_test_node',
            parameters=[{'rate': 10.0}],
            output='screen',
        ),
    ])
'''


@pytest.fixture
def minimal_config_yaml():
    """Minimal config YAML for testing."""
    return '''
test_node:
  ros__parameters:
    rate: 10.0
    enabled: true
    name: "test"
'''


class TestLaunchFileAnalyzer:
    """Tests for LaunchFileAnalyzer."""

    def test_analyzer_init(self):
        """Test analyzer initialization."""
        analyzer = LaunchFileAnalyzer()
        assert analyzer is not None

    def test_analyze_sample_launch(self, sample_launch_path):
        """Test analyzing sample launch file."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        assert info is not None
        assert info.file_path == sample_launch_path
        assert len(info.parse_errors) == 0

        # Check arguments
        assert len(info.arguments) >= 3
        arg_names = [a.name for a in info.arguments]
        assert "use_sim_time" in arg_names
        assert "enable_voice" in arg_names
        assert "robot_name" in arg_names

        # Check nodes
        assert len(info.nodes) >= 4  # odometry, motor, rplidar, voice

        # Check containers
        assert len(info.containers) >= 1
        container = info.containers[0]
        assert container.name == "hardware_container"
        assert len(container.nodes) >= 1

        # Check processes
        assert len(info.processes) >= 1

    def test_analyze_minimal_launch(self, minimal_launch_code):
        """Test analyzing minimal launch file."""
        analyzer = LaunchFileAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write(minimal_launch_code)
            temp_path = Path(f.name)

        try:
            info = analyzer.analyze_file(temp_path)

            assert info is not None
            assert len(info.parse_errors) == 0

            # Check argument
            assert len(info.arguments) == 1
            assert info.arguments[0].name == "use_sim"
            assert info.arguments[0].default_value == "false"

            # Check node
            assert len(info.nodes) == 1
            node = info.nodes[0]
            assert node.package == "test_pkg"
            assert node.executable == "test_node"
            assert node.name == "my_test_node"
            assert len(node.parameters) >= 1

        finally:
            temp_path.unlink()

    def test_launch_arguments_extraction(self, sample_launch_path):
        """Test extracting launch arguments with defaults and descriptions."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        # Find use_sim_time argument
        use_sim_arg = None
        for arg in info.arguments:
            if arg.name == "use_sim_time":
                use_sim_arg = arg
                break

        assert use_sim_arg is not None
        assert use_sim_arg.default_value == "false"
        assert "simulation" in use_sim_arg.description.lower()

    def test_node_parameters_extraction(self, sample_launch_path):
        """Test extracting node parameters."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        # Find motor_controller node
        motor_node = None
        for node in info.nodes:
            if node.name == "motor_controller":
                motor_node = node
                break

        assert motor_node is not None
        assert motor_node.package == "test_pkg"

        # Check parameters
        param_names = [p.name for p in motor_node.parameters]
        assert "max_speed" in param_names
        assert "wheel_base" in param_names

    def test_node_remappings_extraction(self, sample_launch_path):
        """Test extracting node remappings."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        # Find motor_controller node with remappings
        motor_node = None
        for node in info.nodes:
            if node.name == "motor_controller":
                motor_node = node
                break

        assert motor_node is not None
        assert len(motor_node.remappings) >= 2

        remap_from = [r.from_topic for r in motor_node.remappings]
        assert "/cmd_vel" in remap_from

    def test_timer_action_delay(self, sample_launch_path):
        """Test that TimerAction delays are captured."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        # Find rplidar_node which should have 3.0s delay
        rplidar_node = None
        for node in info.nodes:
            if node.name == "rplidar_node":
                rplidar_node = node
                break

        assert rplidar_node is not None
        assert rplidar_node.delay == 3.0

    def test_conditional_node(self, sample_launch_path):
        """Test that conditional nodes are detected."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        # Find voice_control node which has IfCondition
        voice_node = None
        for node in info.nodes:
            if node.name == "voice_control":
                voice_node = node
                break

        assert voice_node is not None
        assert voice_node.condition is not None
        assert "IfCondition" in voice_node.condition

    def test_composable_container(self, sample_launch_path):
        """Test composable node container extraction."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        # Check container
        assert len(info.containers) >= 1
        container = info.containers[0]

        assert container.name == "hardware_container"
        assert container.package == "rclcpp_components"
        assert len(container.nodes) >= 1

        # Check composable node
        comp_node = container.nodes[0]
        assert comp_node.is_composable
        assert comp_node.container_name == "hardware_container"
        assert comp_node.package == "robot_state_publisher"

    def test_execute_process(self, sample_launch_path):
        """Test ExecuteProcess extraction."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        assert len(info.processes) >= 1
        process = info.processes[0]

        assert "ros2" in process.cmd
        assert "launch" in process.cmd
        assert process.delay == 10.0

    def test_launch_sequence(self, sample_launch_path):
        """Test launch sequence ordering by delay."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        sequence = info.get_launch_sequence()

        # Should be sorted by delay
        delays = [item["delay"] for item in sequence]
        assert delays == sorted(delays)

    def test_summary(self, sample_launch_path):
        """Test summary generation."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        summary = info.summary()

        assert "file" in summary
        assert "arguments" in summary
        assert "nodes" in summary
        assert "max_delay" in summary
        assert summary["max_delay"] >= 10.0  # Nav2 launch is at 10s

    def test_to_dict(self, sample_launch_path):
        """Test conversion to dictionary."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        d = info.to_dict()

        assert "file_path" in d
        assert "arguments" in d
        assert "nodes" in d
        assert "containers" in d
        assert "processes" in d
        assert "total_nodes" in d

    def test_invalid_launch_file(self):
        """Test handling of invalid launch file."""
        analyzer = LaunchFileAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("this is not valid python {{{")
            temp_path = Path(f.name)

        try:
            info = analyzer.analyze_file(temp_path)
            assert len(info.parse_errors) > 0
        finally:
            temp_path.unlink()


class TestParameterExtractor:
    """Tests for ParameterExtractor."""

    def test_extractor_init(self):
        """Test extractor initialization."""
        extractor = ParameterExtractor()
        assert extractor is not None

    def test_extract_sample_config(self, sample_config_path):
        """Test extracting from sample config file."""
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(sample_config_path)

        assert info is not None
        assert len(info.parse_errors) == 0
        assert len(info.nodes) >= 3  # slam_toolbox, motor_controller, voice_system

    def test_extract_minimal_config(self, minimal_config_yaml):
        """Test extracting from minimal config."""
        extractor = ParameterExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(minimal_config_yaml)
            temp_path = Path(f.name)

        try:
            info = extractor.extract_from_file(temp_path)

            assert info is not None
            assert len(info.parse_errors) == 0
            assert len(info.nodes) == 1

            node = info.nodes[0]
            assert node.node_name == "test_node"
            assert len(node.parameters) == 3

            # Check parameter values
            rate_param = node.get_parameter("rate")
            assert rate_param is not None
            assert rate_param.value == 10.0
            assert rate_param.value_type == "float"

            enabled_param = node.get_parameter("enabled")
            assert enabled_param is not None
            assert enabled_param.value is True
            assert enabled_param.value_type == "bool"

        finally:
            temp_path.unlink()

    def test_nested_parameters(self, sample_config_path):
        """Test extracting nested parameters."""
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(sample_config_path)

        # Find slam_toolbox node
        slam_node = info.get_node("slam_toolbox")
        assert slam_node is not None

        # Check for nested correlation parameters
        nested_params = slam_node.get_parameters_by_prefix("correlation")
        assert len(nested_params) == 0  # They should be flattened

        # Check flattened path
        all_paths = [p.path for p in slam_node.parameters]
        assert "correlation.search_space_dimension" in all_paths

    def test_parameter_types(self, sample_config_path):
        """Test parameter type detection."""
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(sample_config_path)

        motor_node = info.get_node("motor_controller")
        assert motor_node is not None

        # Check integer
        max_speed = motor_node.get_parameter("max_speed")
        assert max_speed.value_type == "int"

        # Check float
        wheel_base = motor_node.get_parameter("wheel_base")
        assert wheel_base.value_type == "float"

        # Check bool
        enable_pid = motor_node.get_parameter("enable_pid")
        assert enable_pid.value_type == "bool"

        # Check list
        pid_gains = motor_node.get_parameter("pid_gains")
        assert pid_gains.value_type == "list"

    def test_string_parameters(self, sample_config_path):
        """Test string parameter extraction."""
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(sample_config_path)

        voice_node = info.get_node("voice_system")
        assert voice_node is not None

        wake_word = voice_node.get_parameter("wake_word")
        assert wake_word is not None
        assert wake_word.value == "hey robot"
        assert wake_word.value_type == "string"

    def test_summary(self, sample_config_path):
        """Test summary generation."""
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(sample_config_path)

        summary = info.summary()

        assert "file" in summary
        assert "nodes" in summary
        assert "total_parameters" in summary
        assert summary["nodes"] >= 3

    def test_to_dict(self, sample_config_path):
        """Test conversion to dictionary."""
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(sample_config_path)

        d = info.to_dict()

        assert "file_path" in d
        assert "nodes" in d
        assert "summary" in d

    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        extractor = ParameterExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: {{{}}")
            temp_path = Path(f.name)

        try:
            info = extractor.extract_from_file(temp_path)
            assert len(info.parse_errors) > 0
        finally:
            temp_path.unlink()

    def test_extract_from_dict(self):
        """Test extracting from a dictionary."""
        extractor = ParameterExtractor()

        data = {
            "my_node": {
                "ros__parameters": {
                    "param1": 10,
                    "param2": "value",
                }
            }
        }

        info = extractor.extract_from_dict(data, "test_source")

        assert len(info.nodes) == 1
        assert info.nodes[0].node_name == "my_node"
        assert len(info.nodes[0].parameters) == 2


class TestLaunchTopology:
    """Tests for LaunchTopology."""

    def test_topology_init(self):
        """Test topology initialization."""
        topology = LaunchTopology()
        assert topology is not None
        assert len(topology.launch_files) == 0

    def test_topology_with_files(self, sample_launch_path):
        """Test topology with launch files."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        topology = LaunchTopology(launch_files=[info])

        assert len(topology.all_nodes) > 0
        assert len(topology.all_arguments) > 0
        assert len(topology.get_packages()) > 0

    def test_topology_summary(self, sample_launch_path):
        """Test topology summary."""
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(sample_launch_path)

        topology = LaunchTopology(launch_files=[info])
        summary = topology.summary()

        assert "launch_files" in summary
        assert "total_nodes" in summary
        assert "unique_packages" in summary
