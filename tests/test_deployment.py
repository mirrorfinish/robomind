"""
Tests for deployment module.

Tests deployment manifest parsing, launch file tracing, and systemd discovery.
"""

import pytest
from pathlib import Path
import tempfile
import os

from robomind.deployment.manifest import (
    DeploymentManifest,
    JetsonConfig,
    load_deployment_manifest,
    create_example_manifest,
)
from robomind.deployment.launch_tracer import (
    LaunchTracer,
    TracedNode,
    LaunchTrace,
    trace_launch_file,
)
from robomind.deployment.systemd_discovery import (
    SystemdService,
    SystemdDiscovery,
)


class TestJetsonConfig:
    """Test JetsonConfig dataclass."""

    def test_from_dict_basic(self):
        """Test basic creation from dictionary."""
        data = {
            "hostname": "nav.local",
            "ros_domain_id": 0,
            "systemd_services": ["nav-service"],
            "launch_files": ["robot.launch.py"],
        }

        config = JetsonConfig.from_dict("nav_jetson", data)

        assert config.name == "nav_jetson"
        assert config.hostname == "nav.local"
        assert config.ros_domain_id == 0
        assert config.systemd_services == ["nav-service"]
        assert config.launch_files == ["robot.launch.py"]
        assert config.http_only is False

    def test_from_dict_http_only(self):
        """Test HTTP-only configuration."""
        data = {
            "hostname": "ai.local",
            "http_only": True,
            "http_ports": [8080, 8081],
        }

        config = JetsonConfig.from_dict("ai_jetson", data)

        assert config.http_only is True
        assert config.http_ports == [8080, 8081]
        assert config.ros_domain_id is None

    def test_has_ros2(self):
        """Test ROS2 detection."""
        ros2_config = JetsonConfig.from_dict("nav", {
            "hostname": "nav.local",
            "ros_domain_id": 0,
            "launch_files": ["robot.launch.py"],
        })
        assert ros2_config.has_ros2() is True

        http_config = JetsonConfig.from_dict("ai", {
            "hostname": "ai.local",
            "http_only": True,
        })
        assert http_config.has_ros2() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = JetsonConfig(
            name="nav",
            hostname="nav.local",
            ros_domain_id=0,
            systemd_services=["nav-service"],
        )

        result = config.to_dict()

        assert result["name"] == "nav"
        assert result["hostname"] == "nav.local"
        assert result["ros_domain_id"] == 0


class TestDeploymentManifest:
    """Test DeploymentManifest class."""

    def test_from_dict_basic(self):
        """Test basic manifest creation."""
        data = {
            "default_ros_domain_id": 0,
            "jetsons": {
                "nav": {"hostname": "nav.local", "ros_domain_id": 0},
                "ai": {"hostname": "ai.local", "http_only": True},
            }
        }

        manifest = DeploymentManifest.from_dict(data)

        assert len(manifest.jetsons) == 2
        assert "nav" in manifest.jetsons
        assert "ai" in manifest.jetsons

    def test_aggregate_deployed_items(self):
        """Test aggregation of deployed items."""
        data = {
            "jetsons": {
                "nav": {
                    "hostname": "nav.local",
                    "systemd_services": ["svc1", "svc2"],
                    "launch_files": ["launch1.py"],
                    "packages": ["pkg1"],
                },
                "ai": {
                    "hostname": "ai.local",
                    "systemd_services": ["svc3"],
                    "packages": ["pkg2", "pkg1"],  # Duplicate pkg1
                }
            }
        }

        manifest = DeploymentManifest.from_dict(data)

        assert manifest.deployed_services == ["svc1", "svc2", "svc3"]
        assert manifest.deployed_launch_files == ["launch1.py"]
        assert manifest.deployed_packages == ["pkg1", "pkg2"]  # Deduplicated

    def test_get_host_for_service(self):
        """Test finding host for a service."""
        data = {
            "jetsons": {
                "nav": {
                    "hostname": "nav.local",
                    "systemd_services": ["nav-service"],
                },
                "ai": {
                    "hostname": "ai.local",
                    "systemd_services": ["ai-service"],
                }
            }
        }

        manifest = DeploymentManifest.from_dict(data)

        assert manifest.get_host_for_service("nav-service") == "nav"
        assert manifest.get_host_for_service("ai-service") == "ai"
        assert manifest.get_host_for_service("unknown") is None

    def test_get_ros2_hosts(self):
        """Test getting ROS2 hosts."""
        data = {
            "jetsons": {
                "nav": {"hostname": "nav.local", "ros_domain_id": 0},
                "ai": {"hostname": "ai.local", "http_only": True},
                "vision": {"hostname": "vision.local", "http_only": True},
            }
        }

        manifest = DeploymentManifest.from_dict(data)

        ros2_hosts = manifest.get_ros2_hosts()
        assert len(ros2_hosts) == 1
        assert ros2_hosts[0].name == "nav"

    def test_is_launch_file_deployed(self):
        """Test checking if launch file is deployed."""
        data = {
            "jetsons": {
                "nav": {
                    "hostname": "nav.local",
                    "launch_files": ["robot.launch.py", "full/path/to/bringup.launch.py"],
                }
            }
        }

        manifest = DeploymentManifest.from_dict(data)

        assert manifest.is_launch_file_deployed("robot.launch.py") is True
        assert manifest.is_launch_file_deployed("bringup.launch.py") is True
        assert manifest.is_launch_file_deployed("unknown.launch.py") is False

    def test_summary(self):
        """Test manifest summary."""
        data = {
            "jetsons": {
                "nav": {"hostname": "nav.local", "ros_domain_id": 0},
                "ai": {"hostname": "ai.local", "http_only": True},
            }
        }

        manifest = DeploymentManifest.from_dict(data)
        summary = manifest.summary()

        assert summary["total_hosts"] == 2
        assert summary["ros2_hosts"] == 1
        assert summary["http_only_hosts"] == 1


class TestLoadDeploymentManifest:
    """Test loading manifest from file."""

    def test_load_valid_manifest(self):
        """Test loading a valid manifest file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
jetsons:
  nav_jetson:
    hostname: nav.local
    ros_domain_id: 0
    systemd_services:
      - betaray-navigation
    launch_files:
      - robot.launch.py
""")
            f.flush()

            manifest = load_deployment_manifest(Path(f.name))

            assert len(manifest.jetsons) == 1
            assert "nav_jetson" in manifest.jetsons
            assert manifest.jetsons["nav_jetson"].hostname == "nav.local"

        os.unlink(f.name)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_deployment_manifest(Path("/nonexistent/manifest.yaml"))


class TestCreateExampleManifest:
    """Test example manifest generation."""

    def test_example_is_valid_yaml(self):
        """Test that example manifest is valid YAML."""
        import yaml

        example = create_example_manifest()
        data = yaml.safe_load(example)

        assert "jetsons" in data
        assert len(data["jetsons"]) > 0


class TestTracedNode:
    """Test TracedNode dataclass."""

    def test_basic_creation(self):
        """Test basic node creation."""
        node = TracedNode(
            name="motor_controller",
            package="betaray_control",
            executable="motor_controller_node",
        )

        assert node.name == "motor_controller"
        assert node.package == "betaray_control"
        assert node.is_conditional() is False

    def test_conditional_node(self):
        """Test conditional node."""
        node = TracedNode(
            name="debug_node",
            package="betaray_debug",
            executable="debug",
            condition="IfCondition(...)",
        )

        assert node.is_conditional() is True

    def test_get_full_name(self):
        """Test getting full name with namespace."""
        node_no_ns = TracedNode(
            name="motor",
            package="pkg",
            executable="exec",
        )
        assert node_no_ns.get_full_name() == "motor"

        node_with_ns = TracedNode(
            name="motor",
            package="pkg",
            executable="exec",
            namespace="/robot",
        )
        assert node_with_ns.get_full_name() == "/robot/motor"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        node = TracedNode(
            name="motor",
            package="betaray",
            executable="motor_node",
            namespace="/robot",
            source_launch_file="robot.launch.py",
            line_number=42,
        )

        result = node.to_dict()

        assert result["name"] == "motor"
        assert result["package"] == "betaray"
        assert result["namespace"] == "/robot"
        assert result["line_number"] == 42


class TestLaunchTrace:
    """Test LaunchTrace class."""

    def test_basic_trace(self):
        """Test basic trace creation."""
        trace = LaunchTrace(root_launch_file=Path("robot.launch.py"))

        assert len(trace.nodes) == 0
        assert len(trace.errors) == 0

    def test_get_node_names(self):
        """Test getting node names."""
        trace = LaunchTrace(root_launch_file=Path("robot.launch.py"))
        trace.nodes = [
            TracedNode(name="motor", package="pkg", executable="exec"),
            TracedNode(name="sensor", package="pkg", executable="exec"),
        ]

        names = trace.get_node_names()

        assert names == ["motor", "sensor"]

    def test_get_unconditional_nodes(self):
        """Test filtering unconditional nodes."""
        trace = LaunchTrace(root_launch_file=Path("robot.launch.py"))
        trace.nodes = [
            TracedNode(name="motor", package="pkg", executable="exec"),
            TracedNode(name="debug", package="pkg", executable="exec", condition="If"),
        ]

        unconditional = trace.get_unconditional_nodes()

        assert len(unconditional) == 1
        assert unconditional[0].name == "motor"

    def test_summary(self):
        """Test trace summary."""
        trace = LaunchTrace(root_launch_file=Path("robot.launch.py"))
        trace.nodes = [
            TracedNode(name="motor", package="pkg1", executable="exec"),
            TracedNode(name="debug", package="pkg2", executable="exec", condition="If"),
        ]
        trace.packages_used = {"pkg1", "pkg2"}

        summary = trace.summary()

        assert summary["total_nodes"] == 2
        assert summary["unconditional_nodes"] == 1
        assert summary["conditional_nodes"] == 1
        assert summary["packages"] == 2


class TestLaunchTracer:
    """Test LaunchTracer class."""

    def test_trace_basic_launch_file(self):
        """Test tracing a basic launch file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='betaray_control',
            executable='motor_controller_node',
            name='motor_controller',
        ),
        Node(
            package='betaray_navigation',
            executable='navigation_node',
            name='navigation',
        ),
    ])
""")
            f.flush()

            tracer = LaunchTracer()
            trace = tracer.trace(Path(f.name))

            assert len(trace.nodes) == 2
            assert trace.nodes[0].name == "motor_controller"
            assert trace.nodes[0].package == "betaray_control"
            assert trace.nodes[1].name == "navigation"

        os.unlink(f.name)

    def test_trace_with_namespace(self):
        """Test tracing launch file with namespaces."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='betaray',
            executable='motor',
            name='motor_controller',
            namespace='robot',
        ),
    ])
""")
            f.flush()

            tracer = LaunchTracer()
            trace = tracer.trace(Path(f.name))

            assert len(trace.nodes) == 1
            assert trace.nodes[0].namespace == "robot"

        os.unlink(f.name)

    def test_trace_with_condition(self):
        """Test tracing launch file with conditional nodes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='betaray',
            executable='debug_node',
            name='debug',
            condition=IfCondition(LaunchConfiguration('debug')),
        ),
    ])
""")
            f.flush()

            tracer = LaunchTracer()
            trace = tracer.trace(Path(f.name))

            assert len(trace.nodes) == 1
            assert trace.nodes[0].is_conditional()

        os.unlink(f.name)

    def test_trace_composable_nodes(self):
        """Test tracing composable nodes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("""
from launch import LaunchDescription
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='betaray_perception',
                    plugin='betaray::LidarProcessor',
                    name='lidar_processor',
                ),
            ],
        ),
    ])
""")
            f.flush()

            tracer = LaunchTracer()
            trace = tracer.trace(Path(f.name))

            # Should find the composable node
            composable_nodes = [n for n in trace.nodes if "lidar" in n.name.lower()]
            assert len(composable_nodes) >= 1

        os.unlink(f.name)

    def test_trace_nonexistent_file(self):
        """Test tracing nonexistent file."""
        tracer = LaunchTracer()
        trace = tracer.trace(Path("/nonexistent/robot.launch.py"))

        assert len(trace.errors) > 0
        assert "not found" in trace.errors[0].lower()

    def test_trace_with_launch_arguments(self):
        """Test extracting launch arguments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('debug', default_value='false'),
        DeclareLaunchArgument('robot_name', default_value='betaray'),
        Node(package='pkg', executable='node', name='node'),
    ])
""")
            f.flush()

            tracer = LaunchTracer()
            trace = tracer.trace(Path(f.name))

            assert "debug" in trace.arguments
            assert "robot_name" in trace.arguments

        os.unlink(f.name)


class TestTraceLaunchFile:
    """Test convenience function."""

    def test_trace_launch_file_function(self):
        """Test the trace_launch_file convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".launch.py", delete=False) as f:
            f.write("""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='pkg', executable='node', name='test_node'),
    ])
""")
            f.flush()

            trace = trace_launch_file(Path(f.name))

            assert len(trace.nodes) == 1
            assert trace.nodes[0].name == "test_node"

        os.unlink(f.name)


class TestSystemdService:
    """Test SystemdService dataclass."""

    def test_basic_creation(self):
        """Test basic service creation."""
        service = SystemdService(
            name="betaray-navigation",
            exec_start="/usr/bin/ros2 launch betaray robot.launch.py",
        )

        assert service.name == "betaray-navigation"
        assert service.is_enabled is False
        assert service.is_active is False

    def test_uses_ros2_launch(self):
        """Test ROS2 launch detection."""
        service = SystemdService(
            name="betaray-navigation",
            launch_file="robot.launch.py",
        )

        assert service.uses_ros2_launch() is True

    def test_uses_python(self):
        """Test Python script detection."""
        service = SystemdService(
            name="ai-server",
            python_script="ai_server.py",
        )

        assert service.uses_python() is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        service = SystemdService(
            name="test-service",
            exec_start="/usr/bin/test",
            is_enabled=True,
            is_active=True,
        )

        result = service.to_dict()

        assert result["name"] == "test-service"
        assert result["is_enabled"] is True
        assert result["is_active"] is True


class TestSystemdDiscovery:
    """Test SystemdDiscovery class."""

    def test_parse_ros2_launch_service(self):
        """Test parsing a service that uses ros2 launch."""
        discovery = SystemdDiscovery()

        # Create a test service file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".service", delete=False) as f:
            f.write("""[Unit]
Description=BetaRay Navigation System
After=network.target

[Service]
Type=simple
User=robot
WorkingDirectory=/home/robot/betaray
ExecStart=/usr/bin/ros2 launch betaray_bringup robot.launch.py
Restart=always

[Install]
WantedBy=multi-user.target
""")
            f.flush()

            service = discovery._parse_service_file(Path(f.name))

            assert service.name == Path(f.name).stem.replace(".service", "")
            assert service.working_directory == "/home/robot/betaray"
            assert service.user == "robot"
            assert service.launch_file == "robot.launch.py"

        os.unlink(f.name)

    def test_parse_python_service(self):
        """Test parsing a service that runs Python directly."""
        discovery = SystemdDiscovery()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".service", delete=False) as f:
            f.write("""[Unit]
Description=AI HTTP Server

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/robot/ai_server.py
Restart=always

[Install]
WantedBy=multi-user.target
""")
            f.flush()

            service = discovery._parse_service_file(Path(f.name))

            assert service.python_script == "/home/robot/ai_server.py"
            assert service.executable == "python"

        os.unlink(f.name)

    def test_parse_service_with_environment(self):
        """Test parsing service with environment variables."""
        discovery = SystemdDiscovery()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".service", delete=False) as f:
            f.write("""[Unit]
Description=Test Service

[Service]
Type=simple
Environment=ROS_DOMAIN_ID=0
Environment=PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages
ExecStart=/usr/bin/test

[Install]
WantedBy=multi-user.target
""")
            f.flush()

            service = discovery._parse_service_file(Path(f.name))

            assert "ROS_DOMAIN_ID" in service.environment
            assert service.environment["ROS_DOMAIN_ID"] == "0"

        os.unlink(f.name)

    def test_parse_ros2_run_service(self):
        """Test parsing a service that uses ros2 run."""
        discovery = SystemdDiscovery()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".service", delete=False) as f:
            f.write("""[Unit]
Description=Motor Controller

[Service]
Type=simple
ExecStart=/usr/bin/ros2 run betaray_control motor_controller_node

[Install]
WantedBy=multi-user.target
""")
            f.flush()

            service = discovery._parse_service_file(Path(f.name))

            assert service.executable == "motor_controller_node"

        os.unlink(f.name)
