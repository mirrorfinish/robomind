"""Tests for remote SSH analysis module.

Note: Most tests are marked with pytest.mark.skip when they require
actual SSH connectivity. These tests validate the API and logic
without requiring real remote hosts.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from robomind.remote.ssh_analyzer import (
    RemoteHost,
    SSHConnection,
    SSHAnalyzer,
    DistributedAnalyzer,
    RemoteAnalysisResult,
    DistributedAnalysisResult,
    parse_remote_specs,
)


class TestRemoteHost:
    """Tests for RemoteHost dataclass."""

    def test_basic_init(self):
        """Test basic initialization."""
        host = RemoteHost(
            hostname="jetson.local",
            username="robot",
        )
        assert host.hostname == "jetson.local"
        assert host.username == "robot"
        assert host.port == 22
        assert host.project_path == "~"
        assert host.hardware_target == "jetson"

    def test_full_init(self):
        """Test initialization with all parameters."""
        host = RemoteHost(
            hostname="nav.local",
            username="robot",
            port=2222,
            key_file=Path("/home/user/.ssh/id_rsa"),
            project_path="/home/robot/betaray",
            hardware_target="nav_orin",
        )
        assert host.port == 2222
        assert host.key_file == Path("/home/user/.ssh/id_rsa")
        assert host.project_path == "/home/robot/betaray"
        assert host.hardware_target == "nav_orin"

    def test_connection_string(self):
        """Test connection_string property."""
        host = RemoteHost("jetson.local", "robot")
        assert host.connection_string == "robot@jetson.local"

    def test_full_path(self):
        """Test full_path property."""
        host = RemoteHost("jetson.local", "robot", project_path="~/betaray")
        assert host.full_path == "robot@jetson.local:~/betaray"

    def test_str(self):
        """Test string representation."""
        host = RemoteHost("jetson.local", "robot", project_path="~/project")
        assert "jetson.local" in str(host)
        assert "~/project" in str(host)

    def test_hardware_target_default(self):
        """Test hardware_target defaults to hostname."""
        host = RemoteHost("nav-jetson.local", "robot")
        assert host.hardware_target == "nav-jetson"

    def test_key_file_string_conversion(self):
        """Test key_file string is converted to Path."""
        host = RemoteHost("jetson.local", "robot", key_file="/home/user/.ssh/id_rsa")
        assert isinstance(host.key_file, Path)


class TestRemoteHostFromString:
    """Tests for RemoteHost.from_string class method."""

    def test_simple_spec(self):
        """Test simple user@host format."""
        host = RemoteHost.from_string("robot@jetson.local")
        assert host.hostname == "jetson.local"
        assert host.username == "robot"
        assert host.port == 22
        assert host.project_path == "~"

    def test_spec_with_path(self):
        """Test user@host:path format."""
        host = RemoteHost.from_string("robot@nav.local:~/betaray")
        assert host.hostname == "nav.local"
        assert host.project_path == "~/betaray"

    def test_spec_with_port_and_path(self):
        """Test user@host:port:path format."""
        host = RemoteHost.from_string("robot@jetson:2222:/home/robot/project")
        assert host.hostname == "jetson"
        assert host.port == 2222
        assert host.project_path == "/home/robot/project"

    def test_spec_with_key_file(self):
        """Test spec with key file."""
        host = RemoteHost.from_string(
            "robot@jetson.local:~/project",
            key_file=Path("/tmp/key")
        )
        assert host.key_file == Path("/tmp/key")

    def test_invalid_spec_no_at(self):
        """Test invalid spec without @."""
        with pytest.raises(ValueError):
            RemoteHost.from_string("jetson.local")


class TestParseRemoteSpecs:
    """Tests for parse_remote_specs function."""

    def test_single_spec(self):
        """Test parsing single spec."""
        hosts = parse_remote_specs(["robot@jetson.local"])
        assert len(hosts) == 1
        assert hosts[0].hostname == "jetson.local"

    def test_multiple_specs(self):
        """Test parsing multiple specs."""
        hosts = parse_remote_specs([
            "robot@nav.local:~/betaray",
            "robot@ai.local:~/betaray",
            "voice@vision.local",
        ])
        assert len(hosts) == 3
        assert hosts[0].hostname == "nav.local"
        assert hosts[1].hostname == "ai.local"
        assert hosts[2].hostname == "vision.local"

    def test_invalid_spec_skipped(self):
        """Test that invalid specs are skipped."""
        hosts = parse_remote_specs([
            "robot@valid.local",
            "invalid_no_at",
            "robot@also_valid.local",
        ])
        assert len(hosts) == 2

    def test_with_key_file(self):
        """Test parsing with key file."""
        hosts = parse_remote_specs(
            ["robot@jetson.local"],
            key_file=Path("/tmp/key")
        )
        assert hosts[0].key_file == Path("/tmp/key")


class TestSSHConnection:
    """Tests for SSHConnection class."""

    def test_init(self):
        """Test connection initialization."""
        host = RemoteHost("jetson.local", "robot")
        conn = SSHConnection(host)
        assert conn.host == host
        assert conn.connected is False
        assert conn._last_error is None

    def test_build_ssh_args(self):
        """Test SSH argument building."""
        host = RemoteHost("jetson.local", "robot", port=2222)
        conn = SSHConnection(host)
        args = conn._build_ssh_args()

        assert "-p" in args
        assert "2222" in args
        assert "StrictHostKeyChecking=accept-new" in " ".join(args)

    def test_build_ssh_args_with_key(self, tmp_path):
        """Test SSH argument building with key file."""
        key_file = tmp_path / "id_rsa"
        key_file.write_text("fake key")

        host = RemoteHost("jetson.local", "robot", key_file=key_file)
        conn = SSHConnection(host)
        args = conn._build_ssh_args()

        assert "-i" in args
        assert str(key_file) in args

    @patch("subprocess.run")
    def test_test_connection_success(self, mock_run):
        """Test successful connection test."""
        mock_run.return_value = Mock(returncode=0, stdout="OK\n", stderr="")

        host = RemoteHost("jetson.local", "robot")
        conn = SSHConnection(host)
        result = conn.test_connection()

        assert result is True
        assert conn.connected is True

    @patch("subprocess.run")
    def test_test_connection_failure(self, mock_run):
        """Test failed connection test."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Connection refused")

        host = RemoteHost("jetson.local", "robot")
        conn = SSHConnection(host)
        result = conn.test_connection()

        assert result is False
        assert conn.connected is False
        assert "Connection refused" in conn.last_error

    @patch("subprocess.run")
    def test_test_connection_timeout(self, mock_run):
        """Test connection timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("ssh", 15)

        host = RemoteHost("jetson.local", "robot")
        conn = SSHConnection(host)
        result = conn.test_connection()

        assert result is False
        assert "timeout" in conn.last_error.lower()

    @patch("subprocess.run")
    def test_execute_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value = Mock(returncode=0, stdout="hello", stderr="")

        host = RemoteHost("jetson.local", "robot")
        conn = SSHConnection(host)
        rc, stdout, stderr = conn.execute("echo hello")

        assert rc == 0
        assert stdout == "hello"
        assert stderr == ""

    @patch("subprocess.run")
    def test_execute_failure(self, mock_run):
        """Test failed command execution."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="command not found")

        host = RemoteHost("jetson.local", "robot")
        conn = SSHConnection(host)
        rc, stdout, stderr = conn.execute("invalid_command")

        assert rc == 1
        assert "command not found" in stderr


class TestSSHAnalyzer:
    """Tests for SSHAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        host = RemoteHost("jetson.local", "robot")
        analyzer = SSHAnalyzer(host)
        assert analyzer.host == host
        assert analyzer._temp_dir is None

    @patch.object(SSHConnection, "test_connection")
    def test_connect(self, mock_connect):
        """Test connect method."""
        mock_connect.return_value = True

        host = RemoteHost("jetson.local", "robot")
        analyzer = SSHAnalyzer(host)
        result = analyzer.connect()

        assert result is True
        mock_connect.assert_called_once()

    @patch.object(SSHConnection, "test_connection")
    def test_analyze_connection_failure(self, mock_connect):
        """Test analyze returns error when connection fails."""
        mock_connect.return_value = False

        host = RemoteHost("jetson.local", "robot")
        analyzer = SSHAnalyzer(host)
        analyzer.connection._last_error = "Connection refused"

        result = analyzer.analyze()

        assert result.success is False
        assert "Connection failed" in result.error

    @patch.object(SSHConnection, "execute")
    def test_get_remote_ros2_info(self, mock_execute):
        """Test getting remote ROS2 info."""
        mock_execute.side_effect = [
            (0, "/node1\n/node2\n", ""),  # ros2 node list
            (0, "/topic1\n/topic2\n/topic3\n", ""),  # ros2 topic list
            (0, "/service1\n", ""),  # ros2 service list
        ]

        host = RemoteHost("jetson.local", "robot")
        analyzer = SSHAnalyzer(host)
        info = analyzer.get_remote_ros2_info()

        assert len(info["nodes"]) == 2
        assert len(info["topics"]) == 3
        assert len(info["services"]) == 1


class TestRemoteAnalysisResult:
    """Tests for RemoteAnalysisResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        host = RemoteHost("jetson.local", "robot")
        result = RemoteAnalysisResult(
            host=host,
            success=True,
            nodes=["node1", "node2"],
            stats={"ros2_nodes": 2},
        )

        assert result.success is True
        assert len(result.nodes) == 2
        assert result.error is None

    def test_failure_result(self):
        """Test failed result."""
        host = RemoteHost("jetson.local", "robot")
        result = RemoteAnalysisResult(
            host=host,
            success=False,
            error="Connection failed",
        )

        assert result.success is False
        assert result.error == "Connection failed"

    def test_to_dict(self):
        """Test to_dict method."""
        host = RemoteHost("jetson.local", "robot", hardware_target="nav_orin")
        result = RemoteAnalysisResult(
            host=host,
            success=True,
            nodes=["node1", "node2"],
        )

        d = result.to_dict()
        assert d["hostname"] == "jetson.local"
        assert d["hardware_target"] == "nav_orin"
        assert d["success"] is True
        assert d["node_count"] == 2


class TestDistributedAnalysisResult:
    """Tests for DistributedAnalysisResult dataclass."""

    def test_default(self):
        """Test default initialization."""
        result = DistributedAnalysisResult()
        assert result.hosts_analyzed == 0
        assert result.hosts_failed == 0
        assert len(result.host_results) == 0
        assert len(result.merged_nodes) == 0

    def test_to_dict(self):
        """Test to_dict method."""
        result = DistributedAnalysisResult(
            hosts_analyzed=2,
            hosts_failed=1,
            merged_nodes=["n1", "n2", "n3"],
            errors=["host3: Connection failed"],
        )

        d = result.to_dict()
        assert d["hosts_analyzed"] == 2
        assert d["hosts_failed"] == 1
        assert d["total_nodes"] == 3
        assert len(d["errors"]) == 1


class TestDistributedAnalyzer:
    """Tests for DistributedAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        hosts = [
            RemoteHost("nav.local", "robot"),
            RemoteHost("ai.local", "robot"),
        ]
        analyzer = DistributedAnalyzer(hosts)

        assert len(analyzer.hosts) == 2
        assert analyzer.max_workers == 4

    @patch.object(SSHAnalyzer, "connect")
    def test_test_connections(self, mock_connect):
        """Test connection testing."""
        mock_connect.side_effect = [True, False]

        hosts = [
            RemoteHost("nav.local", "robot"),
            RemoteHost("ai.local", "robot"),
        ]
        analyzer = DistributedAnalyzer(hosts)
        results = analyzer.test_connections()

        assert results["nav.local"] is True
        assert results["ai.local"] is False


class TestIntegration:
    """Integration tests (require mocking or real SSH)."""

    @pytest.mark.skip(reason="Requires real SSH connection")
    def test_real_connection(self):
        """Test with real SSH connection (manual test)."""
        host = RemoteHost("betaray-nav.local", "robot", project_path="~/betaray")
        analyzer = SSHAnalyzer(host)

        assert analyzer.connect() is True

        info = analyzer.get_remote_ros2_info()
        print(f"Nodes: {info['nodes']}")
        print(f"Topics: {info['topics']}")

    @pytest.mark.skip(reason="Requires real SSH connection")
    def test_real_distributed_analysis(self):
        """Test distributed analysis with real hosts (manual test)."""
        hosts = [
            RemoteHost("betaray-nav.local", "robot", project_path="~/betaray"),
            RemoteHost("betaray-ai.local", "robot", project_path="~/betaray"),
        ]
        analyzer = DistributedAnalyzer(hosts)
        result = analyzer.analyze_all()

        print(f"Analyzed: {result.hosts_analyzed} hosts")
        print(f"Nodes found: {len(result.merged_nodes)}")
