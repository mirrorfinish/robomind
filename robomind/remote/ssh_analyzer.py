"""
RoboMind SSH Analyzer - Remote analysis of distributed ROS2 systems.

Enables analyzing ROS2 projects spread across multiple machines via SSH.
Supports:
- SSH connection management
- Remote code synchronization (rsync)
- Remote command execution
- Aggregating results from multiple hosts

Example:
    analyzer = DistributedAnalyzer([
        RemoteHost("nav.local", "robot", project_path="~/betaray"),
        RemoteHost("ai.local", "robot", project_path="~/betaray"),
    ])
    result = analyzer.analyze_all(output_dir)
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class RemoteHost:
    """Configuration for a remote host to analyze."""
    hostname: str
    username: str
    port: int = 22
    key_file: Optional[Path] = None
    project_path: str = "~"
    hardware_target: Optional[str] = None  # e.g., "nav_orin", "ai_orin"

    def __post_init__(self):
        if self.key_file and isinstance(self.key_file, str):
            self.key_file = Path(self.key_file)

        # Default hardware target to hostname if not specified
        if not self.hardware_target:
            self.hardware_target = self.hostname.split(".")[0]

    @property
    def connection_string(self) -> str:
        """Get SSH connection string (user@host)."""
        return f"{self.username}@{self.hostname}"

    @property
    def full_path(self) -> str:
        """Get full remote path (user@host:path)."""
        return f"{self.connection_string}:{self.project_path}"

    def __str__(self) -> str:
        return f"RemoteHost({self.connection_string}, path={self.project_path})"

    @classmethod
    def from_string(cls, spec: str, key_file: Optional[Path] = None) -> "RemoteHost":
        """
        Parse remote host from string specification.

        Formats:
            user@host
            user@host:path
            user@host:port:path

        Examples:
            RemoteHost.from_string("robot@nav.local")
            RemoteHost.from_string("robot@ai.local:~/betaray")
            RemoteHost.from_string("robot@jetson:22:~/project")
        """
        # Handle port in hostname (user@host:port:path)
        if spec.count(":") >= 2:
            user_host, port_str, path = spec.split(":", 2)
            port = int(port_str)
        elif ":" in spec:
            user_host, path = spec.split(":", 1)
            port = 22
        else:
            user_host = spec
            path = "~"
            port = 22

        if "@" not in user_host:
            raise ValueError(f"Invalid remote spec: {spec}. Expected user@host")

        username, hostname = user_host.split("@", 1)

        return cls(
            hostname=hostname,
            username=username,
            port=port,
            key_file=key_file,
            project_path=path,
        )


@dataclass
class SSHConnection:
    """
    Manages SSH connection to a remote host.

    Uses subprocess to call ssh/scp/rsync commands rather than paramiko
    for better compatibility with SSH agent and config files.
    """
    host: RemoteHost
    connected: bool = False
    ros2_distro: str = "humble"
    _last_error: Optional[str] = None

    def _build_ssh_args(self) -> List[str]:
        """Build common SSH arguments."""
        args = [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-p", str(self.host.port),
        ]
        if self.host.key_file and self.host.key_file.exists():
            args.extend(["-i", str(self.host.key_file)])
        return args

    def test_connection(self) -> bool:
        """Test SSH connection to host."""
        try:
            cmd = ["ssh"] + self._build_ssh_args() + [
                self.host.connection_string,
                "echo", "OK"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.connected = result.returncode == 0 and "OK" in result.stdout
            if not self.connected:
                self._last_error = result.stderr.strip() or "Connection failed"
            return self.connected
        except subprocess.TimeoutExpired:
            self._last_error = "Connection timeout"
            self.connected = False
            return False
        except Exception as e:
            self._last_error = str(e)
            self.connected = False
            return False

    def execute(self, command: str, timeout: int = 60, source_ros2: bool = False) -> Tuple[int, str, str]:
        """
        Execute command on remote host.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            source_ros2: If True, source ROS2 setup.bash before running command

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            if source_ros2:
                command = f"source /opt/ros/{self.ros2_distro}/setup.bash && {command}"
            cmd = ["ssh"] + self._build_ssh_args() + [
                self.host.connection_string,
                command,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timeout"
        except Exception as e:
            return -1, "", str(e)

    def rsync_from_remote(
        self,
        local_path: Path,
        exclude_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Sync remote project to local path using rsync.

        Args:
            local_path: Local directory to sync to
            exclude_patterns: Patterns to exclude (e.g., ["__pycache__", ".git"])

        Returns:
            True if sync successful
        """
        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__",
                "*.pyc",
                ".git",
                ".venv",
                "venv",
                "build",
                "install",
                "log",
                "*.bag",
                "*.db3",
            ]

        # Ensure local path exists
        local_path.mkdir(parents=True, exist_ok=True)

        # Build rsync command
        cmd = ["rsync", "-avz", "--delete"]

        # Add excludes
        for pattern in exclude_patterns:
            cmd.extend(["--exclude", pattern])

        # SSH options
        ssh_opts = " ".join(self._build_ssh_args())
        cmd.extend(["-e", f"ssh {ssh_opts}"])

        # Source and destination
        remote_path = self.host.project_path
        if not remote_path.endswith("/"):
            remote_path += "/"
        cmd.append(f"{self.host.connection_string}:{remote_path}")
        cmd.append(str(local_path) + "/")

        try:
            logger.info(f"Syncing from {self.host.connection_string}:{remote_path} to {local_path}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large repos
            )
            if result.returncode != 0:
                self._last_error = result.stderr.strip()
                logger.error(f"rsync failed: {self._last_error}")
                return False
            return True
        except subprocess.TimeoutExpired:
            self._last_error = "rsync timeout"
            return False
        except Exception as e:
            self._last_error = str(e)
            return False

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error


@dataclass
class RemoteAnalysisResult:
    """Result from analyzing a single remote host."""
    host: RemoteHost
    success: bool
    nodes: List[Any] = field(default_factory=list)
    topic_graph: Optional[Any] = None
    system_graph: Optional[Any] = None
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    local_copy_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hostname": self.host.hostname,
            "hardware_target": self.host.hardware_target,
            "success": self.success,
            "error": self.error,
            "stats": self.stats,
            "node_count": len(self.nodes),
        }


@dataclass
class DistributedAnalysisResult:
    """Result from analyzing all remote hosts."""
    hosts_analyzed: int = 0
    hosts_failed: int = 0
    host_results: Dict[str, RemoteAnalysisResult] = field(default_factory=dict)
    merged_nodes: List[Any] = field(default_factory=list)
    merged_topic_graph: Optional[Any] = None
    merged_system_graph: Optional[Any] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hosts_analyzed": self.hosts_analyzed,
            "hosts_failed": self.hosts_failed,
            "total_nodes": len(self.merged_nodes),
            "host_results": {k: v.to_dict() for k, v in self.host_results.items()},
            "errors": self.errors,
        }


class SSHAnalyzer:
    """
    Analyze a single remote host.

    Workflow:
    1. Test SSH connection
    2. Rsync code to local temp directory
    3. Run RoboMind analysis on local copy
    4. Tag results with hardware target
    """

    def __init__(self, host: RemoteHost, ros2_distro: str = "humble"):
        self.host = host
        self.connection = SSHConnection(host, ros2_distro=ros2_distro)
        self._temp_dir: Optional[Path] = None

    def connect(self) -> bool:
        """Test connection to remote host."""
        return self.connection.test_connection()

    def sync_code(self, local_path: Optional[Path] = None) -> Optional[Path]:
        """
        Sync remote code to local directory.

        Args:
            local_path: Optional local path. If None, uses temp directory.

        Returns:
            Path to local copy, or None if failed.
        """
        if local_path is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix=f"robomind_{self.host.hostname}_"))
            local_path = self._temp_dir

        if self.connection.rsync_from_remote(local_path):
            return local_path
        return None

    def analyze(self, local_path: Optional[Path] = None) -> RemoteAnalysisResult:
        """
        Full analysis of remote host.

        Args:
            local_path: Optional path to already-synced code.

        Returns:
            RemoteAnalysisResult with nodes and graphs.
        """
        # Import here to avoid circular imports
        from robomind.core.scanner import ProjectScanner
        from robomind.core.parser import PythonParser
        from robomind.ros2.node_extractor import ROS2NodeExtractor
        from robomind.ros2.topic_extractor import TopicExtractor
        from robomind.core.graph import build_system_graph

        # Test connection
        if not self.connect():
            return RemoteAnalysisResult(
                host=self.host,
                success=False,
                error=f"Connection failed: {self.connection.last_error}",
            )

        # Sync code
        if local_path is None:
            local_path = self.sync_code()
            if local_path is None:
                return RemoteAnalysisResult(
                    host=self.host,
                    success=False,
                    error=f"Code sync failed: {self.connection.last_error}",
                )

        try:
            # Run analysis
            scanner = ProjectScanner(local_path)
            scan_result = scanner.scan()

            parser = PythonParser()
            node_extractor = ROS2NodeExtractor()
            topic_extractor = TopicExtractor()
            all_nodes = []

            for pf in scan_result.python_files:
                pr = parser.parse_file(pf.path)
                if pr and pr.has_ros2_imports():
                    nodes = node_extractor.extract_from_file(pf.path, pf.package_name)
                    # Tag nodes with hardware target
                    for node in nodes:
                        node.hardware_target = self.host.hardware_target
                    all_nodes.extend(nodes)
                    topic_extractor.add_nodes(nodes)

            topic_graph = topic_extractor.build()
            system_graph = build_system_graph(all_nodes, topic_graph)

            stats = {
                "python_files": len(scan_result.python_files),
                "packages": len(scan_result.packages),
                "ros2_nodes": len(all_nodes),
                "topics": len(topic_graph.topics),
                "connected_topics": len(topic_graph.get_connected_topics()),
            }

            return RemoteAnalysisResult(
                host=self.host,
                success=True,
                nodes=all_nodes,
                topic_graph=topic_graph,
                system_graph=system_graph,
                stats=stats,
                local_copy_path=local_path,
            )

        except Exception as e:
            logger.exception(f"Analysis failed for {self.host.hostname}")
            return RemoteAnalysisResult(
                host=self.host,
                success=False,
                error=str(e),
            )

    def cleanup(self):
        """Clean up temporary directory."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def get_remote_ros2_info(self) -> Dict[str, Any]:
        """
        Get ROS2 runtime information from remote host.

        Executes ros2 commands to get live system state.
        """
        info = {
            "hostname": self.host.hostname,
            "nodes": [],
            "topics": [],
            "services": [],
        }

        # Get node list
        rc, stdout, stderr = self.connection.execute("ros2 node list 2>/dev/null || true", source_ros2=True)
        if rc == 0:
            info["nodes"] = [n.strip() for n in stdout.strip().split("\n") if n.strip()]

        # Get topic list
        rc, stdout, stderr = self.connection.execute("ros2 topic list 2>/dev/null || true", source_ros2=True)
        if rc == 0:
            info["topics"] = [t.strip() for t in stdout.strip().split("\n") if t.strip()]

        # Get service list
        rc, stdout, stderr = self.connection.execute("ros2 service list 2>/dev/null || true", source_ros2=True)
        if rc == 0:
            info["services"] = [s.strip() for s in stdout.strip().split("\n") if s.strip()]

        return info


class DistributedAnalyzer:
    """
    Analyze distributed ROS2 system across multiple hosts.

    Coordinates analysis of multiple remote hosts, merging results
    into a unified system graph.
    """

    def __init__(self, hosts: List[RemoteHost], max_workers: int = 4):
        """
        Initialize distributed analyzer.

        Args:
            hosts: List of remote hosts to analyze
            max_workers: Maximum parallel SSH connections
        """
        self.hosts = hosts
        self.max_workers = max_workers
        self._analyzers: Dict[str, SSHAnalyzer] = {}

    def analyze_all(
        self,
        output_dir: Optional[Path] = None,
        keep_local_copies: bool = False,
    ) -> DistributedAnalysisResult:
        """
        Analyze all remote hosts.

        Args:
            output_dir: Optional directory for local copies
            keep_local_copies: If True, don't delete synced code

        Returns:
            DistributedAnalysisResult with merged analysis
        """
        result = DistributedAnalysisResult()

        # Create analyzers
        for host in self.hosts:
            self._analyzers[host.hostname] = SSHAnalyzer(host)

        # Analyze hosts (parallel)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_host, host, output_dir): host
                for host in self.hosts
            }

            for future in as_completed(futures):
                host = futures[future]
                try:
                    host_result = future.result()
                    result.host_results[host.hostname] = host_result

                    if host_result.success:
                        result.hosts_analyzed += 1
                        result.merged_nodes.extend(host_result.nodes)
                    else:
                        result.hosts_failed += 1
                        if host_result.error:
                            result.errors.append(f"{host.hostname}: {host_result.error}")

                except Exception as e:
                    result.hosts_failed += 1
                    result.errors.append(f"{host.hostname}: {str(e)}")

        # Merge topic graphs and system graphs
        if result.merged_nodes:
            result.merged_topic_graph, result.merged_system_graph = self._merge_graphs(result)

        # Cleanup
        if not keep_local_copies:
            self.cleanup()

        return result

    def _analyze_host(
        self,
        host: RemoteHost,
        output_dir: Optional[Path],
    ) -> RemoteAnalysisResult:
        """Analyze a single host."""
        analyzer = self._analyzers[host.hostname]

        local_path = None
        if output_dir:
            local_path = output_dir / f"remote_{host.hostname}"

        return analyzer.analyze(local_path)

    def _merge_graphs(
        self,
        result: DistributedAnalysisResult,
    ) -> Tuple[Any, Any]:
        """Merge topic graphs and system graphs from all hosts."""
        from robomind.ros2.topic_extractor import TopicExtractor
        from robomind.core.graph import build_system_graph

        # Rebuild topic graph with all nodes
        topic_extractor = TopicExtractor()
        for node in result.merged_nodes:
            topic_extractor.add_nodes([node])

        merged_topic_graph = topic_extractor.build()
        merged_system_graph = build_system_graph(result.merged_nodes, merged_topic_graph)

        return merged_topic_graph, merged_system_graph

    def test_connections(self) -> Dict[str, bool]:
        """Test connections to all hosts."""
        results = {}
        for host in self.hosts:
            analyzer = SSHAnalyzer(host)
            results[host.hostname] = analyzer.connect()
        return results

    def cleanup(self):
        """Clean up all temporary directories."""
        for analyzer in self._analyzers.values():
            analyzer.cleanup()


def parse_remote_specs(
    specs: List[str],
    key_file: Optional[Path] = None,
) -> List[RemoteHost]:
    """
    Parse list of remote specifications into RemoteHost objects.

    Args:
        specs: List of "user@host:path" strings
        key_file: Optional SSH key file

    Returns:
        List of RemoteHost objects
    """
    hosts = []
    for spec in specs:
        try:
            host = RemoteHost.from_string(spec, key_file)
            hosts.append(host)
        except ValueError as e:
            logger.warning(f"Invalid remote spec '{spec}': {e}")
    return hosts
