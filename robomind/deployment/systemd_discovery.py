"""
Systemd Service Discovery - Auto-detect deployed services.

This module discovers what code is actually deployed by parsing
systemd service files and mapping executables back to source files.

Features:
- Parse systemd service files for ExecStart commands
- Handle ros2 launch commands in ExecStart
- Map executables to source files
- Mark discovered entry points as confirmed_deployed
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SystemdService:
    """Information about a discovered systemd service."""
    name: str
    unit_file: Optional[Path] = None
    exec_start: Optional[str] = None
    working_directory: Optional[str] = None
    user: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)

    # Parsed from ExecStart
    executable: Optional[str] = None
    executable_args: List[str] = field(default_factory=list)
    launch_file: Optional[str] = None  # If ExecStart uses ros2 launch
    python_script: Optional[str] = None  # If ExecStart runs Python directly

    # Status (from systemctl)
    is_enabled: bool = False
    is_active: bool = False
    status: str = "unknown"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "unit_file": str(self.unit_file) if self.unit_file else None,
            "exec_start": self.exec_start,
            "working_directory": self.working_directory,
            "user": self.user,
            "environment": self.environment,
            "executable": self.executable,
            "executable_args": self.executable_args,
            "launch_file": self.launch_file,
            "python_script": self.python_script,
            "is_enabled": self.is_enabled,
            "is_active": self.is_active,
            "status": self.status,
        }

    def uses_ros2_launch(self) -> bool:
        """Check if service uses ros2 launch."""
        return self.launch_file is not None

    def uses_python(self) -> bool:
        """Check if service runs Python directly."""
        return self.python_script is not None


class SystemdDiscovery:
    """
    Discover systemd services and map them to source code.

    Usage:
        discovery = SystemdDiscovery()
        services = discovery.discover("/etc/systemd/system/betaray*.service")
        for svc in services:
            print(f"{svc.name}: {svc.exec_start}")
    """

    # Patterns for parsing ExecStart
    ROS2_LAUNCH_PATTERN = re.compile(
        r'ros2\s+launch\s+(?P<package>\S+)\s+(?P<launch_file>\S+)'
    )
    PYTHON_SCRIPT_PATTERN = re.compile(
        r'(?:python3?|/usr/bin/python3?)\s+(?P<script>\S+\.py)'
    )
    ROS2_RUN_PATTERN = re.compile(
        r'ros2\s+run\s+(?P<package>\S+)\s+(?P<executable>\S+)'
    )

    def __init__(
        self,
        service_dirs: Optional[List[Path]] = None,
        ssh_host: Optional[str] = None,
    ):
        """
        Initialize systemd discovery.

        Args:
            service_dirs: Directories to search for service files
            ssh_host: SSH host for remote discovery (user@host)
        """
        self.service_dirs = service_dirs or [
            Path("/etc/systemd/system"),
            Path("/usr/lib/systemd/system"),
            Path.home() / ".config/systemd/user",
        ]
        self.ssh_host = ssh_host

    def discover(self, pattern: str = "*.service") -> List[SystemdService]:
        """
        Discover systemd services matching a pattern.

        Args:
            pattern: Glob pattern for service files (e.g., "betaray*.service")

        Returns:
            List of discovered SystemdService objects
        """
        services = []

        if self.ssh_host:
            services = self._discover_remote(pattern)
        else:
            services = self._discover_local(pattern)

        # Get status for each service
        for service in services:
            self._get_service_status(service)

        logger.info(f"Discovered {len(services)} services matching '{pattern}'")

        return services

    def _discover_local(self, pattern: str) -> List[SystemdService]:
        """Discover services on local system."""
        services = []

        for service_dir in self.service_dirs:
            if not service_dir.exists():
                continue

            for service_file in service_dir.glob(pattern):
                if service_file.is_file():
                    service = self._parse_service_file(service_file)
                    if service:
                        services.append(service)

        return services

    def _discover_remote(self, pattern: str) -> List[SystemdService]:
        """Discover services on remote system via SSH."""
        services = []

        try:
            # List service files matching pattern
            for service_dir in ["/etc/systemd/system", "/usr/lib/systemd/system"]:
                cmd = f"ssh {self.ssh_host} 'ls {service_dir}/{pattern} 2>/dev/null || true'"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30
                )

                for line in result.stdout.strip().split("\n"):
                    if line and line.endswith(".service"):
                        # Fetch service file content
                        service = self._fetch_remote_service(Path(line))
                        if service:
                            services.append(service)

        except subprocess.TimeoutExpired:
            logger.error(f"SSH timeout connecting to {self.ssh_host}")
        except Exception as e:
            logger.error(f"Remote discovery failed: {e}")

        return services

    def _fetch_remote_service(self, service_path: Path) -> Optional[SystemdService]:
        """Fetch and parse a service file from remote host."""
        try:
            cmd = f"ssh {self.ssh_host} 'cat {service_path}'"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                return self._parse_service_content(
                    service_path.stem.replace(".service", ""),
                    result.stdout,
                    service_path,
                )
        except Exception as e:
            logger.error(f"Failed to fetch {service_path}: {e}")

        return None

    def _parse_service_file(self, path: Path) -> Optional[SystemdService]:
        """Parse a local systemd service file."""
        try:
            with open(path, "r") as f:
                content = f.read()

            return self._parse_service_content(
                path.stem.replace(".service", ""),
                content,
                path,
            )

        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
            return None

    def _parse_service_content(
        self,
        name: str,
        content: str,
        unit_file: Path,
    ) -> SystemdService:
        """Parse systemd service file content."""
        service = SystemdService(
            name=name,
            unit_file=unit_file,
        )

        current_section = None

        for line in content.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            # Section header
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                continue

            # Key=Value
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if current_section == "Service":
                    if key == "ExecStart":
                        service.exec_start = value
                        self._parse_exec_start(service, value)
                    elif key == "WorkingDirectory":
                        service.working_directory = value
                    elif key == "User":
                        service.user = value
                    elif key == "Environment":
                        # Parse Environment=KEY=VALUE
                        if "=" in value:
                            env_key, env_val = value.split("=", 1)
                            service.environment[env_key] = env_val.strip('"\'')

        return service

    def _parse_exec_start(self, service: SystemdService, exec_start: str):
        """Parse ExecStart to extract executable info."""
        # Remove prefix characters like - or @
        exec_start = exec_start.lstrip("-@!")

        # Check for ros2 launch
        match = self.ROS2_LAUNCH_PATTERN.search(exec_start)
        if match:
            service.launch_file = match.group("launch_file")
            service.executable = "ros2 launch"
            return

        # Check for ros2 run
        match = self.ROS2_RUN_PATTERN.search(exec_start)
        if match:
            service.executable = match.group("executable")
            return

        # Check for Python script
        match = self.PYTHON_SCRIPT_PATTERN.search(exec_start)
        if match:
            service.python_script = match.group("script")
            service.executable = "python"
            return

        # Generic executable
        parts = exec_start.split()
        if parts:
            service.executable = parts[0]
            service.executable_args = parts[1:]

    def _get_service_status(self, service: SystemdService):
        """Get service status from systemctl."""
        try:
            if self.ssh_host:
                cmd = f"ssh {self.ssh_host} 'systemctl is-enabled {service.name}.service 2>/dev/null || echo disabled'"
            else:
                cmd = f"systemctl is-enabled {service.name}.service 2>/dev/null || echo disabled"

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            service.is_enabled = result.stdout.strip() == "enabled"

            if self.ssh_host:
                cmd = f"ssh {self.ssh_host} 'systemctl is-active {service.name}.service 2>/dev/null || echo inactive'"
            else:
                cmd = f"systemctl is-active {service.name}.service 2>/dev/null || echo inactive"

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            status = result.stdout.strip()
            service.is_active = status == "active"
            service.status = status

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout getting status for {service.name}")
        except Exception as e:
            logger.warning(f"Failed to get status for {service.name}: {e}")

    def discover_from_manifest(
        self,
        service_names: List[str],
    ) -> List[SystemdService]:
        """
        Discover specific services by name.

        Args:
            service_names: List of service names to find

        Returns:
            List of discovered services (with status checked)
        """
        services = []

        for name in service_names:
            found = False
            # Search in standard locations
            for service_dir in self.service_dirs:
                service_file = service_dir / f"{name}.service"
                if service_file.exists():
                    service = self._parse_service_file(service_file)
                    if service:
                        services.append(service)
                        found = True
                        break

            # If file not found locally, still check status (service may exist remotely)
            if not found:
                service = SystemdService(name=name)
                services.append(service)

        # Get status for all services
        for service in services:
            self._get_service_status(service)

        return services


def discover_services(
    pattern: str = "*.service",
    ssh_host: Optional[str] = None,
) -> List[SystemdService]:
    """
    Convenience function to discover systemd services.

    Args:
        pattern: Glob pattern for service files
        ssh_host: Optional SSH host for remote discovery

    Returns:
        List of discovered SystemdService objects
    """
    discovery = SystemdDiscovery(ssh_host=ssh_host)
    return discovery.discover(pattern)


def map_service_to_source(
    service: SystemdService,
    project_root: Path,
) -> Optional[Path]:
    """
    Try to map a systemd service to its source file.

    Args:
        service: The systemd service
        project_root: Root of the project to search

    Returns:
        Path to source file if found, None otherwise
    """
    project_root = Path(project_root)

    # If it's a Python script, search for it
    if service.python_script:
        script_name = Path(service.python_script).name
        for match in project_root.rglob(script_name):
            if match.is_file():
                return match

    # If it's a launch file, search for it
    if service.launch_file:
        launch_name = Path(service.launch_file).name
        for match in project_root.rglob(f"*{launch_name}*"):
            if match.is_file():
                return match

    # If it's a ros2 run executable, search for setup.py entry points
    if service.executable and service.executable != "python":
        # Search for the executable name in Python files
        for py_file in project_root.rglob("*.py"):
            try:
                content = py_file.read_text(errors="replace")
                if f"def {service.executable}(" in content or f"def main(" in content:
                    # Check if this file's name matches
                    if service.executable in py_file.stem:
                        return py_file
            except Exception:
                continue

    return None
