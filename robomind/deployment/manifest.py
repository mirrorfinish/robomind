"""
Deployment Manifest Parser - Parse YAML files describing actual deployments.

A deployment manifest tells RoboMind what code is actually running in production,
enabling it to distinguish deployed code from dead/archived code.

Example manifest:
```yaml
jetsons:
  nav_jetson:
    hostname: betaray-nav.local
    ros_domain_id: 0
    systemd_services:
      - betaray-navigation-system
    launch_files:
      - betaray_navigation_jetson.launch.py

  ai_jetson:
    hostname: betaray-ai.local
    http_only: true  # No external ROS2
```
"""

import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class JetsonConfig:
    """Configuration for a single Jetson/host in the deployment."""
    name: str
    hostname: str
    ros_domain_id: Optional[int] = None
    systemd_services: List[str] = field(default_factory=list)
    launch_files: List[str] = field(default_factory=list)
    http_only: bool = False
    http_ports: List[int] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)  # Python entry scripts
    packages: List[str] = field(default_factory=list)  # ROS2 packages deployed here

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "JetsonConfig":
        """Create JetsonConfig from dictionary."""
        return cls(
            name=name,
            hostname=data.get("hostname", name),
            ros_domain_id=data.get("ros_domain_id"),
            systemd_services=data.get("systemd_services", []),
            launch_files=data.get("launch_files", []),
            http_only=data.get("http_only", False),
            http_ports=data.get("http_ports", []),
            entry_points=data.get("entry_points", []),
            packages=data.get("packages", []),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "hostname": self.hostname,
        }
        if self.ros_domain_id is not None:
            result["ros_domain_id"] = self.ros_domain_id
        if self.systemd_services:
            result["systemd_services"] = self.systemd_services
        if self.launch_files:
            result["launch_files"] = self.launch_files
        if self.http_only:
            result["http_only"] = self.http_only
        if self.http_ports:
            result["http_ports"] = self.http_ports
        if self.entry_points:
            result["entry_points"] = self.entry_points
        if self.packages:
            result["packages"] = self.packages
        return result

    def has_ros2(self) -> bool:
        """Check if this host has ROS2 deployment."""
        return not self.http_only and (
            self.ros_domain_id is not None or
            bool(self.launch_files) or
            any("ros2" in s.lower() for s in self.systemd_services)
        )


@dataclass
class DeploymentManifest:
    """
    Complete deployment manifest describing what runs in production.

    This enables RoboMind to:
    1. Filter analysis to only deployed code
    2. Map nodes to their target hosts
    3. Understand communication patterns (ROS2 internal vs HTTP external)
    """
    jetsons: Dict[str, JetsonConfig] = field(default_factory=dict)

    # Global settings
    project_root: Optional[Path] = None
    default_ros_domain_id: int = 0

    # Derived data (populated during analysis)
    deployed_launch_files: List[str] = field(default_factory=list)
    deployed_services: List[str] = field(default_factory=list)
    deployed_packages: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> "DeploymentManifest":
        """Create manifest from dictionary."""
        manifest = cls()

        # Parse global settings
        manifest.default_ros_domain_id = data.get("default_ros_domain_id", 0)
        if "project_root" in data:
            manifest.project_root = Path(data["project_root"])

        # Parse jetson configs
        jetsons_data = data.get("jetsons", {})
        for name, config in jetsons_data.items():
            manifest.jetsons[name] = JetsonConfig.from_dict(name, config)

        # Aggregate deployed items
        manifest._aggregate_deployed_items()

        return manifest

    def _aggregate_deployed_items(self):
        """Aggregate all deployed items from all hosts."""
        self.deployed_launch_files = []
        self.deployed_services = []
        self.deployed_packages = []

        for jetson in self.jetsons.values():
            self.deployed_launch_files.extend(jetson.launch_files)
            self.deployed_services.extend(jetson.systemd_services)
            self.deployed_packages.extend(jetson.packages)

        # Remove duplicates while preserving order
        self.deployed_launch_files = list(dict.fromkeys(self.deployed_launch_files))
        self.deployed_services = list(dict.fromkeys(self.deployed_services))
        self.deployed_packages = list(dict.fromkeys(self.deployed_packages))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "default_ros_domain_id": self.default_ros_domain_id,
            "project_root": str(self.project_root) if self.project_root else None,
            "jetsons": {
                name: config.to_dict()
                for name, config in self.jetsons.items()
            },
            "deployed_launch_files": self.deployed_launch_files,
            "deployed_services": self.deployed_services,
            "deployed_packages": self.deployed_packages,
        }

    def get_host_for_service(self, service_name: str) -> Optional[str]:
        """Find which host runs a given systemd service."""
        for name, config in self.jetsons.items():
            if service_name in config.systemd_services:
                return name
        return None

    def get_host_for_launch_file(self, launch_file: str) -> Optional[str]:
        """Find which host runs a given launch file."""
        launch_basename = Path(launch_file).name
        for name, config in self.jetsons.items():
            for lf in config.launch_files:
                if Path(lf).name == launch_basename or lf == launch_file:
                    return name
        return None

    def get_ros2_hosts(self) -> List[JetsonConfig]:
        """Get all hosts that have ROS2 deployments."""
        return [config for config in self.jetsons.values() if config.has_ros2()]

    def get_http_only_hosts(self) -> List[JetsonConfig]:
        """Get all hosts that are HTTP-only (no external ROS2)."""
        return [config for config in self.jetsons.values() if config.http_only]

    def is_launch_file_deployed(self, launch_file: str) -> bool:
        """Check if a launch file is in the deployment manifest."""
        launch_basename = Path(launch_file).name
        for lf in self.deployed_launch_files:
            if Path(lf).name == launch_basename or lf == launch_file:
                return True
        return False

    def is_service_deployed(self, service_name: str) -> bool:
        """Check if a systemd service is in the deployment manifest."""
        return service_name in self.deployed_services

    def summary(self) -> Dict:
        """Get summary of deployment manifest."""
        return {
            "total_hosts": len(self.jetsons),
            "ros2_hosts": len(self.get_ros2_hosts()),
            "http_only_hosts": len(self.get_http_only_hosts()),
            "launch_files": len(self.deployed_launch_files),
            "systemd_services": len(self.deployed_services),
            "packages": len(self.deployed_packages),
        }


def load_deployment_manifest(path: Path) -> DeploymentManifest:
    """
    Load a deployment manifest from a YAML file.

    Args:
        path: Path to the manifest YAML file

    Returns:
        DeploymentManifest object

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Deployment manifest not found: {path}")

    logger.info(f"Loading deployment manifest: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    manifest = DeploymentManifest.from_dict(data)

    logger.info(f"Loaded manifest: {manifest.summary()}")

    return manifest


def create_example_manifest() -> str:
    """Generate an example deployment manifest YAML."""
    return """# RoboMind Deployment Manifest
# Describes what code is actually running in production

# Global settings
default_ros_domain_id: 0
project_root: ~/betaray

# Host configurations
jetsons:
  nav_jetson:
    hostname: betaray-nav.local
    ros_domain_id: 0
    systemd_services:
      - betaray-navigation-system
      - canary-http-server
    launch_files:
      - betaray_navigation_jetson.launch.py
    packages:
      - betaray_navigation
      - betaray_control

  ai_jetson:
    hostname: betaray-ai.local
    http_only: true  # No external ROS2 communication
    http_ports:
      - 8080
    systemd_services:
      - ai-http-server
    entry_points:
      - ai_server.py

  vision_jetson:
    hostname: vision-jetson.local
    http_only: true
    http_ports:
      - 9091
    systemd_services:
      - betaray-vision-system
    entry_points:
      - vision_server.py

  thor:
    hostname: thor.local
    http_only: true
    http_ports:
      - 8087
      - 8088
      - 8089
      - 30000
    systemd_services:
      - thor-deep-reasoning
      - thor-guardian
      - thor-memory-api
"""
