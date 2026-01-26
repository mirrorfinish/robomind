"""
ROS2 Parameter File Extractor for RoboMind

Parses ROS2 YAML parameter files to extract:
- Node parameter configurations (ros__parameters sections)
- Parameter names, types, and values
- Nested parameter structures

Standard ROS2 parameter file format:
    node_name:
      ros__parameters:
        param_name: value
        nested:
          sub_param: value
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ParameterValue:
    """A parameter with its value and metadata."""
    name: str
    value: Any
    value_type: str  # int, float, string, bool, list, dict
    path: str  # Full path like "node.ros__parameters.nested.param"
    line_number: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.value_type,
            "path": self.path,
        }


@dataclass
class NodeParameters:
    """Parameters for a specific node."""
    node_name: str
    parameters: List[ParameterValue] = field(default_factory=list)
    file_path: Optional[Path] = None

    def get_parameter(self, name: str) -> Optional[ParameterValue]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_parameters_by_prefix(self, prefix: str) -> List[ParameterValue]:
        """Get all parameters starting with a prefix."""
        return [p for p in self.parameters if p.name.startswith(prefix)]

    def to_dict(self) -> Dict:
        return {
            "node_name": self.node_name,
            "parameters": [p.to_dict() for p in self.parameters],
            "file_path": str(self.file_path) if self.file_path else None,
            "parameter_count": len(self.parameters),
        }


@dataclass
class ParameterFileInfo:
    """Complete information about a parameter file."""
    file_path: Path
    nodes: List[NodeParameters] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)
    raw_yaml: Optional[Dict] = None

    @property
    def total_parameters(self) -> int:
        """Total number of parameters across all nodes."""
        return sum(len(n.parameters) for n in self.nodes)

    def get_node(self, node_name: str) -> Optional[NodeParameters]:
        """Get parameters for a specific node."""
        for node in self.nodes:
            if node.node_name == node_name:
                return node
        return None

    def summary(self) -> Dict:
        return {
            "file": self.file_path.name,
            "nodes": len(self.nodes),
            "total_parameters": self.total_parameters,
            "node_names": [n.node_name for n in self.nodes],
            "has_errors": len(self.parse_errors) > 0,
        }

    def to_dict(self) -> Dict:
        return {
            "file_path": str(self.file_path),
            "nodes": [n.to_dict() for n in self.nodes],
            "summary": self.summary(),
            "parse_errors": self.parse_errors,
        }


class ParameterExtractor:
    """
    Extract parameters from ROS2 YAML parameter files.

    Handles standard ROS2 parameter file format:
        node_name:
          ros__parameters:
            param1: value1
            nested:
              param2: value2

    Usage:
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(Path("config.yaml"))
        print(info.summary())
    """

    def __init__(self):
        # Note: Order matters! bool must come before int since bool is a subclass of int in Python
        self.type_checks = [
            (bool, "bool"),  # Check bool FIRST (before int)
            (int, "int"),
            (float, "float"),
            (str, "string"),
            (list, "list"),
            (dict, "dict"),
            (type(None), "null"),
        ]

    def extract_from_file(self, file_path: Path) -> ParameterFileInfo:
        """
        Extract parameters from a YAML file.

        Args:
            file_path: Path to the YAML parameter file

        Returns:
            ParameterFileInfo with all extracted parameters
        """
        info = ParameterFileInfo(file_path=file_path)

        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            if data is None:
                info.parse_errors.append("Empty YAML file")
                return info

            if not isinstance(data, dict):
                info.parse_errors.append(f"Expected dict at root, got {type(data).__name__}")
                return info

            info.raw_yaml = data
            self._extract_nodes(data, info)

        except yaml.YAMLError as e:
            info.parse_errors.append(f"YAML parse error: {e}")
            logger.error(f"Failed to parse {file_path}: {e}")
        except Exception as e:
            info.parse_errors.append(f"Error: {e}")
            logger.error(f"Error extracting parameters from {file_path}: {e}")

        return info

    def _extract_nodes(self, data: Dict, info: ParameterFileInfo):
        """Extract node parameter configurations from parsed YAML."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Check if this is a node configuration
                if "ros__parameters" in value:
                    node_params = self._extract_node_parameters(
                        key, value["ros__parameters"], info.file_path
                    )
                    info.nodes.append(node_params)
                else:
                    # Could be nested node configuration or other structure
                    # Try to find ros__parameters deeper
                    self._search_for_parameters(key, value, info)

    def _search_for_parameters(self, prefix: str, data: Dict, info: ParameterFileInfo):
        """Recursively search for ros__parameters sections."""
        for key, value in data.items():
            if key == "ros__parameters" and isinstance(value, dict):
                # Found parameters at this level
                node_name = prefix.rstrip(".")
                node_params = self._extract_node_parameters(
                    node_name, value, info.file_path
                )
                info.nodes.append(node_params)
            elif isinstance(value, dict):
                # Continue searching
                self._search_for_parameters(f"{prefix}.{key}", value, info)

    def _extract_node_parameters(
        self, node_name: str, params_dict: Dict, file_path: Path
    ) -> NodeParameters:
        """Extract parameters for a single node."""
        node_params = NodeParameters(
            node_name=node_name,
            file_path=file_path,
        )

        self._flatten_parameters(params_dict, "", node_params)

        return node_params

    def _flatten_parameters(
        self, data: Dict, prefix: str, node_params: NodeParameters
    ):
        """Flatten nested parameter structure into list of ParameterValue."""
        for key, value in data.items():
            full_path = f"{prefix}.{key}".lstrip(".")

            if isinstance(value, dict):
                # Nested parameter group - recurse
                self._flatten_parameters(value, full_path, node_params)
            else:
                # Leaf parameter
                param = ParameterValue(
                    name=key,
                    value=value,
                    value_type=self._get_type_name(value),
                    path=full_path,
                )
                node_params.parameters.append(param)

    def _get_type_name(self, value: Any) -> str:
        """Get the type name for a value."""
        # Check in order - bool must be checked before int
        for python_type, type_name in self.type_checks:
            if isinstance(value, python_type):
                return type_name
        return "unknown"

    def extract_from_dict(self, data: Dict, source_name: str = "inline") -> ParameterFileInfo:
        """
        Extract parameters from a dictionary (e.g., from launch file).

        Args:
            data: Dictionary with parameter data
            source_name: Name to use for the file path

        Returns:
            ParameterFileInfo with extracted parameters
        """
        info = ParameterFileInfo(file_path=Path(source_name))
        info.raw_yaml = data

        try:
            self._extract_nodes(data, info)
        except Exception as e:
            info.parse_errors.append(f"Error: {e}")
            logger.error(f"Error extracting parameters: {e}")

        return info


@dataclass
class ParameterCollection:
    """Collection of parameters from multiple files."""
    files: List[ParameterFileInfo] = field(default_factory=list)

    @property
    def all_nodes(self) -> List[NodeParameters]:
        """Get all node parameters from all files."""
        nodes = []
        for f in self.files:
            nodes.extend(f.nodes)
        return nodes

    def get_node_names(self) -> Set[str]:
        """Get all unique node names."""
        return {n.node_name for n in self.all_nodes}

    def get_parameters_for_node(self, node_name: str) -> List[ParameterValue]:
        """Get all parameters for a specific node from all files."""
        params = []
        for f in self.files:
            node = f.get_node(node_name)
            if node:
                params.extend(node.parameters)
        return params

    def summary(self) -> Dict:
        return {
            "files": len(self.files),
            "nodes": len(self.all_nodes),
            "unique_node_names": len(self.get_node_names()),
            "total_parameters": sum(f.total_parameters for f in self.files),
        }

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary(),
            "files": [f.to_dict() for f in self.files],
            "node_names": sorted(self.get_node_names()),
        }


class ConfigScanner:
    """
    Scan a project for ROS2 parameter files.

    Looks for:
    - config/*.yaml
    - params/*.yaml
    - *_params.yaml
    - *_config.yaml
    """

    CONFIG_PATTERNS = [
        "config/*.yaml",
        "params/*.yaml",
        "parameters/*.yaml",
        "*_params.yaml",
        "*_config.yaml",
        "config/**/*.yaml",
        "params/**/*.yaml",
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.extractor = ParameterExtractor()

    def scan(self) -> ParameterCollection:
        """Scan project for parameter files and extract parameters."""
        collection = ParameterCollection()
        found_files: Set[Path] = set()

        # Find all matching files
        for pattern in self.CONFIG_PATTERNS:
            for match in self.project_root.rglob(pattern.replace("**", "*")):
                if match.is_file() and match not in found_files:
                    # Skip common non-parameter files
                    if self._is_parameter_file(match):
                        found_files.add(match)

        # Extract parameters from each file
        for file_path in sorted(found_files):
            info = self.extractor.extract_from_file(file_path)
            if info.nodes or info.parse_errors:
                collection.files.append(info)

        return collection

    def _is_parameter_file(self, file_path: Path) -> bool:
        """Check if a file is likely a ROS2 parameter file."""
        # Skip build/install directories
        path_str = str(file_path)
        if "/build/" in path_str or "/install/" in path_str:
            return False
        if "/__pycache__/" in path_str:
            return False

        # Try to peek at the file to check for ros__parameters
        try:
            with open(file_path, "r") as f:
                content = f.read(2000)  # Read first 2KB
                return "ros__parameters" in content or "parameters:" in content.lower()
        except Exception:
            return False


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python param_extractor.py <config.yaml or project_path>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        extractor = ParameterExtractor()
        info = extractor.extract_from_file(path)

        print("\n" + "=" * 60)
        print(f"PARAMETER FILE: {path.name}")
        print("=" * 60)
        print(json.dumps(info.summary(), indent=2))

        for node in info.nodes:
            print(f"\n  Node: {node.node_name}")
            print(f"  Parameters: {len(node.parameters)}")
            for param in node.parameters[:10]:
                print(f"    {param.path}: {param.value} ({param.value_type})")
            if len(node.parameters) > 10:
                print(f"    ... and {len(node.parameters) - 10} more")

    elif path.is_dir():
        scanner = ConfigScanner(path)
        collection = scanner.scan()

        print("\n" + "=" * 60)
        print(f"PROJECT SCAN: {path}")
        print("=" * 60)
        print(json.dumps(collection.summary(), indent=2))

        print("\n" + "=" * 60)
        print("PARAMETER FILES")
        print("=" * 60)
        for f in collection.files:
            print(f"\n  {f.file_path.relative_to(path)}")
            for node in f.nodes:
                print(f"    Node: {node.node_name} ({len(node.parameters)} params)")
