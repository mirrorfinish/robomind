"""
Launch File Tracer - Trace ROS2 launch files to find deployed nodes.

This module follows the launch file graph to determine which nodes
actually get launched, enabling RoboMind to filter out dead code.

Features:
- Parse Python launch files (LaunchDescription, Node, IncludeLaunchDescription)
- Follow IncludeLaunchDescription calls recursively
- Handle conditional launches (IfCondition, UnlessCondition)
- Extract package/executable mappings
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class TracedNode:
    """A node discovered through launch file tracing."""
    name: str
    package: str
    executable: str
    namespace: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    remappings: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None  # IfCondition/UnlessCondition
    source_launch_file: Optional[str] = None
    line_number: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "package": self.package,
            "executable": self.executable,
            "namespace": self.namespace,
            "parameters": self.parameters,
            "remappings": self.remappings,
            "condition": self.condition,
            "source_launch_file": self.source_launch_file,
            "line_number": self.line_number,
        }

    def is_conditional(self) -> bool:
        """Check if this node has a launch condition."""
        return self.condition is not None

    def get_full_name(self) -> str:
        """Get fully qualified node name with namespace."""
        if self.namespace:
            return f"{self.namespace}/{self.name}"
        return self.name


@dataclass
class LaunchTrace:
    """Result of tracing a launch file hierarchy."""
    root_launch_file: Path
    nodes: List[TracedNode] = field(default_factory=list)
    included_launch_files: List[Path] = field(default_factory=list)
    packages_used: Set[str] = field(default_factory=set)
    arguments: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "root_launch_file": str(self.root_launch_file),
            "nodes": [n.to_dict() for n in self.nodes],
            "included_launch_files": [str(f) for f in self.included_launch_files],
            "packages_used": list(self.packages_used),
            "arguments": self.arguments,
            "errors": self.errors,
        }

    def get_node_names(self) -> List[str]:
        """Get list of all node names."""
        return [n.name for n in self.nodes]

    def get_unconditional_nodes(self) -> List[TracedNode]:
        """Get nodes that always launch (no conditions)."""
        return [n for n in self.nodes if not n.is_conditional()]

    def get_conditional_nodes(self) -> List[TracedNode]:
        """Get nodes that have launch conditions."""
        return [n for n in self.nodes if n.is_conditional()]

    def get_packages(self) -> List[str]:
        """Get list of ROS2 packages used."""
        return list(self.packages_used)

    def summary(self) -> Dict:
        """Get summary of trace results."""
        return {
            "total_nodes": len(self.nodes),
            "unconditional_nodes": len(self.get_unconditional_nodes()),
            "conditional_nodes": len(self.get_conditional_nodes()),
            "included_files": len(self.included_launch_files),
            "packages": len(self.packages_used),
            "errors": len(self.errors),
        }


class LaunchTracer:
    """
    Trace ROS2 launch files to discover deployed nodes.

    Usage:
        tracer = LaunchTracer(project_root=Path("~/betaray"))
        trace = tracer.trace(Path("robot.launch.py"))
        print(f"Found {len(trace.nodes)} nodes")
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        max_depth: int = 10,
    ):
        """
        Initialize launch tracer.

        Args:
            project_root: Root path to search for included launch files
            max_depth: Maximum recursion depth for included files
        """
        self.project_root = Path(project_root) if project_root else None
        self.max_depth = max_depth
        self._visited_files: Set[Path] = set()

    def trace(self, launch_file: Path) -> LaunchTrace:
        """
        Trace a launch file and all its includes.

        Args:
            launch_file: Path to the root launch file

        Returns:
            LaunchTrace with all discovered nodes
        """
        launch_file = Path(launch_file).resolve()

        if not launch_file.exists():
            return LaunchTrace(
                root_launch_file=launch_file,
                errors=[f"Launch file not found: {launch_file}"],
            )

        logger.info(f"Tracing launch file: {launch_file}")

        self._visited_files.clear()
        trace = LaunchTrace(root_launch_file=launch_file)

        self._trace_recursive(launch_file, trace, depth=0)

        logger.info(f"Trace complete: {trace.summary()}")

        return trace

    def _trace_recursive(
        self,
        launch_file: Path,
        trace: LaunchTrace,
        depth: int,
        namespace: Optional[str] = None,
    ):
        """Recursively trace launch files."""
        if depth > self.max_depth:
            trace.errors.append(f"Max depth exceeded at: {launch_file}")
            return

        if launch_file in self._visited_files:
            return  # Avoid cycles

        self._visited_files.add(launch_file)

        if launch_file != trace.root_launch_file:
            trace.included_launch_files.append(launch_file)

        try:
            with open(launch_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Parse as Python AST
            tree = ast.parse(content, filename=str(launch_file))

            # Find nodes and includes
            self._extract_from_ast(tree, launch_file, trace, namespace)

        except SyntaxError as e:
            trace.errors.append(f"Syntax error in {launch_file}: {e}")
        except Exception as e:
            trace.errors.append(f"Error parsing {launch_file}: {e}")

    def _extract_from_ast(
        self,
        tree: ast.AST,
        launch_file: Path,
        trace: LaunchTrace,
        namespace: Optional[str],
    ):
        """Extract nodes and includes from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._process_call(node, launch_file, trace, namespace)

    def _process_call(
        self,
        node: ast.Call,
        launch_file: Path,
        trace: LaunchTrace,
        namespace: Optional[str],
    ):
        """Process a function call node."""
        func_name = self._get_func_name(node)

        if func_name in ("Node", "launch_ros.actions.Node"):
            traced = self._extract_node(node, launch_file, namespace)
            if traced:
                trace.nodes.append(traced)
                trace.packages_used.add(traced.package)

        elif func_name in ("ComposableNode", "launch_ros.descriptions.ComposableNode"):
            traced = self._extract_composable_node(node, launch_file, namespace)
            if traced:
                trace.nodes.append(traced)
                trace.packages_used.add(traced.package)

        elif func_name in ("IncludeLaunchDescription", "launch.actions.IncludeLaunchDescription"):
            included_file = self._extract_include_path(node, launch_file)
            if included_file and included_file.exists():
                self._trace_recursive(included_file, trace, depth=1, namespace=namespace)

        elif func_name in ("GroupAction", "launch.actions.GroupAction"):
            # Extract namespace from group if present
            group_ns = self._extract_namespace_from_group(node)
            new_ns = f"{namespace}/{group_ns}" if namespace and group_ns else (group_ns or namespace)
            # Process nested actions
            for arg in node.args:
                if isinstance(arg, ast.List):
                    for item in arg.elts:
                        if isinstance(item, ast.Call):
                            self._process_call(item, launch_file, trace, new_ns)

        elif func_name in ("DeclareLaunchArgument", "launch.actions.DeclareLaunchArgument"):
            arg_info = self._extract_launch_argument(node)
            if arg_info:
                trace.arguments[arg_info["name"]] = arg_info.get("default")

    def _get_func_name(self, node: ast.Call) -> str:
        """Get the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _extract_node(
        self,
        node: ast.Call,
        launch_file: Path,
        namespace: Optional[str],
    ) -> Optional[TracedNode]:
        """Extract node information from a Node() call."""
        kwargs = self._get_kwargs(node)

        package = kwargs.get("package")
        executable = kwargs.get("executable")
        name = kwargs.get("name", kwargs.get("node_name"))
        node_namespace = kwargs.get("namespace")

        if not package or not executable:
            return None

        # Combine namespaces
        full_namespace = namespace
        if node_namespace:
            full_namespace = f"{namespace}/{node_namespace}" if namespace else node_namespace

        # Extract condition
        condition = None
        if "condition" in kwargs:
            condition = kwargs["condition"]

        return TracedNode(
            name=name or executable,
            package=package,
            executable=executable,
            namespace=full_namespace,
            condition=condition,
            source_launch_file=str(launch_file),
            line_number=node.lineno,
        )

    def _extract_composable_node(
        self,
        node: ast.Call,
        launch_file: Path,
        namespace: Optional[str],
    ) -> Optional[TracedNode]:
        """Extract node information from a ComposableNode() call."""
        kwargs = self._get_kwargs(node)

        package = kwargs.get("package")
        plugin = kwargs.get("plugin")
        name = kwargs.get("name", kwargs.get("node_name"))
        node_namespace = kwargs.get("namespace")

        if not package or not plugin:
            return None

        full_namespace = namespace
        if node_namespace:
            full_namespace = f"{namespace}/{node_namespace}" if namespace else node_namespace

        return TracedNode(
            name=name or plugin.split("::")[-1] if plugin else "unknown",
            package=package,
            executable=plugin or "composable",
            namespace=full_namespace,
            source_launch_file=str(launch_file),
            line_number=node.lineno,
        )

    def _extract_include_path(
        self,
        node: ast.Call,
        current_file: Path,
    ) -> Optional[Path]:
        """Extract the included launch file path."""
        # Look for PythonLaunchDescriptionSource or get_package_share_directory patterns
        for arg in node.args:
            path = self._resolve_launch_path(arg, current_file)
            if path:
                return path

        for kw in node.keywords:
            if kw.arg == "launch_description_source":
                path = self._resolve_launch_path(kw.value, current_file)
                if path:
                    return path

        return None

    def _resolve_launch_path(
        self,
        node: ast.AST,
        current_file: Path,
    ) -> Optional[Path]:
        """Resolve a launch file path from AST node."""
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node)

            if "PythonLaunchDescriptionSource" in func_name:
                # Get the first argument
                if node.args:
                    return self._resolve_launch_path(node.args[0], current_file)

            elif "PathJoinSubstitution" in func_name or "JoinPathSubstitution" in func_name:
                # Try to resolve path components
                parts = []
                for arg in node.args:
                    if isinstance(arg, ast.List):
                        for item in arg.elts:
                            part = self._get_string_value(item)
                            if part:
                                parts.append(part)
                if parts:
                    # This is approximate - real resolution needs FindPackageShare
                    return self._find_launch_file(parts[-1], current_file)

        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Direct string path
            path_str = node.value
            if path_str.endswith(".launch.py") or path_str.endswith(".launch.xml"):
                return self._find_launch_file(path_str, current_file)

        elif isinstance(node, ast.JoinedStr):
            # f-string - try to extract literal parts
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
            joined = "".join(parts)
            if ".launch" in joined:
                return self._find_launch_file(joined, current_file)

        return None

    def _find_launch_file(
        self,
        name: str,
        current_file: Path,
    ) -> Optional[Path]:
        """Try to find a launch file by name."""
        name = Path(name).name  # Get just the filename

        # Search in same directory as current file
        same_dir = current_file.parent / name
        if same_dir.exists():
            return same_dir

        # Search in project root
        if self.project_root:
            for match in self.project_root.rglob(name):
                if match.is_file():
                    return match

        # Search relative to current file's parent directories
        for parent in current_file.parents:
            for match in parent.rglob(name):
                if match.is_file():
                    return match
            # Don't search too far up
            if parent == self.project_root or str(parent) == "/":
                break

        return None

    def _extract_namespace_from_group(self, node: ast.Call) -> Optional[str]:
        """Extract namespace from GroupAction kwargs."""
        kwargs = self._get_kwargs(node)

        if "namespace" in kwargs:
            return kwargs["namespace"]

        # Check for PushRosNamespace in actions
        for arg in node.args:
            if isinstance(arg, ast.List):
                for item in arg.elts:
                    if isinstance(item, ast.Call):
                        func_name = self._get_func_name(item)
                        if "PushRosNamespace" in func_name:
                            ns_kwargs = self._get_kwargs(item)
                            if "namespace" in ns_kwargs:
                                return ns_kwargs["namespace"]
                            if item.args:
                                return self._get_string_value(item.args[0])

        return None

    def _extract_launch_argument(self, node: ast.Call) -> Optional[Dict]:
        """Extract launch argument information."""
        kwargs = self._get_kwargs(node)

        name = kwargs.get("name")
        if not name:
            # Try first positional arg
            if node.args:
                name = self._get_string_value(node.args[0])

        if name:
            return {
                "name": name,
                "default": kwargs.get("default_value"),
                "description": kwargs.get("description"),
            }

        return None

    def _get_kwargs(self, node: ast.Call) -> Dict[str, Any]:
        """Extract keyword arguments from a Call node."""
        kwargs = {}
        for kw in node.keywords:
            if kw.arg:
                kwargs[kw.arg] = self._get_value(kw.value)
        return kwargs

    def _get_value(self, node: ast.AST) -> Any:
        """Extract a value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.NameConstant):  # Python 3.7 compatibility
            return node.value
        elif isinstance(node, ast.Name):
            return f"${{{node.id}}}"  # Variable reference
        elif isinstance(node, ast.List):
            return [self._get_value(item) for item in node.elts]
        elif isinstance(node, ast.Call):
            func_name = self._get_func_name(node)
            if "LaunchConfiguration" in func_name:
                if node.args:
                    return f"${{arg:{self._get_string_value(node.args[0])}}}"
            elif "IfCondition" in func_name or "UnlessCondition" in func_name:
                return func_name
            return f"<{func_name}(...)>"
        return None

    def _get_string_value(self, node: ast.AST) -> Optional[str]:
        """Get string value from an AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        return None


def trace_launch_file(
    launch_file: Path,
    project_root: Optional[Path] = None,
) -> LaunchTrace:
    """
    Convenience function to trace a launch file.

    Args:
        launch_file: Path to the launch file
        project_root: Optional project root for finding includes

    Returns:
        LaunchTrace with discovered nodes
    """
    tracer = LaunchTracer(project_root=project_root)
    return tracer.trace(launch_file)
