"""
ROS2 Launch File Analyzer for RoboMind

Parses ROS2 Python launch files (.launch.py) to extract:
- Launch arguments (DeclareLaunchArgument)
- Node declarations with parameters, remappings
- Composable nodes and containers
- Timed actions (delayed node starts)
- Conditional logic (IfCondition, UnlessCondition)
- Execution processes

Limitations:
- Static analysis only - cannot resolve runtime values
- LaunchConfiguration substitutions shown as references
- Complex conditional logic may not be fully captured
"""

import ast
import logging
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LaunchArgument:
    """A declared launch argument."""
    name: str
    default_value: Optional[str] = None
    description: str = ""
    line_number: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "default_value": self.default_value,
            "description": self.description,
            "line_number": self.line_number,
        }


@dataclass
class LaunchParameter:
    """A parameter passed to a node."""
    name: str
    value: Any
    is_substitution: bool = False  # True if value is LaunchConfiguration etc.

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": str(self.value) if self.is_substitution else self.value,
            "is_substitution": self.is_substitution,
        }


@dataclass
class LaunchRemapping:
    """A topic remapping."""
    from_topic: str
    to_topic: str

    def to_dict(self) -> Dict:
        return {"from": self.from_topic, "to": self.to_topic}


@dataclass
class LaunchNode:
    """A node declared in a launch file."""
    package: str
    executable: str
    name: Optional[str] = None
    namespace: str = ""
    parameters: List[LaunchParameter] = field(default_factory=list)
    remappings: List[LaunchRemapping] = field(default_factory=list)
    condition: Optional[str] = None  # e.g., "IfCondition(enable_voice)"
    respawn: bool = False
    respawn_delay: float = 0.0
    output: str = "screen"
    delay: float = 0.0  # Delay from TimerAction wrapper
    line_number: int = 0
    is_composable: bool = False
    container_name: Optional[str] = None  # For composable nodes

    def to_dict(self) -> Dict:
        return {
            "package": self.package,
            "executable": self.executable,
            "name": self.name,
            "namespace": self.namespace,
            "parameters": [p.to_dict() for p in self.parameters],
            "remappings": [r.to_dict() for r in self.remappings],
            "condition": self.condition,
            "respawn": self.respawn,
            "respawn_delay": self.respawn_delay,
            "output": self.output,
            "delay": self.delay,
            "line_number": self.line_number,
            "is_composable": self.is_composable,
            "container_name": self.container_name,
        }


@dataclass
class LaunchExecuteProcess:
    """An ExecuteProcess action."""
    cmd: List[str]
    name: Optional[str] = None
    condition: Optional[str] = None
    delay: float = 0.0
    line_number: int = 0

    def to_dict(self) -> Dict:
        return {
            "cmd": self.cmd,
            "name": self.name,
            "condition": self.condition,
            "delay": self.delay,
            "line_number": self.line_number,
        }


@dataclass
class LaunchInclude:
    """An included launch file."""
    package: str
    launch_file: str
    launch_arguments: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    delay: float = 0.0
    line_number: int = 0

    def to_dict(self) -> Dict:
        return {
            "package": self.package,
            "launch_file": self.launch_file,
            "launch_arguments": self.launch_arguments,
            "condition": self.condition,
            "delay": self.delay,
            "line_number": self.line_number,
        }


@dataclass
class ComposableNodeContainer:
    """A composable node container."""
    name: str
    namespace: str = ""
    package: str = "rclcpp_components"
    executable: str = "component_container"
    nodes: List[LaunchNode] = field(default_factory=list)
    line_number: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "package": self.package,
            "executable": self.executable,
            "nodes": [n.to_dict() for n in self.nodes],
            "line_number": self.line_number,
        }


@dataclass
class LaunchFileInfo:
    """Complete information about a launch file."""
    file_path: Path
    arguments: List[LaunchArgument] = field(default_factory=list)
    nodes: List[LaunchNode] = field(default_factory=list)
    containers: List[ComposableNodeContainer] = field(default_factory=list)
    processes: List[LaunchExecuteProcess] = field(default_factory=list)
    includes: List[LaunchInclude] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        """Total nodes including composable nodes."""
        composable_count = sum(len(c.nodes) for c in self.containers)
        return len(self.nodes) + composable_count

    def get_launch_sequence(self) -> List[Dict]:
        """Get nodes in launch sequence order (by delay)."""
        items = []

        for node in self.nodes:
            items.append({
                "type": "node",
                "delay": node.delay,
                "name": node.name or node.executable,
                "package": node.package,
                "item": node,
            })

        for container in self.containers:
            items.append({
                "type": "container",
                "delay": 0.0,
                "name": container.name,
                "package": container.package,
                "item": container,
            })

        for process in self.processes:
            items.append({
                "type": "process",
                "delay": process.delay,
                "name": process.name or "process",
                "package": "",
                "item": process,
            })

        return sorted(items, key=lambda x: x["delay"])

    def to_dict(self) -> Dict:
        return {
            "file_path": str(self.file_path),
            "arguments": [a.to_dict() for a in self.arguments],
            "nodes": [n.to_dict() for n in self.nodes],
            "containers": [c.to_dict() for c in self.containers],
            "processes": [p.to_dict() for p in self.processes],
            "includes": [i.to_dict() for i in self.includes],
            "total_nodes": self.total_nodes,
            "parse_errors": self.parse_errors,
        }

    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "file": self.file_path.name,
            "arguments": len(self.arguments),
            "nodes": len(self.nodes),
            "composable_containers": len(self.containers),
            "composable_nodes": sum(len(c.nodes) for c in self.containers),
            "processes": len(self.processes),
            "includes": len(self.includes),
            "total_nodes": self.total_nodes,
            "max_delay": max(
                [n.delay for n in self.nodes] +
                [p.delay for p in self.processes] +
                [0.0]
            ),
            "has_conditions": any(n.condition for n in self.nodes),
            "has_errors": len(self.parse_errors) > 0,
        }


class LaunchFileAnalyzer:
    """
    Analyze ROS2 Python launch files (.launch.py).

    Uses AST parsing to extract launch file structure without executing the code.

    Usage:
        analyzer = LaunchFileAnalyzer()
        info = analyzer.analyze_file(Path("robot.launch.py"))
        print(info.summary())
    """

    def __init__(self):
        self.current_delay = 0.0  # Track delay from TimerAction wrappers

    def analyze_file(self, file_path: Path) -> LaunchFileInfo:
        """
        Analyze a launch file and extract all launch information.

        Args:
            file_path: Path to the .launch.py file

        Returns:
            LaunchFileInfo with all extracted information
        """
        info = LaunchFileInfo(file_path=file_path)

        try:
            with open(file_path, "r") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))
            self._analyze_tree(tree, info)

        except SyntaxError as e:
            info.parse_errors.append(f"Syntax error: {e}")
            logger.error(f"Failed to parse {file_path}: {e}")
        except Exception as e:
            info.parse_errors.append(f"Parse error: {e}")
            logger.error(f"Error analyzing {file_path}: {e}")

        return info

    def _analyze_tree(self, tree: ast.AST, info: LaunchFileInfo):
        """Analyze the AST tree for launch components."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._analyze_call(node, info)

    def _analyze_call(self, node: ast.Call, info: LaunchFileInfo, delay: float = 0.0):
        """Analyze a function call for launch constructs."""
        func_name = self._get_call_name(node)

        if func_name == "DeclareLaunchArgument":
            arg = self._parse_declare_launch_argument(node)
            if arg:
                info.arguments.append(arg)

        elif func_name == "Node":
            launch_node = self._parse_node(node, delay)
            if launch_node:
                info.nodes.append(launch_node)

        elif func_name == "ComposableNodeContainer":
            container = self._parse_composable_container(node)
            if container:
                info.containers.append(container)

        elif func_name == "ExecuteProcess":
            process = self._parse_execute_process(node, delay)
            if process:
                info.processes.append(process)

        elif func_name == "IncludeLaunchDescription":
            include = self._parse_include_launch(node, delay)
            if include:
                info.includes.append(include)

        elif func_name == "TimerAction":
            self._parse_timer_action(node, info)

        elif func_name == "GroupAction":
            self._parse_group_action(node, info, delay)

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _parse_declare_launch_argument(self, node: ast.Call) -> Optional[LaunchArgument]:
        """Parse DeclareLaunchArgument call."""
        try:
            name = None
            default_value = None
            description = ""

            # First positional argument is the name
            if node.args:
                name = self._get_string_value(node.args[0])

            # Look for keyword arguments
            for kw in node.keywords:
                if kw.arg == "default_value":
                    default_value = self._get_value(kw.value)
                elif kw.arg == "description":
                    description = self._get_string_value(kw.value) or ""
                elif kw.arg is None and isinstance(kw.value, ast.Constant):
                    # Positional-like keyword (e.g., first unnamed arg)
                    if name is None:
                        name = self._get_string_value(kw.value)

            if name:
                return LaunchArgument(
                    name=name,
                    default_value=str(default_value) if default_value is not None else None,
                    description=description,
                    line_number=node.lineno,
                )
        except Exception as e:
            logger.debug(f"Failed to parse DeclareLaunchArgument: {e}")
        return None

    def _parse_node(self, node: ast.Call, delay: float = 0.0) -> Optional[LaunchNode]:
        """Parse Node call."""
        try:
            launch_node = LaunchNode(
                package="",
                executable="",
                delay=delay,
                line_number=node.lineno,
            )

            for kw in node.keywords:
                if kw.arg == "package":
                    launch_node.package = self._get_string_value(kw.value) or ""
                elif kw.arg == "executable":
                    launch_node.executable = self._get_string_value(kw.value) or ""
                elif kw.arg == "name":
                    launch_node.name = self._get_string_value(kw.value)
                elif kw.arg == "namespace":
                    launch_node.namespace = self._get_string_value(kw.value) or ""
                elif kw.arg == "parameters":
                    launch_node.parameters = self._parse_parameters(kw.value)
                elif kw.arg == "remappings":
                    launch_node.remappings = self._parse_remappings(kw.value)
                elif kw.arg == "condition":
                    launch_node.condition = self._parse_condition(kw.value)
                elif kw.arg == "respawn":
                    launch_node.respawn = self._get_bool_value(kw.value)
                elif kw.arg == "respawn_delay":
                    launch_node.respawn_delay = self._get_number_value(kw.value)
                elif kw.arg == "output":
                    launch_node.output = self._get_string_value(kw.value) or "screen"

            if launch_node.package and launch_node.executable:
                return launch_node
        except Exception as e:
            logger.debug(f"Failed to parse Node: {e}")
        return None

    def _parse_composable_container(self, node: ast.Call) -> Optional[ComposableNodeContainer]:
        """Parse ComposableNodeContainer call."""
        try:
            container = ComposableNodeContainer(
                name="",
                line_number=node.lineno,
            )

            for kw in node.keywords:
                if kw.arg == "name":
                    container.name = self._get_string_value(kw.value) or ""
                elif kw.arg == "namespace":
                    container.namespace = self._get_string_value(kw.value) or ""
                elif kw.arg == "package":
                    container.package = self._get_string_value(kw.value) or "rclcpp_components"
                elif kw.arg == "executable":
                    container.executable = self._get_string_value(kw.value) or "component_container"
                elif kw.arg == "composable_node_descriptions":
                    container.nodes = self._parse_composable_nodes(kw.value, container.name)

            if container.name:
                return container
        except Exception as e:
            logger.debug(f"Failed to parse ComposableNodeContainer: {e}")
        return None

    def _parse_composable_nodes(self, node: ast.AST, container_name: str) -> List[LaunchNode]:
        """Parse composable node descriptions."""
        nodes = []

        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Call):
                    comp_node = self._parse_composable_node(item, container_name)
                    if comp_node:
                        nodes.append(comp_node)

        return nodes

    def _parse_composable_node(self, node: ast.Call, container_name: str) -> Optional[LaunchNode]:
        """Parse a ComposableNode call."""
        try:
            func_name = self._get_call_name(node)
            if func_name != "ComposableNode":
                return None

            launch_node = LaunchNode(
                package="",
                executable="",
                is_composable=True,
                container_name=container_name,
                line_number=node.lineno,
            )

            for kw in node.keywords:
                if kw.arg == "package":
                    launch_node.package = self._get_string_value(kw.value) or ""
                elif kw.arg == "plugin":
                    launch_node.executable = self._get_string_value(kw.value) or ""
                elif kw.arg == "name":
                    launch_node.name = self._get_string_value(kw.value)
                elif kw.arg == "namespace":
                    launch_node.namespace = self._get_string_value(kw.value) or ""
                elif kw.arg == "parameters":
                    launch_node.parameters = self._parse_parameters(kw.value)
                elif kw.arg == "remappings":
                    launch_node.remappings = self._parse_remappings(kw.value)

            if launch_node.package:
                return launch_node
        except Exception as e:
            logger.debug(f"Failed to parse ComposableNode: {e}")
        return None

    def _parse_execute_process(self, node: ast.Call, delay: float = 0.0) -> Optional[LaunchExecuteProcess]:
        """Parse ExecuteProcess call."""
        try:
            process = LaunchExecuteProcess(
                cmd=[],
                delay=delay,
                line_number=node.lineno,
            )

            for kw in node.keywords:
                if kw.arg == "cmd":
                    process.cmd = self._parse_cmd_list(kw.value)
                elif kw.arg == "name":
                    process.name = self._get_string_value(kw.value)
                elif kw.arg == "condition":
                    process.condition = self._parse_condition(kw.value)

            if process.cmd:
                return process
        except Exception as e:
            logger.debug(f"Failed to parse ExecuteProcess: {e}")
        return None

    def _parse_include_launch(self, node: ast.Call, delay: float = 0.0) -> Optional[LaunchInclude]:
        """Parse IncludeLaunchDescription call."""
        try:
            include = LaunchInclude(
                package="",
                launch_file="",
                delay=delay,
                line_number=node.lineno,
            )

            # Look for PythonLaunchDescriptionSource in args
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    func_name = self._get_call_name(arg)
                    if "LaunchDescriptionSource" in func_name:
                        # Try to extract file path from the source
                        include.launch_file = self._extract_launch_file_path(arg)

            # Look for launch_arguments in keywords
            for kw in node.keywords:
                if kw.arg == "launch_arguments":
                    include.launch_arguments = self._parse_launch_arguments_dict(kw.value)
                elif kw.arg == "condition":
                    include.condition = self._parse_condition(kw.value)

            return include if include.launch_file else None
        except Exception as e:
            logger.debug(f"Failed to parse IncludeLaunchDescription: {e}")
        return None

    def _parse_timer_action(self, node: ast.Call, info: LaunchFileInfo):
        """Parse TimerAction and extract delayed actions."""
        try:
            period = 0.0
            actions = []

            for kw in node.keywords:
                if kw.arg == "period":
                    period = self._get_number_value(kw.value)
                elif kw.arg == "actions":
                    actions = self._get_action_list(kw.value)

            # Process each action with the delay
            for action_node in actions:
                if isinstance(action_node, ast.Call):
                    self._analyze_call(action_node, info, delay=period)

        except Exception as e:
            logger.debug(f"Failed to parse TimerAction: {e}")

    def _parse_group_action(self, node: ast.Call, info: LaunchFileInfo, delay: float = 0.0):
        """Parse GroupAction and extract grouped actions."""
        try:
            for kw in node.keywords:
                if kw.arg == "actions":
                    actions = self._get_action_list(kw.value)
                    for action_node in actions:
                        if isinstance(action_node, ast.Call):
                            self._analyze_call(action_node, info, delay=delay)

            # Also check positional args
            for arg in node.args:
                if isinstance(arg, ast.List):
                    for action_node in arg.elts:
                        if isinstance(action_node, ast.Call):
                            self._analyze_call(action_node, info, delay=delay)

        except Exception as e:
            logger.debug(f"Failed to parse GroupAction: {e}")

    def _parse_parameters(self, node: ast.AST) -> List[LaunchParameter]:
        """Parse node parameters list."""
        params = []

        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Dict):
                    params.extend(self._parse_param_dict(item))
                elif isinstance(item, ast.Call):
                    # Could be PathJoinSubstitution for a parameter file
                    params.append(LaunchParameter(
                        name="<file>",
                        value=self._stringify_ast(item),
                        is_substitution=True,
                    ))

        return params

    def _parse_param_dict(self, node: ast.Dict) -> List[LaunchParameter]:
        """Parse a parameter dictionary."""
        params = []

        for key, value in zip(node.keys, node.values):
            if key is None:
                continue

            name = self._get_string_value(key) or ""
            param_value = self._get_value(value)
            is_sub = isinstance(value, ast.Call)

            params.append(LaunchParameter(
                name=name,
                value=param_value,
                is_substitution=is_sub,
            ))

        return params

    def _parse_remappings(self, node: ast.AST) -> List[LaunchRemapping]:
        """Parse remappings list."""
        remappings = []

        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) == 2:
                    from_topic = self._get_string_value(item.elts[0]) or ""
                    to_topic = self._get_string_value(item.elts[1]) or ""
                    if from_topic and to_topic:
                        remappings.append(LaunchRemapping(from_topic, to_topic))

        return remappings

    def _parse_condition(self, node: ast.AST) -> Optional[str]:
        """Parse a condition (IfCondition, UnlessCondition)."""
        if isinstance(node, ast.Call):
            func_name = self._get_call_name(node)
            if func_name in ("IfCondition", "UnlessCondition"):
                if node.args:
                    arg_str = self._stringify_ast(node.args[0])
                    return f"{func_name}({arg_str})"
        return None

    def _parse_cmd_list(self, node: ast.AST) -> List[str]:
        """Parse command list for ExecuteProcess."""
        cmd = []

        if isinstance(node, ast.List):
            for item in node.elts:
                value = self._get_string_value(item)
                if value:
                    cmd.append(value)
                elif isinstance(item, ast.Call):
                    cmd.append(f"<{self._get_call_name(item)}>")

        return cmd

    def _parse_launch_arguments_dict(self, node: ast.AST) -> Dict[str, str]:
        """Parse launch arguments dictionary."""
        args = {}

        if isinstance(node, ast.List):
            for item in node.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) == 2:
                    key = self._get_string_value(item.elts[0])
                    value = self._get_value(item.elts[1])
                    if key:
                        args[key] = str(value) if value else ""

        return args

    def _get_action_list(self, node: ast.AST) -> List[ast.AST]:
        """Get a list of action nodes."""
        if isinstance(node, ast.List):
            return node.elts
        return []

    def _extract_launch_file_path(self, node: ast.Call) -> str:
        """Extract launch file path from LaunchDescriptionSource."""
        for arg in node.args:
            if isinstance(arg, ast.Call):
                # PathJoinSubstitution, etc.
                return self._stringify_ast(arg)
            elif isinstance(arg, ast.Constant):
                return str(arg.value)
        return ""

    def _get_string_value(self, node: ast.AST) -> Optional[str]:
        """Get string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        return None

    def _get_bool_value(self, node: ast.AST) -> bool:
        """Get boolean value from AST node."""
        if isinstance(node, ast.Constant):
            return bool(node.value)
        elif isinstance(node, ast.NameConstant):  # Python 3.7
            return bool(node.value)
        return False

    def _get_number_value(self, node: ast.AST) -> float:
        """Get numeric value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.Num):  # Python 3.7
            return float(node.n)
        return 0.0

    def _get_value(self, node: ast.AST) -> Any:
        """Get any value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Call):
            return self._stringify_ast(node)
        elif isinstance(node, ast.List):
            return [self._get_value(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._get_value(k): self._get_value(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        return None

    def _stringify_ast(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, ast.Call):
            func_name = self._get_call_name(node)
            if node.args:
                arg_str = ", ".join(self._stringify_ast(a) for a in node.args[:2])
                return f"{func_name}({arg_str})"
            return f"{func_name}()"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._stringify_ast(node.value)}.{node.attr}"
        return str(type(node).__name__)


@dataclass
class LaunchTopology:
    """Complete launch topology from multiple launch files."""
    launch_files: List[LaunchFileInfo] = field(default_factory=list)

    @property
    def all_nodes(self) -> List[LaunchNode]:
        """Get all nodes from all launch files."""
        nodes = []
        for lf in self.launch_files:
            nodes.extend(lf.nodes)
            for container in lf.containers:
                nodes.extend(container.nodes)
        return nodes

    @property
    def all_arguments(self) -> List[LaunchArgument]:
        """Get all arguments from all launch files."""
        args = []
        for lf in self.launch_files:
            args.extend(lf.arguments)
        return args

    def get_packages(self) -> Set[str]:
        """Get all unique packages referenced."""
        packages = set()
        for node in self.all_nodes:
            if node.package:
                packages.add(node.package)
        return packages

    def summary(self) -> Dict:
        """Generate summary statistics."""
        all_nodes = self.all_nodes
        return {
            "launch_files": len(self.launch_files),
            "total_nodes": len(all_nodes),
            "composable_nodes": sum(1 for n in all_nodes if n.is_composable),
            "conditional_nodes": sum(1 for n in all_nodes if n.condition),
            "unique_packages": len(self.get_packages()),
            "total_arguments": len(self.all_arguments),
            "max_delay": max(
                [n.delay for n in all_nodes] + [0.0]
            ),
        }

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary(),
            "launch_files": [lf.to_dict() for lf in self.launch_files],
            "packages": sorted(self.get_packages()),
        }


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python launch_analyzer.py <launch_file.py>")
        sys.exit(1)

    launch_file = Path(sys.argv[1])

    analyzer = LaunchFileAnalyzer()
    info = analyzer.analyze_file(launch_file)

    print("\n" + "=" * 60)
    print(f"LAUNCH FILE: {launch_file.name}")
    print("=" * 60)
    print(json.dumps(info.summary(), indent=2))

    print("\n" + "=" * 60)
    print("LAUNCH ARGUMENTS")
    print("=" * 60)
    for arg in info.arguments:
        print(f"  {arg.name}: {arg.default_value} - {arg.description}")

    print("\n" + "=" * 60)
    print("LAUNCH SEQUENCE")
    print("=" * 60)
    for item in info.get_launch_sequence():
        delay_str = f"[+{item['delay']:.1f}s]" if item['delay'] > 0 else "[0s]"
        print(f"  {delay_str} {item['type']}: {item['package']}/{item['name']}")
