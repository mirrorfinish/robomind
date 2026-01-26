"""
ROS2 Node Extractor for RoboMind

Extracts detailed ROS2 node information from Python source files:
- Node class detection (inherits from rclpy.node.Node)
- Publishers (topic, message type, QoS)
- Subscribers (topic, message type, callback)
- Timers (period, callback)
- Services (name, type, callback)
- Service clients
- Action servers/clients
- Parameters (name, default value, type)

Uses AST analysis to statically extract this information without
requiring ROS2 to be running.
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field

from robomind.core.parser import PythonParser, ParseResult, ClassInfo

logger = logging.getLogger(__name__)


@dataclass
class PublisherInfo:
    """Information about a ROS2 publisher."""
    topic: str
    msg_type: str
    qos: int = 10
    variable_name: Optional[str] = None
    line_number: int = 0


@dataclass
class SubscriberInfo:
    """Information about a ROS2 subscriber."""
    topic: str
    msg_type: str
    callback: str
    qos: int = 10
    variable_name: Optional[str] = None
    line_number: int = 0


@dataclass
class TimerInfo:
    """Information about a ROS2 timer."""
    period: float  # seconds
    callback: str
    variable_name: Optional[str] = None
    line_number: int = 0

    @property
    def frequency_hz(self) -> float:
        """Get timer frequency in Hz."""
        if self.period > 0:
            return 1.0 / self.period
        return 0.0


@dataclass
class ServiceInfo:
    """Information about a ROS2 service server."""
    name: str
    srv_type: str
    callback: str
    variable_name: Optional[str] = None
    line_number: int = 0


@dataclass
class ServiceClientInfo:
    """Information about a ROS2 service client."""
    name: str
    srv_type: str
    variable_name: Optional[str] = None
    line_number: int = 0


@dataclass
class ActionServerInfo:
    """Information about a ROS2 action server."""
    name: str
    action_type: str
    callback: str
    variable_name: Optional[str] = None
    line_number: int = 0


@dataclass
class ActionClientInfo:
    """Information about a ROS2 action client."""
    name: str
    action_type: str
    variable_name: Optional[str] = None
    line_number: int = 0


@dataclass
class ParameterInfo:
    """Information about a ROS2 parameter."""
    name: str
    default_value: Any = None
    param_type: Optional[str] = None  # Inferred type
    line_number: int = 0


@dataclass
class ROS2NodeInfo:
    """Complete information about a ROS2 node."""
    name: str  # Node name from super().__init__('node_name')
    class_name: str  # Python class name
    file_path: Path
    line_number: int
    end_line: int

    # Communication
    publishers: List[PublisherInfo] = field(default_factory=list)
    subscribers: List[SubscriberInfo] = field(default_factory=list)
    timers: List[TimerInfo] = field(default_factory=list)
    services: List[ServiceInfo] = field(default_factory=list)
    service_clients: List[ServiceClientInfo] = field(default_factory=list)
    action_servers: List[ActionServerInfo] = field(default_factory=list)
    action_clients: List[ActionClientInfo] = field(default_factory=list)

    # Configuration
    parameters: List[ParameterInfo] = field(default_factory=list)

    # Metadata
    base_classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    package_name: Optional[str] = None

    # TF related
    has_tf_broadcaster: bool = False
    has_tf_listener: bool = False

    def get_published_topics(self) -> List[str]:
        """Get list of topics this node publishes to."""
        return [pub.topic for pub in self.publishers]

    def get_subscribed_topics(self) -> List[str]:
        """Get list of topics this node subscribes to."""
        return [sub.topic for sub in self.subscribers]

    def summary(self) -> Dict:
        """Generate summary of this node."""
        return {
            "name": self.name,
            "class": self.class_name,
            "file": str(self.file_path),
            "line": self.line_number,
            "publishers": len(self.publishers),
            "subscribers": len(self.subscribers),
            "timers": len(self.timers),
            "services": len(self.services),
            "service_clients": len(self.service_clients),
            "parameters": len(self.parameters),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "class_name": self.class_name,
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "package": self.package_name,
            "docstring": self.docstring,
            "publishers": [
                {"topic": p.topic, "msg_type": p.msg_type, "qos": p.qos}
                for p in self.publishers
            ],
            "subscribers": [
                {"topic": s.topic, "msg_type": s.msg_type, "callback": s.callback, "qos": s.qos}
                for s in self.subscribers
            ],
            "timers": [
                {"period": t.period, "frequency_hz": t.frequency_hz, "callback": t.callback}
                for t in self.timers
            ],
            "services": [
                {"name": s.name, "srv_type": s.srv_type, "callback": s.callback}
                for s in self.services
            ],
            "service_clients": [
                {"name": c.name, "srv_type": c.srv_type}
                for c in self.service_clients
            ],
            "parameters": [
                {"name": p.name, "default": p.default_value, "type": p.param_type}
                for p in self.parameters
            ],
            "has_tf_broadcaster": self.has_tf_broadcaster,
            "has_tf_listener": self.has_tf_listener,
        }


class ROS2NodeExtractor:
    """
    Extract ROS2 node information from Python source files.

    Uses AST analysis to detect:
    - Classes inheriting from rclpy.node.Node
    - create_publisher, create_subscription calls
    - create_timer calls
    - create_service, create_client calls
    - declare_parameter calls
    - TF broadcaster/listener usage

    Usage:
        extractor = ROS2NodeExtractor()
        nodes = extractor.extract_from_file('/path/to/node.py')
        for node in nodes:
            print(f"Node: {node.name}, publishers: {len(node.publishers)}")
    """

    # ROS2 node base class patterns
    NODE_BASES = {"Node", "rclpy.node.Node", "LifecycleNode", "rclpy_lifecycle.LifecycleNode"}

    def __init__(self):
        self.parser = PythonParser()

    def extract_from_file(
        self,
        file_path: Path,
        package_name: Optional[str] = None
    ) -> List[ROS2NodeInfo]:
        """
        Extract all ROS2 nodes from a Python file.

        Args:
            file_path: Path to Python file
            package_name: Optional ROS2 package name

        Returns:
            List of ROS2NodeInfo objects found in file
        """
        file_path = Path(file_path)

        # Read source code
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return []

        # Parse AST
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        # Also get parsed info for docstrings
        parse_result = self.parser.parse_file(file_path)

        nodes = []

        # Find classes that inherit from Node
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if self._is_ros2_node_class(node):
                    ros2_node = self._extract_node_info(
                        node, file_path, source, package_name, parse_result
                    )
                    if ros2_node:
                        nodes.append(ros2_node)
                        logger.info(f"Found ROS2 node: {ros2_node.name} in {file_path}")

        return nodes

    def _is_ros2_node_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class inherits from a ROS2 Node class."""
        for base in class_node.bases:
            base_name = self._get_base_name(base)
            if base_name in self.NODE_BASES:
                return True
        return False

    def _get_base_name(self, base: ast.expr) -> str:
        """Extract the name of a base class."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            # Handle rclpy.node.Node style
            parts = []
            node = base
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            parts.reverse()
            return ".".join(parts)
        return ""

    def _extract_node_info(
        self,
        class_node: ast.ClassDef,
        file_path: Path,
        source: str,
        package_name: Optional[str],
        parse_result: ParseResult,
    ) -> Optional[ROS2NodeInfo]:
        """Extract complete ROS2 node information from a class."""

        # Get base classes
        bases = [self._get_base_name(b) for b in class_node.bases]

        # Try to get node name from __init__
        node_name = self._extract_node_name(class_node) or class_node.name.lower()

        # Get docstring from parse result
        docstring = None
        cls_info = parse_result.get_class(class_node.name)
        if cls_info:
            docstring = cls_info.docstring

        node_info = ROS2NodeInfo(
            name=node_name,
            class_name=class_node.name,
            file_path=file_path,
            line_number=class_node.lineno,
            end_line=class_node.end_lineno or class_node.lineno,
            base_classes=bases,
            docstring=docstring,
            package_name=package_name,
        )

        # Extract from entire class body
        self._extract_from_class(class_node, node_info)

        return node_info

    def _extract_node_name(self, class_node: ast.ClassDef) -> Optional[str]:
        """
        Try to extract node name from super().__init__('node_name') call.
        """
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call):
                # Look for super().__init__(...) pattern
                if self._is_super_init_call(node):
                    # First string argument is typically the node name
                    for arg in node.args:
                        name = self._extract_string_value(arg)
                        if name:
                            return name
                    # Check keyword arguments
                    for kw in node.keywords:
                        if kw.arg in ("node_name", "name"):
                            name = self._extract_string_value(kw.value)
                            if name:
                                return name
        return None

    def _is_super_init_call(self, call_node: ast.Call) -> bool:
        """Check if this is a super().__init__() call."""
        if isinstance(call_node.func, ast.Attribute):
            if call_node.func.attr == "__init__":
                if isinstance(call_node.func.value, ast.Call):
                    if isinstance(call_node.func.value.func, ast.Name):
                        return call_node.func.value.func.id == "super"
        return False

    def _extract_from_class(self, class_node: ast.ClassDef, node_info: ROS2NodeInfo):
        """Extract all ROS2 constructs from a class."""
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call):
                self._process_call(node, node_info)

            # Check for TF usage
            if isinstance(node, ast.Name):
                if node.id in ("TransformBroadcaster", "StaticTransformBroadcaster"):
                    node_info.has_tf_broadcaster = True
                elif node.id in ("TransformListener", "Buffer"):
                    node_info.has_tf_listener = True

    def _process_call(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Process a function call to extract ROS2 constructs."""
        if not isinstance(call_node.func, ast.Attribute):
            # Check for ActionClient(...) style calls
            if isinstance(call_node.func, ast.Name):
                if call_node.func.id == "ActionClient":
                    self._extract_action_client(call_node, node_info)
                elif call_node.func.id == "ActionServer":
                    self._extract_action_server(call_node, node_info)
            return

        method_name = call_node.func.attr

        try:
            if method_name == "create_publisher":
                self._extract_publisher(call_node, node_info)
            elif method_name == "create_subscription":
                self._extract_subscriber(call_node, node_info)
            elif method_name == "create_timer":
                self._extract_timer(call_node, node_info)
            elif method_name == "create_service":
                self._extract_service(call_node, node_info)
            elif method_name == "create_client":
                self._extract_service_client(call_node, node_info)
            elif method_name == "declare_parameter":
                self._extract_parameter(call_node, node_info)
        except Exception as e:
            logger.debug(f"Error extracting {method_name}: {e}")

    def _extract_publisher(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract publisher information from create_publisher call."""
        # create_publisher(msg_type, topic, qos_profile)
        if len(call_node.args) < 2:
            return

        msg_type = self._extract_type_name(call_node.args[0])
        topic = self._extract_string_value(call_node.args[1])

        if not topic:
            return

        qos = 10  # Default
        if len(call_node.args) > 2:
            qos_val = self._extract_int_value(call_node.args[2])
            if qos_val is not None:
                qos = qos_val

        node_info.publishers.append(PublisherInfo(
            topic=topic,
            msg_type=msg_type,
            qos=qos,
            line_number=call_node.lineno,
        ))

    def _extract_subscriber(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract subscriber information from create_subscription call."""
        # create_subscription(msg_type, topic, callback, qos_profile)
        if len(call_node.args) < 3:
            return

        msg_type = self._extract_type_name(call_node.args[0])
        topic = self._extract_string_value(call_node.args[1])
        callback = self._extract_callback_name(call_node.args[2])

        if not topic:
            return

        qos = 10  # Default
        if len(call_node.args) > 3:
            qos_val = self._extract_int_value(call_node.args[3])
            if qos_val is not None:
                qos = qos_val

        node_info.subscribers.append(SubscriberInfo(
            topic=topic,
            msg_type=msg_type,
            callback=callback,
            qos=qos,
            line_number=call_node.lineno,
        ))

    def _extract_timer(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract timer information from create_timer call."""
        # create_timer(period, callback)
        if len(call_node.args) < 2:
            return

        period = self._extract_float_value(call_node.args[0])
        callback = self._extract_callback_name(call_node.args[1])

        if period is None:
            # Try to get from expression like 1.0 / rate
            period = self._estimate_timer_period(call_node.args[0])

        if period is None:
            period = 0.0  # Unknown

        node_info.timers.append(TimerInfo(
            period=period,
            callback=callback,
            line_number=call_node.lineno,
        ))

    def _extract_service(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract service server information."""
        # create_service(srv_type, service_name, callback)
        if len(call_node.args) < 3:
            return

        srv_type = self._extract_type_name(call_node.args[0])
        name = self._extract_string_value(call_node.args[1])
        callback = self._extract_callback_name(call_node.args[2])

        if not name:
            return

        node_info.services.append(ServiceInfo(
            name=name,
            srv_type=srv_type,
            callback=callback,
            line_number=call_node.lineno,
        ))

    def _extract_service_client(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract service client information."""
        # create_client(srv_type, service_name)
        if len(call_node.args) < 2:
            return

        srv_type = self._extract_type_name(call_node.args[0])
        name = self._extract_string_value(call_node.args[1])

        if not name:
            return

        node_info.service_clients.append(ServiceClientInfo(
            name=name,
            srv_type=srv_type,
            line_number=call_node.lineno,
        ))

    def _extract_action_client(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract action client information."""
        # ActionClient(node, action_type, action_name)
        if len(call_node.args) < 3:
            return

        action_type = self._extract_type_name(call_node.args[1])
        name = self._extract_string_value(call_node.args[2])

        if not name:
            return

        node_info.action_clients.append(ActionClientInfo(
            name=name,
            action_type=action_type,
            line_number=call_node.lineno,
        ))

    def _extract_action_server(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract action server information."""
        # ActionServer(node, action_type, action_name, callback)
        if len(call_node.args) < 4:
            return

        action_type = self._extract_type_name(call_node.args[1])
        name = self._extract_string_value(call_node.args[2])
        callback = self._extract_callback_name(call_node.args[3])

        if not name:
            return

        node_info.action_servers.append(ActionServerInfo(
            name=name,
            action_type=action_type,
            callback=callback,
            line_number=call_node.lineno,
        ))

    def _extract_parameter(self, call_node: ast.Call, node_info: ROS2NodeInfo):
        """Extract parameter declaration."""
        # declare_parameter(name, default_value)
        if len(call_node.args) < 1:
            return

        name = self._extract_string_value(call_node.args[0])
        if not name:
            return

        default_value = None
        param_type = None

        if len(call_node.args) > 1:
            default_value = self._extract_literal_value(call_node.args[1])
            if default_value is not None:
                param_type = type(default_value).__name__

        node_info.parameters.append(ParameterInfo(
            name=name,
            default_value=default_value,
            param_type=param_type,
            line_number=call_node.lineno,
        ))

    # Helper methods for value extraction

    def _extract_string_value(self, node: ast.expr) -> Optional[str]:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            # f-string - extract literal parts
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                else:
                    parts.append("{...}")
            return "".join(parts) if parts else None
        return None

    def _extract_int_value(self, node: ast.expr) -> Optional[int]:
        """Extract integer value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    def _extract_float_value(self, node: ast.expr) -> Optional[float]:
        """Extract float value from AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
        return None

    def _extract_literal_value(self, node: ast.expr) -> Any:
        """Extract any literal value from AST node."""
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return None

    def _extract_type_name(self, node: ast.expr) -> str:
        """Extract type name from AST node."""
        try:
            return ast.unparse(node)
        except Exception:
            if isinstance(node, ast.Name):
                return node.id
            return "Unknown"

    def _extract_callback_name(self, node: ast.expr) -> str:
        """Extract callback function name from AST node."""
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        try:
            return ast.unparse(node)
        except Exception:
            return "unknown"

    def _estimate_timer_period(self, node: ast.expr) -> Optional[float]:
        """Try to estimate timer period from expression like 1.0 / rate."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # Pattern: 1.0 / something
            numerator = self._extract_float_value(node.left)
            denominator = self._extract_float_value(node.right)
            if numerator is not None and denominator is not None and denominator != 0:
                return numerator / denominator
        return None


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python node_extractor.py <python_file>")
        sys.exit(1)

    extractor = ROS2NodeExtractor()
    nodes = extractor.extract_from_file(Path(sys.argv[1]))

    for node in nodes:
        print("\n" + "=" * 60)
        print(f"Node: {node.name} ({node.class_name})")
        print("=" * 60)
        print(json.dumps(node.to_dict(), indent=2, default=str))
