"""
ROS2 Message Definition Parser - Parse .msg, .srv, and .action files.

Parses standard and custom ROS2 interface definitions into structured data,
enabling field lookup, type checking, and schema documentation.

Searches:
- Standard ROS2 installs: /opt/ros/*/share/*/msg/*.msg
- Source-built ROS2: ~/ros2_humble/install/share/*/msg/*.msg
- Project custom messages: {project}/**/msg/*.msg, **/srv/*.srv, **/action/*.action
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ROS2 builtin primitive types
BUILTIN_TYPES = frozenset({
    "bool", "byte", "char",
    "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "string", "wstring",
})

# Common standard ROS2 message packages
STANDARD_MSG_PACKAGES = frozenset({
    "std_msgs", "geometry_msgs", "sensor_msgs", "nav_msgs",
    "diagnostic_msgs", "visualization_msgs", "shape_msgs",
    "trajectory_msgs", "stereo_msgs", "tf2_msgs",
    "action_msgs", "builtin_interfaces", "rcl_interfaces",
    "std_srvs", "nav2_msgs", "vision_msgs", "unique_identifier_msgs",
    "lifecycle_msgs", "rosgraph_msgs", "composition_interfaces",
    "statistics_msgs", "type_description_interfaces",
})


@dataclass
class MessageField:
    """A single field in a message definition."""
    name: str
    field_type: str
    is_array: bool = False
    array_size: Optional[int] = None  # None = unbounded, int = bounded
    is_builtin: bool = False
    comment: str = ""
    default_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.field_type,
        }
        if self.is_array:
            d["is_array"] = True
            if self.array_size is not None:
                d["array_size"] = self.array_size
        if self.comment:
            d["comment"] = self.comment
        if self.default_value is not None:
            d["default"] = self.default_value
        return d


@dataclass
class MessageConstant:
    """A constant defined in a message."""
    name: str
    constant_type: str
    value: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.constant_type, "value": self.value}


@dataclass
class MessageDefinition:
    """A parsed ROS2 message, service, or action definition."""
    name: str
    package: str
    full_name: str  # e.g., "sensor_msgs/msg/LaserScan"
    kind: str  # "msg", "srv", "action"
    fields: List[MessageField] = field(default_factory=list)
    constants: List[MessageConstant] = field(default_factory=list)
    raw_text: str = ""
    file_path: str = ""

    # For .srv files
    request_fields: List[MessageField] = field(default_factory=list)
    response_fields: List[MessageField] = field(default_factory=list)

    # For .action files
    goal_fields: List[MessageField] = field(default_factory=list)
    result_fields: List[MessageField] = field(default_factory=list)
    feedback_fields: List[MessageField] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "package": self.package,
            "full_name": self.full_name,
            "kind": self.kind,
        }
        if self.kind == "msg":
            d["fields"] = [f.to_dict() for f in self.fields]
        elif self.kind == "srv":
            d["request"] = [f.to_dict() for f in self.request_fields]
            d["response"] = [f.to_dict() for f in self.response_fields]
        elif self.kind == "action":
            d["goal"] = [f.to_dict() for f in self.goal_fields]
            d["result"] = [f.to_dict() for f in self.result_fields]
            d["feedback"] = [f.to_dict() for f in self.feedback_fields]

        if self.constants:
            d["constants"] = [c.to_dict() for c in self.constants]
        return d


def _parse_field_line(line: str) -> Optional[MessageField]:
    """Parse a single field line like 'float32[] ranges  # Range data'."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Extract inline comment
    comment = ""
    if "#" in line:
        parts = line.split("#", 1)
        line = parts[0].strip()
        comment = parts[1].strip()

    if not line:
        return None

    # Split into tokens
    tokens = line.split()
    if len(tokens) < 2:
        return None

    field_type_raw = tokens[0]
    field_name = tokens[1]

    # Check for constant: "uint8 SOLID=0" or "uint8 SOLID = 0"
    if "=" in field_name or (len(tokens) >= 3 and tokens[2] == "="):
        return None  # Will be parsed as constant separately

    # Parse array notation
    is_array = False
    array_size = None
    if "[]" in field_type_raw:
        is_array = True
        field_type_raw = field_type_raw.replace("[]", "")
    elif "[" in field_type_raw and "]" in field_type_raw:
        is_array = True
        match = re.search(r"\[(\d+)\]", field_type_raw)
        if match:
            array_size = int(match.group(1))
        field_type_raw = re.sub(r"\[\d*\]", "", field_type_raw)

    # Check if builtin
    base_type = field_type_raw.split("/")[-1] if "/" in field_type_raw else field_type_raw
    is_builtin = base_type in BUILTIN_TYPES

    # Parse default value
    default_value = None
    if len(tokens) >= 3 and not tokens[2].startswith("#"):
        default_value = " ".join(tokens[2:])

    return MessageField(
        name=field_name,
        field_type=field_type_raw,
        is_array=is_array,
        array_size=array_size,
        is_builtin=is_builtin,
        comment=comment,
        default_value=default_value,
    )


def _parse_constant_line(line: str) -> Optional[MessageConstant]:
    """Parse a constant line like 'uint8 SOLID=0'."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Remove comment
    if "#" in line:
        line = line.split("#", 1)[0].strip()

    if "=" not in line:
        return None

    # Split on = to get type+name and value
    before_eq, value = line.split("=", 1)
    tokens = before_eq.strip().split()
    if len(tokens) < 2:
        return None

    const_type = tokens[0]
    const_name = tokens[1]
    value = value.strip()

    # Only uppercase names are constants by convention
    if not const_name[0].isupper():
        return None

    return MessageConstant(name=const_name, constant_type=const_type, value=value)


def parse_message_file(file_path: Path, package: str, kind: str) -> Optional[MessageDefinition]:
    """
    Parse a .msg, .srv, or .action file.

    Args:
        file_path: Path to the definition file
        package: ROS2 package name (e.g., "sensor_msgs")
        kind: "msg", "srv", or "action"

    Returns:
        MessageDefinition or None on error
    """
    try:
        content = file_path.read_text(errors="replace")
    except Exception as e:
        logger.debug(f"Could not read {file_path}: {e}")
        return None

    name = file_path.stem  # e.g., "LaserScan"
    full_name = f"{package}/{kind}/{name}"

    msg_def = MessageDefinition(
        name=name,
        package=package,
        full_name=full_name,
        kind=kind,
        raw_text=content,
        file_path=str(file_path),
    )

    if kind == "msg":
        fields, constants = _parse_section(content)
        msg_def.fields = fields
        msg_def.constants = constants
    elif kind == "srv":
        sections = content.split("---")
        if len(sections) >= 1:
            req_fields, req_consts = _parse_section(sections[0])
            msg_def.request_fields = req_fields
            msg_def.constants = req_consts
        if len(sections) >= 2:
            resp_fields, _ = _parse_section(sections[1])
            msg_def.response_fields = resp_fields
        # Combine for convenience
        msg_def.fields = msg_def.request_fields + msg_def.response_fields
    elif kind == "action":
        sections = content.split("---")
        if len(sections) >= 1:
            goal_fields, goal_consts = _parse_section(sections[0])
            msg_def.goal_fields = goal_fields
            msg_def.constants = goal_consts
        if len(sections) >= 2:
            result_fields, _ = _parse_section(sections[1])
            msg_def.result_fields = result_fields
        if len(sections) >= 3:
            feedback_fields, _ = _parse_section(sections[2])
            msg_def.feedback_fields = feedback_fields
        msg_def.fields = msg_def.goal_fields + msg_def.result_fields + msg_def.feedback_fields

    return msg_def


def _parse_section(text: str):
    """Parse a section of a message file into fields and constants."""
    fields = []
    constants = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Try constant first
        const = _parse_constant_line(line)
        if const:
            constants.append(const)
            continue

        # Try field
        f = _parse_field_line(line)
        if f:
            fields.append(f)

    return fields, constants


class MessageDatabase:
    """
    Database of parsed ROS2 message definitions.

    Loads from standard ROS2 installs and project custom messages.
    Provides lookup by full name or short name with fuzzy matching.
    """

    def __init__(self):
        self.messages: Dict[str, MessageDefinition] = {}  # full_name -> definition
        self._name_index: Dict[str, List[str]] = {}  # short_name -> [full_names]

    def _index(self, msg_def: MessageDefinition):
        """Add to lookup indices."""
        self.messages[msg_def.full_name] = msg_def
        short = msg_def.name
        if short not in self._name_index:
            self._name_index[short] = []
        if msg_def.full_name not in self._name_index[short]:
            self._name_index[short].append(msg_def.full_name)

    def load_standard_messages(self, extra_paths: Optional[List[Path]] = None):
        """Load standard ROS2 message definitions from installed distros."""
        search_paths = []

        # Auto-detect installed ROS2 distros
        opt_ros = Path("/opt/ros")
        if opt_ros.exists():
            for distro_dir in opt_ros.iterdir():
                if distro_dir.is_dir():
                    share_dir = distro_dir / "share"
                    if share_dir.exists():
                        search_paths.append(share_dir)

        # Source-built ROS2
        home_ros = Path.home() / "ros2_humble" / "install" / "share"
        if home_ros.exists():
            search_paths.append(home_ros)

        if extra_paths:
            search_paths.extend(extra_paths)

        for share_dir in search_paths:
            self._load_from_share_dir(share_dir)

        logger.info(f"Loaded {len(self.messages)} standard message definitions")

    def _load_from_share_dir(self, share_dir: Path):
        """Load messages from a ROS2 share directory."""
        if not share_dir.exists():
            return

        for pkg_dir in share_dir.iterdir():
            if not pkg_dir.is_dir():
                continue
            pkg_name = pkg_dir.name

            # Load .msg files
            msg_dir = pkg_dir / "msg"
            if msg_dir.exists():
                for msg_file in msg_dir.glob("*.msg"):
                    msg_def = parse_message_file(msg_file, pkg_name, "msg")
                    if msg_def:
                        self._index(msg_def)

            # Load .srv files
            srv_dir = pkg_dir / "srv"
            if srv_dir.exists():
                for srv_file in srv_dir.glob("*.srv"):
                    srv_def = parse_message_file(srv_file, pkg_name, "srv")
                    if srv_def:
                        self._index(srv_def)

            # Load .action files
            action_dir = pkg_dir / "action"
            if action_dir.exists():
                for action_file in action_dir.glob("*.action"):
                    action_def = parse_message_file(action_file, pkg_name, "action")
                    if action_def:
                        self._index(action_def)

    def load_project_messages(self, project_path: Path):
        """Load custom message definitions from a project directory."""
        project_path = Path(project_path)
        count = 0

        for pattern, kind in [("**/msg/*.msg", "msg"), ("**/srv/*.srv", "srv"), ("**/action/*.action", "action")]:
            for def_file in project_path.rglob(pattern.split("/")[-1] if "/" not in pattern else ""):
                pass  # rglob doesn't work well with ** patterns containing directories

        # Search for msg files
        for msg_file in project_path.rglob("*.msg"):
            if "/msg/" in str(msg_file):
                pkg_name = self._guess_package(msg_file)
                msg_def = parse_message_file(msg_file, pkg_name, "msg")
                if msg_def:
                    self._index(msg_def)
                    count += 1

        # Search for srv files
        for srv_file in project_path.rglob("*.srv"):
            if "/srv/" in str(srv_file):
                pkg_name = self._guess_package(srv_file)
                srv_def = parse_message_file(srv_file, pkg_name, "srv")
                if srv_def:
                    self._index(srv_def)
                    count += 1

        # Search for action files
        for action_file in project_path.rglob("*.action"):
            if "/action/" in str(action_file):
                pkg_name = self._guess_package(action_file)
                action_def = parse_message_file(action_file, pkg_name, "action")
                if action_def:
                    self._index(action_def)
                    count += 1

        logger.info(f"Loaded {count} project message definitions from {project_path}")

    def _guess_package(self, def_file: Path) -> str:
        """Guess the package name for a message definition file."""
        # Walk up to find package.xml
        current = def_file.parent
        for _ in range(5):
            current = current.parent
            pkg_xml = current / "package.xml"
            if pkg_xml.exists():
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(pkg_xml)
                    name_elem = tree.find("name")
                    if name_elem is not None and name_elem.text:
                        return name_elem.text.strip()
                except Exception:
                    pass
                break
        # Fallback: use parent of msg/srv/action directory
        return def_file.parent.parent.name

    def lookup(self, type_name: str) -> Optional[MessageDefinition]:
        """
        Look up a message definition by name.

        Supports:
            - Full name: "sensor_msgs/msg/LaserScan"
            - Import style: "sensor_msgs.msg.LaserScan" (Python import)
            - Short name: "LaserScan" (returns first match)
            - Partial: "sensor_msgs/LaserScan"
        """
        # Normalize Python import style
        if "." in type_name and "/" not in type_name:
            type_name = type_name.replace(".", "/")

        # Direct full match
        if type_name in self.messages:
            return self.messages[type_name]

        # Try adding /msg/ if missing
        if "/" in type_name and "/msg/" not in type_name and "/srv/" not in type_name and "/action/" not in type_name:
            parts = type_name.split("/")
            if len(parts) == 2:
                for kind in ["msg", "srv", "action"]:
                    candidate = f"{parts[0]}/{kind}/{parts[1]}"
                    if candidate in self.messages:
                        return self.messages[candidate]

        # Short name lookup
        if type_name in self._name_index:
            full_names = self._name_index[type_name]
            if full_names:
                return self.messages[full_names[0]]

        return None

    def search(self, pattern: str) -> List[MessageDefinition]:
        """Search for message definitions matching a pattern."""
        pattern_lower = pattern.lower()
        results = []
        for full_name, msg_def in self.messages.items():
            if pattern_lower in full_name.lower() or pattern_lower in msg_def.name.lower():
                results.append(msg_def)
        return results

    def get_used_messages(self, type_names: List[str]) -> Dict[str, MessageDefinition]:
        """Get definitions for a list of type names (filtering to only used types)."""
        result = {}
        for type_name in type_names:
            msg_def = self.lookup(type_name)
            if msg_def and msg_def.full_name not in result:
                result[msg_def.full_name] = msg_def
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Export all messages to dict."""
        return {
            full_name: msg_def.to_dict()
            for full_name, msg_def in sorted(self.messages.items())
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        by_kind = {"msg": 0, "srv": 0, "action": 0}
        by_package: Dict[str, int] = {}
        for msg in self.messages.values():
            by_kind[msg.kind] = by_kind.get(msg.kind, 0) + 1
            by_package[msg.package] = by_package.get(msg.package, 0) + 1

        return {
            "total": len(self.messages),
            "by_kind": by_kind,
            "packages": len(by_package),
        }


def load_message_database(
    project_path: Optional[Path] = None,
    load_standard: bool = True,
) -> MessageDatabase:
    """
    Convenience function to create and populate a MessageDatabase.

    Args:
        project_path: Optional project path for custom messages
        load_standard: Whether to load standard ROS2 messages

    Returns:
        Populated MessageDatabase
    """
    db = MessageDatabase()
    if load_standard:
        db.load_standard_messages()
    if project_path:
        db.load_project_messages(project_path)
    return db
