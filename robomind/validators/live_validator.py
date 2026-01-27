"""
RoboMind Live Validator - Compare static analysis against running ROS2 system.

This module queries a live ROS2 system and compares it against static analysis
to find discrepancies:
- Topics that exist in code but not at runtime (orphaned publishers/subscribers)
- Topics that exist at runtime but not in code (dynamic/external topics)
- Node mismatches between code and runtime
- Topic type mismatches
"""

import logging
import subprocess
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)


class DiffType(Enum):
    """Types of validation differences."""
    TOPIC_IN_CODE_NOT_LIVE = "topic_in_code_not_live"
    TOPIC_IN_LIVE_NOT_CODE = "topic_in_live_not_code"
    NODE_IN_CODE_NOT_LIVE = "node_in_code_not_live"
    NODE_IN_LIVE_NOT_CODE = "node_in_live_not_code"
    TYPE_MISMATCH = "type_mismatch"
    PUBLISHER_COUNT_MISMATCH = "publisher_count_mismatch"
    SUBSCRIBER_COUNT_MISMATCH = "subscriber_count_mismatch"


class Severity(Enum):
    """Severity levels for validation differences."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationDiff:
    """A single validation difference."""
    diff_type: DiffType
    severity: Severity
    name: str  # Topic or node name
    message: str
    code_value: Any = None
    live_value: Any = None
    recommendation: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.diff_type.value,
            "severity": self.severity.value,
            "name": self.name,
            "message": self.message,
            "code_value": str(self.code_value) if self.code_value else None,
            "live_value": str(self.live_value) if self.live_value else None,
            "recommendation": self.recommendation,
        }


@dataclass
class HTTPHealthResult:
    """Result of an HTTP health check."""
    endpoint: str
    available: bool
    status_code: int = 0
    response_time_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "endpoint": self.endpoint,
            "available": self.available,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
        }


@dataclass
class LiveSystemInfo:
    """Information from a live ROS2 system."""
    nodes: List[str] = field(default_factory=list)
    topics: Dict[str, Dict] = field(default_factory=dict)  # topic -> {type, publishers, subscribers}
    services: List[str] = field(default_factory=list)
    http_endpoints: Dict[str, HTTPHealthResult] = field(default_factory=dict)
    available: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "available": self.available,
            "nodes": self.nodes,
            "topics": self.topics,
            "services": self.services,
            "http_endpoints": {k: v.to_dict() for k, v in self.http_endpoints.items()},
            "error": self.error,
        }


@dataclass
class ValidationResult:
    """Result of validation comparison."""
    diffs: List[ValidationDiff] = field(default_factory=list)
    live_info: Optional[LiveSystemInfo] = None
    code_stats: Dict = field(default_factory=dict)
    validated: bool = False
    error: Optional[str] = None

    @property
    def has_critical(self) -> bool:
        """Check if any critical issues exist."""
        return any(d.severity == Severity.CRITICAL for d in self.diffs)

    @property
    def has_errors(self) -> bool:
        """Check if any error-level issues exist."""
        return any(d.severity in (Severity.ERROR, Severity.CRITICAL) for d in self.diffs)

    def get_by_severity(self, severity: Severity) -> List[ValidationDiff]:
        """Get diffs filtered by severity."""
        return [d for d in self.diffs if d.severity == severity]

    def get_by_type(self, diff_type: DiffType) -> List[ValidationDiff]:
        """Get diffs filtered by type."""
        return [d for d in self.diffs if d.diff_type == diff_type]

    def summary(self) -> Dict:
        """Generate summary statistics."""
        by_severity = {s.value: 0 for s in Severity}
        by_type = {t.value: 0 for t in DiffType}

        for diff in self.diffs:
            by_severity[diff.severity.value] += 1
            by_type[diff.diff_type.value] += 1

        return {
            "validated": self.validated,
            "total_diffs": len(self.diffs),
            "by_severity": by_severity,
            "by_type": by_type,
            "code_stats": self.code_stats,
            "live_available": self.live_info.available if self.live_info else False,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.summary(),
            "diffs": [d.to_dict() for d in self.diffs],
            "live_system": self.live_info.to_dict() if self.live_info else None,
            "error": self.error,
        }


class LiveValidator:
    """
    Validate static analysis against a live ROS2 system.

    Usage:
        validator = LiveValidator(nodes, topic_graph)
        result = validator.validate()
        print(result.summary())
    """

    def __init__(
        self,
        nodes: List[ROS2NodeInfo],
        topic_graph: Optional[TopicGraphResult] = None,
        ssh_host: Optional[str] = None,
        ros2_distro: str = "humble",
        http_comm_map=None,
    ):
        self.nodes = nodes
        self.topic_graph = topic_graph
        self.ssh_host = ssh_host
        self.ros2_distro = ros2_distro
        self.http_comm_map = http_comm_map

        # Build code-based maps
        self._code_topics = self._build_code_topics()
        self._code_nodes = {n.name for n in nodes}

    def _build_code_topics(self) -> Dict[str, Dict]:
        """Build topic info from static analysis."""
        topics = {}

        if self.topic_graph:
            for topic_name, topic in self.topic_graph.topics.items():
                topics[topic_name] = {
                    "type": topic.msg_type,
                    "publishers": list(topic.publishers),
                    "subscribers": list(topic.subscribers),
                }
        else:
            # Build from nodes directly
            for node in self.nodes:
                for pub in node.publishers:
                    if pub.topic not in topics:
                        topics[pub.topic] = {"type": pub.msg_type, "publishers": [], "subscribers": []}
                    topics[pub.topic]["publishers"].append(node.name)
                    if pub.msg_type and not topics[pub.topic]["type"]:
                        topics[pub.topic]["type"] = pub.msg_type

                for sub in node.subscribers:
                    if sub.topic not in topics:
                        topics[sub.topic] = {"type": sub.msg_type, "publishers": [], "subscribers": []}
                    topics[sub.topic]["subscribers"].append(node.name)
                    if sub.msg_type and not topics[sub.topic]["type"]:
                        topics[sub.topic]["type"] = sub.msg_type

        return topics

    def _run_command(self, cmd: List[str], timeout: int = 10) -> Optional[str]:
        """Run a command locally or via SSH."""
        try:
            if self.ssh_host:
                full_cmd = ["ssh", self.ssh_host] + cmd
            else:
                full_cmd = cmd

            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                logger.warning(f"Command failed: {' '.join(cmd)}: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {' '.join(cmd)}")
            return None
        except Exception as e:
            logger.error(f"Command error: {' '.join(cmd)}: {e}")
            return None

    def get_live_system_info(self) -> LiveSystemInfo:
        """Query the live ROS2 system for current state."""
        info = LiveSystemInfo()

        # Get node list
        output = self._run_command(["ros2", "node", "list"])
        if output is None:
            info.error = "Could not connect to ROS2 system. Is ROS2 running?"
            return info

        info.available = True
        info.nodes = [n.strip() for n in output.strip().split("\n") if n.strip()]

        # Get topic list with types
        output = self._run_command(["ros2", "topic", "list", "-t"])
        if output:
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                # Format: /topic [type/Name]
                match = re.match(r"(/\S+)\s+\[(\S+)\]", line.strip())
                if match:
                    topic_name = match.group(1)
                    topic_type = match.group(2)
                    info.topics[topic_name] = {
                        "type": topic_type,
                        "publishers": [],
                        "subscribers": [],
                    }
                elif line.strip().startswith("/"):
                    # Just topic name without type
                    topic_name = line.strip().split()[0]
                    info.topics[topic_name] = {
                        "type": None,
                        "publishers": [],
                        "subscribers": [],
                    }

        # Get detailed topic info for each topic
        for topic_name in list(info.topics.keys())[:50]:  # Limit to avoid timeout
            output = self._run_command(["ros2", "topic", "info", topic_name], timeout=5)
            if output:
                pubs = re.search(r"Publisher count:\s*(\d+)", output)
                subs = re.search(r"Subscription count:\s*(\d+)", output)
                if pubs:
                    info.topics[topic_name]["publisher_count"] = int(pubs.group(1))
                if subs:
                    info.topics[topic_name]["subscriber_count"] = int(subs.group(1))

        # Get service list
        output = self._run_command(["ros2", "service", "list"])
        if output:
            info.services = [s.strip() for s in output.strip().split("\n") if s.strip()]

        return info

    def check_http_endpoints(self, endpoints: List[str], timeout: float = 5.0) -> Dict[str, HTTPHealthResult]:
        """
        Check HTTP endpoint health.

        Args:
            endpoints: List of HTTP URLs to check
            timeout: Request timeout in seconds

        Returns:
            Dict mapping endpoint URL to HTTPHealthResult
        """
        results = {}

        for endpoint in endpoints:
            result = HTTPHealthResult(endpoint=endpoint, available=False)

            try:
                start_time = time.time()

                # Create request with HEAD method for efficiency
                req = urllib.request.Request(endpoint, method='HEAD')
                req.add_header('User-Agent', 'RoboMind/1.0')

                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result.status_code = response.status
                    result.available = 200 <= response.status < 400
                    result.response_time_ms = (time.time() - start_time) * 1000

            except urllib.error.HTTPError as e:
                result.status_code = e.code
                result.available = False
                result.error = f"HTTP {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                result.available = False
                result.error = f"Connection failed: {e.reason}"
            except Exception as e:
                result.available = False
                result.error = str(e)

            results[endpoint] = result

        return results

    def get_http_endpoints_from_comm_map(self) -> List[str]:
        """Extract unique HTTP endpoints to check from communication map."""
        endpoints = set()

        if self.http_comm_map:
            # Get target hosts from client calls
            for client in self.http_comm_map.http_clients:
                if client.target_url and client.target_url.startswith("http"):
                    endpoints.add(client.target_url)

            # Construct health endpoints from detected hosts
            summary = self.http_comm_map.summary()
            for host in summary.get("http_target_hosts", []):
                if ":" in host:
                    endpoints.add(f"http://{host}/health")
                    endpoints.add(f"http://{host}/")

        return list(endpoints)

    def validate(self, check_http: bool = True) -> ValidationResult:
        """
        Perform validation comparing code analysis to live system.

        Args:
            check_http: Whether to validate HTTP endpoints

        Returns:
            ValidationResult with all differences found
        """
        result = ValidationResult()
        result.code_stats = {
            "nodes": len(self._code_nodes),
            "topics": len(self._code_topics),
        }

        # Get live system info
        live = self.get_live_system_info()
        result.live_info = live

        if not live.available:
            result.error = live.error
            # Still check HTTP even if ROS2 is not available
            if check_http and self.http_comm_map:
                endpoints = self.get_http_endpoints_from_comm_map()
                if endpoints:
                    live.http_endpoints = self.check_http_endpoints(endpoints[:20])  # Limit
                    result.validated = True  # Partial validation

            return result

        result.validated = True

        # Compare topics
        self._compare_topics(result, live)

        # Compare nodes
        self._compare_nodes(result, live)

        # Check HTTP endpoints
        if check_http and self.http_comm_map:
            endpoints = self.get_http_endpoints_from_comm_map()
            if endpoints:
                live.http_endpoints = self.check_http_endpoints(endpoints[:20])  # Limit
                self._report_http_health(result, live)

        return result

    def _report_http_health(self, result: ValidationResult, live: LiveSystemInfo):
        """Add HTTP health check results to validation."""
        for endpoint, health in live.http_endpoints.items():
            if not health.available:
                result.diffs.append(ValidationDiff(
                    diff_type=DiffType.TOPIC_IN_CODE_NOT_LIVE,  # Reuse type
                    severity=Severity.WARNING,
                    name=endpoint,
                    message=f"HTTP endpoint '{endpoint}' is not responding",
                    code_value="expected available",
                    live_value=health.error or f"status {health.status_code}",
                    recommendation="Check if the HTTP service is running",
                ))

    def _compare_topics(self, result: ValidationResult, live: LiveSystemInfo):
        """Compare topics between code and live system."""
        code_topics = set(self._code_topics.keys())
        live_topics = set(live.topics.keys())

        # Normalize topics for comparison (handle leading slash)
        def normalize(topic: str) -> str:
            return topic if topic.startswith("/") else f"/{topic}"

        code_normalized = {normalize(t): t for t in code_topics}
        live_normalized = {normalize(t): t for t in live_topics}

        # Topics in code but not live (potential orphans)
        for norm_topic, orig_topic in code_normalized.items():
            if norm_topic not in live_normalized:
                code_info = self._code_topics[orig_topic]
                has_pub = bool(code_info.get("publishers"))
                has_sub = bool(code_info.get("subscribers"))

                if has_pub and has_sub:
                    severity = Severity.WARNING
                    msg = f"Topic '{orig_topic}' has both publishers and subscribers in code but not active"
                    rec = "Nodes may not be running, or topic names may differ at runtime"
                elif has_pub:
                    severity = Severity.INFO
                    msg = f"Topic '{orig_topic}' has publishers in code but topic not active"
                    rec = "Publisher node may not be running"
                else:
                    severity = Severity.INFO
                    msg = f"Topic '{orig_topic}' has subscribers in code but topic not active"
                    rec = "External publisher may not be running"

                result.diffs.append(ValidationDiff(
                    diff_type=DiffType.TOPIC_IN_CODE_NOT_LIVE,
                    severity=severity,
                    name=orig_topic,
                    message=msg,
                    code_value=code_info,
                    recommendation=rec,
                ))

        # Topics in live but not code (external/dynamic topics)
        for norm_topic, orig_topic in live_normalized.items():
            if norm_topic not in code_normalized:
                # Skip common ROS2 system topics
                if any(orig_topic.startswith(p) for p in ["/rosout", "/parameter_events", "/tf"]):
                    continue

                result.diffs.append(ValidationDiff(
                    diff_type=DiffType.TOPIC_IN_LIVE_NOT_CODE,
                    severity=Severity.INFO,
                    name=orig_topic,
                    message=f"Topic '{orig_topic}' exists at runtime but not found in code",
                    live_value=live.topics.get(orig_topic),
                    recommendation="May be from external nodes or dynamically created",
                ))

        # Check for type mismatches on common topics
        for norm_topic in code_normalized.keys() & live_normalized.keys():
            code_topic = code_normalized[norm_topic]
            live_topic = live_normalized[norm_topic]

            code_type = self._code_topics[code_topic].get("type", "").split("/")[-1]
            live_type = live.topics[live_topic].get("type", "").split("/")[-1]

            if code_type and live_type and code_type != live_type:
                result.diffs.append(ValidationDiff(
                    diff_type=DiffType.TYPE_MISMATCH,
                    severity=Severity.ERROR,
                    name=code_topic,
                    message=f"Topic type mismatch: code has '{code_type}', live has '{live_type}'",
                    code_value=code_type,
                    live_value=live_type,
                    recommendation="Fix message type in code or verify correct topic",
                ))

    def _compare_nodes(self, result: ValidationResult, live: LiveSystemInfo):
        """Compare nodes between code and live system."""
        # Normalize node names (remove leading /)
        code_nodes = {n.lstrip("/") for n in self._code_nodes}
        live_nodes = {n.lstrip("/") for n in live.nodes}

        # Nodes in code but not running
        for node in code_nodes - live_nodes:
            # Check for partial matches (might have namespace prefix)
            partial_match = any(node in ln for ln in live_nodes)
            if partial_match:
                continue

            result.diffs.append(ValidationDiff(
                diff_type=DiffType.NODE_IN_CODE_NOT_LIVE,
                severity=Severity.WARNING,
                name=node,
                message=f"Node '{node}' found in code but not running",
                recommendation="Node may need to be launched or is conditionally loaded",
            ))

        # Nodes running but not in code
        for node in live_nodes - code_nodes:
            # Skip common system nodes
            if any(node.startswith(p) for p in ["_ros2cli", "rviz", "rqt"]):
                continue

            # Check for partial matches
            partial_match = any(cn in node for cn in code_nodes)
            if partial_match:
                continue

            result.diffs.append(ValidationDiff(
                diff_type=DiffType.NODE_IN_LIVE_NOT_CODE,
                severity=Severity.INFO,
                name=node,
                message=f"Node '{node}' running but not found in analyzed code",
                recommendation="May be from external package or not in analyzed path",
            ))


def validate_against_live(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    ssh_host: Optional[str] = None,
    http_comm_map=None,
    check_http: bool = True,
) -> ValidationResult:
    """
    Convenience function to validate static analysis against live system.

    Args:
        nodes: List of ROS2NodeInfo from static analysis
        topic_graph: Optional TopicGraphResult
        ssh_host: Optional SSH host to connect to remote ROS2 system
        http_comm_map: Optional HTTP CommunicationMap for endpoint validation
        check_http: Whether to check HTTP endpoints

    Returns:
        ValidationResult with all differences found
    """
    validator = LiveValidator(nodes, topic_graph, ssh_host, http_comm_map=http_comm_map)
    return validator.validate(check_http=check_http)


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from robomind.ros2.node_extractor import ROS2NodeExtractor
    from robomind.ros2.topic_extractor import TopicExtractor

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python live_validator.py <project_path> [ssh_host]")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    ssh_host = sys.argv[2] if len(sys.argv) > 2 else None

    # Extract nodes
    node_extractor = ROS2NodeExtractor()
    topic_extractor = TopicExtractor()
    all_nodes = []

    print(f"Analyzing code in {project_path}...")
    for py_file in project_path.rglob("*.py"):
        if "__pycache__" in str(py_file) or "/build/" in str(py_file):
            continue
        nodes = node_extractor.extract_from_file(py_file)
        all_nodes.extend(nodes)
        topic_extractor.add_nodes(nodes)

    topic_graph = topic_extractor.build()
    print(f"Found {len(all_nodes)} nodes, {len(topic_graph.topics)} topics in code")

    # Validate
    print("\nValidating against live system...")
    result = validate_against_live(all_nodes, topic_graph, ssh_host)

    if not result.validated:
        print(f"Validation failed: {result.error}")
        sys.exit(1)

    print(f"\nLive system: {len(result.live_info.nodes)} nodes, {len(result.live_info.topics)} topics")
    print(f"\nFound {len(result.diffs)} differences:")

    summary = result.summary()
    for sev, count in summary["by_severity"].items():
        if count > 0:
            print(f"  {sev.upper()}: {count}")

    print("\nDetails:")
    for diff in sorted(result.diffs, key=lambda d: d.severity.value, reverse=True):
        print(f"  [{diff.severity.value.upper()}] {diff.message}")
        if diff.recommendation:
            print(f"    -> {diff.recommendation}")
