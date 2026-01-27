"""
RoboMind Timing Chain Analyzer

Analyzes timing characteristics of ROS2 callback chains to identify:
- End-to-end latency paths (sensor → actuator)
- Timer frequency conflicts
- Potential deadline misses
- Blocking operations in callbacks
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robomind.ros2.node_extractor import ROS2NodeInfo, TimerInfo
from robomind.ros2.topic_extractor import TopicGraphResult


@dataclass
class CallbackChain:
    """A chain of callbacks from input to output."""
    name: str
    nodes: List[str]  # Node names in order
    topics: List[str]  # Topics connecting them
    total_hops: int
    estimated_latency_ms: Optional[float] = None
    bottleneck: Optional[str] = None
    chain_type: str = "unknown"  # "sensor_to_actuator", "perception", "control"


@dataclass
class TimingIssue:
    """A timing-related issue."""
    issue_type: str  # "high_frequency", "frequency_mismatch", "blocking_call", "deadline_risk"
    severity: str
    node: str
    file_path: str
    line_number: int
    description: str
    recommendation: str


@dataclass
class TimingAnalysisResult:
    """Results of timing analysis."""
    chains: List[CallbackChain]
    issues: List[TimingIssue]
    timer_frequencies: Dict[str, float]  # node -> Hz
    critical_path: Optional[CallbackChain] = None


class TimingAnalyzer:
    """
    Analyzes timing characteristics of ROS2 systems.

    Key analyses:
    1. Callback chain tracing (sensor → processing → actuator)
    2. Timer frequency analysis
    3. Blocking call detection
    4. Latency estimation
    """

    # Patterns indicating blocking operations (bad in callbacks)
    BLOCKING_PATTERNS = [
        (r'time\.sleep\s*\(', 'time.sleep() blocks the executor'),
        (r'\.sleep\s*\(', 'sleep() call blocks the executor'),
        (r'input\s*\(', 'input() waits for user input'),
        (r'\.read\s*\(', 'Synchronous file/socket read may block'),
        (r'requests\.(get|post|put|delete)\s*\(', 'Synchronous HTTP request blocks'),
        (r'\.recv\s*\(', 'Synchronous socket recv blocks'),
        (r'subprocess\.(run|call|check_output)\s*\(', 'Subprocess call may block'),
        (r'\.wait\s*\(', 'wait() call may block'),
        (r'\.join\s*\(', 'Thread join may block'),
        (r'Lock\(\)\.acquire', 'Lock acquire may block'),
    ]

    # Patterns indicating memory allocation (bad in real-time callbacks)
    ALLOCATION_PATTERNS = [
        (r'\[\s*\]\s*\*\s*\d+', 'List multiplication allocates memory'),
        (r'\.append\s*\(', 'List append may reallocate'),
        (r'np\.zeros\s*\(', 'NumPy allocation in callback'),
        (r'np\.ones\s*\(', 'NumPy allocation in callback'),
        (r'np\.empty\s*\(', 'NumPy allocation in callback'),
        (r'torch\.zeros\s*\(', 'PyTorch tensor allocation in callback'),
        (r'cv2\.imread\s*\(', 'OpenCV image load allocates memory'),
    ]

    # Known sensor topics (input of chains)
    SENSOR_TOPICS = {'/scan', '/camera/image_raw', '/imu', '/odom', '/gps', '/lidar',
                     'scan', 'camera', 'image', 'imu', 'odom', 'gps', 'lidar',
                     '/depth', '/pointcloud', '/ultrasonic'}

    # Known actuator topics (output of chains)
    ACTUATOR_TOPICS = {'/cmd_vel', '/joint_commands', '/motor_command', '/servo',
                       'cmd_vel', 'motor', 'actuator', 'joint', 'servo',
                       '/gpio', '/pwm'}

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.file_contents: Dict[str, str] = {}

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def add_topic_graph(self, topic_graph: TopicGraphResult):
        """Add topic graph for chain analysis."""
        self.topic_graph = topic_graph

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content with caching."""
        if not file_path:
            return None
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r') as f:
                    self.file_contents[file_path] = f.read()
            except Exception:
                return None
        return self.file_contents.get(file_path)

    def _get_callback_content(self, file_path: str, callback_name: str) -> Optional[str]:
        """Extract callback function content."""
        content = self._get_file_content(file_path)
        if not content:
            return None

        # Simple extraction - find function definition and grab body
        pattern = rf'def\s+{re.escape(callback_name)}\s*\([^)]*\):[^\n]*\n((?:[ \t]+[^\n]*\n)*)'
        match = re.search(pattern, content)
        if match:
            return match.group(0)
        return None

    def _analyze_callback_for_blocking(self, node: ROS2NodeInfo, callback_name: str) -> List[TimingIssue]:
        """Check a callback for blocking operations."""
        issues = []
        content = self._get_callback_content(node.file_path, callback_name)
        if not content:
            return issues

        lines = content.split('\n')
        for i, line in enumerate(lines):
            for pattern, description in self.BLOCKING_PATTERNS:
                if re.search(pattern, line):
                    issues.append(TimingIssue(
                        issue_type='blocking_call',
                        severity='high',
                        node=node.name,
                        file_path=node.file_path or '',
                        line_number=i + 1,  # Approximate
                        description=f"Blocking call in callback '{callback_name}': {description}",
                        recommendation="Move blocking operations to a separate thread or use async patterns",
                    ))

            for pattern, description in self.ALLOCATION_PATTERNS:
                if re.search(pattern, line):
                    issues.append(TimingIssue(
                        issue_type='allocation_in_callback',
                        severity='medium',
                        node=node.name,
                        file_path=node.file_path or '',
                        line_number=i + 1,
                        description=f"Memory allocation in callback '{callback_name}': {description}",
                        recommendation="Pre-allocate buffers outside callbacks for real-time performance",
                    ))

        return issues

    def _analyze_timer_frequencies(self) -> Tuple[Dict[str, float], List[TimingIssue]]:
        """Analyze timer frequencies for issues."""
        frequencies: Dict[str, float] = {}
        issues: List[TimingIssue] = []

        for node in self.nodes:
            for timer in node.timers:
                if timer.period and timer.period > 0:
                    freq = 1.0 / timer.period
                    frequencies[node.name] = freq

                    # Check for problematic frequencies
                    if freq > 1000:  # >1kHz
                        issues.append(TimingIssue(
                            issue_type='high_frequency',
                            severity='medium',
                            node=node.name,
                            file_path=node.file_path or '',
                            line_number=timer.line_number,
                            description=f"Very high timer frequency: {freq:.0f} Hz (period={timer.period}s)",
                            recommendation="Consider if this frequency is necessary; high rates increase CPU load",
                        ))
                    elif freq < 1:  # <1Hz
                        # Check if this is a control loop
                        if any(t in node.name.lower() for t in ['control', 'motor', 'cmd']):
                            issues.append(TimingIssue(
                                issue_type='low_frequency',
                                severity='low',
                                node=node.name,
                                file_path=node.file_path or '',
                                line_number=timer.line_number,
                                description=f"Low control loop frequency: {freq:.2f} Hz",
                                recommendation="Control loops typically need higher frequencies (10-100 Hz)",
                            ))

        return frequencies, issues

    def _trace_callback_chains(self) -> List[CallbackChain]:
        """Trace callback chains through the topic graph."""
        chains = []

        if not self.topic_graph:
            return chains

        # Build node -> subscribed topics and node -> published topics maps
        node_subs: Dict[str, Set[str]] = {}
        node_pubs: Dict[str, Set[str]] = {}

        for node in self.nodes:
            node_subs[node.name] = {s.topic for s in node.subscribers if s.topic}
            node_pubs[node.name] = {p.topic for p in node.publishers if p.topic}

        # Find sensor-to-actuator chains using BFS
        def find_chain(start_topic: str, visited: Set[str], path: List[str], topics: List[str]) -> List[CallbackChain]:
            found_chains = []

            # Find nodes that subscribe to this topic
            for node in self.nodes:
                if start_topic not in node_subs.get(node.name, set()):
                    continue
                if node.name in visited:
                    continue

                new_visited = visited | {node.name}
                new_path = path + [node.name]
                new_topics = topics + [start_topic]

                # Check if this node publishes to actuator topics
                for pub_topic in node_pubs.get(node.name, set()):
                    if any(act in pub_topic.lower() for act in ['cmd_vel', 'motor', 'actuator', 'joint', 'servo']):
                        # Found complete chain!
                        found_chains.append(CallbackChain(
                            name=f"{path[0] if path else 'sensor'}→{node.name}",
                            nodes=new_path,
                            topics=new_topics + [pub_topic],
                            total_hops=len(new_path),
                            chain_type='sensor_to_actuator',
                        ))
                    else:
                        # Continue searching
                        found_chains.extend(find_chain(pub_topic, new_visited, new_path, new_topics))

            return found_chains

        # Start from sensor topics
        for topic in self.topic_graph.topics:
            if any(sens in topic.lower() for sens in ['scan', 'camera', 'image', 'imu', 'odom', 'lidar', 'sensor']):
                chains.extend(find_chain(topic, set(), [], []))

        # Deduplicate and find longest/critical chains
        unique_chains = []
        seen = set()
        for chain in chains:
            key = tuple(chain.nodes)
            if key not in seen:
                seen.add(key)
                unique_chains.append(chain)

        return unique_chains

    def analyze(self) -> TimingAnalysisResult:
        """Run full timing analysis."""
        issues = []

        # Analyze timer frequencies
        frequencies, freq_issues = self._analyze_timer_frequencies()
        issues.extend(freq_issues)

        # Analyze callbacks for blocking operations
        for node in self.nodes:
            for sub in node.subscribers:
                if sub.callback:
                    issues.extend(self._analyze_callback_for_blocking(node, sub.callback))
            for timer in node.timers:
                if timer.callback:
                    issues.extend(self._analyze_callback_for_blocking(node, timer.callback))

        # Trace callback chains
        chains = self._trace_callback_chains()

        # Find critical path (longest chain)
        critical_path = None
        if chains:
            critical_path = max(chains, key=lambda c: c.total_hops)

        return TimingAnalysisResult(
            chains=chains,
            issues=issues,
            timer_frequencies=frequencies,
            critical_path=critical_path,
        )


def analyze_timing(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
) -> TimingAnalysisResult:
    """Convenience function for timing analysis."""
    analyzer = TimingAnalyzer()
    analyzer.add_nodes(nodes)
    if topic_graph:
        analyzer.add_topic_graph(topic_graph)
    return analyzer.analyze()
