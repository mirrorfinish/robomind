"""
RoboMind Deep Analyzer

Combines all analysis modules into a comprehensive deep analysis:
- QoS compatibility
- Timing chains
- Security vulnerabilities
- Architecture patterns
- Callback complexity
- Message types
- Parameters

Generates a unified report with prioritized findings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

from robomind.analyzers.qos_analyzer import QoSAnalyzer, QoSFinding
from robomind.analyzers.timing_analyzer import TimingAnalyzer, TimingAnalysisResult, TimingIssue
from robomind.analyzers.security_analyzer import SecurityAnalyzer, SecurityFinding
from robomind.analyzers.architecture_analyzer import ArchitectureAnalyzer, ArchitectureFinding
from robomind.analyzers.complexity_analyzer import ComplexityAnalyzer, ComplexityFinding
from robomind.analyzers.message_analyzer import MessageTypeAnalyzer, MessageTypeFinding
from robomind.analyzers.parameter_analyzer import ParameterAnalyzer, ParameterFinding


@dataclass
class UnifiedFinding:
    """A unified finding from any analyzer."""
    category: str  # qos, timing, security, architecture, complexity, message, parameter
    severity: str  # critical, high, medium, low
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    affected_nodes: List[str] = field(default_factory=list)
    recommendation: str = ""
    fix_available: bool = False
    raw_finding: Any = None  # Original finding object


@dataclass
class DeepAnalysisReport:
    """Complete deep analysis report."""
    # Summary
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # By category
    qos_findings: List[UnifiedFinding] = field(default_factory=list)
    timing_findings: List[UnifiedFinding] = field(default_factory=list)
    security_findings: List[UnifiedFinding] = field(default_factory=list)
    architecture_findings: List[UnifiedFinding] = field(default_factory=list)
    complexity_findings: List[UnifiedFinding] = field(default_factory=list)
    message_findings: List[UnifiedFinding] = field(default_factory=list)
    parameter_findings: List[UnifiedFinding] = field(default_factory=list)

    # Timing-specific
    callback_chains: List[Any] = field(default_factory=list)
    critical_path: Optional[Any] = None

    # All findings sorted by severity
    all_findings: List[UnifiedFinding] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "summary": {
                "total_findings": self.total_findings,
                "by_severity": {
                    "critical": self.critical_count,
                    "high": self.high_count,
                    "medium": self.medium_count,
                    "low": self.low_count,
                },
                "by_category": {
                    "qos": len(self.qos_findings),
                    "timing": len(self.timing_findings),
                    "security": len(self.security_findings),
                    "architecture": len(self.architecture_findings),
                    "complexity": len(self.complexity_findings),
                    "message": len(self.message_findings),
                    "parameter": len(self.parameter_findings),
                },
            },
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "title": f.title,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "affected_nodes": f.affected_nodes,
                    "recommendation": f.recommendation,
                    "fix_available": f.fix_available,
                }
                for f in self.all_findings
            ],
            "timing_analysis": {
                "callback_chains": len(self.callback_chains),
                "critical_path": {
                    "nodes": self.critical_path.nodes if self.critical_path else [],
                    "hops": self.critical_path.total_hops if self.critical_path else 0,
                } if self.critical_path else None,
            },
        }


class DeepAnalyzer:
    """
    Orchestrates all analysis modules for comprehensive code review.
    """

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.topic_graph: Optional[TopicGraphResult] = None
        self.launched_nodes: Set[str] = set()

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def add_topic_graph(self, topic_graph: TopicGraphResult):
        """Add topic graph."""
        self.topic_graph = topic_graph

    def set_launched_nodes(self, launched: Set[str]):
        """Set nodes that are actually launched (for dead code detection)."""
        self.launched_nodes = launched

    def _convert_qos_finding(self, f: QoSFinding) -> UnifiedFinding:
        """Convert QoS finding to unified format."""
        return UnifiedFinding(
            category='qos',
            severity=f.severity,
            title=f'QoS mismatch on topic: {f.topic}',
            description='; '.join(f.issues),
            file_path=f.publisher_file,
            affected_nodes=[f.publisher_node, f.subscriber_node],
            recommendation='Ensure publisher and subscriber QoS profiles are compatible',
            raw_finding=f,
        )

    def _convert_timing_issue(self, f: TimingIssue) -> UnifiedFinding:
        """Convert timing issue to unified format."""
        return UnifiedFinding(
            category='timing',
            severity=f.severity,
            title=f.description[:60] + '...' if len(f.description) > 60 else f.description,
            description=f.description,
            file_path=f.file_path,
            line_number=f.line_number,
            affected_nodes=[f.node],
            recommendation=f.recommendation,
            raw_finding=f,
        )

    def _convert_security_finding(self, f: SecurityFinding) -> UnifiedFinding:
        """Convert security finding to unified format."""
        cwe_str = f' ({f.cwe_id})' if f.cwe_id else ''
        return UnifiedFinding(
            category='security',
            severity=f.severity,
            title=f'{f.description}{cwe_str}',
            description=f'Found in {f.node}: {f.code_snippet}',
            file_path=f.file_path,
            line_number=f.line_number,
            affected_nodes=[f.node],
            recommendation=f.recommendation,
            raw_finding=f,
        )

    def _convert_architecture_finding(self, f: ArchitectureFinding) -> UnifiedFinding:
        """Convert architecture finding to unified format."""
        return UnifiedFinding(
            category='architecture',
            severity=f.severity,
            title=f.title,
            description=f.description,
            affected_nodes=f.affected_nodes,
            recommendation=f.recommendation,
            raw_finding=f,
        )

    def _convert_complexity_finding(self, f: ComplexityFinding) -> UnifiedFinding:
        """Convert complexity finding to unified format."""
        return UnifiedFinding(
            category='complexity',
            severity=f.severity,
            title=f'{f.finding_type}: {f.function_name}',
            description=f.description,
            file_path=f.file_path,
            line_number=f.line_number,
            affected_nodes=[f.node_name],
            recommendation=f.recommendation,
            raw_finding=f,
        )

    def _convert_message_finding(self, f: MessageTypeFinding) -> UnifiedFinding:
        """Convert message type finding to unified format."""
        return UnifiedFinding(
            category='message',
            severity=f.severity,
            title=f'{f.finding_type}: {f.topic}',
            description=f.description,
            affected_nodes=f.affected_nodes,
            recommendation=f.recommendation,
            raw_finding=f,
        )

    def _convert_parameter_finding(self, f: ParameterFinding) -> UnifiedFinding:
        """Convert parameter finding to unified format."""
        return UnifiedFinding(
            category='parameter',
            severity=f.severity,
            title=f'{f.finding_type}: {f.parameter_name}',
            description=f.description,
            file_path=f.file_path,
            line_number=f.line_number,
            affected_nodes=[f.node_name],
            recommendation=f.recommendation,
            raw_finding=f,
        )

    def analyze(self,
                enable_qos: bool = True,
                enable_timing: bool = True,
                enable_security: bool = True,
                enable_architecture: bool = True,
                enable_complexity: bool = True,
                enable_message: bool = True,
                enable_parameter: bool = True) -> DeepAnalysisReport:
        """Run all enabled analyses and generate unified report."""

        report = DeepAnalysisReport()

        # QoS Analysis
        if enable_qos:
            qos_analyzer = QoSAnalyzer()
            qos_analyzer.add_nodes(self.nodes)
            qos_findings = qos_analyzer.analyze()
            report.qos_findings = [self._convert_qos_finding(f) for f in qos_findings]

        # Timing Analysis
        if enable_timing:
            timing_analyzer = TimingAnalyzer()
            timing_analyzer.add_nodes(self.nodes)
            if self.topic_graph:
                timing_analyzer.add_topic_graph(self.topic_graph)
            timing_result = timing_analyzer.analyze()
            report.timing_findings = [self._convert_timing_issue(f) for f in timing_result.issues]
            report.callback_chains = timing_result.chains
            report.critical_path = timing_result.critical_path

        # Security Analysis
        if enable_security:
            security_analyzer = SecurityAnalyzer()
            security_analyzer.add_nodes(self.nodes)
            security_findings = security_analyzer.analyze()
            report.security_findings = [self._convert_security_finding(f) for f in security_findings]

        # Architecture Analysis
        if enable_architecture:
            arch_analyzer = ArchitectureAnalyzer()
            arch_analyzer.add_nodes(self.nodes)
            if self.topic_graph:
                arch_analyzer.add_topic_graph(self.topic_graph)
            if self.launched_nodes:
                arch_analyzer.launched_nodes = self.launched_nodes
            arch_findings = arch_analyzer.analyze()
            report.architecture_findings = [self._convert_architecture_finding(f) for f in arch_findings]

        # Complexity Analysis
        if enable_complexity:
            complexity_analyzer = ComplexityAnalyzer()
            complexity_analyzer.add_nodes(self.nodes)
            complexity_findings = complexity_analyzer.analyze()
            report.complexity_findings = [self._convert_complexity_finding(f) for f in complexity_findings]

        # Message Type Analysis
        if enable_message:
            message_analyzer = MessageTypeAnalyzer()
            message_analyzer.add_nodes(self.nodes)
            if self.topic_graph:
                message_analyzer.add_topic_graph(self.topic_graph)
            message_findings = message_analyzer.analyze()
            report.message_findings = [self._convert_message_finding(f) for f in message_findings]

        # Parameter Analysis
        if enable_parameter:
            param_analyzer = ParameterAnalyzer()
            param_analyzer.add_nodes(self.nodes)
            param_findings = param_analyzer.analyze()
            report.parameter_findings = [self._convert_parameter_finding(f) for f in param_findings]

        # Combine all findings
        all_findings = (
            report.qos_findings +
            report.timing_findings +
            report.security_findings +
            report.architecture_findings +
            report.complexity_findings +
            report.message_findings +
            report.parameter_findings
        )

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_findings.sort(key=lambda f: severity_order.get(f.severity, 4))
        report.all_findings = all_findings

        # Calculate summary
        report.total_findings = len(all_findings)
        report.critical_count = sum(1 for f in all_findings if f.severity == 'critical')
        report.high_count = sum(1 for f in all_findings if f.severity == 'high')
        report.medium_count = sum(1 for f in all_findings if f.severity == 'medium')
        report.low_count = sum(1 for f in all_findings if f.severity == 'low')

        return report


def deep_analyze(
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    launched_nodes: Optional[Set[str]] = None,
) -> DeepAnalysisReport:
    """Convenience function for deep analysis."""
    analyzer = DeepAnalyzer()
    analyzer.add_nodes(nodes)
    if topic_graph:
        analyzer.add_topic_graph(topic_graph)
    if launched_nodes:
        analyzer.set_launched_nodes(launched_nodes)
    return analyzer.analyze()
