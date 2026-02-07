"""
RoboMind Analyzers - System analysis modules

Includes:
- coupling.py - Calculate coupling strength between components
- flow_tracer.py - Trace data flow paths through the system
- confidence.py - Calculate confidence scores for findings
- recommendations.py - Generate actionable recommendations
- qos_analyzer.py - QoS compatibility checking
- timing_analyzer.py - Timing chain analysis
- security_analyzer.py - Security vulnerability scanning
- architecture_analyzer.py - Architecture pattern detection
- complexity_analyzer.py - Callback complexity metrics
- message_analyzer.py - Message type checking
- parameter_analyzer.py - Parameter validation
- deep_analyzer.py - Unified deep analysis
- ai_service_analyzer.py - AI/ML inference service detection
"""

from robomind.analyzers.coupling import (
    CouplingAnalyzer,
    CouplingMatrix,
    CouplingScore,
    analyze_coupling,
)
from robomind.analyzers.flow_tracer import (
    FlowTracer,
    FlowPath,
    FlowTraceResult,
    trace_flow,
)
from robomind.analyzers.confidence import (
    ConfidenceCalculator,
    ConfidenceScore,
    ConfidenceLevel,
    ConfidenceFactor,
    NodeConfidenceContext,
    get_confidence_summary,
)
from robomind.analyzers.recommendations import (
    RecommendationEngine,
    Recommendation,
    RecommendationReport,
    generate_recommendations,
)
from robomind.analyzers.qos_analyzer import (
    QoSAnalyzer,
    QoSProfile,
    QoSFinding,
    analyze_qos,
)
from robomind.analyzers.timing_analyzer import (
    TimingAnalyzer,
    TimingAnalysisResult,
    TimingIssue,
    CallbackChain,
    analyze_timing,
)
from robomind.analyzers.security_analyzer import (
    SecurityAnalyzer,
    SecurityFinding,
    analyze_security,
)
from robomind.analyzers.architecture_analyzer import (
    ArchitectureAnalyzer,
    ArchitectureFinding,
    analyze_architecture,
)
from robomind.analyzers.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityFinding,
    ComplexityMetrics,
    analyze_complexity,
)
from robomind.analyzers.message_analyzer import (
    MessageTypeAnalyzer,
    MessageTypeFinding,
    analyze_message_types,
)
from robomind.analyzers.parameter_analyzer import (
    ParameterAnalyzer,
    ParameterFinding,
    analyze_parameters,
)
from robomind.analyzers.deep_analyzer import (
    DeepAnalyzer,
    DeepAnalysisReport,
    UnifiedFinding,
    deep_analyze,
)
from robomind.analyzers.ai_service_analyzer import (
    AIServiceAnalyzer,
    AIServiceInfo,
    AIServiceAnalysisResult,
    analyze_ai_services,
)
from robomind.analyzers.impact_analyzer import (
    ImpactAnalyzer,
    ImpactItem,
    ImpactResult,
    analyze_impact,
)

__all__ = [
    # Coupling
    "CouplingAnalyzer",
    "CouplingMatrix",
    "CouplingScore",
    "analyze_coupling",
    # Flow
    "FlowTracer",
    "FlowPath",
    "FlowTraceResult",
    "trace_flow",
    # Confidence
    "ConfidenceCalculator",
    "ConfidenceScore",
    "ConfidenceLevel",
    "ConfidenceFactor",
    "NodeConfidenceContext",
    "get_confidence_summary",
    # Recommendations
    "RecommendationEngine",
    "Recommendation",
    "RecommendationReport",
    "generate_recommendations",
    # QoS
    "QoSAnalyzer",
    "QoSProfile",
    "QoSFinding",
    "analyze_qos",
    # Timing
    "TimingAnalyzer",
    "TimingAnalysisResult",
    "TimingIssue",
    "CallbackChain",
    "analyze_timing",
    # Security
    "SecurityAnalyzer",
    "SecurityFinding",
    "analyze_security",
    # Architecture
    "ArchitectureAnalyzer",
    "ArchitectureFinding",
    "analyze_architecture",
    # Complexity
    "ComplexityAnalyzer",
    "ComplexityFinding",
    "ComplexityMetrics",
    "analyze_complexity",
    # Message Types
    "MessageTypeAnalyzer",
    "MessageTypeFinding",
    "analyze_message_types",
    # Parameters
    "ParameterAnalyzer",
    "ParameterFinding",
    "analyze_parameters",
    # Deep Analysis
    "DeepAnalyzer",
    "DeepAnalysisReport",
    "UnifiedFinding",
    "deep_analyze",
    # AI Services
    "AIServiceAnalyzer",
    "AIServiceInfo",
    "AIServiceAnalysisResult",
    "analyze_ai_services",
    # Impact Analysis
    "ImpactAnalyzer",
    "ImpactItem",
    "ImpactResult",
    "analyze_impact",
]
