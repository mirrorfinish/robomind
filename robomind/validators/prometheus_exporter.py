"""
RoboMind Prometheus Exporter - Export metrics for monitoring.

Generates Prometheus-compatible metrics from RoboMind analysis and validation:
- Node counts (expected vs running)
- Topic connection status
- Validation findings
- HTTP endpoint health

Usage:
    robomind analyze ~/betaray --export-prometheus metrics.prom
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from robomind.ros2.node_extractor import ROS2NodeInfo
from robomind.ros2.topic_extractor import TopicGraphResult

logger = logging.getLogger(__name__)


@dataclass
class PrometheusMetric:
    """A single Prometheus metric."""
    name: str
    value: float
    help_text: str
    metric_type: str = "gauge"  # gauge, counter, histogram
    labels: Dict[str, str] = None

    def format(self) -> str:
        """Format as Prometheus exposition format."""
        lines = []
        lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        if self.labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
            lines.append(f"{self.name}{{{label_str}}} {self.value}")
        else:
            lines.append(f"{self.name} {self.value}")

        return "\n".join(lines)


class PrometheusExporter:
    """
    Export RoboMind analysis results as Prometheus metrics.

    Generates metrics for:
    - Static analysis counts
    - Validation results
    - HTTP endpoint health
    - Confidence scores

    Usage:
        exporter = PrometheusExporter()
        exporter.add_node_metrics(nodes)
        exporter.add_topic_metrics(topic_graph)
        exporter.add_validation_metrics(validation_result)
        exporter.export("metrics.prom")
    """

    def __init__(self, project_name: str = "robomind"):
        self.project_name = project_name
        self.metrics: List[PrometheusMetric] = []
        self.timestamp = datetime.now()

    def add_metric(
        self,
        name: str,
        value: float,
        help_text: str,
        metric_type: str = "gauge",
        labels: Dict[str, str] = None,
    ):
        """Add a metric."""
        full_name = f"robomind_{name}"
        self.metrics.append(PrometheusMetric(
            name=full_name,
            value=value,
            help_text=help_text,
            metric_type=metric_type,
            labels=labels,
        ))

    def add_node_metrics(self, nodes: List[ROS2NodeInfo]):
        """Add node-related metrics."""
        self.add_metric(
            "nodes_total",
            len(nodes),
            "Total ROS2 nodes detected in static analysis",
        )

        # Count by package
        packages = {}
        for node in nodes:
            pkg = node.package_name or "unknown"
            packages[pkg] = packages.get(pkg, 0) + 1

        self.add_metric(
            "packages_total",
            len(packages),
            "Total ROS2 packages found",
        )

        # Publishers and subscribers
        total_publishers = sum(len(n.publishers) for n in nodes)
        total_subscribers = sum(len(n.subscribers) for n in nodes)
        total_timers = sum(len(n.timers) for n in nodes)
        total_services = sum(len(n.services) for n in nodes)

        self.add_metric(
            "publishers_total",
            total_publishers,
            "Total publishers declared in code",
        )
        self.add_metric(
            "subscribers_total",
            total_subscribers,
            "Total subscribers declared in code",
        )
        self.add_metric(
            "timers_total",
            total_timers,
            "Total timers declared in code",
        )
        self.add_metric(
            "services_total",
            total_services,
            "Total services declared in code",
        )

    def add_topic_metrics(self, topic_graph: TopicGraphResult):
        """Add topic-related metrics."""
        self.add_metric(
            "topics_total",
            len(topic_graph.topics),
            "Total topics found in static analysis",
        )

        connected = topic_graph.get_connected_topics()
        self.add_metric(
            "topics_connected",
            len(connected),
            "Topics with matched publisher and subscriber",
        )

        orphaned = len(topic_graph.topics) - len(connected)
        self.add_metric(
            "topics_orphaned",
            orphaned,
            "Topics missing either publisher or subscriber",
        )

    def add_validation_metrics(self, validation_result):
        """Add validation-related metrics."""
        if not validation_result:
            return

        summary = validation_result.summary()

        self.add_metric(
            "validation_diffs_total",
            summary.get("total_diffs", 0),
            "Total validation differences found",
        )

        # By severity
        by_severity = summary.get("by_severity", {})
        for severity, count in by_severity.items():
            self.add_metric(
                "validation_diffs",
                count,
                f"Validation differences by severity",
                labels={"severity": severity},
            )

        # Live system availability
        self.add_metric(
            "validation_live_available",
            1 if summary.get("live_available") else 0,
            "Whether live system validation was available",
        )

        # Nodes running vs expected
        if validation_result.live_info:
            live_nodes = len(validation_result.live_info.nodes)
            self.add_metric(
                "nodes_running",
                live_nodes,
                "Nodes confirmed running at validation time",
            )

            live_topics = len(validation_result.live_info.topics)
            self.add_metric(
                "topics_live",
                live_topics,
                "Topics active at validation time",
            )

    def add_http_metrics(self, http_comm_map):
        """Add HTTP communication metrics."""
        if not http_comm_map:
            return

        summary = http_comm_map.summary()

        self.add_metric(
            "http_endpoints_total",
            summary.get("http_endpoints", 0),
            "HTTP endpoints detected in code",
        )
        self.add_metric(
            "http_clients_total",
            summary.get("http_clients", 0),
            "HTTP client calls detected in code",
        )
        self.add_metric(
            "http_target_hosts_total",
            len(summary.get("http_target_hosts", [])),
            "Unique HTTP target hosts",
        )

    def add_confidence_metrics(self, confidence_scores: List):
        """Add confidence score metrics."""
        if not confidence_scores:
            return

        scores = [s.score for s in confidence_scores]
        avg_score = sum(scores) / len(scores) if scores else 0

        self.add_metric(
            "confidence_average",
            round(avg_score, 3),
            "Average confidence score across all findings",
        )

        high_conf = len([s for s in scores if s >= 0.7])
        low_conf = len([s for s in scores if s < 0.3])

        self.add_metric(
            "confidence_high",
            high_conf,
            "Findings with high confidence (>= 0.7)",
        )
        self.add_metric(
            "confidence_low",
            low_conf,
            "Findings with low confidence (< 0.3)",
        )

    def add_http_health_metrics(self, health_results: Dict[str, Dict]):
        """Add HTTP endpoint health check results."""
        for endpoint, result in health_results.items():
            status_code = result.get("status_code", 0)
            response_time = result.get("response_time_ms", 0)
            available = 1 if status_code == 200 else 0

            # Clean endpoint for label (remove http://)
            label = endpoint.replace("http://", "").replace("https://", "")

            self.add_metric(
                "http_endpoint_available",
                available,
                "HTTP endpoint availability (1=available, 0=unavailable)",
                labels={"endpoint": label},
            )

            if response_time > 0:
                self.add_metric(
                    "http_endpoint_response_ms",
                    response_time,
                    "HTTP endpoint response time in milliseconds",
                    labels={"endpoint": label},
                )

    def add_systemd_metrics(self, systemd_results: List[Dict]):
        """Add systemd service status metrics.

        Args:
            systemd_results: List of dicts with 'host' and 'service' keys,
                           where 'service' is a SystemdService object.
        """
        for entry in systemd_results:
            host = entry.get("host", "unknown")
            svc = entry.get("service")
            if not svc:
                continue

            self.add_metric(
                "systemd_service_enabled",
                1 if svc.is_enabled else 0,
                "Whether systemd service is enabled",
                labels={"service": svc.name, "host": host},
            )
            self.add_metric(
                "systemd_service_active",
                1 if svc.is_active else 0,
                "Whether systemd service is currently active",
                labels={"service": svc.name, "host": host},
            )

    def add_ai_service_metrics(self, ai_services):
        """Add AI/ML service metrics.

        Args:
            ai_services: AIServiceAnalysisResult from the ai_service_analyzer.
        """
        if not ai_services or not ai_services.services:
            return

        self.add_metric(
            "ai_services_total",
            len(ai_services.services),
            "Total AI/ML services detected",
        )

        gpu_count = sum(1 for s in ai_services.services if s.gpu_required)
        self.add_metric(
            "ai_services_gpu_required",
            gpu_count,
            "AI services requiring GPU",
        )

        for svc in ai_services.services:
            labels = {
                "name": svc.name,
                "framework": svc.framework,
            }
            if svc.model_name:
                labels["model"] = svc.model_name
            if svc.port:
                labels["port"] = str(svc.port)

            self.add_metric(
                "ai_service_detected",
                1,
                "AI service detected in codebase",
                labels=labels,
            )
            self.add_metric(
                "ai_service_callers",
                len(svc.caller_files),
                "Number of files calling this AI service",
                labels={"name": svc.name},
            )

    def build(self) -> str:
        """Build complete Prometheus metrics output."""
        lines = []
        lines.append(f"# RoboMind Prometheus Metrics")
        lines.append(f"# Generated: {self.timestamp.isoformat()}")
        lines.append(f"# Project: {self.project_name}")
        lines.append("")

        for metric in self.metrics:
            lines.append(metric.format())
            lines.append("")

        return "\n".join(lines)

    def export(self, output_path: Path) -> bool:
        """Export metrics to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            content = self.build()
            with open(output_path, "w") as f:
                f.write(content)

            logger.info(f"Exported Prometheus metrics to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            return False


def export_prometheus_metrics(
    output_path: Path,
    nodes: List[ROS2NodeInfo],
    topic_graph: Optional[TopicGraphResult] = None,
    validation_result=None,
    http_comm_map=None,
    confidence_scores: List = None,
    http_health_results: Dict[str, Dict] = None,
    systemd_results: List[Dict] = None,
    ai_services=None,
    project_name: str = "robomind",
) -> bool:
    """
    Convenience function to export Prometheus metrics.

    Args:
        output_path: Path to output .prom file
        nodes: List of ROS2NodeInfo
        topic_graph: Optional TopicGraphResult
        validation_result: Optional ValidationResult
        http_comm_map: Optional HTTP CommunicationMap
        confidence_scores: Optional list of confidence scores
        http_health_results: Optional dict of HTTP health check results
        systemd_results: Optional list of systemd service results
        ai_services: Optional AIServiceAnalysisResult
        project_name: Project name for metrics

    Returns:
        True if export succeeded
    """
    exporter = PrometheusExporter(project_name)
    exporter.add_node_metrics(nodes)

    if topic_graph:
        exporter.add_topic_metrics(topic_graph)
    if validation_result:
        exporter.add_validation_metrics(validation_result)
    if http_comm_map:
        exporter.add_http_metrics(http_comm_map)
    if confidence_scores:
        exporter.add_confidence_metrics(confidence_scores)
    if http_health_results:
        exporter.add_http_health_metrics(http_health_results)
    if systemd_results:
        exporter.add_systemd_metrics(systemd_results)
    if ai_services:
        exporter.add_ai_service_metrics(ai_services)

    return exporter.export(output_path)
