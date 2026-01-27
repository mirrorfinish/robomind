"""
Tests for Prometheus metrics exporter.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from robomind.validators.prometheus_exporter import (
    PrometheusExporter,
    PrometheusMetric,
    export_prometheus_metrics,
)
from robomind.ros2.node_extractor import ROS2NodeInfo, PublisherInfo, SubscriberInfo, TimerInfo


class TestPrometheusMetric:
    """Tests for PrometheusMetric dataclass."""

    def test_format_simple(self):
        """Test formatting a simple metric."""
        metric = PrometheusMetric(
            name="test_metric",
            value=42.0,
            help_text="A test metric",
            metric_type="gauge",
        )
        output = metric.format()
        assert "# HELP test_metric A test metric" in output
        assert "# TYPE test_metric gauge" in output
        assert "test_metric 42.0" in output

    def test_format_with_labels(self):
        """Test formatting a metric with labels."""
        metric = PrometheusMetric(
            name="test_metric",
            value=10.0,
            help_text="A labeled metric",
            metric_type="gauge",
            labels={"severity": "high", "type": "error"},
        )
        output = metric.format()
        assert 'severity="high"' in output
        assert 'type="error"' in output


class TestPrometheusExporter:
    """Tests for PrometheusExporter class."""

    def test_init(self):
        """Test exporter initialization."""
        exporter = PrometheusExporter("test_project")
        assert exporter.project_name == "test_project"
        assert exporter.metrics == []

    def test_add_metric(self):
        """Test adding a metric."""
        exporter = PrometheusExporter()
        exporter.add_metric(
            "test_count",
            100,
            "A test counter",
        )
        assert len(exporter.metrics) == 1
        assert exporter.metrics[0].name == "robomind_test_count"
        assert exporter.metrics[0].value == 100

    def test_add_node_metrics(self):
        """Test adding node-related metrics."""
        exporter = PrometheusExporter()

        nodes = [
            ROS2NodeInfo(
                name="node1",
                class_name="Node1",
                file_path=Path("/test/node1.py"),
                line_number=1,
                end_line=50,
                package_name="pkg1",
                publishers=[
                    PublisherInfo(topic="/topic1", msg_type="std_msgs/String", line_number=10),
                ],
                subscribers=[
                    SubscriberInfo(topic="/topic2", msg_type="std_msgs/Int32", callback="cb", line_number=15),
                ],
            ),
            ROS2NodeInfo(
                name="node2",
                class_name="Node2",
                file_path=Path("/test/node2.py"),
                line_number=1,
                end_line=30,
                package_name="pkg1",
            ),
        ]

        exporter.add_node_metrics(nodes)

        # Check that expected metrics exist
        metric_names = [m.name for m in exporter.metrics]
        assert "robomind_nodes_total" in metric_names
        assert "robomind_publishers_total" in metric_names
        assert "robomind_subscribers_total" in metric_names

    def test_add_topic_metrics(self):
        """Test adding topic-related metrics."""
        exporter = PrometheusExporter()

        topic_graph = MagicMock()
        topic_graph.topics = {"/topic1": {}, "/topic2": {}, "/topic3": {}}
        topic_graph.get_connected_topics.return_value = [MagicMock(), MagicMock()]

        exporter.add_topic_metrics(topic_graph)

        metric_names = [m.name for m in exporter.metrics]
        assert "robomind_topics_total" in metric_names
        assert "robomind_topics_connected" in metric_names
        assert "robomind_topics_orphaned" in metric_names

    def test_add_validation_metrics(self):
        """Test adding validation metrics."""
        exporter = PrometheusExporter()

        validation = MagicMock()
        validation.summary.return_value = {
            "total_diffs": 5,
            "by_severity": {"info": 2, "warning": 2, "error": 1, "critical": 0},
            "live_available": True,
        }
        validation.live_info = MagicMock()
        validation.live_info.nodes = ["node1", "node2"]
        validation.live_info.topics = {"/t1": {}, "/t2": {}}

        exporter.add_validation_metrics(validation)

        metric_names = [m.name for m in exporter.metrics]
        assert "robomind_validation_diffs_total" in metric_names
        assert "robomind_nodes_running" in metric_names

    def test_add_http_metrics(self):
        """Test adding HTTP communication metrics."""
        exporter = PrometheusExporter()

        http_comm_map = MagicMock()
        http_comm_map.summary.return_value = {
            "http_endpoints": 5,
            "http_clients": 8,
            "http_target_hosts": ["host1:8080", "host2:9091"],
        }

        exporter.add_http_metrics(http_comm_map)

        metric_names = [m.name for m in exporter.metrics]
        assert "robomind_http_endpoints_total" in metric_names
        assert "robomind_http_clients_total" in metric_names

    def test_add_http_health_metrics(self):
        """Test adding HTTP health check metrics."""
        exporter = PrometheusExporter()

        health_results = {
            "http://host1:8080/health": {
                "status_code": 200,
                "response_time_ms": 45.5,
            },
            "http://host2:9091/health": {
                "status_code": 500,
                "response_time_ms": 100.0,
            },
        }

        exporter.add_http_health_metrics(health_results)

        metric_names = [m.name for m in exporter.metrics]
        assert "robomind_http_endpoint_available" in metric_names
        assert "robomind_http_endpoint_response_ms" in metric_names

    def test_build_output(self):
        """Test building complete output."""
        exporter = PrometheusExporter("test_project")
        exporter.add_metric("test", 42, "Test metric")

        output = exporter.build()

        assert "# RoboMind Prometheus Metrics" in output
        assert "# Project: test_project" in output
        assert "robomind_test 42" in output

    def test_export_to_file(self):
        """Test exporting to file."""
        exporter = PrometheusExporter("test_project")
        exporter.add_metric("test", 42, "Test metric")

        with tempfile.NamedTemporaryFile(suffix=".prom", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = exporter.export(output_path)
            assert result is True
            assert output_path.exists()

            content = output_path.read_text()
            assert "robomind_test 42" in content
        finally:
            if output_path.exists():
                output_path.unlink()


class TestExportFunction:
    """Tests for convenience export function."""

    def test_export_prometheus_metrics(self):
        """Test the convenience function."""
        nodes = [
            ROS2NodeInfo(
                name="node1",
                class_name="Node1",
                file_path=Path("/test/node1.py"),
                line_number=1,
                end_line=50,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".prom", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = export_prometheus_metrics(
                output_path=output_path,
                nodes=nodes,
                project_name="test",
            )
            assert result is True
            assert output_path.exists()

            content = output_path.read_text()
            assert "robomind_nodes_total 1" in content
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_export_with_all_data(self):
        """Test export with all optional data."""
        nodes = [
            ROS2NodeInfo(
                name="node1",
                class_name="Node1",
                file_path=Path("/test/node1.py"),
                line_number=1,
                end_line=50,
                publishers=[PublisherInfo(topic="/t1", msg_type="String", line_number=10)],
            ),
        ]

        topic_graph = MagicMock()
        topic_graph.topics = {"/t1": {}}
        topic_graph.get_connected_topics.return_value = []

        validation = MagicMock()
        validation.summary.return_value = {
            "total_diffs": 1,
            "by_severity": {"info": 1, "warning": 0, "error": 0, "critical": 0},
            "live_available": False,
        }
        validation.live_info = None

        http_comm_map = MagicMock()
        http_comm_map.summary.return_value = {
            "http_endpoints": 2,
            "http_clients": 3,
            "http_target_hosts": [],
        }

        with tempfile.NamedTemporaryFile(suffix=".prom", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = export_prometheus_metrics(
                output_path=output_path,
                nodes=nodes,
                topic_graph=topic_graph,
                validation_result=validation,
                http_comm_map=http_comm_map,
                project_name="full_test",
            )
            assert result is True

            content = output_path.read_text()
            assert "robomind_nodes_total" in content
            assert "robomind_topics_total" in content
            assert "robomind_http_endpoints_total" in content
        finally:
            if output_path.exists():
                output_path.unlink()
