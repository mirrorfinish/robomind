"""
Tests for enhanced exporters (AI Context and SARIF).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from robomind.exporters.ai_context_exporter import (
    AIContextExporter,
    AIFinding,
    export_ai_context,
)
from robomind.exporters.sarif_exporter import (
    SARIFExporter,
    SARIFResult,
    export_sarif,
    SARIF_VERSION,
)
from robomind.ros2.node_extractor import ROS2NodeInfo, PublisherInfo, SubscriberInfo


class TestAIFinding:
    """Tests for AIFinding dataclass."""

    def test_basic_creation(self):
        """Test creating a basic finding."""
        finding = AIFinding(
            id="TEST-001",
            summary="Test finding",
            confidence=0.8,
            risk_level="HIGH",
        )
        assert finding.id == "TEST-001"
        assert finding.confidence == 0.8
        assert finding.actionable is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        finding = AIFinding(
            id="TEST-001",
            summary="Test finding",
            file_path="/test/file.py",
            line_number=42,
            confidence=0.9,
            risk_level="CRITICAL",
            effort_level="LOW",
            suggested_fix="Fix this issue",
            runtime_confirmed=True,
        )
        result = finding.to_dict()

        assert result["id"] == "TEST-001"
        assert result["file"] == "/test/file.py:42"
        assert result["confidence"] == 0.9
        assert result["suggested_fix"] == "Fix this issue"
        assert result["runtime_confirmed"] is True

    def test_non_actionable(self):
        """Test non-actionable finding."""
        finding = AIFinding(
            id="TEST-002",
            summary="Non-actionable",
            confidence=0.3,
            actionable=False,
            non_actionable_reason="Dead code",
        )
        result = finding.to_dict()

        assert result["actionable"] is False
        assert result["reason_non_actionable"] == "Dead code"


class TestAIContextExporter:
    """Tests for AIContextExporter class."""

    def test_init(self):
        """Test exporter initialization."""
        exporter = AIContextExporter()
        assert exporter.project_name == "Unknown"
        assert exporter.nodes == []

    def test_set_project_info(self):
        """Test setting project info."""
        exporter = AIContextExporter()
        exporter.set_project_info("TestProject")
        assert exporter.project_name == "TestProject"

    def test_add_findings(self):
        """Test adding actionable and non-actionable findings."""
        exporter = AIContextExporter()

        actionable = AIFinding(id="A001", summary="Actionable")
        non_actionable = AIFinding(id="N001", summary="Non-actionable")

        exporter.add_actionable_finding(actionable)
        exporter.add_non_actionable_finding(non_actionable)

        assert len(exporter.actionable_findings) == 1
        assert len(exporter.non_actionable_findings) == 1

    def test_build_basic(self):
        """Test building basic context."""
        exporter = AIContextExporter()
        exporter.set_project_info("Test")

        nodes = [
            ROS2NodeInfo(
                name="test_node",
                class_name="TestNode",
                file_path=Path("/test/node.py"),
                line_number=1,
                end_line=50,
                package_name="test_pkg",
            ),
        ]
        exporter.set_nodes(nodes)

        result = exporter.build()

        assert "_comment" in result
        assert "system_summary" in result
        assert "ai_hints" in result
        assert result["system_summary"]["ros2_nodes"] == 1

    def test_build_with_http(self):
        """Test building with HTTP communication."""
        exporter = AIContextExporter()

        http_comm_map = MagicMock()
        http_comm_map.summary.return_value = {
            "http_endpoints": 5,
            "http_clients": 3,
            "http_target_hosts": ["host1:8080"],
        }

        exporter.set_http_communication(http_comm_map)

        result = exporter.build()
        summary = result["system_summary"]

        assert summary["cross_system_protocol"] == "http"
        assert summary["http_endpoints"] == 5

    def test_export_to_file(self):
        """Test exporting to file."""
        exporter = AIContextExporter()
        exporter.set_project_info("TestProject")

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = exporter.export(output_path)
            assert result is True
            assert output_path.exists()

            content = output_path.read_text()
            assert "AI-Optimized Context" in content
            assert "TestProject" in content
        finally:
            if output_path.exists():
                output_path.unlink()


class TestExportAIContext:
    """Tests for convenience function."""

    def test_export_ai_context(self):
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

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = export_ai_context(
                output_path=output_path,
                nodes=nodes,
                project_name="test",
            )
            assert result is True
            assert output_path.exists()

            # Parse and verify YAML
            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert "system_summary" in data
        finally:
            if output_path.exists():
                output_path.unlink()


class TestSARIFResult:
    """Tests for SARIFResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic result."""
        result = SARIFResult(
            rule_id="ROBOMIND001",
            message="Test message",
            level="warning",
        )
        assert result.rule_id == "ROBOMIND001"
        assert result.level == "warning"

    def test_to_sarif(self):
        """Test converting to SARIF format."""
        result = SARIFResult(
            rule_id="ROBOMIND001",
            message="Test message",
            file_path="/test/file.py",
            line_number=42,
            level="error",
            confidence=0.9,
        )
        sarif = result.to_sarif()

        assert sarif["ruleId"] == "ROBOMIND001"
        assert sarif["level"] == "error"
        assert sarif["message"]["text"] == "Test message"
        assert sarif["locations"][0]["physicalLocation"]["region"]["startLine"] == 42
        assert sarif["properties"]["confidence"] == 0.9

    def test_to_sarif_relative_path(self):
        """Test converting with relative path."""
        result = SARIFResult(
            rule_id="ROBOMIND001",
            message="Test",
            file_path="/home/user/project/src/file.py",
        )
        sarif = result.to_sarif(base_path="/home/user/project")

        uri = sarif["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert uri == "src/file.py"


class TestSARIFExporter:
    """Tests for SARIFExporter class."""

    def test_init(self):
        """Test exporter initialization."""
        exporter = SARIFExporter("test_project")
        assert exporter.project_name == "test_project"
        assert exporter.results == []

    def test_add_result(self):
        """Test adding results."""
        exporter = SARIFExporter()
        exporter.add_result(SARIFResult(
            rule_id="ROBOMIND001",
            message="Test",
        ))
        assert len(exporter.results) == 1

    def test_build_structure(self):
        """Test building SARIF structure."""
        exporter = SARIFExporter("test_project")
        exporter.add_result(SARIFResult(
            rule_id="ROBOMIND001",
            message="Test finding",
        ))

        sarif = exporter.build()

        assert sarif["$schema"] == "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
        assert sarif["version"] == SARIF_VERSION
        assert len(sarif["runs"]) == 1

        run = sarif["runs"][0]
        assert run["tool"]["driver"]["name"] == "RoboMind"
        assert len(run["results"]) == 1
        assert len(run["tool"]["driver"]["rules"]) == 1

    def test_export_to_file(self):
        """Test exporting to file."""
        exporter = SARIFExporter("test_project")
        exporter.add_result(SARIFResult(
            rule_id="ROBOMIND001",
            message="Test finding",
        ))

        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = exporter.export(output_path)
            assert result is True
            assert output_path.exists()

            # Parse and verify JSON
            with open(output_path) as f:
                data = json.load(f)

            assert data["version"] == SARIF_VERSION
            assert len(data["runs"][0]["results"]) == 1
        finally:
            if output_path.exists():
                output_path.unlink()


class TestExportSARIF:
    """Tests for convenience function."""

    def test_export_sarif(self):
        """Test the convenience function."""
        nodes = [
            ROS2NodeInfo(
                name="node1",
                class_name="Node1",
                file_path=Path("/test/node1.py"),
                line_number=1,
                end_line=50,
                subscribers=[
                    SubscriberInfo(
                        topic="orphan_topic",
                        msg_type="std_msgs/String",
                        callback="cb",
                        line_number=10,
                    ),
                ],
            ),
        ]

        topic_graph = MagicMock()
        topic_graph.topics = {
            "orphan_topic": MagicMock(
                subscribers=["node1"],
                publishers=[],
            ),
        }

        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = export_sarif(
                output_path=output_path,
                nodes=nodes,
                topic_graph=topic_graph,
                project_name="test",
            )
            assert result is True
            assert output_path.exists()

            # Parse and verify
            with open(output_path) as f:
                data = json.load(f)

            # Should have at least one result for orphaned subscriber
            assert len(data["runs"][0]["results"]) >= 1
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_export_with_relative_topics(self):
        """Test export finds relative topic names."""
        nodes = [
            ROS2NodeInfo(
                name="node1",
                class_name="Node1",
                file_path=Path("/test/node1.py"),
                line_number=1,
                end_line=50,
                publishers=[
                    PublisherInfo(topic="no_slash", msg_type="String", line_number=10),
                ],
            ),
        ]

        topic_graph = MagicMock()
        topic_graph.topics = {
            "no_slash": MagicMock(
                publishers=["node1"],
                subscribers=[],
            ),
        }

        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = export_sarif(
                output_path=output_path,
                nodes=nodes,
                topic_graph=topic_graph,
            )
            assert result is True

            with open(output_path) as f:
                data = json.load(f)

            # Should have ROBOMIND003 for relative topic name
            rule_ids = [r["ruleId"] for r in data["runs"][0]["results"]]
            assert "ROBOMIND003" in rule_ids
        finally:
            if output_path.exists():
                output_path.unlink()
