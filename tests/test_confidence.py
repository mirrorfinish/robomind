"""
Tests for confidence scoring module.

Tests the ConfidenceCalculator and related classes for calculating
confidence scores to filter false positives.
"""

import pytest
from pathlib import Path

from robomind.analyzers.confidence import (
    ConfidenceCalculator,
    ConfidenceScore,
    ConfidenceLevel,
    ConfidenceFactor,
    NodeConfidenceContext,
    get_confidence_summary,
    CONFIDENCE_FACTORS,
)


class TestConfidenceLevel:
    """Test ConfidenceLevel enum and from_score method."""

    def test_from_score_very_high(self):
        """Score >= 0.9 should be VERY_HIGH."""
        assert ConfidenceLevel.from_score(0.9) == ConfidenceLevel.VERY_HIGH
        assert ConfidenceLevel.from_score(1.0) == ConfidenceLevel.VERY_HIGH
        assert ConfidenceLevel.from_score(0.95) == ConfidenceLevel.VERY_HIGH

    def test_from_score_high(self):
        """Score 0.7-0.9 should be HIGH."""
        assert ConfidenceLevel.from_score(0.7) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.8) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.89) == ConfidenceLevel.HIGH

    def test_from_score_medium(self):
        """Score 0.5-0.7 should be MEDIUM."""
        assert ConfidenceLevel.from_score(0.5) == ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.from_score(0.6) == ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.from_score(0.69) == ConfidenceLevel.MEDIUM

    def test_from_score_low(self):
        """Score 0.3-0.5 should be LOW."""
        assert ConfidenceLevel.from_score(0.3) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.4) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.49) == ConfidenceLevel.LOW

    def test_from_score_very_low(self):
        """Score < 0.3 should be VERY_LOW."""
        assert ConfidenceLevel.from_score(0.0) == ConfidenceLevel.VERY_LOW
        assert ConfidenceLevel.from_score(0.1) == ConfidenceLevel.VERY_LOW
        assert ConfidenceLevel.from_score(0.29) == ConfidenceLevel.VERY_LOW


class TestConfidenceScore:
    """Test ConfidenceScore dataclass."""

    def test_to_dict(self):
        """Test converting score to dictionary."""
        factor = ConfidenceFactor(
            name="in_launch_file",
            impact=0.35,
            reason="Node referenced in launch file",
            evidence="robot.launch.py",
        )
        score = ConfidenceScore(
            score=0.85,
            level=ConfidenceLevel.HIGH,
            factors=[factor],
            base_score=0.5,
        )

        result = score.to_dict()

        assert result["score"] == 0.85
        assert result["level"] == "high"
        assert len(result["factors"]) == 1
        assert result["factors"][0]["name"] == "in_launch_file"
        assert result["factors"][0]["impact"] == 0.35
        assert result["factors"][0]["evidence"] == "robot.launch.py"


class TestNodeConfidenceContext:
    """Test NodeConfidenceContext dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        context = NodeConfidenceContext(
            node_name="test_node",
            file_path=Path("/src/test_node.py"),
        )

        assert context.node_name == "test_node"
        assert context.location_confidence == 1.0
        assert context.dead_code_indicators == []
        assert context.in_launch_files == []
        assert context.has_publishers is False
        assert context.has_subscribers is False


class TestConfidenceCalculator:
    """Test ConfidenceCalculator class."""

    def test_basic_node_confidence(self):
        """Test basic confidence calculation for a node."""
        calculator = ConfidenceCalculator()

        context = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            package_name="betaray_control",
        )

        score = calculator.calculate_node_confidence(context)

        # Base score is 0.5, minus no_launch_reference penalty
        assert score.score >= 0.0
        assert score.score <= 1.0
        assert isinstance(score.level, ConfidenceLevel)
        assert len(score.factors) > 0

    def test_launch_file_boosts_confidence(self):
        """Node in launch file should have higher confidence."""
        calculator = ConfidenceCalculator()

        # Node WITHOUT launch file reference
        context_no_launch = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
        )
        score_no_launch = calculator.calculate_node_confidence(context_no_launch)

        # Node WITH launch file reference
        context_with_launch = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
        )
        score_with_launch = calculator.calculate_node_confidence(context_with_launch)

        # Having launch file should increase confidence
        assert score_with_launch.score > score_no_launch.score

    def test_archive_directory_lowers_confidence(self):
        """Node in archive directory should have lower confidence."""
        calculator = ConfidenceCalculator()

        # Normal node
        context_normal = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
        )
        score_normal = calculator.calculate_node_confidence(context_normal)

        # Node in archive directory
        context_archive = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/archive/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
            dead_code_indicators=["directory:archive"],
        )
        score_archive = calculator.calculate_node_confidence(context_archive)

        # Archive should significantly lower confidence
        assert score_archive.score < score_normal.score

    def test_deprecated_filename_lowers_confidence(self):
        """Node with deprecated naming should have lower confidence."""
        calculator = ConfidenceCalculator()

        # Normal node
        context_normal = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
        )
        score_normal = calculator.calculate_node_confidence(context_normal)

        # Node with _old suffix
        context_deprecated = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller_old.py"),
            in_launch_files=["robot.launch.py"],
            dead_code_indicators=["filename:_deprecated.py"],
        )
        score_deprecated = calculator.calculate_node_confidence(context_deprecated)

        # Deprecated naming should lower confidence
        assert score_deprecated.score < score_normal.score

    def test_systemd_service_highly_boosts_confidence(self):
        """Node in systemd service should have high confidence."""
        calculator = ConfidenceCalculator()

        context = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
            in_systemd_services=["betaray-navigation.service"],
        )

        score = calculator.calculate_node_confidence(context)

        # Should be high confidence
        assert score.score >= 0.7
        assert score.level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]

    def test_matching_pubsub_boosts_confidence(self):
        """Node with matching pub/sub should have higher confidence."""
        calculator = ConfidenceCalculator()

        # Node without matching pub/sub
        context_no_match = NodeConfidenceContext(
            node_name="sensor_node",
            file_path=Path("/src/sensor_node.py"),
            in_launch_files=["robot.launch.py"],
            has_publishers=True,
            publisher_topics=["/sensor_data"],
        )
        score_no_match = calculator.calculate_node_confidence(context_no_match)

        # Node with matching pub/sub
        context_matched = NodeConfidenceContext(
            node_name="sensor_node",
            file_path=Path("/src/sensor_node.py"),
            in_launch_files=["robot.launch.py"],
            has_publishers=True,
            has_subscribers=True,
            publisher_topics=["/sensor_data"],
            subscriber_topics=["/cmd"],
            topics_with_both_pubsub=["/sensor_data"],
        )
        score_matched = calculator.calculate_node_confidence(context_matched)

        # Matching should boost confidence
        assert score_matched.score > score_no_match.score

    def test_http_overlap_lowers_confidence(self):
        """Node with HTTP endpoint for same data should have lower confidence."""
        calculator = ConfidenceCalculator()

        # Normal node
        context_normal = NodeConfidenceContext(
            node_name="sensor_node",
            file_path=Path("/src/sensor_node.py"),
            in_launch_files=["robot.launch.py"],
        )
        score_normal = calculator.calculate_node_confidence(context_normal)

        # Node with HTTP overlap
        context_http = NodeConfidenceContext(
            node_name="sensor_node",
            file_path=Path("/src/sensor_node.py"),
            in_launch_files=["robot.launch.py"],
            http_endpoints_for_same_data=["/api/sensor"],
        )
        score_http = calculator.calculate_node_confidence(context_http)

        # HTTP overlap should lower confidence (suggests ROS2 topic may be unused)
        assert score_http.score < score_normal.score

    def test_recently_modified_boosts_confidence(self):
        """Recently modified code should have higher confidence."""
        calculator = ConfidenceCalculator()

        # Old code
        context_old = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
            last_git_commit_days_ago=200,  # 6+ months old
        )
        score_old = calculator.calculate_node_confidence(context_old)

        # Recently modified
        context_recent = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
            last_git_commit_days_ago=10,  # Recent
        )
        score_recent = calculator.calculate_node_confidence(context_recent)

        # Recent should be higher confidence
        assert score_recent.score > score_old.score

    def test_custom_weights(self):
        """Test using custom weights for factors."""
        custom_weights = {
            "in_launch_file": 0.5,  # Increased from 0.35
        }
        calculator = ConfidenceCalculator(custom_weights=custom_weights)

        context = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            in_launch_files=["robot.launch.py"],
        )

        score = calculator.calculate_node_confidence(context)

        # Check that custom weight was applied
        launch_factor = next(
            (f for f in score.factors if f.name == "in_launch_file"),
            None
        )
        assert launch_factor is not None
        assert launch_factor.impact == 0.5

    def test_score_clamped_to_valid_range(self):
        """Score should always be between 0.0 and 1.0."""
        calculator = ConfidenceCalculator()

        # Very negative scenario
        context_bad = NodeConfidenceContext(
            node_name="old_controller",
            file_path=Path("/archive/old_controller_deprecated.py"),
            dead_code_indicators=[
                "directory:archive",
                "filename:_deprecated.py",
            ],
            location_confidence=0.2,
            http_endpoints_for_same_data=["/api/control"],
            last_git_commit_days_ago=365,
        )
        score_bad = calculator.calculate_node_confidence(context_bad)
        assert score_bad.score >= 0.0

        # Very positive scenario
        context_good = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("/src/motor_controller.py"),
            package_name="betaray_control",
            in_launch_files=["robot.launch.py", "main.launch.py"],
            in_systemd_services=["betaray-navigation.service"],
            topics_with_both_pubsub=["/cmd_vel", "/odom"],
            last_git_commit_days_ago=5,
        )
        score_good = calculator.calculate_node_confidence(context_good)
        assert score_good.score <= 1.0


class TestTopicConfidence:
    """Test topic confidence calculation."""

    def test_topic_with_both_pub_and_sub(self):
        """Topic with both publishers and subscribers should have higher confidence."""
        calculator = ConfidenceCalculator()

        # Connected topic
        score_connected = calculator.calculate_topic_confidence(
            topic_name="/cmd_vel",
            publishers=["motor_controller"],
            subscribers=["navigation_node"],
            publisher_confidences={"motor_controller": 0.8},
            subscriber_confidences={"navigation_node": 0.8},
        )

        # Orphaned topic (publisher only)
        score_orphaned = calculator.calculate_topic_confidence(
            topic_name="/unused_data",
            publishers=["sensor_node"],
            subscribers=[],
            publisher_confidences={"sensor_node": 0.8},
            subscriber_confidences={},
        )

        assert score_connected.score > score_orphaned.score

    def test_topic_with_no_connections(self):
        """Topic with no connections should have very low confidence."""
        calculator = ConfidenceCalculator()

        score = calculator.calculate_topic_confidence(
            topic_name="/phantom_topic",
            publishers=[],
            subscribers=[],
            publisher_confidences={},
            subscriber_confidences={},
        )

        assert score.score < 0.3
        assert score.level == ConfidenceLevel.VERY_LOW

    def test_topic_confidence_inherits_from_nodes(self):
        """Topic confidence should be influenced by connected nodes' confidence."""
        calculator = ConfidenceCalculator()

        # Topic with high-confidence nodes
        score_high = calculator.calculate_topic_confidence(
            topic_name="/cmd_vel",
            publishers=["motor_controller"],
            subscribers=["navigation_node"],
            publisher_confidences={"motor_controller": 0.9},
            subscriber_confidences={"navigation_node": 0.85},
        )

        # Topic with low-confidence nodes
        score_low = calculator.calculate_topic_confidence(
            topic_name="/cmd_vel",
            publishers=["old_controller"],
            subscribers=["archived_node"],
            publisher_confidences={"old_controller": 0.2},
            subscriber_confidences={"archived_node": 0.25},
        )

        assert score_high.score > score_low.score


class TestFilterByConfidence:
    """Test filtering items by confidence threshold."""

    def test_filter_by_min_confidence(self):
        """Test filtering items by minimum confidence."""
        calculator = ConfidenceCalculator(min_confidence=0.5)

        items = [
            {"name": "high", "confidence": 0.9},
            {"name": "medium", "confidence": 0.6},
            {"name": "low", "confidence": 0.3},
        ]

        filtered = calculator.filter_by_confidence(
            items,
            score_getter=lambda x: x["confidence"],
        )

        assert len(filtered) == 2
        assert filtered[0]["name"] == "high"
        assert filtered[1]["name"] == "medium"

    def test_filter_with_override_threshold(self):
        """Test filtering with override threshold."""
        calculator = ConfidenceCalculator(min_confidence=0.5)

        items = [
            {"name": "high", "confidence": 0.9},
            {"name": "medium", "confidence": 0.6},
            {"name": "low", "confidence": 0.3},
        ]

        # Override to be more strict
        filtered = calculator.filter_by_confidence(
            items,
            score_getter=lambda x: x["confidence"],
            min_confidence=0.7,
        )

        assert len(filtered) == 1
        assert filtered[0]["name"] == "high"


class TestConfidenceSummary:
    """Test confidence summary generation."""

    def test_summary_empty_list(self):
        """Test summary with empty list."""
        summary = get_confidence_summary([])

        assert summary["total"] == 0
        assert summary["average"] == 0.0
        assert summary["by_level"] == {}

    def test_summary_with_scores(self):
        """Test summary with various scores."""
        scores = [
            ConfidenceScore(score=0.95, level=ConfidenceLevel.VERY_HIGH),
            ConfidenceScore(score=0.8, level=ConfidenceLevel.HIGH),
            ConfidenceScore(score=0.6, level=ConfidenceLevel.MEDIUM),
            ConfidenceScore(score=0.4, level=ConfidenceLevel.LOW),
            ConfidenceScore(score=0.2, level=ConfidenceLevel.VERY_LOW),
        ]

        summary = get_confidence_summary(scores)

        assert summary["total"] == 5
        assert summary["average"] == pytest.approx(0.59, abs=0.01)
        assert summary["by_level"]["very_high"] == 1
        assert summary["by_level"]["high"] == 1
        assert summary["by_level"]["medium"] == 1
        assert summary["by_level"]["low"] == 1
        assert summary["by_level"]["very_low"] == 1
        assert summary["above_0.7"] == 2
        assert summary["below_0.3"] == 1


class TestConfidenceFactorsWeights:
    """Test that confidence factor weights are reasonable."""

    def test_positive_factors_are_positive(self):
        """Positive factors should have positive weights."""
        positive_factors = [
            "in_launch_file",
            "in_systemd_service",
            "has_matching_pubsub",
            "in_ros2_package",
            "recently_modified",
        ]

        for factor in positive_factors:
            assert CONFIDENCE_FACTORS[factor] > 0, f"{factor} should be positive"

    def test_negative_factors_are_negative(self):
        """Negative factors should have negative weights."""
        negative_factors = [
            "in_archive_dir",
            "deprecated_filename",
            "no_launch_reference",
            "no_imports",
            "http_overlap",
            "stale_code",
            "in_test_dir",
        ]

        for factor in negative_factors:
            assert CONFIDENCE_FACTORS[factor] < 0, f"{factor} should be negative"

    def test_systemd_is_strongest_positive(self):
        """Systemd service should be strongest positive indicator."""
        positive_weights = [
            v for k, v in CONFIDENCE_FACTORS.items() if v > 0
        ]
        assert CONFIDENCE_FACTORS["in_systemd_service"] == max(positive_weights)

    def test_archive_is_strongest_negative(self):
        """Archive directory should be among strongest negative indicators."""
        assert CONFIDENCE_FACTORS["in_archive_dir"] <= -0.35
