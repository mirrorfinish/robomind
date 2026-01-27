"""
RoboMind Confidence Scoring - Calculate confidence levels for findings.

Confidence scoring helps distinguish real issues from false positives by
considering multiple factors:
- File location (archive, backup directories = lower confidence)
- Launch file references (in launch file = higher confidence)
- Publisher/subscriber matching
- HTTP endpoint overlap (if HTTP exists for same data, ROS2 may be dead)
- Git commit age (optional)

Confidence scores range from 0.0 to 1.0:
- 0.9-1.0: Very high confidence - almost certainly real
- 0.7-0.9: High confidence - likely real
- 0.5-0.7: Medium confidence - needs verification
- 0.3-0.5: Low confidence - possibly false positive
- 0.0-0.3: Very low confidence - likely false positive
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"            # 0.7-0.9
    MEDIUM = "medium"        # 0.5-0.7
    LOW = "low"              # 0.3-0.5
    VERY_LOW = "very_low"    # 0.0-0.3

    @staticmethod
    def from_score(score: float) -> "ConfidenceLevel":
        """Get confidence level from numeric score."""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


# Weight factors for confidence calculation
CONFIDENCE_FACTORS = {
    # Positive factors (increase confidence)
    "in_launch_file": 0.35,           # Node referenced in a launch file
    "in_systemd_service": 0.40,       # Referenced in systemd service
    "has_matching_pubsub": 0.25,      # Has both publisher and subscriber
    "in_ros2_package": 0.15,          # In a proper ROS2 package
    "recently_modified": 0.10,        # Modified within last 30 days

    # Negative factors (decrease confidence)
    "in_archive_dir": -0.40,          # In archive/backup/old directory
    "deprecated_filename": -0.35,     # Has _old, _deprecated suffix
    "no_launch_reference": -0.20,     # No launch file references this node
    "no_imports": -0.25,              # No other file imports this module
    "http_overlap": -0.30,            # HTTP endpoint exists for same data
    "stale_code": -0.15,              # No git commit in 6+ months
    "in_test_dir": -0.20,             # In test/tests directory
}


@dataclass
class ConfidenceFactor:
    """A single factor contributing to confidence score."""
    name: str
    impact: float  # Positive or negative
    reason: str
    evidence: Optional[str] = None


@dataclass
class ConfidenceScore:
    """Confidence score for a finding or node."""
    score: float  # 0.0 to 1.0
    level: ConfidenceLevel
    factors: List[ConfidenceFactor] = field(default_factory=list)
    base_score: float = 0.5  # Starting point before factors

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "score": round(self.score, 3),
            "level": self.level.value,
            "factors": [
                {
                    "name": f.name,
                    "impact": round(f.impact, 3),
                    "reason": f.reason,
                    "evidence": f.evidence,
                }
                for f in self.factors
            ],
        }


@dataclass
class NodeConfidenceContext:
    """Context information for calculating node confidence."""
    node_name: str
    file_path: Path
    package_name: Optional[str] = None

    # Location factors
    location_confidence: float = 1.0
    dead_code_indicators: List[str] = field(default_factory=list)

    # Reference factors
    in_launch_files: List[str] = field(default_factory=list)
    in_systemd_services: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)

    # Connection factors
    has_publishers: bool = False
    has_subscribers: bool = False
    publisher_topics: List[str] = field(default_factory=list)
    subscriber_topics: List[str] = field(default_factory=list)
    topics_with_both_pubsub: List[str] = field(default_factory=list)

    # HTTP overlap
    http_endpoints_for_same_data: List[str] = field(default_factory=list)

    # Metadata
    last_modified_days_ago: Optional[int] = None
    last_git_commit_days_ago: Optional[int] = None


class ConfidenceCalculator:
    """
    Calculate confidence scores for ROS2 nodes and findings.

    Usage:
        calculator = ConfidenceCalculator()
        context = NodeConfidenceContext(
            node_name="motor_controller",
            file_path=Path("src/motor_controller.py"),
            location_confidence=1.0,
            in_launch_files=["robot.launch.py"],
        )
        score = calculator.calculate_node_confidence(context)
        print(f"Confidence: {score.score} ({score.level.value})")
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        custom_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize confidence calculator.

        Args:
            min_confidence: Minimum confidence threshold for filtering
            custom_weights: Override default factor weights
        """
        self.min_confidence = min_confidence
        self.weights = CONFIDENCE_FACTORS.copy()
        if custom_weights:
            self.weights.update(custom_weights)

    def calculate_node_confidence(
        self,
        context: NodeConfidenceContext,
    ) -> ConfidenceScore:
        """
        Calculate confidence score for a ROS2 node.

        Args:
            context: NodeConfidenceContext with all relevant information

        Returns:
            ConfidenceScore with score, level, and contributing factors
        """
        factors = []
        base_score = 0.5  # Start at neutral

        # === Location-based factors ===

        # Check for archive/backup directories
        if context.dead_code_indicators:
            for indicator in context.dead_code_indicators:
                if "archive" in indicator or "backup" in indicator:
                    factors.append(ConfidenceFactor(
                        name="in_archive_dir",
                        impact=self.weights["in_archive_dir"],
                        reason=f"File is in archive/backup directory",
                        evidence=indicator,
                    ))
                elif "deprecated" in indicator or "_old" in indicator:
                    factors.append(ConfidenceFactor(
                        name="deprecated_filename",
                        impact=self.weights["deprecated_filename"],
                        reason="File has deprecated naming pattern",
                        evidence=indicator,
                    ))
                elif "test" in indicator:
                    factors.append(ConfidenceFactor(
                        name="in_test_dir",
                        impact=self.weights["in_test_dir"],
                        reason="File is in test directory",
                        evidence=indicator,
                    ))

        # Use location_confidence from scanner
        if context.location_confidence < 1.0:
            location_penalty = context.location_confidence - 1.0
            if location_penalty < -0.1:  # Only add if significant
                factors.append(ConfidenceFactor(
                    name="location_penalty",
                    impact=location_penalty,
                    reason="File location suggests non-production code",
                    evidence=str(context.file_path),
                ))

        # === Reference factors ===

        # Launch file reference (strong positive)
        if context.in_launch_files:
            factors.append(ConfidenceFactor(
                name="in_launch_file",
                impact=self.weights["in_launch_file"],
                reason=f"Node referenced in {len(context.in_launch_files)} launch file(s)",
                evidence=", ".join(context.in_launch_files[:3]),
            ))
        else:
            factors.append(ConfidenceFactor(
                name="no_launch_reference",
                impact=self.weights["no_launch_reference"],
                reason="Node not found in any launch file",
            ))

        # Systemd service reference (very strong positive)
        if context.in_systemd_services:
            factors.append(ConfidenceFactor(
                name="in_systemd_service",
                impact=self.weights["in_systemd_service"],
                reason=f"Referenced in systemd service",
                evidence=", ".join(context.in_systemd_services),
            ))

        # Import references
        if not context.imported_by and not context.in_launch_files:
            factors.append(ConfidenceFactor(
                name="no_imports",
                impact=self.weights["no_imports"],
                reason="No other files import this module",
            ))

        # === Connection factors ===

        # Has matching pub/sub (good sign of active use)
        if context.topics_with_both_pubsub:
            factors.append(ConfidenceFactor(
                name="has_matching_pubsub",
                impact=self.weights["has_matching_pubsub"],
                reason=f"Topics have both publisher and subscriber",
                evidence=f"{len(context.topics_with_both_pubsub)} connected topics",
            ))

        # In a ROS2 package
        if context.package_name:
            factors.append(ConfidenceFactor(
                name="in_ros2_package",
                impact=self.weights["in_ros2_package"],
                reason=f"Part of ROS2 package",
                evidence=context.package_name,
            ))

        # === HTTP overlap ===

        if context.http_endpoints_for_same_data:
            factors.append(ConfidenceFactor(
                name="http_overlap",
                impact=self.weights["http_overlap"],
                reason="HTTP endpoint exists for similar data (ROS2 may be unused)",
                evidence=", ".join(context.http_endpoints_for_same_data[:3]),
            ))

        # === Temporal factors ===

        if context.last_git_commit_days_ago is not None:
            if context.last_git_commit_days_ago > 180:  # 6 months
                factors.append(ConfidenceFactor(
                    name="stale_code",
                    impact=self.weights["stale_code"],
                    reason=f"No git commits in {context.last_git_commit_days_ago} days",
                ))
            elif context.last_git_commit_days_ago < 30:
                factors.append(ConfidenceFactor(
                    name="recently_modified",
                    impact=self.weights["recently_modified"],
                    reason="Recently modified in git",
                ))

        # Calculate final score
        total_impact = sum(f.impact for f in factors)
        final_score = max(0.0, min(1.0, base_score + total_impact))

        return ConfidenceScore(
            score=final_score,
            level=ConfidenceLevel.from_score(final_score),
            factors=factors,
            base_score=base_score,
        )

    def calculate_topic_confidence(
        self,
        topic_name: str,
        publishers: List[str],
        subscribers: List[str],
        publisher_confidences: Dict[str, float],
        subscriber_confidences: Dict[str, float],
    ) -> ConfidenceScore:
        """
        Calculate confidence for a topic finding.

        The topic's confidence is influenced by the confidence of its
        publishers and subscribers.
        """
        factors = []
        base_score = 0.5

        # If no publishers or subscribers, this is likely a false finding
        if not publishers and not subscribers:
            return ConfidenceScore(
                score=0.1,
                level=ConfidenceLevel.VERY_LOW,
                factors=[ConfidenceFactor(
                    name="no_connections",
                    impact=-0.4,
                    reason="Topic has no publishers or subscribers",
                )],
            )

        # Calculate average confidence from connected nodes
        all_confidences = []
        for pub in publishers:
            if pub in publisher_confidences:
                all_confidences.append(publisher_confidences[pub])
        for sub in subscribers:
            if sub in subscriber_confidences:
                all_confidences.append(subscriber_confidences[sub])

        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            if avg_confidence < 0.5:
                factors.append(ConfidenceFactor(
                    name="low_node_confidence",
                    impact=avg_confidence - 0.5,
                    reason="Connected nodes have low confidence",
                    evidence=f"Average: {avg_confidence:.2f}",
                ))
            else:
                factors.append(ConfidenceFactor(
                    name="high_node_confidence",
                    impact=(avg_confidence - 0.5) * 0.5,  # Partial boost
                    reason="Connected nodes have good confidence",
                    evidence=f"Average: {avg_confidence:.2f}",
                ))

        # Connected topics (both pub and sub) get a boost
        if publishers and subscribers:
            factors.append(ConfidenceFactor(
                name="topic_connected",
                impact=0.2,
                reason="Topic has both publishers and subscribers",
            ))
        else:
            factors.append(ConfidenceFactor(
                name="topic_orphaned",
                impact=-0.15,
                reason="Topic is orphaned (missing publisher or subscriber)",
            ))

        total_impact = sum(f.impact for f in factors)
        final_score = max(0.0, min(1.0, base_score + total_impact))

        return ConfidenceScore(
            score=final_score,
            level=ConfidenceLevel.from_score(final_score),
            factors=factors,
            base_score=base_score,
        )

    def filter_by_confidence(
        self,
        items: List[Any],
        score_getter,
        min_confidence: Optional[float] = None,
    ) -> List[Any]:
        """
        Filter items by minimum confidence threshold.

        Args:
            items: List of items to filter
            score_getter: Function to get confidence score from item
            min_confidence: Override instance min_confidence

        Returns:
            Filtered list of items above threshold
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        return [item for item in items if score_getter(item) >= threshold]


def get_confidence_summary(scores: List[ConfidenceScore]) -> Dict:
    """Generate summary statistics for a list of confidence scores."""
    if not scores:
        return {
            "total": 0,
            "by_level": {},
            "average": 0.0,
        }

    by_level = {}
    for level in ConfidenceLevel:
        by_level[level.value] = sum(1 for s in scores if s.level == level)

    return {
        "total": len(scores),
        "by_level": by_level,
        "average": round(sum(s.score for s in scores) / len(scores), 3),
        "above_0.7": sum(1 for s in scores if s.score >= 0.7),
        "below_0.3": sum(1 for s in scores if s.score < 0.3),
    }
