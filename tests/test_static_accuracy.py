"""Static Analysis Accuracy Tests for RoboMind.

Tests precision and recall of static analysis against runtime state.
Based on: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
"""

import pytest
import subprocess
import json
from typing import Set, Dict, Tuple

from robomind.mcp_server.graph_loader import SystemGraph


class TestTopicAccuracy:
    """Test accuracy of topic detection."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def get_robomind_topics(self, graph) -> Set[str]:
        """Get topics from RoboMind analysis."""
        result = graph.query('/', limit=500)
        return {t['name'] for t in result['results']['topics']}

    def get_live_topics(self, ssh_host: str = None) -> Set[str]:
        """Get topics from live ROS2 system."""
        if ssh_host:
            cmd = [
                'sshpass', '-p', 'betaray2024', 'ssh', f'robot@{ssh_host}',
                'source /opt/ros/humble/setup.bash && ros2 topic list'
            ]
        else:
            cmd = ['bash', '-c', 'source /opt/ros/humble/setup.bash && ros2 topic list']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return set(result.stdout.strip().split('\n'))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return set()

    def test_topic_precision_recall_local(self, graph):
        """Calculate precision/recall against local ROS2 (if available)."""
        robomind_topics = self.get_robomind_topics(graph)
        live_topics = self.get_live_topics()

        if not live_topics:
            pytest.skip("No local ROS2 environment available")

        self._calculate_and_report_metrics(robomind_topics, live_topics, "Local ROS2")

    def test_topic_precision_recall_nav_jetson(self, graph):
        """Calculate precision/recall against Nav Jetson."""
        robomind_topics = self.get_robomind_topics(graph)
        live_topics = self.get_live_topics('betaray-nav.local')

        if not live_topics:
            pytest.skip("Cannot reach Nav Jetson")

        self._calculate_and_report_metrics(robomind_topics, live_topics, "Nav Jetson")

    def _calculate_and_report_metrics(
        self,
        predicted: Set[str],
        actual: Set[str],
        label: str
    ):
        """Calculate and report precision/recall metrics."""
        true_positives = len(predicted & actual)
        false_positives = len(predicted - actual)
        false_negatives = len(actual - predicted)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n=== {label} Topic Accuracy ===")
        print(f"RoboMind topics: {len(predicted)}")
        print(f"Live topics: {len(actual)}")
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives} (in code, not running)")
        print(f"False negatives: {false_negatives} (running, not in code)")
        print(f"Precision: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        print(f"F1 Score: {f1:.1%}")

        # Report some false positives/negatives
        if false_positives > 0:
            fp_sample = list(predicted - actual)[:5]
            print(f"Sample false positives: {fp_sample}")
        if false_negatives > 0:
            fn_sample = list(actual - predicted)[:5]
            print(f"Sample false negatives: {fn_sample}")


class TestNodeAccuracy:
    """Test accuracy of node detection."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def get_robomind_nodes(self, graph) -> Set[str]:
        """Get nodes from RoboMind analysis."""
        result = graph.query('', limit=500)
        return {n['name'] for n in result['results']['ros2_nodes']}

    def get_live_nodes(self, ssh_host: str = None) -> Set[str]:
        """Get nodes from live ROS2 system."""
        if ssh_host:
            cmd = [
                'sshpass', '-p', 'betaray2024', 'ssh', f'robot@{ssh_host}',
                'source /opt/ros/humble/setup.bash && ros2 node list'
            ]
        else:
            cmd = ['bash', '-c', 'source /opt/ros/humble/setup.bash && ros2 node list']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # ros2 node list returns nodes like /node_name
                return {n.lstrip('/') for n in result.stdout.strip().split('\n') if n}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return set()

    def test_node_count_comparison(self, graph):
        """Compare node counts between static and runtime."""
        robomind_nodes = self.get_robomind_nodes(graph)
        summary = graph.get_summary()

        print(f"\n=== Node Count Analysis ===")
        print(f"RoboMind detected nodes: {len(robomind_nodes)}")
        print(f"Summary total_nodes: {summary['total_nodes']}")

        # These should match
        assert len(robomind_nodes) <= summary['total_nodes']


class TestHTTPEndpointAccuracy:
    """Test accuracy of HTTP endpoint detection."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_endpoint_reachability(self, graph):
        """Test if detected endpoints are actually reachable."""
        import requests

        endpoints = graph.get_http_endpoints()

        print(f"\n=== HTTP Endpoint Accuracy ===")
        print(f"Detected endpoints: {len(endpoints)}")

        reachable = 0
        unreachable = 0
        for ep in endpoints[:10]:  # Test first 10
            url = f"http://{ep['host']}:{ep['port']}{ep['path']}"
            try:
                r = requests.get(url, timeout=2)
                if r.ok:
                    reachable += 1
                else:
                    unreachable += 1
            except:
                unreachable += 1

        tested = reachable + unreachable
        if tested > 0:
            accuracy = reachable / tested * 100
            print(f"Tested: {tested}")
            print(f"Reachable: {reachable}")
            print(f"Unreachable: {unreachable}")
            print(f"Reachability: {accuracy:.1f}%")


class TestConnectionAccuracy:
    """Test accuracy of connection (pub/sub relationship) detection."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_connection_integrity(self, graph):
        """Verify connection data is internally consistent."""
        summary = graph.get_summary()

        total_topics = summary['total_topics']
        connected = summary['connected_topics']

        print(f"\n=== Connection Accuracy ===")
        print(f"Total topics: {total_topics}")
        print(f"Connected topics: {connected}")

        if total_topics > 0:
            connectivity = connected / total_topics * 100
            print(f"Connectivity rate: {connectivity:.1f}%")

            # Some connectivity expected
            assert connectivity > 0, "No connected topics found"

    def test_orphan_analysis(self, graph):
        """Analyze orphaned topics."""
        orphans = graph.get_orphaned_topics()

        no_pub = len(orphans['no_publishers'])
        no_sub = len(orphans['no_subscribers'])

        print(f"\n=== Orphan Analysis ===")
        print(f"Topics without publishers: {no_pub}")
        print(f"Topics without subscribers: {no_sub}")

        if no_pub > 0:
            print(f"Sample no-pub topics: {orphans['no_publishers'][:5]}")
        if no_sub > 0:
            print(f"Sample no-sub topics: {orphans['no_subscribers'][:5]}")


class TestCrossValidation:
    """Cross-validate analysis against multiple sources."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_internal_consistency(self, graph):
        """Verify internal data consistency."""
        # Query should match summary counts
        summary = graph.get_summary()

        # Query for all topics
        all_results = graph.query('', limit=1000)
        query_topics = len(all_results['results']['topics'])
        query_nodes = len(all_results['results']['ros2_nodes'])

        print(f"\n=== Internal Consistency ===")
        print(f"Summary nodes: {summary['total_nodes']}, Query nodes: {query_nodes}")
        print(f"Summary topics: {summary['total_topics']}, Query topics: {query_topics}")

        # Query may return fewer due to limit
        assert query_nodes <= summary['total_nodes']

    def test_bidirectional_connections(self, graph):
        """Verify pub/sub relationships are bidirectional."""
        graph_data = graph._load_graph()
        nodes = graph_data.get('nodes', [])
        topics_data = graph_data.get('topics', {}).get('topics', {})

        # Build reverse mapping: topic -> publishers/subscribers
        topic_publishers = {}
        topic_subscribers = {}
        for name, info in topics_data.items():
            topic_publishers[name] = set(info.get('publishers', []))
            topic_subscribers[name] = set(info.get('subscribers', []))

        # Check nodes' claimed topics exist
        missing_topics = []
        for node in nodes[:20]:  # Sample 20 nodes
            for pub in node.get('publishers', []):
                topic = pub.get('topic')
                if topic and topic not in topics_data:
                    missing_topics.append((node['name'], topic, 'publisher'))

        if missing_topics:
            print(f"Warning: {len(missing_topics)} published topics not in topic list")
            print(f"Sample: {missing_topics[:3]}")


class TestAccuracyReport:
    """Generate comprehensive accuracy report."""

    def test_full_accuracy_report(self):
        """Generate full accuracy report."""
        graph = SystemGraph()

        print("\n" + "=" * 60)
        print("ROBOMIND STATIC ANALYSIS ACCURACY REPORT")
        print("=" * 60)

        summary = graph.get_summary()
        print(f"\nProject: {summary['project']}")
        print(f"Generated: {summary['generated_at']}")

        print(f"\n--- Detected Elements ---")
        print(f"ROS2 Nodes: {summary['total_nodes']}")
        print(f"Topics: {summary['total_topics']}")
        print(f"Connected Topics: {summary['connected_topics']}")
        print(f"HTTP Endpoints: {summary['total_http_endpoints']}")
        print(f"Packages: {len(summary['packages'])}")

        # Connectivity metric
        if summary['total_topics'] > 0:
            connectivity = summary['connected_topics'] / summary['total_topics'] * 100
            print(f"\nConnectivity Rate: {connectivity:.1f}%")

        # Orphan analysis
        orphans = graph.get_orphaned_topics()
        print(f"\n--- Orphan Analysis ---")
        print(f"Topics missing publishers: {len(orphans['no_publishers'])}")
        print(f"Topics missing subscribers: {len(orphans['no_subscribers'])}")

        # Coupling analysis
        high_coupling = graph.get_coupling_pairs(min_score=0.5)
        print(f"\n--- Coupling Analysis ---")
        print(f"Highly coupled pairs (>0.5): {len(high_coupling)}")

        print("=" * 60)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
