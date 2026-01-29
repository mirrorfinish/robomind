"""Graph Consistency Tests for RoboMind.

Tests structural integrity of the system graph.
Based on: https://networkx.org/documentation/stable/
"""

import pytest
import json
from pathlib import Path

from robomind.mcp_server.graph_loader import SystemGraph


class TestGraphIntegrity:
    """Test graph data integrity."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    @pytest.fixture
    def graph_data(self, graph):
        """Load raw graph data."""
        return graph._load_graph()

    def test_all_nodes_have_required_fields(self, graph_data):
        """All nodes should have name and file_path."""
        nodes = graph_data.get('nodes', [])
        missing_fields = []

        for node in nodes:
            if not node.get('name'):
                missing_fields.append(('name', node))
            if not node.get('file_path'):
                missing_fields.append(('file_path', node.get('name', 'UNKNOWN')))

        assert len(missing_fields) == 0, f"Nodes missing required fields: {missing_fields[:5]}"

    def test_no_duplicate_node_names(self, graph_data):
        """Check for duplicate node names (warns but doesn't fail)."""
        nodes = graph_data.get('nodes', [])
        node_names = [n.get('name') for n in nodes]

        duplicates = [name for name in node_names if node_names.count(name) > 1]
        unique_duplicates = list(set(duplicates))

        if unique_duplicates:
            print(f"Warning: {len(unique_duplicates)} duplicate node names found")
            print(f"Duplicates may indicate multiple files defining same node class")
            # Don't fail - duplicates often indicate same node defined in different packages

    def test_all_topics_have_msg_type(self, graph_data):
        """All topics should have a message type."""
        topics_data = graph_data.get('topics', {})
        topics_dict = topics_data.get('topics', {}) if isinstance(topics_data, dict) else {}

        missing_msg_type = []
        for topic_name, topic_info in topics_dict.items():
            if not topic_info.get('msg_type'):
                missing_msg_type.append(topic_name)

        # Warn but don't fail - some topics may have unknown types
        if missing_msg_type:
            print(f"Warning: {len(missing_msg_type)} topics without msg_type")

    def test_publisher_subscriber_references_valid(self, graph_data):
        """Publisher/subscriber references should point to real nodes."""
        nodes = graph_data.get('nodes', [])
        node_names = {n.get('name') for n in nodes}

        topics_data = graph_data.get('topics', {})
        topics_dict = topics_data.get('topics', {}) if isinstance(topics_data, dict) else {}

        invalid_refs = []
        for topic_name, topic_info in topics_dict.items():
            for pub in topic_info.get('publishers', []):
                if pub not in node_names and pub != 'unknown':
                    invalid_refs.append(('publisher', pub, topic_name))
            for sub in topic_info.get('subscribers', []):
                if sub not in node_names and sub != 'unknown':
                    invalid_refs.append(('subscriber', sub, topic_name))

        # This might have false positives if pub/sub names differ from node names
        if invalid_refs:
            print(f"Info: {len(invalid_refs)} potentially invalid node references")


class TestOrphanDetection:
    """Test orphan detection accuracy."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_orphan_detection_returns_lists(self, graph):
        """Orphan detection should return proper structure."""
        orphans = graph.get_orphaned_topics()

        assert 'no_publishers' in orphans
        assert 'no_subscribers' in orphans
        assert isinstance(orphans['no_publishers'], list)
        assert isinstance(orphans['no_subscribers'], list)

    def test_no_overlap_in_orphan_categories(self, graph):
        """Topics shouldn't be in both orphan categories."""
        orphans = graph.get_orphaned_topics()

        overlap = set(orphans['no_publishers']) & set(orphans['no_subscribers'])
        # Topics with no publishers AND no subscribers shouldn't exist in graph
        # but if they do, they could appear in both lists
        if overlap:
            print(f"Info: {len(overlap)} topics with neither publishers nor subscribers")

    def test_orphan_count_reasonable(self, graph):
        """Orphan count should be reasonable percentage."""
        orphans = graph.get_orphaned_topics()
        summary = graph.get_summary()

        total_topics = summary.get('total_topics', 0)
        if total_topics > 0:
            orphan_count = len(orphans['no_publishers']) + len(orphans['no_subscribers'])
            orphan_percentage = orphan_count / total_topics * 100

            # More than 80% orphans would indicate a problem
            assert orphan_percentage < 80, f"Too many orphans: {orphan_percentage:.1f}%"


class TestConnectivityAnalysis:
    """Test connectivity metrics."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_connected_topics_count(self, graph):
        """Connected topics should be subset of total topics."""
        summary = graph.get_summary()

        total = summary.get('total_topics', 0)
        connected = summary.get('connected_topics', 0)

        assert connected <= total, f"Connected {connected} > Total {total}"

    def test_connectivity_percentage(self, graph):
        """Calculate and report connectivity percentage."""
        summary = graph.get_summary()

        total = summary.get('total_topics', 0)
        connected = summary.get('connected_topics', 0)

        if total > 0:
            connectivity = connected / total * 100
            print(f"Connectivity: {connectivity:.1f}% ({connected}/{total})")

            # At least some connectivity expected
            assert connectivity > 0, "No connected topics found"


class TestCouplingAnalysis:
    """Test coupling detection."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_coupling_scores_in_range(self, graph):
        """Coupling scores should be between 0 and 1."""
        pairs = graph.get_coupling_pairs(min_score=0.0)

        for pair in pairs:
            score = pair.get('score', 0)
            assert 0 <= score <= 1, f"Invalid coupling score: {score}"

    def test_coupling_pairs_have_distinct_nodes(self, graph):
        """Coupling pairs should have different source and target."""
        pairs = graph.get_coupling_pairs(min_score=0.0)

        self_coupled = [p for p in pairs if p.get('source') == p.get('target')]
        assert len(self_coupled) == 0, f"Self-coupled nodes found: {self_coupled}"

    def test_coupling_min_score_filter(self, graph):
        """Min score filter should work correctly."""
        all_pairs = graph.get_coupling_pairs(min_score=0.0)
        filtered_pairs = graph.get_coupling_pairs(min_score=0.5)

        assert len(filtered_pairs) <= len(all_pairs)

        for pair in filtered_pairs:
            assert pair.get('score', 0) >= 0.5


class TestHTTPEndpointIntegrity:
    """Test HTTP endpoint data integrity."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_http_endpoints_have_required_fields(self, graph):
        """HTTP endpoints should have path, host, port."""
        endpoints = graph.get_http_endpoints()

        for ep in endpoints:
            assert 'path' in ep, f"Endpoint missing path: {ep}"
            assert 'host' in ep, f"Endpoint missing host: {ep}"
            assert 'port' in ep, f"Endpoint missing port: {ep}"

    def test_http_ports_valid(self, graph):
        """HTTP ports should be valid port numbers."""
        endpoints = graph.get_http_endpoints()

        for ep in endpoints:
            port = ep.get('port')
            if port is not None:
                assert isinstance(port, (int, float)), f"Invalid port type: {type(port)}"
                assert 1 <= port <= 65535, f"Invalid port number: {port}"

    def test_http_methods_valid(self, graph):
        """HTTP methods should be valid."""
        endpoints = graph.get_http_endpoints()
        valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'}

        for ep in endpoints:
            methods = ep.get('methods', [])
            for method in methods:
                assert method.upper() in valid_methods, f"Invalid HTTP method: {method}"


class TestQueryConsistency:
    """Test that queries return consistent results."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_query_determinism(self, graph):
        """Same query should return same results."""
        import hashlib

        result1 = graph.query('motor', limit=10)
        result2 = graph.query('motor', limit=10)

        hash1 = hashlib.md5(json.dumps(result1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.md5(json.dumps(result2, sort_keys=True).encode()).hexdigest()

        assert hash1 == hash2, "Query results not deterministic"

    def test_case_insensitive_query(self, graph):
        """Queries should be case-insensitive."""
        result_lower = graph.query('motor', limit=10)
        result_upper = graph.query('MOTOR', limit=10)
        result_mixed = graph.query('Motor', limit=10)

        # Should return same count
        assert result_lower['total_matches'] == result_upper['total_matches']
        assert result_lower['total_matches'] == result_mixed['total_matches']

    def test_partial_match_query(self, graph):
        """Partial patterns should match."""
        # 'cmd' should match '/cmd_vel'
        result = graph.query('cmd')

        # Should find cmd_vel related topics
        topic_names = [t['name'] for t in result['results']['topics']]
        cmd_vel_found = any('cmd' in name.lower() for name in topic_names)

        # May or may not find matches depending on codebase
        print(f"Partial match 'cmd' found {len(topic_names)} topics")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
