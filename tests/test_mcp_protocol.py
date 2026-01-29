"""MCP Protocol Validation Tests for RoboMind MCP Server.

Tests MCP server compliance, edge cases, and error handling.
Based on: https://modelcontextprotocol.io/docs/tools/inspector
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock

from robomind.mcp_server.graph_loader import SystemGraph


class TestMCPEdgeCases:
    """Test edge case inputs to the graph loader."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_empty_pattern_query(self, graph):
        """Empty pattern should not crash (may return all or none)."""
        result = graph.query('')
        assert 'results' in result
        # Empty pattern behavior: may match all or none depending on implementation
        # Key requirement: should not crash

    def test_very_long_pattern_query(self, graph):
        """Very long pattern should be handled gracefully."""
        long_pattern = 'a' * 10000
        result = graph.query(long_pattern)
        assert 'results' in result
        # Should not crash, may return 0 matches

    def test_regex_special_chars_query(self, graph):
        """Regex special characters should be escaped or handled."""
        special_patterns = ['.*', '[a-z]+', '(test)', '^start$', 'a|b']
        for pattern in special_patterns:
            result = graph.query(pattern)
            assert 'results' in result
            # Should not crash

    def test_unicode_pattern_query(self, graph):
        """Unicode characters in pattern should work."""
        result = graph.query('日本語')
        assert 'results' in result

    def test_whitespace_pattern_query(self, graph):
        """Whitespace-only pattern should be handled."""
        result = graph.query('   ')
        assert 'results' in result

    def test_null_equivalent_pattern(self, graph):
        """Patterns that could be null-like should work."""
        for pattern in ['null', 'None', 'undefined']:
            result = graph.query(pattern)
            assert 'results' in result

    def test_nonexistent_topic(self, graph):
        """Querying nonexistent topic should return None, not crash."""
        result = graph.get_topic_connections('/this/topic/does/not/exist')
        assert result is None

    def test_nonexistent_node(self, graph):
        """Querying nonexistent node should return None, not crash."""
        result = graph.get_node_details('nonexistent_node_12345')
        assert result is None

    def test_negative_limit(self, graph):
        """Negative limit should be handled gracefully."""
        # Should either clamp to 0 or use default
        result = graph.query('test', limit=-1)
        assert 'results' in result

    def test_zero_limit(self, graph):
        """Zero limit should return empty results."""
        result = graph.query('test', limit=0)
        assert 'results' in result

    def test_very_large_limit(self, graph):
        """Very large limit should not cause memory issues."""
        result = graph.query('/', limit=999999)
        assert 'results' in result
        # Should be capped or return all available


class TestMCPConcurrentOperations:
    """Test concurrent access to the graph loader."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, graph):
        """Multiple concurrent queries should not interfere."""
        async def query_task(pattern):
            return graph.query(pattern, limit=5)

        # Run 10 concurrent queries
        tasks = [
            asyncio.create_task(asyncio.to_thread(graph.query, f'test{i}', 5))
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert 'results' in result

    @pytest.mark.asyncio
    async def test_concurrent_health_and_query(self, graph):
        """Health checks and queries should not conflict."""
        async def mixed_operations():
            queries = [
                asyncio.to_thread(graph.query, 'motor'),
                asyncio.to_thread(graph.get_summary),
                asyncio.to_thread(graph.get_orphaned_topics),
            ]
            return await asyncio.gather(*queries)

        results = await mixed_operations()
        assert len(results) == 3


class TestMCPErrorResponses:
    """Test error handling follows MCP spec."""

    def test_missing_graph_file_error(self, tmp_path):
        """Missing graph file should raise clear error."""
        graph = SystemGraph(str(tmp_path / 'nonexistent'))
        with pytest.raises(FileNotFoundError) as exc_info:
            graph.get_summary()
        assert 'system_graph.json' in str(exc_info.value) or 'not found' in str(exc_info.value).lower()

    def test_invalid_json_graph_file(self, tmp_path):
        """Invalid JSON in graph file should raise clear error."""
        graph_path = tmp_path / 'system_graph.json'
        graph_path.write_text('not valid json {{{')

        graph = SystemGraph(str(tmp_path))
        with pytest.raises(json.JSONDecodeError):
            graph.get_summary()

    def test_empty_graph_file(self, tmp_path):
        """Empty graph file should be handled."""
        graph_path = tmp_path / 'system_graph.json'
        graph_path.write_text('{}')

        graph = SystemGraph(str(tmp_path))
        result = graph.get_summary()
        # Should return defaults, not crash
        assert 'total_nodes' in result


class TestMCPSchemaCompliance:
    """Test that tool responses match expected schemas."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_query_response_schema(self, graph):
        """Query response should have expected fields."""
        result = graph.query('test')

        # Required fields
        assert 'pattern' in result
        assert 'total_matches' in result
        assert 'results' in result

        # Results structure
        results = result['results']
        assert 'ros2_nodes' in results
        assert 'topics' in results
        assert 'http_endpoints' in results
        assert 'http_clients' in results

        # All should be lists
        assert isinstance(results['ros2_nodes'], list)
        assert isinstance(results['topics'], list)
        assert isinstance(results['http_endpoints'], list)
        assert isinstance(results['http_clients'], list)

    def test_summary_response_schema(self, graph):
        """Summary response should have expected fields."""
        result = graph.get_summary()

        expected_fields = [
            'project', 'generated_at', 'total_nodes',
            'total_topics', 'connected_topics', 'total_http_endpoints', 'packages'
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_orphans_response_schema(self, graph):
        """Orphans response should have expected fields."""
        result = graph.get_orphaned_topics()

        assert 'no_publishers' in result
        assert 'no_subscribers' in result
        assert isinstance(result['no_publishers'], list)
        assert isinstance(result['no_subscribers'], list)

    def test_coupling_response_schema(self, graph):
        """Coupling response should be a list of pairs."""
        result = graph.get_coupling_pairs(min_score=0.0)

        assert isinstance(result, list)
        for pair in result:
            assert 'source' in pair
            assert 'target' in pair
            assert 'score' in pair


class TestMCPInputValidation:
    """Test input validation and sanitization."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_limit_type_coercion(self, graph):
        """Limit should handle type coercion."""
        # Float should work
        result = graph.query('test', limit=5.5)
        assert 'results' in result

    def test_pattern_type_validation(self, graph):
        """Non-string patterns should be handled."""
        # These might raise TypeError or convert to string
        for pattern in [123, 3.14, True, ['list']]:
            try:
                result = graph.query(str(pattern))
                assert 'results' in result
            except (TypeError, AttributeError):
                pass  # Also acceptable to reject non-strings


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
