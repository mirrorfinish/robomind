"""Performance Benchmark Tests for RoboMind MCP Server.

Tests performance characteristics of the MCP server components.
"""

import pytest
import time
import resource
import json
import hashlib
from typing import List, Dict

from robomind.mcp_server.graph_loader import SystemGraph


class TestGraphLoadPerformance:
    """Test graph loading performance."""

    def test_graph_load_time(self):
        """Graph should load within reasonable time."""
        # Cold load (new instance)
        start = time.time()
        graph = SystemGraph()
        _ = graph._load_graph()
        cold_load_time = time.time() - start

        print(f"Cold graph load time: {cold_load_time:.3f}s")
        assert cold_load_time < 5.0, f"Graph load too slow: {cold_load_time:.2f}s"

    def test_graph_cache_performance(self):
        """Second load should use cache (be faster)."""
        graph = SystemGraph()

        # First load (cold)
        start = time.time()
        _ = graph._load_graph()
        cold_time = time.time() - start

        # Second load (cached)
        start = time.time()
        _ = graph._load_graph()
        cached_time = time.time() - start

        print(f"Cold: {cold_time:.4f}s, Cached: {cached_time:.4f}s")
        assert cached_time < cold_time, "Cache not working"

    def test_context_load_time(self):
        """Context YAML should load quickly."""
        graph = SystemGraph()

        start = time.time()
        try:
            _ = graph._load_context()
            load_time = time.time() - start
            print(f"Context load time: {load_time:.3f}s")
        except FileNotFoundError:
            print("Context file not found (OK)")


class TestQueryPerformance:
    """Test query performance."""

    @pytest.fixture
    def graph(self):
        """Pre-loaded graph for query tests."""
        g = SystemGraph()
        g._load_graph()  # Pre-load
        return g

    def test_single_query_latency(self, graph):
        """Single query should be fast."""
        start = time.time()
        result = graph.query('motor', limit=10)
        query_time = (time.time() - start) * 1000

        print(f"Single query latency: {query_time:.1f}ms")
        assert query_time < 100, f"Query too slow: {query_time:.1f}ms"

    def test_batch_query_performance(self, graph):
        """Batch queries should maintain performance."""
        patterns = ['motor', 'sensor', 'controller', 'voice', 'vision',
                    'camera', 'lidar', 'nav', 'cmd_vel', 'betaray']

        start = time.time()
        for pattern in patterns:
            graph.query(pattern, limit=10)
        total_time = (time.time() - start) * 1000

        avg_time = total_time / len(patterns)
        print(f"Batch query: {total_time:.1f}ms total, {avg_time:.1f}ms avg per query")
        assert avg_time < 50, f"Average query too slow: {avg_time:.1f}ms"

    def test_100_queries_benchmark(self, graph):
        """Benchmark 100 queries."""
        start = time.time()
        for i in range(100):
            graph.query(f'test{i}', limit=5)
        total_time = (time.time() - start) * 1000

        queries_per_second = 100 / (total_time / 1000)
        avg_ms = total_time / 100

        print(f"100 queries: {total_time:.0f}ms total")
        print(f"  {avg_ms:.2f}ms per query")
        print(f"  {queries_per_second:.0f} queries/sec")

        assert avg_ms < 100, f"Average query too slow: {avg_ms:.1f}ms"

    def test_large_limit_query(self, graph):
        """Large limit should not cause excessive slowdown."""
        start = time.time()
        result = graph.query('/', limit=1000)
        large_time = time.time() - start

        print(f"Large limit (1000) query: {large_time:.3f}s, found {result['total_matches']}")
        assert large_time < 2.0, f"Large query too slow: {large_time:.2f}s"

    def test_regex_pattern_performance(self, graph):
        """Complex regex patterns should still be fast."""
        complex_patterns = [
            '.*motor.*',
            'sensor|controller',
            '^/betaray',
            'node$',
            '[a-z]+_[a-z]+',
        ]

        for pattern in complex_patterns:
            start = time.time()
            result = graph.query(pattern, limit=10)
            query_time = (time.time() - start) * 1000
            print(f"Pattern '{pattern}': {query_time:.1f}ms, {result['total_matches']} matches")


class TestMemoryUsage:
    """Test memory usage characteristics."""

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def test_graph_memory_usage(self):
        """Graph loading should use reasonable memory."""
        mem_before = self.get_memory_mb()

        graph = SystemGraph()
        _ = graph._load_graph()

        mem_after = self.get_memory_mb()
        mem_used = mem_after - mem_before

        print(f"Memory before: {mem_before:.1f}MB")
        print(f"Memory after: {mem_after:.1f}MB")
        print(f"Memory used by graph: {mem_used:.1f}MB")

        # Should not use more than 200MB
        assert mem_used < 200, f"Graph uses too much memory: {mem_used:.1f}MB"

    def test_query_memory_stability(self):
        """Repeated queries should not leak memory."""
        graph = SystemGraph()
        graph._load_graph()

        mem_before = self.get_memory_mb()

        # Run 1000 queries
        for i in range(1000):
            graph.query(f'test{i % 100}', limit=10)

        mem_after = self.get_memory_mb()
        mem_growth = mem_after - mem_before

        print(f"Memory growth after 1000 queries: {mem_growth:.1f}MB")

        # Should not grow significantly
        assert mem_growth < 50, f"Memory leak detected: {mem_growth:.1f}MB growth"


class TestConcurrentPerformance:
    """Test performance under concurrent access."""

    def test_sequential_vs_cached_performance(self):
        """Compare sequential access patterns."""
        graph = SystemGraph()

        # Pattern 1: Fresh load each time (not recommended)
        times_fresh = []
        for _ in range(5):
            g = SystemGraph()
            start = time.time()
            g.query('motor', limit=10)
            times_fresh.append(time.time() - start)

        # Pattern 2: Reuse graph instance (recommended)
        graph._load_graph()  # Pre-load
        times_reuse = []
        for _ in range(5):
            start = time.time()
            graph.query('motor', limit=10)
            times_reuse.append(time.time() - start)

        avg_fresh = sum(times_fresh) / len(times_fresh) * 1000
        avg_reuse = sum(times_reuse) / len(times_reuse) * 1000

        print(f"Fresh instance avg: {avg_fresh:.1f}ms")
        print(f"Reused instance avg: {avg_reuse:.1f}ms")
        print(f"Speedup from reuse: {avg_fresh / avg_reuse:.1f}x")


class TestSpecificOperationPerformance:
    """Test specific operation performance."""

    @pytest.fixture
    def graph(self):
        """Pre-loaded graph."""
        g = SystemGraph()
        g._load_graph()
        return g

    def test_summary_performance(self, graph):
        """Summary should be fast."""
        start = time.time()
        result = graph.get_summary()
        summary_time = (time.time() - start) * 1000

        print(f"Summary time: {summary_time:.1f}ms")
        assert summary_time < 50, f"Summary too slow: {summary_time:.1f}ms"

    def test_orphans_performance(self, graph):
        """Orphan detection should be fast."""
        start = time.time()
        result = graph.get_orphaned_topics()
        orphan_time = (time.time() - start) * 1000

        print(f"Orphan detection time: {orphan_time:.1f}ms")
        assert orphan_time < 100, f"Orphan detection too slow: {orphan_time:.1f}ms"

    def test_coupling_performance(self, graph):
        """Coupling analysis should be reasonably fast."""
        start = time.time()
        result = graph.get_coupling_pairs(min_score=0.0)
        coupling_time = (time.time() - start) * 1000

        print(f"Coupling analysis time: {coupling_time:.1f}ms, {len(result)} pairs")
        assert coupling_time < 500, f"Coupling analysis too slow: {coupling_time:.1f}ms"

    def test_http_endpoints_performance(self, graph):
        """HTTP endpoint retrieval should be fast."""
        start = time.time()
        result = graph.get_http_endpoints()
        endpoint_time = (time.time() - start) * 1000

        print(f"HTTP endpoints time: {endpoint_time:.1f}ms, {len(result)} endpoints")
        assert endpoint_time < 50, f"Endpoint retrieval too slow: {endpoint_time:.1f}ms"


class TestPerformanceReport:
    """Generate comprehensive performance report."""

    def test_full_performance_report(self):
        """Generate full performance report."""
        print("\n" + "=" * 60)
        print("ROBOMIND PERFORMANCE REPORT")
        print("=" * 60)

        # Graph load
        start = time.time()
        graph = SystemGraph()
        _ = graph._load_graph()
        load_time = time.time() - start
        print(f"\nGraph Load Time: {load_time:.3f}s")

        # Summary
        summary = graph.get_summary()
        print(f"Nodes: {summary['total_nodes']}")
        print(f"Topics: {summary['total_topics']}")

        # Query benchmark
        print("\nQuery Performance (10 patterns, 10 results each):")
        patterns = ['motor', 'sensor', 'voice', 'nav', '/cmd']
        for pattern in patterns:
            start = time.time()
            result = graph.query(pattern, limit=10)
            query_time = (time.time() - start) * 1000
            print(f"  '{pattern}': {query_time:.1f}ms ({result['total_matches']} matches)")

        # Memory
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"\nMemory Usage: {mem_mb:.1f}MB")

        # Throughput test
        start = time.time()
        for i in range(100):
            graph.query(f'q{i}', limit=5)
        throughput_time = time.time() - start
        qps = 100 / throughput_time
        print(f"\nThroughput: {qps:.0f} queries/sec")

        print("=" * 60)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
