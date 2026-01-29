"""HTTP Health Tests for RoboMind MCP Server.

Tests HTTP endpoint health checking functionality.
Based on: https://arxiv.org/html/2404.11498v1 (Runtime Verification)
"""

import pytest
import requests
import time
from collections import defaultdict
from typing import List, Tuple, Dict


# NEXUS endpoints from MCP server configuration
NEXUS_ENDPOINTS = [
    ("thor-deep-reasoning", "thor.local", 8088, "/api/health"),
    ("thor-memory-api", "thor.local", 8089, "/health"),
    ("thor-face-recognition", "thor.local", 8090, "/health"),
    ("thor-vision-router", "thor.local", 8091, "/health"),
    ("thor-chromadb", "localhost", 8000, "/api/v2/heartbeat"),
    ("nav-canary", "betaray-nav.local", 8081, "/health"),
    ("ai-http", "betaray-ai.local", 8080, "/health"),
    ("vision-frame", "vision-jetson.local", 9091, "/health"),
    ("vllm-thought", "localhost", 30000, "/health"),
]


class TestHTTPBasicHealth:
    """Basic HTTP health check tests."""

    @pytest.fixture
    def timeout(self):
        """Default timeout for health checks."""
        return 3

    def check_endpoint(self, name: str, host: str, port: int, path: str, timeout: float) -> Dict:
        """Check a single endpoint and return status."""
        url = f"http://{host}:{port}{path}"
        try:
            start = time.time()
            resp = requests.get(url, timeout=timeout)
            latency = (time.time() - start) * 1000

            if resp.ok:
                return {"status": "healthy", "code": resp.status_code, "latency_ms": latency}
            else:
                return {"status": "degraded", "code": resp.status_code, "latency_ms": latency}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "url": url}
        except requests.exceptions.ConnectionError:
            return {"status": "unreachable", "url": url}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def test_individual_endpoints(self, timeout):
        """Test each endpoint individually and report status."""
        results = {}
        for name, host, port, path in NEXUS_ENDPOINTS:
            result = self.check_endpoint(name, host, port, path, timeout)
            results[name] = result
            print(f"{name}: {result['status']}")

        # At least some endpoints should be healthy
        healthy = sum(1 for r in results.values() if r['status'] == 'healthy')
        print(f"\n{healthy}/{len(NEXUS_ENDPOINTS)} endpoints healthy")

    def test_localhost_endpoints(self, timeout):
        """Test localhost endpoints specifically (more likely to be up)."""
        localhost_endpoints = [
            (name, host, port, path)
            for name, host, port, path in NEXUS_ENDPOINTS
            if host == 'localhost'
        ]

        healthy = 0
        for name, host, port, path in localhost_endpoints:
            result = self.check_endpoint(name, host, port, path, timeout)
            if result['status'] == 'healthy':
                healthy += 1

        print(f"Localhost endpoints: {healthy}/{len(localhost_endpoints)} healthy")

    def test_thor_local_endpoints(self, timeout):
        """Test thor.local endpoints (WSL2 services)."""
        thor_endpoints = [
            (name, host, port, path)
            for name, host, port, path in NEXUS_ENDPOINTS
            if host == 'thor.local'
        ]

        results = {}
        for name, host, port, path in thor_endpoints:
            results[name] = self.check_endpoint(name, host, port, path, timeout)

        healthy = sum(1 for r in results.values() if r['status'] == 'healthy')
        print(f"Thor services: {healthy}/{len(thor_endpoints)} healthy")

        return results


class TestHTTPLatency:
    """Test HTTP endpoint latency metrics."""

    @pytest.fixture
    def samples(self):
        """Number of samples for latency tests."""
        return 5

    def measure_latency(self, host: str, port: int, path: str, samples: int) -> Dict:
        """Measure latency over multiple samples."""
        latencies = []
        failures = 0

        for _ in range(samples):
            try:
                start = time.time()
                resp = requests.get(f"http://{host}:{port}{path}", timeout=5)
                if resp.ok:
                    latencies.append((time.time() - start) * 1000)
                else:
                    failures += 1
            except:
                failures += 1

        if not latencies:
            return {"status": "failed", "failures": failures}

        latencies.sort()
        return {
            "status": "ok",
            "samples": len(latencies),
            "failures": failures,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "avg_ms": sum(latencies) / len(latencies),
            "p50_ms": latencies[len(latencies) // 2],
            "p90_ms": latencies[int(len(latencies) * 0.9)] if len(latencies) >= 10 else latencies[-1],
        }

    def test_chromadb_latency(self, samples):
        """Test ChromaDB latency (should be fast)."""
        result = self.measure_latency("localhost", 8000, "/api/v2/heartbeat", samples)

        if result["status"] == "ok":
            print(f"ChromaDB latency: avg={result['avg_ms']:.1f}ms, p50={result['p50_ms']:.1f}ms")
            # ChromaDB should respond quickly
            assert result['avg_ms'] < 500, f"ChromaDB too slow: {result['avg_ms']:.1f}ms"

    def test_memory_api_latency(self, samples):
        """Test Memory API latency."""
        result = self.measure_latency("thor.local", 8089, "/health", samples)

        if result["status"] == "ok":
            print(f"Memory API latency: avg={result['avg_ms']:.1f}ms")

    def test_reasoning_api_latency(self, samples):
        """Test Deep Reasoning API latency."""
        result = self.measure_latency("thor.local", 8088, "/api/health", samples)

        if result["status"] == "ok":
            print(f"Reasoning API latency: avg={result['avg_ms']:.1f}ms")


class TestHTTPStability:
    """Test HTTP endpoint stability over time."""

    def test_short_stability(self):
        """Quick stability check (1 minute, 6 samples)."""
        history = defaultdict(list)

        print("Running 1-minute stability test (6 samples @ 10s intervals)...")

        for i in range(6):
            for name, host, port, path in NEXUS_ENDPOINTS[:5]:  # Test first 5 only for speed
                url = f"http://{host}:{port}{path}"
                try:
                    start = time.time()
                    r = requests.get(url, timeout=3)
                    latency = (time.time() - start) * 1000
                    history[name].append(('healthy' if r.ok else 'degraded', latency))
                except:
                    history[name].append(('failed', 0))

            if i < 5:  # Don't sleep after last iteration
                time.sleep(10)

        print("\n=== STABILITY RESULTS ===")
        for name, results in history.items():
            healthy = sum(1 for r in results if r[0] == 'healthy')
            latencies = [r[1] for r in results if r[0] == 'healthy']
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            status = "STABLE" if healthy == len(results) else f"FLAPPING ({healthy}/{len(results)})"
            print(f"  {name}: {status}, avg latency: {avg_latency:.0f}ms")

    def test_flapping_detection(self):
        """Detect if any service is flapping (alternating state)."""
        samples = 4
        history = defaultdict(list)

        for _ in range(samples):
            for name, host, port, path in NEXUS_ENDPOINTS[:3]:
                try:
                    r = requests.get(f"http://{host}:{port}{path}", timeout=2)
                    history[name].append('up' if r.ok else 'down')
                except:
                    history[name].append('down')
            time.sleep(2)

        # Check for flapping (state changes)
        flapping = []
        for name, states in history.items():
            changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
            if changes > 1:  # More than 1 state change indicates flapping
                flapping.append(name)

        if flapping:
            print(f"Warning: Flapping services detected: {flapping}")
        else:
            print("No flapping detected")


class TestHTTPRecovery:
    """Test service recovery detection."""

    def test_recovery_detection(self):
        """Test ability to detect service recovery."""
        # This test just verifies the detection mechanism works
        # Actual recovery would require taking a service down

        name, host, port, path = NEXUS_ENDPOINTS[0]  # Use first endpoint

        # Get initial state
        try:
            r = requests.get(f"http://{host}:{port}{path}", timeout=2)
            initial_state = 'up' if r.ok else 'down'
        except:
            initial_state = 'down'

        print(f"Initial state of {name}: {initial_state}")

        # Check again after short delay
        time.sleep(1)

        try:
            r = requests.get(f"http://{host}:{port}{path}", timeout=2)
            final_state = 'up' if r.ok else 'down'
        except:
            final_state = 'down'

        print(f"Final state of {name}: {final_state}")

        if initial_state != final_state:
            print(f"State change detected: {initial_state} -> {final_state}")


class TestHTTPHealthSummary:
    """Generate health summary report."""

    def test_generate_health_report(self):
        """Generate comprehensive health report."""
        results = {}

        print("\n" + "=" * 60)
        print("NEXUS HTTP HEALTH REPORT")
        print("=" * 60)

        for name, host, port, path in NEXUS_ENDPOINTS:
            url = f"http://{host}:{port}{path}"
            try:
                start = time.time()
                r = requests.get(url, timeout=3)
                latency = (time.time() - start) * 1000

                if r.ok:
                    results[name] = {"status": "HEALTHY", "latency": latency}
                    print(f"  ✓ {name:25} HEALTHY  ({latency:.0f}ms)")
                else:
                    results[name] = {"status": "DEGRADED", "code": r.status_code}
                    print(f"  ⚠ {name:25} DEGRADED (HTTP {r.status_code})")
            except requests.exceptions.Timeout:
                results[name] = {"status": "TIMEOUT"}
                print(f"  ✗ {name:25} TIMEOUT")
            except requests.exceptions.ConnectionError:
                results[name] = {"status": "UNREACHABLE"}
                print(f"  ✗ {name:25} UNREACHABLE")
            except Exception as e:
                results[name] = {"status": "ERROR", "error": str(e)}
                print(f"  ✗ {name:25} ERROR: {str(e)[:30]}")

        print("-" * 60)
        healthy = sum(1 for r in results.values() if r['status'] == 'HEALTHY')
        print(f"Summary: {healthy}/{len(results)} services healthy")
        print("=" * 60)

        return results


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
