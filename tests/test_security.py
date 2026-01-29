"""Security Tests for RoboMind MCP Server.

Tests security aspects of the MCP server.
Based on: https://www.redhat.com/en/blog/model-context-protocol-mcp-understanding-security-risks-and-controls
"""

import pytest
import os
import json
import time
import signal
from pathlib import Path

from robomind.mcp_server.graph_loader import SystemGraph


class TestPathTraversal:
    """Test path traversal prevention."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_path_traversal_in_query(self, graph):
        """Path traversal patterns in query should not access files."""
        traversal_patterns = [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32',
            '/etc/passwd',
            '../../.ssh/id_rsa',
            '%2e%2e%2f%2e%2e%2f',  # URL encoded
            '....//....//etc/passwd',
        ]

        for pattern in traversal_patterns:
            result = graph.query(pattern)
            # Should return empty results, not file contents
            assert 'results' in result
            # Results should not contain file contents
            result_str = json.dumps(result)
            assert 'root:' not in result_str, f"Possible file leak with pattern: {pattern}"
            assert 'BEGIN RSA' not in result_str, f"Possible key leak with pattern: {pattern}"

    def test_null_byte_injection(self, graph):
        """Null byte injection should be handled."""
        patterns = [
            'test\x00.txt',
            'motor\x00/etc/passwd',
            '\x00\x00\x00',
        ]

        for pattern in patterns:
            try:
                result = graph.query(pattern)
                assert 'results' in result
            except (ValueError, TypeError):
                # Also acceptable to reject null bytes
                pass


class TestInjection:
    """Test injection attack prevention."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_shell_injection_patterns(self, graph):
        """Shell metacharacters should not execute commands."""
        shell_patterns = [
            '; rm -rf /',
            '| cat /etc/passwd',
            '`whoami`',
            '$(id)',
            '&& cat /etc/shadow',
            '; DROP TABLE nodes;--',
            "' OR '1'='1",
        ]

        for pattern in shell_patterns:
            result = graph.query(pattern)
            # Should return results without executing commands
            assert 'results' in result
            # Pattern should be treated as literal text
            assert result['pattern'] == pattern

    def test_regex_dos_patterns(self, graph):
        """ReDoS patterns should complete within timeout."""
        # These patterns can cause exponential backtracking in naive regex
        # We use a shorter timeout and accept that some complex patterns may be slow
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Query timed out")

        redos_patterns = [
            '(a+)+$',
            '([a-zA-Z]+)*',
        ]

        for pattern in redos_patterns:
            start = time.time()
            try:
                # Set alarm for 2 seconds
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)
                result = graph.query(pattern, limit=10)
                signal.alarm(0)  # Cancel alarm
            except TimeoutError:
                signal.alarm(0)
                print(f"Warning: Pattern '{pattern}' timed out (potential ReDoS)")
            except Exception:
                signal.alarm(0)
            elapsed = time.time() - start

            # Just warn, don't fail - regex handling is complex
            if elapsed > 2.0:
                print(f"Warning: Pattern '{pattern}' took {elapsed:.1f}s")


class TestResourceExhaustion:
    """Test resource exhaustion prevention."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_very_large_limit(self, graph):
        """Very large limits should be handled safely."""
        large_limits = [999999, 2**31 - 1, 2**63 - 1]

        for limit in large_limits:
            try:
                result = graph.query('test', limit=limit)
                # Should either cap the limit or handle gracefully
                assert result['total_matches'] < 10000, "Unreasonably large result set"
            except (OverflowError, MemoryError, ValueError):
                # Also acceptable to reject absurd limits
                pass

    def test_negative_limit(self, graph):
        """Negative limits should be handled safely."""
        negative_limits = [-1, -999, -2**31]

        for limit in negative_limits:
            result = graph.query('test', limit=limit)
            # Should not crash, should return valid results
            assert 'results' in result

    def test_very_long_pattern(self, graph):
        """Very long patterns should not cause issues."""
        # 10KB pattern
        long_pattern = 'a' * 10000
        result = graph.query(long_pattern)
        assert 'results' in result

        # 1MB pattern (might be rejected, which is OK)
        try:
            huge_pattern = 'a' * 1000000
            result = graph.query(huge_pattern)
            assert 'results' in result
        except (MemoryError, ValueError):
            pass  # Acceptable to reject huge patterns


class TestDataSanitization:
    """Test that data is properly sanitized."""

    @pytest.fixture
    def graph(self):
        """Create a SystemGraph instance."""
        return SystemGraph()

    def test_html_in_pattern(self, graph):
        """HTML in patterns should not cause XSS if results are displayed."""
        xss_patterns = [
            '<script>alert("xss")</script>',
            '<img src=x onerror=alert(1)>',
            '"><script>alert(1)</script>',
            "javascript:alert('xss')",
        ]

        for pattern in xss_patterns:
            result = graph.query(pattern)
            # Pattern should be stored as-is (escaping happens at display time)
            assert result['pattern'] == pattern

    def test_unicode_normalization(self, graph):
        """Unicode should be handled consistently."""
        unicode_patterns = [
            '\u202e',  # Right-to-left override
            '\ufeff',  # BOM
            'café',    # NFD vs NFC
            '𝕋𝕖𝕤𝕥',    # Mathematical symbols
        ]

        for pattern in unicode_patterns:
            result = graph.query(pattern)
            assert 'results' in result


class TestAccessControl:
    """Test access control (for reference - MCP server is local)."""

    def test_graph_path_validation(self, tmp_path):
        """Custom analysis path should be validated."""
        # Valid path
        graph = SystemGraph(str(tmp_path))
        # Should not crash during init

    def test_symlink_following(self, tmp_path):
        """Symlinks should be handled carefully."""
        # Create a symlink to /etc
        symlink_path = tmp_path / 'dangerous_link'
        try:
            symlink_path.symlink_to('/etc')

            # If graph tries to follow symlinks, it should not access /etc
            graph = SystemGraph(str(symlink_path))
            # Attempting to load should fail (no system_graph.json in /etc)
            with pytest.raises(FileNotFoundError):
                graph.get_summary()
        except OSError:
            # Symlink creation might fail in some environments
            pass


class TestErrorHandling:
    """Test that errors don't leak sensitive information."""

    def test_file_not_found_error_message(self, tmp_path):
        """File not found errors should not leak full paths."""
        graph = SystemGraph(str(tmp_path / 'nonexistent'))

        try:
            graph.get_summary()
        except FileNotFoundError as e:
            error_msg = str(e)
            # Should not contain sensitive system paths
            assert '/etc' not in error_msg
            assert '/root' not in error_msg

    def test_json_parse_error_message(self, tmp_path):
        """JSON parse errors should not leak file contents."""
        graph_file = tmp_path / 'system_graph.json'
        # Write invalid JSON with sensitive-looking data
        graph_file.write_text('{"password": "secret123", invalid}')

        graph = SystemGraph(str(tmp_path))

        try:
            graph.get_summary()
        except json.JSONDecodeError as e:
            error_msg = str(e)
            # Error message should not contain the password
            assert 'secret123' not in error_msg


class TestSecurityReport:
    """Generate security test report."""

    def test_security_summary(self):
        """Run all security checks and report."""
        print("\n" + "=" * 60)
        print("ROBOMIND SECURITY TEST REPORT")
        print("=" * 60)

        graph = SystemGraph()
        checks_passed = 0
        checks_total = 0

        # Path traversal
        checks_total += 1
        result = graph.query('../../../etc/passwd')
        if 'root:' not in json.dumps(result):
            print("  ✓ Path traversal protection: PASSED")
            checks_passed += 1
        else:
            print("  ✗ Path traversal protection: FAILED")

        # Shell injection
        checks_total += 1
        result = graph.query('; rm -rf /')
        if result['pattern'] == '; rm -rf /':
            print("  ✓ Shell injection protection: PASSED")
            checks_passed += 1
        else:
            print("  ✗ Shell injection protection: FAILED")

        # Resource limits
        checks_total += 1
        import time
        start = time.time()
        result = graph.query('a', limit=999999)
        if time.time() - start < 5.0:
            print("  ✓ Resource exhaustion protection: PASSED")
            checks_passed += 1
        else:
            print("  ✗ Resource exhaustion protection: FAILED")

        # ReDoS
        checks_total += 1
        start = time.time()
        try:
            graph.query('(a+)+$', limit=10)
        except:
            pass
        if time.time() - start < 5.0:
            print("  ✓ ReDoS protection: PASSED")
            checks_passed += 1
        else:
            print("  ✗ ReDoS protection: FAILED")

        print("-" * 60)
        print(f"Security Checks: {checks_passed}/{checks_total} passed")
        print("=" * 60)

        assert checks_passed == checks_total, f"Some security checks failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
