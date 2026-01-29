#!/usr/bin/env python3
"""Run all RoboMind MCP Server tests.

Usage:
    python3 tests/run_mcp_tests.py           # Run all tests
    python3 tests/run_mcp_tests.py --quick   # Quick tests only (no network)
    python3 tests/run_mcp_tests.py --report  # Generate detailed reports
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_file(test_file: Path, verbose: bool = True) -> tuple[bool, str]:
    """Run a single test file and return (passed, output)."""
    cmd = ['python3', '-m', 'pytest', str(test_file), '-v']
    if not verbose:
        cmd.append('-q')

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    passed = result.returncode == 0
    output = result.stdout + result.stderr
    return passed, output


def main():
    args = sys.argv[1:]
    quick_mode = '--quick' in args
    report_mode = '--report' in args

    tests_dir = Path(__file__).parent

    # Define test files in order
    test_files = [
        ('MCP Protocol', 'test_mcp_protocol.py'),
        ('Graph Consistency', 'test_graph_consistency.py'),
        ('Performance', 'test_performance.py'),
        ('Security', 'test_security.py'),
        ('Static Accuracy', 'test_static_accuracy.py'),
    ]

    if not quick_mode:
        test_files.append(('HTTP Health', 'test_health.py'))

    print("=" * 70)
    print("ROBOMIND MCP SERVER TEST SUITE")
    print("=" * 70)

    if quick_mode:
        print("Mode: QUICK (skipping network tests)")
    else:
        print("Mode: FULL (including network tests)")

    print()

    results = {}
    total_start = time.time()

    for name, filename in test_files:
        test_path = tests_dir / filename

        if not test_path.exists():
            print(f"[SKIP] {name}: File not found")
            continue

        print(f"[TEST] {name}...", end=' ', flush=True)
        start = time.time()

        try:
            passed, output = run_test_file(test_path, verbose=report_mode)
            elapsed = time.time() - start

            if passed:
                print(f"PASSED ({elapsed:.1f}s)")
                results[name] = ('PASS', elapsed)
            else:
                print(f"FAILED ({elapsed:.1f}s)")
                results[name] = ('FAIL', elapsed)

                if report_mode:
                    print("-" * 40)
                    print(output[-2000:] if len(output) > 2000 else output)
                    print("-" * 40)

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            results[name] = ('TIMEOUT', 300)
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = ('ERROR', 0)

    total_time = time.time() - total_start

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r[0] == 'PASS')
    failed = sum(1 for r in results.values() if r[0] == 'FAIL')
    errors = sum(1 for r in results.values() if r[0] in ('TIMEOUT', 'ERROR'))

    for name, (status, elapsed) in results.items():
        symbol = '✓' if status == 'PASS' else '✗' if status == 'FAIL' else '!'
        print(f"  {symbol} {name:25} {status:8} ({elapsed:.1f}s)")

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed, {errors} errors")
    print(f"Time: {total_time:.1f}s")
    print("=" * 70)

    # Return exit code
    return 0 if failed + errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
