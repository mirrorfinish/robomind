"""
RoboMind Security Vulnerability Analyzer

Scans ROS2 code for security issues including:
- Hardcoded credentials/secrets
- Unvalidated input
- Command injection risks
- Network exposure without authentication
- Insecure deserialization
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robomind.ros2.node_extractor import ROS2NodeInfo


@dataclass
class SecurityFinding:
    """A security vulnerability finding."""
    vuln_type: str
    severity: str  # "critical", "high", "medium", "low"
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    node: str
    file_path: str
    line_number: int
    code_snippet: str
    description: str
    recommendation: str


class SecurityAnalyzer:
    """
    Scans ROS2 code for security vulnerabilities.

    Based on OWASP, CWE, and robotics-specific security concerns.
    """

    # Hardcoded secrets patterns
    SECRET_PATTERNS = [
        (r'password\s*=\s*["\'][^"\']+["\']', 'CWE-798', 'Hardcoded password'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'CWE-798', 'Hardcoded API key'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'CWE-798', 'Hardcoded secret'),
        (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', 'CWE-798', 'Hardcoded token'),
        (r'private_key\s*=\s*["\']', 'CWE-798', 'Hardcoded private key'),
        (r'aws_access_key_id\s*=\s*["\']', 'CWE-798', 'Hardcoded AWS credentials'),
        (r'AWS_SECRET_ACCESS_KEY\s*=\s*["\']', 'CWE-798', 'Hardcoded AWS secret'),
    ]

    # Hardcoded IP/network patterns (often security issues in robotics)
    NETWORK_PATTERNS = [
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b(?!\.)', None, 'Hardcoded IP address'),
        (r'http://(?!localhost|127\.0\.0\.1)', 'CWE-319', 'Unencrypted HTTP connection'),
    ]

    # Command injection patterns
    INJECTION_PATTERNS = [
        (r'os\.system\s*\([^)]*\+', 'CWE-78', 'Potential OS command injection'),
        (r'subprocess\.(run|call|Popen)\s*\([^)]*shell\s*=\s*True', 'CWE-78', 'Shell=True enables command injection'),
        (r'eval\s*\(', 'CWE-95', 'eval() can execute arbitrary code'),
        (r'exec\s*\(', 'CWE-95', 'exec() can execute arbitrary code'),
        (r'__import__\s*\(', 'CWE-95', 'Dynamic import can be exploited'),
    ]

    # Insecure deserialization
    DESERIALIZE_PATTERNS = [
        (r'pickle\.loads?\s*\(', 'CWE-502', 'Pickle deserialization is insecure'),
        (r'yaml\.load\s*\([^)]*\)(?!.*Loader)', 'CWE-502', 'yaml.load without SafeLoader is insecure'),
        (r'yaml\.unsafe_load', 'CWE-502', 'yaml.unsafe_load allows code execution'),
        (r'marshal\.loads?\s*\(', 'CWE-502', 'marshal deserialization is insecure'),
    ]

    # Path traversal
    PATH_PATTERNS = [
        (r'open\s*\([^)]*\+[^)]*\)', 'CWE-22', 'Potential path traversal via string concatenation'),
        (r'\.\./', 'CWE-22', 'Path traversal pattern detected'),
    ]

    # ROS-specific security issues
    ROS_SECURITY_PATTERNS = [
        (r'allow_undeclared_parameters\s*=\s*True', None, 'Allows arbitrary parameter injection'),
        (r'automatically_declare_parameters_from_overrides\s*=\s*True', None, 'Auto-declares parameters (potential injection)'),
    ]

    # Patterns that indicate lack of authentication
    NO_AUTH_PATTERNS = [
        (r'Flask\s*\(', None, 'Flask app without visible authentication'),
        (r'@app\.route\s*\([^)]*\)(?!.*login|auth)', None, 'HTTP endpoint without authentication'),
        (r'socket\.bind\s*\(\s*\(["\']0\.0\.0\.0', 'CWE-284', 'Socket bound to all interfaces'),
    ]

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.file_contents: Dict[str, str] = {}
        self.python_files: List[Path] = []

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes
        # Collect unique file paths
        self.python_files = list(set(
            Path(n.file_path) for n in nodes if n.file_path
        ))

    def add_files(self, files: List[Path]):
        """Add additional Python files to scan."""
        self.python_files.extend(files)

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content with caching."""
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r') as f:
                    self.file_contents[file_path] = f.read()
            except Exception:
                return None
        return self.file_contents.get(file_path)

    def _get_node_for_file(self, file_path: str) -> str:
        """Get node name for a file path."""
        for node in self.nodes:
            if node.file_path == file_path:
                return node.name
        return Path(file_path).stem

    def _scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security issues."""
        findings = []
        content = self._get_file_content(str(file_path))
        if not content:
            return findings

        lines = content.split('\n')
        node_name = self._get_node_for_file(str(file_path))

        # Check all pattern categories
        all_patterns = [
            (self.SECRET_PATTERNS, 'critical', 'hardcoded_secret'),
            (self.INJECTION_PATTERNS, 'critical', 'code_injection'),
            (self.DESERIALIZE_PATTERNS, 'high', 'insecure_deserialization'),
            (self.PATH_PATTERNS, 'medium', 'path_traversal'),
            (self.ROS_SECURITY_PATTERNS, 'medium', 'ros_security'),
            (self.NO_AUTH_PATTERNS, 'medium', 'missing_authentication'),
        ]

        for patterns, default_severity, vuln_category in all_patterns:
            for pattern, cwe, description in patterns:
                for i, line in enumerate(lines, 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue

                    if re.search(pattern, line, re.IGNORECASE):
                        # Determine severity based on context
                        severity = default_severity

                        # Lower severity for test files
                        if 'test' in str(file_path).lower():
                            severity = 'low'

                        findings.append(SecurityFinding(
                            vuln_type=vuln_category,
                            severity=severity,
                            cwe_id=cwe,
                            node=node_name,
                            file_path=str(file_path),
                            line_number=i,
                            code_snippet=line.strip()[:100],
                            description=description,
                            recommendation=self._get_recommendation(vuln_category, cwe),
                        ))

        # Check for network patterns (special handling for IPs)
        for i, line in enumerate(lines, 1):
            # Skip comments and strings that look like version numbers
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            # Check for hardcoded IPs (but exclude common safe ones)
            ip_matches = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', line)
            for ip in ip_matches:
                # Skip localhost and link-local
                if ip.startswith('127.') or ip.startswith('0.') or ip == '255.255.255.255':
                    continue
                # Skip version-like patterns
                if re.search(rf'{re.escape(ip)}["\']?\s*[,\)]', line):
                    if 'version' in line.lower():
                        continue

                findings.append(SecurityFinding(
                    vuln_type='hardcoded_network',
                    severity='low',
                    cwe_id=None,
                    node=node_name,
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip()[:100],
                    description=f'Hardcoded IP address: {ip}',
                    recommendation='Use configuration files or environment variables for network addresses',
                ))

        return findings

    def _get_recommendation(self, vuln_type: str, cwe: Optional[str]) -> str:
        """Get remediation recommendation for a vulnerability type."""
        recommendations = {
            'hardcoded_secret': 'Use environment variables or a secrets manager (e.g., HashiCorp Vault)',
            'code_injection': 'Validate and sanitize all inputs; avoid shell=True and eval()',
            'insecure_deserialization': 'Use safe serialization formats (JSON) or yaml.safe_load()',
            'path_traversal': 'Validate file paths; use pathlib and resolve() to canonicalize',
            'ros_security': 'Explicitly declare parameters; enable SROS2 security features',
            'missing_authentication': 'Implement authentication (JWT, OAuth2) for all endpoints',
            'hardcoded_network': 'Use configuration files or ROS2 parameters for network settings',
        }
        return recommendations.get(vuln_type, 'Review and remediate the security issue')

    def analyze(self) -> List[SecurityFinding]:
        """Run security analysis on all files."""
        findings = []

        for file_path in self.python_files:
            findings.extend(self._scan_file(file_path))

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 4))

        return findings


def analyze_security(nodes: List[ROS2NodeInfo]) -> List[SecurityFinding]:
    """Convenience function for security analysis."""
    analyzer = SecurityAnalyzer()
    analyzer.add_nodes(nodes)
    return analyzer.analyze()
