"""
RoboMind Remote Analysis Module

Provides SSH-based analysis of distributed ROS2 systems across multiple hosts.

Example:
    from robomind.remote import DistributedAnalyzer, RemoteHost

    hosts = [
        RemoteHost("nav.local", "robot", project_path="~/betaray"),
        RemoteHost("ai.local", "robot", project_path="~/betaray"),
    ]
    analyzer = DistributedAnalyzer(hosts)
    result = analyzer.analyze_all()
"""

from robomind.remote.ssh_analyzer import (
    RemoteHost,
    SSHConnection,
    SSHAnalyzer,
    DistributedAnalyzer,
    RemoteAnalysisResult,
    DistributedAnalysisResult,
    parse_remote_specs,
)

__all__ = [
    "RemoteHost",
    "SSHConnection",
    "SSHAnalyzer",
    "DistributedAnalyzer",
    "RemoteAnalysisResult",
    "DistributedAnalysisResult",
    "parse_remote_specs",
]
