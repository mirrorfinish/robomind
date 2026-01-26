"""
RoboMind - Rapid Prototyping System for Autonomous ROS2 Robots

A tool to analyze, visualize, and accelerate development of ROS2 robotics projects.
Scans codebases, extracts ROS2 patterns, and generates structured output for
AI-assisted development and human understanding.

Usage:
    robomind analyze ~/my_robot_project
    robomind visualize ~/my_robot_project --output graph.html
    robomind remote robot@jetson.local --ros2-info

Features:
    - Project scanning (Python, launch files, configs)
    - ROS2 pattern detection (nodes, topics, services, parameters)
    - Dependency graph building with coupling analysis
    - Multi-format export (JSON, YAML, HTML)
    - SSH remote analysis for distributed systems
"""

__version__ = "1.0.0"
__author__ = "Justin"

from robomind.core.scanner import ProjectScanner, ScanResult
from robomind.core.parser import PythonParser, ParseResult

__all__ = [
    "__version__",
    "ProjectScanner",
    "ScanResult",
    "PythonParser",
    "ParseResult",
]
