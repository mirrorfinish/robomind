"""Tests for RoboMind Project Scanner."""

import pytest
from pathlib import Path
import tempfile
import os

from robomind.core.scanner import ProjectScanner, ScanResult, ProjectFile


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create package structure
    package_dir = tmp_path / "test_package"
    package_dir.mkdir()

    # Create package.xml
    package_xml = package_dir / "package.xml"
    package_xml.write_text(
        """<?xml version="1.0"?>
<package format="3">
  <name>test_package</name>
  <version>0.1.0</version>
  <description>Test package</description>
  <maintainer email="test@test.com">Test</maintainer>
  <license>MIT</license>
</package>
"""
    )

    # Create Python files
    (package_dir / "node.py").write_text("# Test node\nprint('hello')\n")
    (package_dir / "utils.py").write_text("# Utilities\ndef foo():\n    pass\n")

    # Create launch file
    launch_dir = package_dir / "launch"
    launch_dir.mkdir()
    (launch_dir / "test.launch.py").write_text("# Launch file\n")

    # Create config
    config_dir = package_dir / "config"
    config_dir.mkdir()
    (config_dir / "params.yaml").write_text("param: value\n")

    # Create stuff to ignore
    pycache = package_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "test.pyc").write_text("compiled")

    return tmp_path


def test_scanner_init(tmp_path):
    """Test scanner initialization."""
    scanner = ProjectScanner(tmp_path)
    assert scanner.root_path.exists()


def test_scanner_nonexistent_path():
    """Test scanner with non-existent path."""
    with pytest.raises(ValueError):
        ProjectScanner("/nonexistent/path/that/does/not/exist")


def test_find_packages(temp_project):
    """Test finding ROS2 packages."""
    scanner = ProjectScanner(temp_project)
    packages = scanner.find_ros2_packages()

    assert len(packages) == 1
    assert "test_package" in packages


def test_scan_python_files(temp_project):
    """Test scanning for Python files."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    assert len(result.python_files) == 2
    assert result.total_files >= 2

    # Check package assignment
    for pf in result.python_files:
        assert pf.package_name == "test_package"


def test_scan_launch_files(temp_project):
    """Test scanning for launch files."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    assert len(result.launch_files) == 1
    assert result.launch_files[0].file_type == "launch"


def test_scan_config_files(temp_project):
    """Test scanning for config files."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    assert len(result.config_files) == 1
    assert result.config_files[0].file_type == "config"


def test_ignore_pycache(temp_project):
    """Test that __pycache__ is ignored."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    # Should not find any files in __pycache__
    for pf in result.python_files:
        assert "__pycache__" not in str(pf.path)


def test_get_files_by_package(temp_project):
    """Test filtering files by package."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    pkg_files = result.get_files_by_package("test_package")
    assert len(pkg_files) == 2

    non_pkg_files = result.get_files_by_package("nonexistent")
    assert len(non_pkg_files) == 0


def test_summary(temp_project):
    """Test summary generation."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    summary = result.summary()

    assert "packages" in summary
    assert "python_files" in summary
    assert summary["packages"] == 1
    assert summary["python_files"] == 2


def test_line_counting(temp_project):
    """Test line of code counting."""
    scanner = ProjectScanner(temp_project)
    result = scanner.scan()

    # At least some lines should be counted
    assert result.total_lines > 0

    # Each file should have lines counted
    for pf in result.python_files:
        assert pf.lines_of_code > 0
