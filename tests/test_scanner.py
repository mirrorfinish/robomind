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


# ============== Exclusion Pattern Tests ==============

@pytest.fixture
def temp_project_with_archive(tmp_path):
    """Create a project with archive directories for exclusion testing."""
    # Create main package
    main_pkg = tmp_path / "main_package"
    main_pkg.mkdir()

    (main_pkg / "package.xml").write_text(
        """<?xml version="1.0"?>
<package format="3">
  <name>main_package</name>
  <version>0.1.0</version>
  <description>Main package</description>
  <maintainer email="test@test.com">Test</maintainer>
  <license>MIT</license>
</package>
"""
    )
    (main_pkg / "node.py").write_text("# Main node\nprint('main')\n")

    # Create archive with old code
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    old_pkg = archive_dir / "old_package"
    old_pkg.mkdir()

    (old_pkg / "package.xml").write_text(
        """<?xml version="1.0"?>
<package format="3">
  <name>old_package</name>
  <version>0.0.1</version>
  <description>Old package</description>
  <maintainer email="test@test.com">Test</maintainer>
  <license>MIT</license>
</package>
"""
    )
    (old_pkg / "old_node.py").write_text("# Old archived code\nprint('old')\n")

    # Create backup directory
    backup_dir = tmp_path / "backup_2024"
    backup_dir.mkdir()
    (backup_dir / "backup_file.py").write_text("# Backup file\nprint('backup')\n")

    # Create nested archive
    nested = tmp_path / "packages" / "archive" / "deprecated"
    nested.mkdir(parents=True)
    (nested / "deprecated.py").write_text("# Deprecated\nprint('deprecated')\n")

    return tmp_path


def test_exclude_patterns_archive(temp_project_with_archive):
    """Test excluding archive directories with glob patterns."""
    # Without exclusion
    scanner_all = ProjectScanner(temp_project_with_archive)
    result_all = scanner_all.scan()

    # With exclusion
    scanner_exclude = ProjectScanner(
        temp_project_with_archive,
        exclude_patterns=["*/archive/*", "**/archive/**"]
    )
    result_exclude = scanner_exclude.scan()

    # Should find fewer files when excluding archive
    assert len(result_all.python_files) > len(result_exclude.python_files)

    # Should not find old_node.py or deprecated.py in excluded results
    excluded_names = [pf.path.name for pf in result_exclude.python_files]
    assert "old_node.py" not in excluded_names
    assert "deprecated.py" not in excluded_names

    # Should still find main node
    assert "node.py" in excluded_names


def test_exclude_patterns_backup(temp_project_with_archive):
    """Test excluding backup directories."""
    scanner = ProjectScanner(
        temp_project_with_archive,
        exclude_patterns=["*backup*"]
    )
    result = scanner.scan()

    # Should not find backup files
    excluded_names = [pf.path.name for pf in result.python_files]
    assert "backup_file.py" not in excluded_names


def test_exclude_patterns_package_discovery(temp_project_with_archive):
    """Test that excluded patterns also exclude package discovery."""
    # Without exclusion - should find both packages
    scanner_all = ProjectScanner(temp_project_with_archive)
    packages_all = scanner_all.find_ros2_packages()
    assert "main_package" in packages_all
    assert "old_package" in packages_all

    # With exclusion - should only find main package
    scanner_exclude = ProjectScanner(
        temp_project_with_archive,
        exclude_patterns=["*/archive/*", "**/archive/**"]
    )
    packages_exclude = scanner_exclude.find_ros2_packages()
    assert "main_package" in packages_exclude
    assert "old_package" not in packages_exclude


def test_exclude_patterns_empty_list(temp_project):
    """Test that empty exclude patterns work normally."""
    scanner = ProjectScanner(temp_project, exclude_patterns=[])
    result = scanner.scan()

    assert len(result.python_files) == 2


def test_exclude_patterns_none(temp_project):
    """Test that None exclude patterns work normally."""
    scanner = ProjectScanner(temp_project, exclude_patterns=None)
    result = scanner.scan()

    assert len(result.python_files) == 2


def test_matches_exclude_pattern(temp_project_with_archive):
    """Test the matches_exclude_pattern method directly."""
    scanner = ProjectScanner(
        temp_project_with_archive,
        exclude_patterns=["*/archive/*"]
    )

    # Should match archive paths
    archive_path = temp_project_with_archive / "archive" / "old_package" / "old_node.py"
    assert scanner.matches_exclude_pattern(archive_path) is True

    # Should not match main paths
    main_path = temp_project_with_archive / "main_package" / "node.py"
    assert scanner.matches_exclude_pattern(main_path) is False


def test_exclude_multiple_patterns(temp_project_with_archive):
    """Test multiple exclusion patterns."""
    scanner = ProjectScanner(
        temp_project_with_archive,
        exclude_patterns=["*/archive/*", "**/archive/**", "*backup*"]
    )
    result = scanner.scan()

    # Should only find main node
    names = [pf.path.name for pf in result.python_files]
    assert len(names) == 1
    assert "node.py" in names
