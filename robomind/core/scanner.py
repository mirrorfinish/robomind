"""
Project Scanner for RoboMind

Scans project directories to discover Python files, ROS2 packages,
launch files, and configuration files.

Designed for robotics projects but works on any Python codebase.
"""

import os
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProjectFile:
    """Represents a single file in the project."""

    path: Path
    relative_path: Path
    file_type: str  # 'python', 'launch', 'config', 'other'
    size_bytes: int
    package_name: Optional[str] = None
    lines_of_code: int = 0
    last_modified: Optional[str] = None


@dataclass
class ScanResult:
    """Results from scanning a project directory."""

    root_path: Path
    python_files: List[ProjectFile] = field(default_factory=list)
    launch_files: List[ProjectFile] = field(default_factory=list)
    config_files: List[ProjectFile] = field(default_factory=list)
    packages: Dict[str, Path] = field(default_factory=dict)  # package_name -> path
    total_files: int = 0
    total_lines: int = 0

    def get_files_by_package(self, package_name: str) -> List[ProjectFile]:
        """Get all files belonging to a specific ROS2 package."""
        return [f for f in self.python_files if f.package_name == package_name]

    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "root": str(self.root_path),
            "packages": len(self.packages),
            "python_files": len(self.python_files),
            "launch_files": len(self.launch_files),
            "config_files": len(self.config_files),
            "total_files": self.total_files,
            "total_lines": self.total_lines,
        }


class ProjectScanner:
    """
    Scan a project directory to discover all relevant files.

    Features:
    - Recursive directory traversal
    - ROS2 package detection (via package.xml)
    - Python file discovery
    - Launch file discovery (.launch.py, .launch.xml)
    - Config file discovery (.yaml, .yml, .json)
    - Exclusion patterns (pycache, .git, build, etc.)

    Usage:
        scanner = ProjectScanner('/path/to/robot_project')
        result = scanner.scan()
        print(result.summary())
    """

    # Directories to always ignore
    IGNORE_DIRS = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "venv",
        "env",
        ".venv",
        ".env",
        "build",
        "install",
        "log",
        "dist",
        ".tox",
        ".eggs",
    }

    # File patterns to ignore
    IGNORE_PATTERNS = {
        "*.pyc",
        "*.pyo",
        "*.egg-info",
    }

    def __init__(
        self,
        root_path: Path,
        exclude_dirs: Optional[Set[str]] = None,
        exclude_files: Optional[Set[str]] = None,
        include_tests: bool = True,
    ):
        """
        Initialize project scanner.

        Args:
            root_path: Root directory to scan
            exclude_dirs: Additional directories to exclude
            exclude_files: Additional file patterns to exclude
            include_tests: Whether to include test files
        """
        self.root_path = Path(root_path).resolve()

        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {root_path}")

        if not self.root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_path}")

        # Build exclusion sets
        self.exclude_dirs = self.IGNORE_DIRS.copy()
        if exclude_dirs:
            self.exclude_dirs.update(exclude_dirs)

        if not include_tests:
            self.exclude_dirs.add("tests")
            self.exclude_dirs.add("test")

        self.exclude_files = exclude_files or set()

        logger.info(f"ProjectScanner initialized for: {self.root_path}")

    def should_ignore_dir(self, dir_name: str) -> bool:
        """Check if a directory should be ignored."""
        return dir_name in self.exclude_dirs or dir_name.startswith(".")

    def should_ignore_file(self, file_name: str) -> bool:
        """Check if a file should be ignored."""
        if file_name in self.exclude_files:
            return True

        for pattern in self.IGNORE_PATTERNS:
            if pattern.startswith("*"):
                if file_name.endswith(pattern[1:]):
                    return True
            elif file_name == pattern:
                return True

        return False

    def find_ros2_packages(self) -> Dict[str, Path]:
        """
        Find all ROS2 packages by looking for package.xml files.

        Returns:
            Dict mapping package_name -> package_path
        """
        packages = {}

        for package_xml in self.root_path.rglob("package.xml"):
            # Skip if in excluded directory
            if any(part in self.exclude_dirs for part in package_xml.parts):
                continue

            package_path = package_xml.parent
            package_name = self._extract_package_name(package_xml)

            if package_name:
                packages[package_name] = package_path
                logger.debug(f"Found ROS2 package: {package_name} at {package_path}")

        return packages

    def _extract_package_name(self, package_xml: Path) -> Optional[str]:
        """Extract package name from package.xml."""
        try:
            tree = ET.parse(package_xml)
            root = tree.getroot()
            name_elem = root.find("name")
            if name_elem is not None and name_elem.text:
                return name_elem.text.strip()
        except Exception as e:
            logger.warning(f"Failed to parse {package_xml}: {e}")
        return None

    def _find_package_for_file(
        self, file_path: Path, packages: Dict[str, Path]
    ) -> Optional[str]:
        """Determine which ROS2 package a file belongs to."""
        for package_name, package_path in packages.items():
            try:
                file_path.relative_to(package_path)
                return package_name
            except ValueError:
                continue
        return None

    def _count_lines(self, file_path: Path) -> int:
        """Count lines of code in a file (excluding blank lines)."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def scan(self) -> ScanResult:
        """
        Perform complete scan of project directory.

        Returns:
            ScanResult containing all discovered files and metadata
        """
        logger.info(f"Scanning project at {self.root_path}")

        result = ScanResult(root_path=self.root_path)

        # Step 1: Find all ROS2 packages
        result.packages = self.find_ros2_packages()
        logger.info(f"Found {len(result.packages)} ROS2 packages")

        total_lines = 0

        # Step 2: Walk directory tree
        for root, dirs, files in os.walk(self.root_path):
            # Filter out excluded directories (modify in-place for os.walk)
            dirs[:] = [d for d in dirs if not self.should_ignore_dir(d)]

            root_path = Path(root)

            for file_name in files:
                if self.should_ignore_file(file_name):
                    continue

                file_path = root_path / file_name

                try:
                    stats = file_path.stat()
                    size_bytes = stats.st_size
                    last_modified = datetime.fromtimestamp(stats.st_mtime).isoformat()
                except Exception:
                    size_bytes = 0
                    last_modified = None

                relative_path = file_path.relative_to(self.root_path)
                package_name = self._find_package_for_file(file_path, result.packages)

                # Categorize by file type
                # Check launch files FIRST (before .py check) since .launch.py ends with .py
                if file_name.endswith((".launch.py", ".launch.xml", ".launch")):
                    pf = ProjectFile(
                        path=file_path,
                        relative_path=relative_path,
                        file_type="launch",
                        size_bytes=size_bytes,
                        package_name=package_name,
                        last_modified=last_modified,
                    )
                    result.launch_files.append(pf)
                    result.total_files += 1

                elif file_name.endswith(".py"):
                    loc = self._count_lines(file_path)
                    total_lines += loc

                    pf = ProjectFile(
                        path=file_path,
                        relative_path=relative_path,
                        file_type="python",
                        size_bytes=size_bytes,
                        package_name=package_name,
                        lines_of_code=loc,
                        last_modified=last_modified,
                    )
                    result.python_files.append(pf)
                    result.total_files += 1

                elif file_name.endswith((".yaml", ".yml", ".json")):
                    pf = ProjectFile(
                        path=file_path,
                        relative_path=relative_path,
                        file_type="config",
                        size_bytes=size_bytes,
                        package_name=package_name,
                        last_modified=last_modified,
                    )
                    result.config_files.append(pf)
                    result.total_files += 1

        result.total_lines = total_lines

        # Sort files by path for consistent output
        result.python_files.sort(key=lambda f: f.relative_path)
        result.launch_files.sort(key=lambda f: f.relative_path)
        result.config_files.sort(key=lambda f: f.relative_path)

        logger.info(f"Scan complete: {result.summary()}")

        return result


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python scanner.py <project_path>")
        sys.exit(1)

    scanner = ProjectScanner(sys.argv[1])
    result = scanner.scan()

    print("\n" + "=" * 60)
    print("SCAN RESULTS")
    print("=" * 60)
    print(json.dumps(result.summary(), indent=2))

    print("\nPackages found:")
    for pkg_name, pkg_path in sorted(result.packages.items()):
        files = result.get_files_by_package(pkg_name)
        print(f"  {pkg_name}: {len(files)} Python files")
