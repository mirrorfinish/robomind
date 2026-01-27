"""
Project Scanner for RoboMind

Scans project directories to discover Python files, ROS2 packages,
launch files, and configuration files.

Designed for robotics projects but works on any Python codebase.
"""

import os
import fnmatch
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Default exclude patterns for dead/archived code (applied unless --no-default-excludes)
DEFAULT_EXCLUDE_PATTERNS = [
    # Archive and backup directories
    "**/archive/**",
    "**/backup/**",
    "**/old/**",
    "**/deprecated/**",
    "**/obsolete/**",
    "**/unused/**",
    "**/legacy/**",
    # Deprecated file suffixes
    "**/*_old.py",
    "**/*_backup.py",
    "**/*_deprecated.py",
    "**/*_legacy.py",
    "**/*_unused.py",
    "**/*_bak.py",
    "**/*.bak",
    "**/*.orig",
    # Test directories (usually not production code)
    "**/test/**",
    "**/tests/**",
    "**/testing/**",
    # Examples and docs
    "**/examples/**",
    "**/example/**",
    "**/docs/**",
    "**/documentation/**",
    # Build artifacts
    "**/__pycache__/**",
    "**/build/**",
    "**/install/**",
    "**/log/**",
    "**/logs/**",
    "**/.git/**",
    "**/.venv/**",
    "**/venv/**",
    "**/*.egg-info/**",
    "**/dist/**",
    # IDE and temp
    "**/.idea/**",
    "**/.vscode/**",
    "**/*.pyc",
]


# Patterns that indicate a file/directory is likely dead code
DEAD_CODE_INDICATORS = {
    # Directory names (strong indicators)
    "archive": -0.4,
    "backup": -0.4,
    "old": -0.3,
    "deprecated": -0.4,
    "obsolete": -0.4,
    "unused": -0.4,
    "legacy": -0.3,
    "test": -0.2,
    "tests": -0.2,
    "examples": -0.2,
    "example": -0.2,
}

# File name patterns that indicate dead code
DEAD_CODE_FILE_PATTERNS = {
    "_old.py": -0.3,
    "_backup.py": -0.3,
    "_deprecated.py": -0.4,
    "_legacy.py": -0.3,
    "_unused.py": -0.4,
    "_bak.py": -0.3,
    "_v1.py": -0.1,  # Weak indicator
    "_v2.py": -0.1,
}


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
    # Confidence-related fields
    location_confidence: float = 1.0  # Adjusted based on directory location
    dead_code_indicators: List[str] = field(default_factory=list)

    def get_location_confidence_penalty(self) -> float:
        """Calculate confidence penalty based on file location."""
        penalty = 0.0
        path_str = str(self.relative_path).lower()
        path_parts = path_str.split("/")

        # Check directory names
        for part in path_parts[:-1]:  # Exclude filename
            if part in DEAD_CODE_INDICATORS:
                penalty += DEAD_CODE_INDICATORS[part]

        # Check filename patterns
        filename = path_parts[-1] if path_parts else ""
        for pattern, pen in DEAD_CODE_FILE_PATTERNS.items():
            if filename.endswith(pattern):
                penalty += pen
                break

        return max(-0.8, penalty)  # Cap at -0.8 (don't go below 0.2 base confidence)


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
        exclude_patterns: Optional[List[str]] = None,
        include_tests: bool = True,
        use_default_excludes: bool = False,
    ):
        """
        Initialize project scanner.

        Args:
            root_path: Root directory to scan
            exclude_dirs: Additional directories to exclude
            exclude_files: Additional file patterns to exclude
            exclude_patterns: Glob patterns for paths to exclude (e.g., "*/archive/*")
            include_tests: Whether to include test files
            use_default_excludes: Whether to apply DEFAULT_EXCLUDE_PATTERNS automatically
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

        # Glob patterns for path exclusion (e.g., "*/archive/*", "**/backup/**")
        self.exclude_patterns = list(exclude_patterns) if exclude_patterns else []

        # Apply default excludes if enabled
        self.use_default_excludes = use_default_excludes
        if use_default_excludes:
            self.exclude_patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

        logger.info(f"ProjectScanner initialized for: {self.root_path}")
        if use_default_excludes:
            logger.info(f"  Using default excludes (archive, backup, test, etc.)")
        if exclude_patterns:
            logger.info(f"  Additional exclude patterns: {exclude_patterns}")

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

    def matches_exclude_pattern(self, path: Path) -> bool:
        """
        Check if a path matches any exclusion glob pattern.

        Supports patterns like:
        - "archive" - exclude any path containing "archive" as a component
        - "*archive*" - exclude paths containing "archive" anywhere
        - "*/archive/*" - exclude paths with archive directory

        Args:
            path: Path to check (relative or absolute)

        Returns:
            True if path matches any exclude pattern
        """
        if not self.exclude_patterns:
            return False

        # Get path relative to root for matching
        try:
            rel_path = path.relative_to(self.root_path)
        except ValueError:
            rel_path = path

        # Convert to string with forward slashes for consistent matching
        path_str = str(rel_path).replace("\\", "/")
        path_parts = path_str.split("/")

        for pattern in self.exclude_patterns:
            # Normalize pattern slashes
            norm_pattern = pattern.replace("\\", "/")

            # Direct fnmatch on full path
            if fnmatch.fnmatch(path_str, norm_pattern):
                return True

            # For patterns like "*/archive/*" or "**/archive/**"
            # Extract the core component name and check if it's in the path
            core_pattern = norm_pattern.strip("*/")
            if "/" not in core_pattern:
                # Simple pattern like "archive" or "*archive*"
                # Check if any path component matches
                for part in path_parts:
                    if fnmatch.fnmatch(part, core_pattern):
                        return True
                    # Also check with wildcards
                    if fnmatch.fnmatch(part, f"*{core_pattern}*"):
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

            # Skip if matches exclusion pattern
            if self.matches_exclude_pattern(package_xml):
                logger.debug(f"Skipping package (excluded pattern): {package_xml}")
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

    def _get_dead_code_indicators(self, relative_path: Path) -> List[str]:
        """Get list of dead code indicators found in the file path."""
        indicators = []
        path_str = str(relative_path).lower()
        path_parts = path_str.split("/")

        # Check directory names
        for part in path_parts[:-1]:
            if part in DEAD_CODE_INDICATORS:
                indicators.append(f"directory:{part}")

        # Check filename patterns
        filename = path_parts[-1] if path_parts else ""
        for pattern in DEAD_CODE_FILE_PATTERNS:
            if filename.endswith(pattern):
                indicators.append(f"filename:{pattern}")
                break

        return indicators

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
            root_path = Path(root)

            # Filter out excluded directories (modify in-place for os.walk)
            dirs[:] = [d for d in dirs if not self.should_ignore_dir(d)]

            # Also filter directories matching exclusion patterns
            if self.exclude_patterns:
                dirs[:] = [d for d in dirs if not self.matches_exclude_pattern(root_path / d)]

            for file_name in files:
                if self.should_ignore_file(file_name):
                    continue

                file_path = root_path / file_name

                # Skip if matches exclusion pattern
                if self.matches_exclude_pattern(file_path):
                    continue

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
                    # Calculate location-based confidence penalty
                    penalty = pf.get_location_confidence_penalty()
                    pf.location_confidence = max(0.2, 1.0 + penalty)

                    # Track dead code indicators for this file
                    pf.dead_code_indicators = self._get_dead_code_indicators(relative_path)

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
