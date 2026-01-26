"""
Python AST Parser for RoboMind

Parses Python source files to extract structural information:
- Classes and their methods
- Functions and their signatures
- Imports and dependencies
- Global variables
- Decorators

Designed to support ROS2 pattern detection in Day 2.
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    line_number: int
    end_line: int
    args: List[str]
    returns: Optional[str] = None
    is_method: bool = False
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)  # Functions called within


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    line_number: int
    end_line: int
    bases: List[str]
    methods: List[FunctionInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None

    def get_method(self, name: str) -> Optional[FunctionInfo]:
        """Get a method by name."""
        for method in self.methods:
            if method.name == name:
                return method
        return None


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    names: List[str] = field(default_factory=list)  # What's imported
    alias: Optional[str] = None
    is_from_import: bool = False
    line_number: int = 0


@dataclass
class ParseResult:
    """Results from parsing a Python file."""

    file_path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    global_variables: Dict[str, Any] = field(default_factory=dict)
    module_docstring: Optional[str] = None
    parse_error: Optional[str] = None
    lines_of_code: int = 0
    package_name: Optional[str] = None  # Set by caller

    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Get a class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None

    def get_imports_from_module(self, module: str) -> List[ImportInfo]:
        """Get all imports from a specific module (prefix match)."""
        return [imp for imp in self.imports if imp.module.startswith(module)]

    def has_ros2_imports(self) -> bool:
        """Check if file imports ROS2 libraries."""
        ros2_modules = {"rclpy", "std_msgs", "geometry_msgs", "sensor_msgs", "nav_msgs"}
        for imp in self.imports:
            module_base = imp.module.split(".")[0]
            if module_base in ros2_modules:
                return True
        return False

    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "file": str(self.file_path),
            "imports": len(self.imports),
            "classes": len(self.classes),
            "functions": len(self.functions),
            "global_variables": len(self.global_variables),
            "lines_of_code": self.lines_of_code,
            "has_ros2": self.has_ros2_imports(),
        }


class PythonParser:
    """
    Parse Python source files using AST.

    Extracts:
    - Module docstring
    - Imports (import X, from X import Y)
    - Class definitions with methods
    - Function definitions
    - Global variables (top-level assignments)
    - Decorators

    Usage:
        parser = PythonParser()
        result = parser.parse_file('/path/to/file.py')
        for cls in result.classes:
            print(f"Class: {cls.name}, bases: {cls.bases}")
    """

    def __init__(self):
        pass

    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse a Python file and extract structural information.

        Args:
            file_path: Path to Python file

        Returns:
            ParseResult with extracted information
        """
        file_path = Path(file_path)
        result = ParseResult(file_path=file_path)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()

            # Count lines
            result.lines_of_code = len([l for l in source.split("\n") if l.strip()])

            # Parse AST
            tree = ast.parse(source, filename=str(file_path))

            # Extract module docstring
            result.module_docstring = ast.get_docstring(tree)

            # Process top-level nodes
            for node in tree.body:
                if isinstance(node, ast.Import):
                    result.imports.extend(self._extract_import(node))

                elif isinstance(node, ast.ImportFrom):
                    result.imports.extend(self._extract_from_import(node))

                elif isinstance(node, ast.ClassDef):
                    result.classes.append(self._extract_class(node, source))

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result.functions.append(self._extract_function(node, source))

                elif isinstance(node, ast.Assign):
                    self._extract_global_variable(node, result.global_variables)

            logger.debug(f"Parsed {file_path}: {result.summary()}")

        except SyntaxError as e:
            result.parse_error = f"SyntaxError: {e}"
            logger.warning(f"Syntax error in {file_path}: {e}")

        except Exception as e:
            result.parse_error = f"ParseError: {e}"
            logger.warning(f"Error parsing {file_path}: {e}")

        return result

    def _extract_import(self, node: ast.Import) -> List[ImportInfo]:
        """Extract from 'import X' statement."""
        imports = []
        for alias in node.names:
            imports.append(
                ImportInfo(
                    module=alias.name,
                    alias=alias.asname,
                    is_from_import=False,
                    line_number=node.lineno,
                )
            )
        return imports

    def _extract_from_import(self, node: ast.ImportFrom) -> List[ImportInfo]:
        """Extract from 'from X import Y' statement."""
        module = node.module or ""
        names = [alias.name for alias in node.names]

        return [
            ImportInfo(
                module=module,
                names=names,
                is_from_import=True,
                line_number=node.lineno,
            )
        ]

    def _extract_class(self, node: ast.ClassDef, source: str) -> ClassInfo:
        """Extract class information."""
        # Extract base classes
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append(str(type(base).__name__))

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function(item, source)
                func_info.is_method = True
                methods.append(func_info)

        # Extract decorators
        decorators = self._extract_decorators(node.decorator_list)

        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            bases=bases,
            methods=methods,
            decorators=decorators,
            docstring=ast.get_docstring(node),
        )

    def _extract_function(self, node: ast.FunctionDef, source: str) -> FunctionInfo:
        """Extract function information."""
        # Extract arguments
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # Extract return type annotation
        returns = None
        if node.returns:
            try:
                returns = ast.unparse(node.returns)
            except Exception:
                pass

        # Extract decorators
        decorators = self._extract_decorators(node.decorator_list)

        # Extract function calls within this function
        calls = self._extract_calls(node)

        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            args=args,
            returns=returns,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,
            docstring=ast.get_docstring(node),
            calls=calls,
        )

    def _extract_decorators(self, decorator_list: List[ast.expr]) -> List[str]:
        """Extract decorator names."""
        decorators = []
        for dec in decorator_list:
            try:
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Name):
                        decorators.append(dec.func.id)
                    elif isinstance(dec.func, ast.Attribute):
                        decorators.append(ast.unparse(dec.func))
                elif isinstance(dec, ast.Attribute):
                    decorators.append(ast.unparse(dec))
            except Exception:
                decorators.append("unknown")
        return decorators

    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function/method calls within a node."""
        calls = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    # Simple call: func()
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # Method call: obj.method() or self.method()
                    calls.add(child.func.attr)

        return list(calls)

    def _extract_global_variable(
        self, node: ast.Assign, variables: Dict[str, Any]
    ) -> None:
        """Extract global variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id

                # Skip private/dunder names
                if name.startswith("_"):
                    continue

                # Try to get literal value
                try:
                    value = ast.literal_eval(node.value)
                    variables[name] = value
                except (ValueError, TypeError, RecursionError):
                    # Can't evaluate - just note the variable exists
                    variables[name] = None


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python parser.py <python_file>")
        sys.exit(1)

    parser = PythonParser()
    result = parser.parse_file(Path(sys.argv[1]))

    print("\n" + "=" * 60)
    print("PARSE RESULTS")
    print("=" * 60)
    print(json.dumps(result.summary(), indent=2))

    if result.classes:
        print(f"\nClasses ({len(result.classes)}):")
        for cls in result.classes:
            print(f"  {cls.name} (extends: {', '.join(cls.bases) or 'None'})")
            for method in cls.methods:
                print(f"    - {method.name}({', '.join(method.args)})")

    if result.functions:
        print(f"\nFunctions ({len(result.functions)}):")
        for func in result.functions:
            print(f"  {func.name}({', '.join(func.args)})")

    if result.has_ros2_imports():
        print("\n[ROS2 imports detected]")
        ros2_imports = result.get_imports_from_module("rclpy")
        ros2_imports.extend(result.get_imports_from_module("std_msgs"))
        ros2_imports.extend(result.get_imports_from_module("geometry_msgs"))
        for imp in ros2_imports:
            print(f"  {imp.module}: {', '.join(imp.names) if imp.names else '*'}")
