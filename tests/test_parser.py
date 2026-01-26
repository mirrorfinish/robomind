"""Tests for RoboMind Python Parser."""

import pytest
from pathlib import Path
import tempfile

from robomind.core.parser import (
    PythonParser,
    ParseResult,
    ClassInfo,
    FunctionInfo,
    ImportInfo,
)


@pytest.fixture
def sample_node_path():
    """Path to sample ROS2 node."""
    return Path(__file__).parent / "fixtures" / "sample_ros2_node.py"


@pytest.fixture
def simple_class_code():
    """Simple Python code with a class."""
    return '''
"""Module docstring."""

import os
from pathlib import Path

CONSTANT = 42

class MyClass:
    """A simple class."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        """Get the value."""
        return self.value

    def set_value(self, new_value):
        self.value = new_value


def standalone_function(x, y):
    """Add two numbers."""
    return x + y
'''


def test_parser_init():
    """Test parser initialization."""
    parser = PythonParser()
    assert parser is not None


def test_parse_sample_node(sample_node_path):
    """Test parsing sample ROS2 node."""
    parser = PythonParser()
    result = parser.parse_file(sample_node_path)

    assert result.parse_error is None
    assert len(result.imports) > 0
    assert len(result.classes) == 1
    assert len(result.functions) == 1  # main()

    # Check class
    cls = result.get_class("SampleNode")
    assert cls is not None
    assert cls.name == "SampleNode"
    assert "Node" in cls.bases
    assert len(cls.methods) >= 3  # __init__, command_callback, timer_callback


def test_parse_imports(sample_node_path):
    """Test import extraction."""
    parser = PythonParser()
    result = parser.parse_file(sample_node_path)

    # Check for rclpy import
    rclpy_imports = result.get_imports_from_module("rclpy")
    assert len(rclpy_imports) > 0

    # Check ROS2 detection
    assert result.has_ros2_imports()


def test_parse_class_methods(sample_node_path):
    """Test method extraction from class."""
    parser = PythonParser()
    result = parser.parse_file(sample_node_path)

    cls = result.get_class("SampleNode")

    # Find __init__ method
    init_method = cls.get_method("__init__")
    assert init_method is not None
    assert init_method.is_method
    assert "self" in init_method.args

    # Find callback methods
    cmd_callback = cls.get_method("command_callback")
    assert cmd_callback is not None
    assert "msg" in cmd_callback.args


def test_parse_simple_class(simple_class_code):
    """Test parsing simple class code."""
    parser = PythonParser()

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(simple_class_code)
        temp_path = Path(f.name)

    try:
        result = parser.parse_file(temp_path)

        # Check module docstring
        assert result.module_docstring == "Module docstring."

        # Check imports
        assert len(result.imports) == 2

        # Check class
        assert len(result.classes) == 1
        cls = result.get_class("MyClass")
        assert cls is not None
        assert cls.docstring == "A simple class."
        assert len(cls.methods) == 3

        # Check standalone function
        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "standalone_function"
        assert func.args == ["x", "y"]
        assert func.docstring == "Add two numbers."

        # Check global variable
        assert "CONSTANT" in result.global_variables
        assert result.global_variables["CONSTANT"] == 42

    finally:
        temp_path.unlink()


def test_parse_syntax_error():
    """Test handling of syntax errors."""
    parser = PythonParser()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def broken(\n")  # Syntax error
        temp_path = Path(f.name)

    try:
        result = parser.parse_file(temp_path)
        assert result.parse_error is not None
        assert "SyntaxError" in result.parse_error
    finally:
        temp_path.unlink()


def test_extract_function_calls(simple_class_code):
    """Test extraction of function calls."""
    parser = PythonParser()

    code_with_calls = '''
def outer():
    inner()
    obj.method()
    module.func()
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code_with_calls)
        temp_path = Path(f.name)

    try:
        result = parser.parse_file(temp_path)
        func = result.functions[0]

        # Should detect calls
        assert "inner" in func.calls
        assert "method" in func.calls
        assert "func" in func.calls

    finally:
        temp_path.unlink()


def test_summary(sample_node_path):
    """Test summary generation."""
    parser = PythonParser()
    result = parser.parse_file(sample_node_path)

    summary = result.summary()

    assert "imports" in summary
    assert "classes" in summary
    assert "functions" in summary
    assert "has_ros2" in summary
    assert summary["has_ros2"] is True


def test_decorators():
    """Test decorator extraction."""
    parser = PythonParser()

    code = '''
@decorator
class MyClass:
    @staticmethod
    def static_method():
        pass

    @property
    def my_property(self):
        return 42
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = Path(f.name)

    try:
        result = parser.parse_file(temp_path)

        cls = result.get_class("MyClass")
        assert "decorator" in cls.decorators

        static_method = cls.get_method("static_method")
        assert "staticmethod" in static_method.decorators

        prop = cls.get_method("my_property")
        assert "property" in prop.decorators

    finally:
        temp_path.unlink()
