"""
RoboMind Callback Complexity Analyzer

Analyzes callback function complexity to identify:
- High cyclomatic complexity
- Deep nesting
- Long functions
- Real-time violations
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robomind.ros2.node_extractor import ROS2NodeInfo


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a function."""
    name: str
    file_path: str
    line_number: int
    lines_of_code: int
    cyclomatic_complexity: int
    max_nesting_depth: int
    num_parameters: int
    num_returns: int
    has_exception_handling: bool


@dataclass
class ComplexityFinding:
    """A complexity-related finding."""
    finding_type: str
    severity: str
    function_name: str
    node_name: str
    file_path: str
    line_number: int
    metrics: ComplexityMetrics
    description: str
    recommendation: str


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate complexity metrics."""

    def __init__(self):
        self.complexity = 1  # Base complexity
        self.max_depth = 0
        self.current_depth = 0
        self.num_returns = 0
        self.has_exception = False

    def visit_If(self, node):
        self.complexity += 1
        self._visit_nested(node)

    def visit_For(self, node):
        self.complexity += 1
        self._visit_nested(node)

    def visit_While(self, node):
        self.complexity += 1
        self._visit_nested(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.has_exception = True
        self._visit_nested(node)

    def visit_With(self, node):
        self._visit_nested(node)

    def visit_BoolOp(self, node):
        # Each and/or adds to complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_Return(self, node):
        self.num_returns += 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def _visit_nested(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1


class ComplexityAnalyzer:
    """
    Analyzes callback complexity in ROS2 nodes.

    Thresholds (based on industry standards):
    - Cyclomatic complexity > 10: Complex, hard to test
    - Cyclomatic complexity > 20: Very complex, should refactor
    - Nesting depth > 4: Too deeply nested
    - Lines of code > 50: Consider splitting
    """

    # Complexity thresholds
    COMPLEXITY_WARNING = 10
    COMPLEXITY_HIGH = 20
    NESTING_WARNING = 4
    NESTING_HIGH = 6
    LOC_WARNING = 50
    LOC_HIGH = 100
    PARAMS_WARNING = 5

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.file_contents: Dict[str, str] = {}

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content with caching."""
        if not file_path:
            return None
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r') as f:
                    self.file_contents[file_path] = f.read()
            except Exception:
                return None
        return self.file_contents.get(file_path)

    def _analyze_function(self, func_node: ast.FunctionDef, file_path: str) -> ComplexityMetrics:
        """Analyze a single function for complexity metrics."""
        # Calculate lines of code
        if hasattr(func_node, 'end_lineno'):
            loc = func_node.end_lineno - func_node.lineno + 1
        else:
            loc = len(ast.unparse(func_node).split('\n')) if hasattr(ast, 'unparse') else 10

        # Calculate cyclomatic complexity using visitor
        visitor = ComplexityVisitor()
        visitor.visit(func_node)

        # Count parameters
        num_params = len(func_node.args.args)
        if func_node.args.vararg:
            num_params += 1
        if func_node.args.kwarg:
            num_params += 1

        return ComplexityMetrics(
            name=func_node.name,
            file_path=file_path,
            line_number=func_node.lineno,
            lines_of_code=loc,
            cyclomatic_complexity=visitor.complexity,
            max_nesting_depth=visitor.max_depth,
            num_parameters=num_params,
            num_returns=visitor.num_returns,
            has_exception_handling=visitor.has_exception,
        )

    def _find_callbacks(self, file_path: str) -> List[Tuple[str, ast.FunctionDef]]:
        """Find callback functions in a file."""
        content = self._get_file_content(file_path)
        if not content:
            return []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        callbacks = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a callback (common patterns)
                is_callback = (
                    node.name.endswith('_callback') or
                    node.name.startswith('callback_') or
                    node.name.startswith('on_') or
                    node.name.startswith('handle_') or
                    '_cb' in node.name or
                    'timer' in node.name.lower()
                )

                # Also check if it's referenced in create_subscription/create_timer
                if not is_callback:
                    # Simple heuristic: has 'msg' parameter and is in a Node class
                    for arg in node.args.args:
                        if arg.arg in ('msg', 'message', 'request', 'goal_handle'):
                            is_callback = True
                            break

                if is_callback:
                    callbacks.append((node.name, node))

        return callbacks

    def analyze(self) -> List[ComplexityFinding]:
        """Analyze all nodes for complexity issues."""
        findings = []

        for node in self.nodes:
            if not node.file_path:
                continue

            callbacks = self._find_callbacks(node.file_path)

            for func_name, func_node in callbacks:
                metrics = self._analyze_function(func_node, node.file_path)

                # Check cyclomatic complexity
                if metrics.cyclomatic_complexity >= self.COMPLEXITY_HIGH:
                    findings.append(ComplexityFinding(
                        finding_type='high_complexity',
                        severity='high',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback has very high cyclomatic complexity: {metrics.cyclomatic_complexity}',
                        recommendation='Split into smaller functions; aim for complexity < 10',
                    ))
                elif metrics.cyclomatic_complexity >= self.COMPLEXITY_WARNING:
                    findings.append(ComplexityFinding(
                        finding_type='moderate_complexity',
                        severity='medium',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback has high cyclomatic complexity: {metrics.cyclomatic_complexity}',
                        recommendation='Consider simplifying logic; high complexity reduces testability',
                    ))

                # Check nesting depth
                if metrics.max_nesting_depth >= self.NESTING_HIGH:
                    findings.append(ComplexityFinding(
                        finding_type='deep_nesting',
                        severity='high',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback has deep nesting: {metrics.max_nesting_depth} levels',
                        recommendation='Reduce nesting using early returns, guard clauses, or extract methods',
                    ))
                elif metrics.max_nesting_depth >= self.NESTING_WARNING:
                    findings.append(ComplexityFinding(
                        finding_type='moderate_nesting',
                        severity='medium',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback has significant nesting: {metrics.max_nesting_depth} levels',
                        recommendation='Consider flattening nested conditionals',
                    ))

                # Check lines of code
                if metrics.lines_of_code >= self.LOC_HIGH:
                    findings.append(ComplexityFinding(
                        finding_type='long_function',
                        severity='medium',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback is very long: {metrics.lines_of_code} lines',
                        recommendation='Split into smaller, focused functions',
                    ))
                elif metrics.lines_of_code >= self.LOC_WARNING:
                    findings.append(ComplexityFinding(
                        finding_type='long_function',
                        severity='low',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback is long: {metrics.lines_of_code} lines',
                        recommendation='Consider extracting helper functions',
                    ))

                # Check parameter count
                if metrics.num_parameters >= self.PARAMS_WARNING:
                    findings.append(ComplexityFinding(
                        finding_type='many_parameters',
                        severity='low',
                        function_name=func_name,
                        node_name=node.name,
                        file_path=node.file_path,
                        line_number=metrics.line_number,
                        metrics=metrics,
                        description=f'Callback has many parameters: {metrics.num_parameters}',
                        recommendation='Consider using a data class or config object',
                    ))

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 4))

        return findings


def analyze_complexity(nodes: List[ROS2NodeInfo]) -> List[ComplexityFinding]:
    """Convenience function for complexity analysis."""
    analyzer = ComplexityAnalyzer()
    analyzer.add_nodes(nodes)
    return analyzer.analyze()
