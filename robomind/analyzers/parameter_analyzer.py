"""
RoboMind Parameter Analyzer

Analyzes ROS2 parameters for:
- Missing validation/bounds checking
- Inconsistent parameter naming
- Undeclared parameters
- Type mismatches
- Duplicate parameters across nodes
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from robomind.ros2.node_extractor import ROS2NodeInfo, ParameterInfo
from robomind.ros2.param_extractor import ParameterCollection


@dataclass
class ParameterFinding:
    """A parameter-related finding."""
    finding_type: str
    severity: str
    parameter_name: str
    node_name: str
    file_path: str
    line_number: int
    description: str
    recommendation: str
    current_value: Optional[Any] = None
    suggested_value: Optional[Any] = None


class ParameterAnalyzer:
    """
    Analyzes ROS2 parameters for issues.
    """

    # Common parameters that should have bounds
    BOUNDED_PARAMS = {
        'max_velocity': (0.0, 10.0),
        'max_speed': (0.0, 10.0),
        'min_velocity': (-10.0, 0.0),
        'max_angular_velocity': (0.0, 6.28),
        'frequency': (0.1, 1000.0),
        'rate': (0.1, 1000.0),
        'timeout': (0.0, 300.0),
        'queue_size': (1, 1000),
        'buffer_size': (1, 10000),
    }

    # Parameter naming conventions
    NAMING_PATTERNS = [
        (r'^[A-Z]', 'Parameters should use snake_case, not PascalCase'),
        (r'-', 'Parameters should use underscores, not hyphens'),
        (r'\s', 'Parameters should not contain spaces'),
    ]

    def __init__(self):
        self.nodes: List[ROS2NodeInfo] = []
        self.param_collection: Optional[ParameterCollection] = None
        self.file_contents: Dict[str, str] = {}

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add nodes to analyze."""
        self.nodes = nodes

    def add_param_collection(self, params: ParameterCollection):
        """Add parameter collection from YAML files."""
        self.param_collection = params

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

    def _check_parameter_validation(self, node: ROS2NodeInfo, param: ParameterInfo) -> List[ParameterFinding]:
        """Check if a parameter has proper validation."""
        findings = []

        content = self._get_file_content(node.file_path)
        if not content:
            return findings

        # Look for the parameter declaration context
        param_name = param.name
        if not param_name:
            return findings

        # Check if there's range validation near the declaration
        lines = content.split('\n')
        start_line = max(0, param.line_number - 1)
        end_line = min(len(lines), param.line_number + 20)
        context = '\n'.join(lines[start_line:end_line])

        # Check for common validation patterns
        has_range_check = any(pattern in context for pattern in [
            'ParameterDescriptor',
            'FloatingPointRange',
            'IntegerRange',
            f'if {param_name}',
            f'assert {param_name}',
            f'{param_name} <',
            f'{param_name} >',
            f'{param_name} <=',
            f'{param_name} >=',
            f'min({param_name}',
            f'max({param_name}',
            f'clamp({param_name}',
        ])

        # Check if this parameter typically needs bounds
        for bounded_name, (min_val, max_val) in self.BOUNDED_PARAMS.items():
            if bounded_name in param_name.lower():
                if not has_range_check:
                    findings.append(ParameterFinding(
                        finding_type='missing_validation',
                        severity='medium',
                        parameter_name=param_name,
                        node_name=node.name,
                        file_path=node.file_path or '',
                        line_number=param.line_number,
                        description=f'Parameter "{param_name}" should have range validation',
                        recommendation=f'Add bounds checking: {min_val} to {max_val}',
                        suggested_value=f'FloatingPointRange(from_value={min_val}, to_value={max_val})',
                    ))
                break

        return findings

    def _check_naming_convention(self, node: ROS2NodeInfo, param: ParameterInfo) -> List[ParameterFinding]:
        """Check parameter naming conventions."""
        findings = []

        param_name = param.name
        if not param_name:
            return findings

        for pattern, message in self.NAMING_PATTERNS:
            if re.search(pattern, param_name):
                findings.append(ParameterFinding(
                    finding_type='naming_convention',
                    severity='low',
                    parameter_name=param_name,
                    node_name=node.name,
                    file_path=node.file_path or '',
                    line_number=param.line_number,
                    description=f'Parameter "{param_name}": {message}',
                    recommendation='Use lowercase snake_case for parameter names',
                ))
                break

        return findings

    def _check_duplicate_parameters(self) -> List[ParameterFinding]:
        """Check for parameters with same name but different defaults across nodes."""
        findings = []

        # Build param_name -> [(node, default_value, line)] map
        param_map: Dict[str, List[Tuple[str, Any, str, int]]] = {}

        for node in self.nodes:
            for param in node.parameters:
                if param.name:
                    if param.name not in param_map:
                        param_map[param.name] = []
                    param_map[param.name].append((
                        node.name,
                        param.default_value,
                        node.file_path or '',
                        param.line_number,
                    ))

        # Find params with inconsistent defaults
        for param_name, usages in param_map.items():
            if len(usages) > 1:
                # Get unique default values (excluding None)
                defaults = set()
                for _, default, _, _ in usages:
                    if default is not None:
                        defaults.add(str(default))

                if len(defaults) > 1:
                    nodes = [u[0] for u in usages]
                    findings.append(ParameterFinding(
                        finding_type='inconsistent_defaults',
                        severity='low',
                        parameter_name=param_name,
                        node_name=', '.join(nodes),
                        file_path=usages[0][2],
                        line_number=usages[0][3],
                        description=f'Parameter "{param_name}" has different defaults across nodes: {defaults}',
                        recommendation='Consider centralizing parameter defaults in a config file',
                    ))

        return findings

    def _check_undeclared_usage(self, node: ROS2NodeInfo) -> List[ParameterFinding]:
        """Check for parameters used without declaration."""
        findings = []

        content = self._get_file_content(node.file_path)
        if not content:
            return findings

        # Get declared parameters
        declared = {p.name for p in node.parameters if p.name}

        # Look for get_parameter calls
        get_pattern = r'\.get_parameter\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(get_pattern, content):
            param_name = match.group(1)
            if param_name not in declared:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1

                findings.append(ParameterFinding(
                    finding_type='undeclared_parameter',
                    severity='medium',
                    parameter_name=param_name,
                    node_name=node.name,
                    file_path=node.file_path or '',
                    line_number=line_num,
                    description=f'Parameter "{param_name}" used but not declared',
                    recommendation='Add declare_parameter() before using get_parameter()',
                ))

        return findings

    def analyze(self) -> List[ParameterFinding]:
        """Run all parameter analyses."""
        findings = []

        for node in self.nodes:
            for param in node.parameters:
                findings.extend(self._check_parameter_validation(node, param))
                findings.extend(self._check_naming_convention(node, param))

            findings.extend(self._check_undeclared_usage(node))

        findings.extend(self._check_duplicate_parameters())

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        findings.sort(key=lambda f: severity_order.get(f.severity, 4))

        return findings


def analyze_parameters(nodes: List[ROS2NodeInfo]) -> List[ParameterFinding]:
    """Convenience function for parameter analysis."""
    analyzer = ParameterAnalyzer()
    analyzer.add_nodes(nodes)
    return analyzer.analyze()
