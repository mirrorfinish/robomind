"""
HTTP Endpoint Extractor - Detect Flask/FastAPI/aiohttp server endpoints.

This module detects HTTP server endpoints in Python code by parsing:
- Flask: @app.route(), @app.get(), @app.post(), etc.
- FastAPI: @app.get(), @router.post(), app.add_api_route()
- aiohttp: web.get(), web.post(), app.router.add_route()
- Starlette: Route(), Mount()

Example detections:
```python
@app.route('/api/detections')      # Flask
@app.get('/health')                 # FastAPI
@router.post('/api/speak')          # FastAPI router
app.add_api_route('/api/tts/speak', handler)  # FastAPI
```
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class HTTPEndpoint:
    """A detected HTTP server endpoint."""
    path: str
    method: str  # GET, POST, PUT, DELETE, etc.
    file_path: Path
    line_number: int
    handler_name: Optional[str] = None
    framework: str = "unknown"  # flask, fastapi, aiohttp, starlette
    inferred_host: Optional[str] = None
    parameters: List[str] = field(default_factory=list)  # Path parameters like {id}
    response_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "handler_name": self.handler_name,
            "framework": self.framework,
            "inferred_host": self.inferred_host,
            "parameters": self.parameters,
            "response_type": self.response_type,
            "tags": self.tags,
        }

    def get_path_params(self) -> List[str]:
        """Extract path parameters like {id} or <id>."""
        params = []
        # FastAPI/Starlette style: {param}
        params.extend(re.findall(r'\{(\w+)\}', self.path))
        # Flask style: <param> or <type:param>
        params.extend(re.findall(r'<(?:\w+:)?(\w+)>', self.path))
        return params


class HTTPEndpointExtractor:
    """
    Extract HTTP endpoints from Python source files.

    Supports Flask, FastAPI, aiohttp, and Starlette frameworks.

    Usage:
        extractor = HTTPEndpointExtractor()
        endpoints = extractor.extract_from_file(Path("server.py"))
        for ep in endpoints:
            print(f"{ep.method} {ep.path}")
    """

    # HTTP methods
    HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}

    # Framework-specific patterns
    FLASK_DECORATORS = {"route", "get", "post", "put", "delete", "patch"}
    FASTAPI_DECORATORS = {"get", "post", "put", "delete", "patch", "head", "options", "api_route"}
    AIOHTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}

    def __init__(self):
        """Initialize endpoint extractor."""
        self._current_file: Optional[Path] = None
        self._framework: str = "unknown"

    def extract_from_file(self, file_path: Path) -> List[HTTPEndpoint]:
        """
        Extract HTTP endpoints from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of HTTPEndpoint objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        self._current_file = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Detect framework from imports
            self._framework = self._detect_framework(content)

            tree = ast.parse(content, filename=str(file_path))
            endpoints = []

            # Walk the AST looking for decorated functions and method calls
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    endpoints.extend(self._extract_from_function(node))
                elif isinstance(node, ast.Call):
                    endpoint = self._extract_from_call(node)
                    if endpoint:
                        endpoints.append(endpoint)

            return endpoints

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return []

    def _detect_framework(self, content: str) -> str:
        """Detect the HTTP framework from imports."""
        if "from fastapi" in content or "import fastapi" in content:
            return "fastapi"
        elif "from flask" in content or "import flask" in content:
            return "flask"
        elif "from aiohttp" in content or "import aiohttp" in content:
            return "aiohttp"
        elif "from starlette" in content or "import starlette" in content:
            return "starlette"
        return "unknown"

    def _extract_from_function(self, node) -> List[HTTPEndpoint]:
        """Extract endpoints from decorated functions."""
        endpoints = []

        for decorator in node.decorator_list:
            endpoint = self._parse_decorator(decorator, node.name, node.lineno)
            if endpoint:
                endpoints.append(endpoint)

        return endpoints

    def _parse_decorator(
        self,
        decorator: ast.AST,
        handler_name: str,
        line_number: int,
    ) -> Optional[HTTPEndpoint]:
        """Parse a decorator to extract endpoint info."""
        if isinstance(decorator, ast.Call):
            func = decorator.func

            # Get the decorator name and method
            if isinstance(func, ast.Attribute):
                method_name = func.attr.lower()
                # Check if it's a route decorator
                if method_name in self.HTTP_METHODS:
                    # @app.get('/path'), @router.post('/path')
                    path = self._get_first_string_arg(decorator)
                    if path:
                        return HTTPEndpoint(
                            path=path,
                            method=method_name.upper(),
                            file_path=self._current_file,
                            line_number=line_number,
                            handler_name=handler_name,
                            framework=self._framework,
                            parameters=self._extract_params_from_path(path),
                        )
                elif method_name == "route":
                    # Flask @app.route('/path', methods=['GET', 'POST'])
                    path = self._get_first_string_arg(decorator)
                    methods = self._get_methods_kwarg(decorator)
                    if path:
                        # Create an endpoint for each method
                        for method in methods:
                            return HTTPEndpoint(
                                path=path,
                                method=method,
                                file_path=self._current_file,
                                line_number=line_number,
                                handler_name=handler_name,
                                framework="flask",
                                parameters=self._extract_params_from_path(path),
                            )
                elif method_name == "api_route":
                    # FastAPI @app.api_route('/path', methods=['GET'])
                    path = self._get_first_string_arg(decorator)
                    methods = self._get_methods_kwarg(decorator)
                    if path:
                        for method in methods:
                            return HTTPEndpoint(
                                path=path,
                                method=method,
                                file_path=self._current_file,
                                line_number=line_number,
                                handler_name=handler_name,
                                framework="fastapi",
                                parameters=self._extract_params_from_path(path),
                            )

            elif isinstance(func, ast.Name):
                # Could be from aiohttp web.get, web.post etc.
                pass

        elif isinstance(decorator, ast.Attribute):
            # Unparenthesized decorator like @app.get (no args) - uncommon
            pass

        return None

    def _extract_from_call(self, node: ast.Call) -> Optional[HTTPEndpoint]:
        """Extract endpoint from method calls like app.add_api_route()."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr.lower()

        # app.add_api_route('/path', handler)
        if method_name == "add_api_route" and len(node.args) >= 2:
            path = self._get_value(node.args[0])
            if isinstance(path, str):
                methods = self._get_methods_kwarg(node)
                return HTTPEndpoint(
                    path=path,
                    method=methods[0] if methods else "GET",
                    file_path=self._current_file,
                    line_number=node.lineno,
                    framework="fastapi",
                    parameters=self._extract_params_from_path(path),
                )

        # app.router.add_route('GET', '/path', handler)
        elif method_name == "add_route" and len(node.args) >= 3:
            method = self._get_value(node.args[0])
            path = self._get_value(node.args[1])
            if isinstance(method, str) and isinstance(path, str):
                return HTTPEndpoint(
                    path=path,
                    method=method.upper(),
                    file_path=self._current_file,
                    line_number=node.lineno,
                    framework="aiohttp",
                    parameters=self._extract_params_from_path(path),
                )

        # aiohttp web.get('/path', handler), web.post('/path', handler)
        elif method_name in self.AIOHTTP_METHODS and len(node.args) >= 2:
            path = self._get_value(node.args[0])
            if isinstance(path, str):
                return HTTPEndpoint(
                    path=path,
                    method=method_name.upper(),
                    file_path=self._current_file,
                    line_number=node.lineno,
                    framework="aiohttp",
                    parameters=self._extract_params_from_path(path),
                )

        return None

    def _get_first_string_arg(self, node: ast.Call) -> Optional[str]:
        """Get the first string argument from a call."""
        if node.args:
            return self._get_value(node.args[0])
        return None

    def _get_methods_kwarg(self, node: ast.Call) -> List[str]:
        """Get methods from methods= keyword argument."""
        for kw in node.keywords:
            if kw.arg == "methods":
                value = self._get_value(kw.value)
                if isinstance(value, list):
                    return [m.upper() for m in value if isinstance(m, str)]
                elif isinstance(value, str):
                    return [value.upper()]
        return ["GET"]  # Default

    def _get_value(self, node: ast.AST) -> Any:
        """Get a Python value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.List):
            return [self._get_value(item) for item in node.elts]
        elif isinstance(node, ast.JoinedStr):
            # f-string - extract static parts
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    parts.append("{...}")
            return "".join(parts)
        return None

    def _extract_params_from_path(self, path: str) -> List[str]:
        """Extract path parameters."""
        params = []
        # FastAPI style: {param}
        params.extend(re.findall(r'\{(\w+)\}', path))
        # Flask style: <param> or <type:param>
        params.extend(re.findall(r'<(?:\w+:)?(\w+)>', path))
        return params


def extract_http_endpoints(file_path: Path) -> List[HTTPEndpoint]:
    """
    Convenience function to extract HTTP endpoints from a file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of HTTPEndpoint objects
    """
    extractor = HTTPEndpointExtractor()
    return extractor.extract_from_file(file_path)
