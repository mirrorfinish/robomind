"""
HTTP Client Extractor - Detect outbound HTTP calls in Python code.

This module detects HTTP client calls by parsing:
- requests: requests.get(), requests.post(), etc.
- aiohttp: aiohttp.ClientSession().get()
- httpx: httpx.get(), httpx.AsyncClient().get()
- urllib: urllib.request.urlopen()

Example detections:
```python
requests.get('http://vision-jetson.local:9091/detections')
requests.post(f'{VISION_URL}/api/capture')
aiohttp.ClientSession().get(url)
httpx.AsyncClient().post(...)
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
class HTTPClientCall:
    """A detected HTTP client call."""
    file_path: Path
    line_number: int
    method: str  # GET, POST, PUT, DELETE, etc.
    target_url: Optional[str] = None  # Literal URL if detectable
    target_variable: Optional[str] = None  # Variable name if URL is dynamic
    library: str = "unknown"  # requests, aiohttp, httpx, urllib
    is_async: bool = False
    context: Optional[str] = None  # Function/method where call occurs

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "method": self.method,
            "target_url": self.target_url,
            "target_variable": self.target_variable,
            "library": self.library,
            "is_async": self.is_async,
            "context": self.context,
        }

    def get_host(self) -> Optional[str]:
        """Extract host from URL if available."""
        if self.target_url:
            match = re.match(r'https?://([^/:]+)(:\d+)?', self.target_url)
            if match:
                host = match.group(1)
                port = match.group(2) or ""
                return f"{host}{port}"
        return None


class HTTPClientExtractor:
    """
    Extract HTTP client calls from Python source files.

    Supports requests, aiohttp, httpx, and urllib libraries.

    Usage:
        extractor = HTTPClientExtractor()
        calls = extractor.extract_from_file(Path("client.py"))
        for call in calls:
            print(f"{call.method} {call.target_url or call.target_variable}")
    """

    # HTTP methods
    HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "request"}

    # Library method patterns
    REQUESTS_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "request"}
    AIOHTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "request"}
    HTTPX_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "request"}

    def __init__(self):
        """Initialize client extractor."""
        self._current_file: Optional[Path] = None
        self._current_context: Optional[str] = None
        self._has_requests: bool = False
        self._has_aiohttp: bool = False
        self._has_httpx: bool = False
        self._has_urllib: bool = False

    def extract_from_file(self, file_path: Path) -> List[HTTPClientCall]:
        """
        Extract HTTP client calls from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of HTTPClientCall objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        self._current_file = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Detect which HTTP libraries are imported
            self._detect_imports(content)

            tree = ast.parse(content, filename=str(file_path))
            calls = []

            # Walk the AST looking for HTTP calls
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._current_context = node.name
                elif isinstance(node, ast.Call):
                    call = self._extract_from_call(node)
                    if call:
                        calls.append(call)

            return calls

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return []

    def _detect_imports(self, content: str):
        """Detect which HTTP libraries are imported."""
        self._has_requests = "import requests" in content or "from requests" in content
        self._has_aiohttp = "import aiohttp" in content or "from aiohttp" in content
        self._has_httpx = "import httpx" in content or "from httpx" in content
        self._has_urllib = "urllib.request" in content

    def _extract_from_call(self, node: ast.Call) -> Optional[HTTPClientCall]:
        """Extract HTTP client call from an AST Call node."""
        # Try different patterns
        call = self._try_requests_pattern(node)
        if call:
            return call

        call = self._try_aiohttp_pattern(node)
        if call:
            return call

        call = self._try_httpx_pattern(node)
        if call:
            return call

        call = self._try_urllib_pattern(node)
        if call:
            return call

        return None

    def _try_requests_pattern(self, node: ast.Call) -> Optional[HTTPClientCall]:
        """Try to match requests.get(), requests.post(), etc."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr.lower()

        # requests.get(), requests.post(), etc.
        if isinstance(node.func.value, ast.Name):
            if node.func.value.id == "requests" and method_name in self.HTTP_METHODS:
                url_arg = self._get_url_arg(node)
                is_literal = self._is_literal_url(url_arg)
                return HTTPClientCall(
                    file_path=self._current_file,
                    line_number=node.lineno,
                    method=self._normalize_method(method_name),
                    target_url=url_arg if is_literal else None,
                    target_variable=url_arg if not is_literal and url_arg else None,
                    library="requests",
                    context=self._current_context,
                )

        # session.get(), session.post() where session = requests.Session()
        if method_name in self.HTTP_METHODS:
            # Could be a session object - check for any .get(), .post() pattern
            # This is heuristic but catches common patterns
            if self._has_requests:
                url_arg = self._get_url_arg(node)
                if url_arg:
                    is_literal = self._is_literal_url(url_arg)
                    return HTTPClientCall(
                        file_path=self._current_file,
                        line_number=node.lineno,
                        method=self._normalize_method(method_name),
                        target_url=url_arg if is_literal else None,
                        target_variable=url_arg if not is_literal else None,
                        library="requests",
                        context=self._current_context,
                    )

        return None

    def _try_aiohttp_pattern(self, node: ast.Call) -> Optional[HTTPClientCall]:
        """Try to match aiohttp.ClientSession().get(), etc."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr.lower()

        # Check for .get(), .post() on something that looks like aiohttp session
        if method_name in self.AIOHTTP_METHODS:
            # Check if it's a method call on a ClientSession
            if self._has_aiohttp:
                # aiohttp.ClientSession().get() or session.get()
                if isinstance(node.func.value, ast.Call):
                    # Could be ClientSession().get()
                    url_arg = self._get_url_arg(node)
                    if url_arg:
                        is_literal = self._is_literal_url(url_arg)
                        return HTTPClientCall(
                            file_path=self._current_file,
                            line_number=node.lineno,
                            method=self._normalize_method(method_name),
                            target_url=url_arg if is_literal else None,
                            target_variable=url_arg if not is_literal else None,
                            library="aiohttp",
                            is_async=True,
                            context=self._current_context,
                        )
                elif isinstance(node.func.value, ast.Name):
                    # session.get() where session is probably an aiohttp session
                    url_arg = self._get_url_arg(node)
                    if url_arg:
                        is_literal = self._is_literal_url(url_arg)
                        return HTTPClientCall(
                            file_path=self._current_file,
                            line_number=node.lineno,
                            method=self._normalize_method(method_name),
                            target_url=url_arg if is_literal else None,
                            target_variable=url_arg if not is_literal else None,
                            library="aiohttp",
                            is_async=True,
                            context=self._current_context,
                        )

        return None

    def _try_httpx_pattern(self, node: ast.Call) -> Optional[HTTPClientCall]:
        """Try to match httpx.get(), httpx.AsyncClient().get(), etc."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr.lower()

        # httpx.get(), httpx.post()
        if isinstance(node.func.value, ast.Name):
            if node.func.value.id == "httpx" and method_name in self.HTTPX_METHODS:
                url_arg = self._get_url_arg(node)
                is_literal = self._is_literal_url(url_arg)
                return HTTPClientCall(
                    file_path=self._current_file,
                    line_number=node.lineno,
                    method=self._normalize_method(method_name),
                    target_url=url_arg if is_literal else None,
                    target_variable=url_arg if not is_literal and url_arg else None,
                    library="httpx",
                    context=self._current_context,
                )

        # client.get(), client.post() for httpx.AsyncClient()
        if self._has_httpx and method_name in self.HTTPX_METHODS:
            url_arg = self._get_url_arg(node)
            if url_arg and isinstance(url_arg, str) and url_arg.startswith("http"):
                return HTTPClientCall(
                    file_path=self._current_file,
                    line_number=node.lineno,
                    method=self._normalize_method(method_name),
                    target_url=url_arg,
                    library="httpx",
                    is_async=True,
                    context=self._current_context,
                )

        return None

    def _try_urllib_pattern(self, node: ast.Call) -> Optional[HTTPClientCall]:
        """Try to match urllib.request.urlopen(), etc."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr.lower()

        # urllib.request.urlopen()
        if method_name == "urlopen":
            if isinstance(node.func.value, ast.Attribute):
                if node.func.value.attr == "request":
                    url_arg = self._get_url_arg(node)
                    is_literal = self._is_literal_url(url_arg)
                    return HTTPClientCall(
                        file_path=self._current_file,
                        line_number=node.lineno,
                        method="GET",  # urlopen is typically GET
                        target_url=url_arg if is_literal else None,
                        target_variable=url_arg if not is_literal and url_arg else None,
                        library="urllib",
                        context=self._current_context,
                    )

        return None

    def _get_url_arg(self, node: ast.Call) -> Optional[Any]:
        """Get the URL argument from a call."""
        # Try first positional argument
        if node.args:
            value = self._get_value(node.args[0])
            if value:
                return value

        # Try url= keyword argument
        for kw in node.keywords:
            if kw.arg == "url":
                return self._get_value(kw.value)

        return None

    def _get_value(self, node: ast.AST) -> Optional[Any]:
        """Get a Python value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.JoinedStr):
            # f-string - try to extract the URL pattern
            parts = []
            has_format = False
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    has_format = True
                    # Try to get variable name
                    if isinstance(value.value, ast.Name):
                        parts.append(f"${{{value.value.id}}}")
                    else:
                        parts.append("{...}")
            result = "".join(parts)
            # If it looks like a URL, return it
            if "http" in result or "/" in result:
                return result
            return None
        elif isinstance(node, ast.Name):
            # Variable reference
            return f"${{{node.id}}}"
        elif isinstance(node, ast.BinOp):
            # String concatenation
            if isinstance(node.op, ast.Add):
                left = self._get_value(node.left)
                right = self._get_value(node.right)
                if left and right:
                    return f"{left}{right}"
        return None

    def _normalize_method(self, method: str) -> str:
        """Normalize HTTP method name."""
        method = method.upper()
        if method == "REQUEST":
            return "GET"  # Default for generic request
        return method

    def _is_literal_url(self, url: Optional[str]) -> bool:
        """Check if URL contains usable URL components vs pure variable reference.

        Returns True for:
        - Full literal URLs: "http://example.com/api"
        - F-string URLs with path parts: "${BASE_URL}/items/${id}"
        - Partial URLs with paths: "/api/items"

        Returns False for:
        - Pure variable references: "${api_url}" (no slashes, no http)
        """
        if not url or not isinstance(url, str):
            return False
        # URLs with http:// or https:// are considered useful
        if "http://" in url or "https://" in url:
            return True
        # URLs with path components (slashes) are useful even with variable parts
        if "/" in url:
            return True
        # Pure variable references have no URL structure
        return False


def extract_http_clients(file_path: Path) -> List[HTTPClientCall]:
    """
    Convenience function to extract HTTP client calls from a file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of HTTPClientCall objects
    """
    extractor = HTTPClientExtractor()
    return extractor.extract_from_file(file_path)
