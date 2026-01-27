"""
Tests for HTTP/REST communication detection module.

Tests endpoint extraction, client detection, and communication mapping.
"""

import pytest
from pathlib import Path
import tempfile
import os

from robomind.http.endpoint_extractor import (
    HTTPEndpoint,
    HTTPEndpointExtractor,
    extract_http_endpoints,
)
from robomind.http.client_extractor import (
    HTTPClientCall,
    HTTPClientExtractor,
    extract_http_clients,
)
from robomind.http.communication_map import (
    CommunicationLink,
    CommunicationMap,
    build_communication_map,
)


class TestHTTPEndpoint:
    """Test HTTPEndpoint dataclass."""

    def test_basic_creation(self):
        """Test basic endpoint creation."""
        endpoint = HTTPEndpoint(
            path="/api/health",
            method="GET",
            file_path=Path("server.py"),
            line_number=42,
            framework="fastapi",
        )

        assert endpoint.path == "/api/health"
        assert endpoint.method == "GET"
        assert endpoint.framework == "fastapi"

    def test_get_path_params_fastapi(self):
        """Test extracting FastAPI-style path parameters."""
        endpoint = HTTPEndpoint(
            path="/api/users/{user_id}/posts/{post_id}",
            method="GET",
            file_path=Path("server.py"),
            line_number=1,
        )

        params = endpoint.get_path_params()
        assert "user_id" in params
        assert "post_id" in params

    def test_get_path_params_flask(self):
        """Test extracting Flask-style path parameters."""
        endpoint = HTTPEndpoint(
            path="/api/users/<int:user_id>/posts/<post_id>",
            method="GET",
            file_path=Path("server.py"),
            line_number=1,
        )

        params = endpoint.get_path_params()
        assert "user_id" in params
        assert "post_id" in params

    def test_to_dict(self):
        """Test conversion to dictionary."""
        endpoint = HTTPEndpoint(
            path="/api/health",
            method="GET",
            file_path=Path("server.py"),
            line_number=42,
            handler_name="health_check",
        )

        result = endpoint.to_dict()

        assert result["path"] == "/api/health"
        assert result["method"] == "GET"
        assert result["handler_name"] == "health_check"


class TestHTTPEndpointExtractor:
    """Test HTTPEndpointExtractor class."""

    def test_extract_fastapi_decorators(self):
        """Test extracting FastAPI route decorators."""
        code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/items")
def create_item(item: dict):
    return item

@app.put("/api/items/{item_id}")
def update_item(item_id: int, item: dict):
    return item
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            endpoints = extract_http_endpoints(Path(f.name))

            assert len(endpoints) == 3
            assert any(ep.path == "/health" and ep.method == "GET" for ep in endpoints)
            assert any(ep.path == "/api/items" and ep.method == "POST" for ep in endpoints)
            assert any(ep.path == "/api/items/{item_id}" and ep.method == "PUT" for ep in endpoints)

        os.unlink(f.name)

    def test_extract_flask_route(self):
        """Test extracting Flask route decorator."""
        code = '''
from flask import Flask

app = Flask(__name__)

@app.route('/api/users', methods=['GET', 'POST'])
def users():
    pass

@app.route('/health')
def health():
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            endpoints = extract_http_endpoints(Path(f.name))

            assert len(endpoints) >= 2
            assert any(ep.path == "/health" for ep in endpoints)

        os.unlink(f.name)

    def test_extract_fastapi_router(self):
        """Test extracting FastAPI router decorators."""
        code = '''
from fastapi import APIRouter

router = APIRouter()

@router.get("/items")
def list_items():
    return []

@router.post("/items")
def create_item():
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            endpoints = extract_http_endpoints(Path(f.name))

            assert len(endpoints) == 2
            assert any(ep.path == "/items" and ep.method == "GET" for ep in endpoints)
            assert any(ep.path == "/items" and ep.method == "POST" for ep in endpoints)

        os.unlink(f.name)

    def test_extract_add_api_route(self):
        """Test extracting app.add_api_route() calls."""
        code = '''
from fastapi import FastAPI

app = FastAPI()

def my_handler():
    pass

app.add_api_route("/api/custom", my_handler, methods=["POST"])
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            endpoints = extract_http_endpoints(Path(f.name))

            assert len(endpoints) >= 1
            assert any(ep.path == "/api/custom" for ep in endpoints)

        os.unlink(f.name)

    def test_extract_async_endpoints(self):
        """Test extracting async endpoint handlers."""
        code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/async-health")
async def async_health():
    return {"status": "ok"}
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            endpoints = extract_http_endpoints(Path(f.name))

            assert len(endpoints) == 1
            assert endpoints[0].path == "/async-health"
            assert endpoints[0].handler_name == "async_health"

        os.unlink(f.name)

    def test_detect_framework(self):
        """Test framework detection from imports."""
        extractor = HTTPEndpointExtractor()

        assert extractor._detect_framework("from fastapi import FastAPI") == "fastapi"
        assert extractor._detect_framework("from flask import Flask") == "flask"
        assert extractor._detect_framework("import aiohttp") == "aiohttp"
        assert extractor._detect_framework("from starlette.routing import Route") == "starlette"
        assert extractor._detect_framework("import os") == "unknown"


class TestHTTPClientCall:
    """Test HTTPClientCall dataclass."""

    def test_basic_creation(self):
        """Test basic client call creation."""
        call = HTTPClientCall(
            file_path=Path("client.py"),
            line_number=42,
            method="GET",
            target_url="http://example.com/api",
            library="requests",
        )

        assert call.method == "GET"
        assert call.target_url == "http://example.com/api"
        assert call.library == "requests"

    def test_get_host(self):
        """Test extracting host from URL."""
        call = HTTPClientCall(
            file_path=Path("client.py"),
            line_number=1,
            method="GET",
            target_url="http://vision-jetson.local:9091/detections",
            library="requests",
        )

        assert call.get_host() == "vision-jetson.local:9091"

    def test_get_host_no_port(self):
        """Test extracting host without port."""
        call = HTTPClientCall(
            file_path=Path("client.py"),
            line_number=1,
            method="GET",
            target_url="https://api.example.com/data",
            library="requests",
        )

        assert call.get_host() == "api.example.com"

    def test_get_host_no_url(self):
        """Test get_host with no URL."""
        call = HTTPClientCall(
            file_path=Path("client.py"),
            line_number=1,
            method="GET",
            target_variable="${API_URL}",
            library="requests",
        )

        assert call.get_host() is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        call = HTTPClientCall(
            file_path=Path("client.py"),
            line_number=42,
            method="POST",
            target_url="http://example.com/api",
            library="httpx",
            is_async=True,
            context="send_data",
        )

        result = call.to_dict()

        assert result["method"] == "POST"
        assert result["target_url"] == "http://example.com/api"
        assert result["library"] == "httpx"
        assert result["is_async"] is True
        assert result["context"] == "send_data"


class TestHTTPClientExtractor:
    """Test HTTPClientExtractor class."""

    def test_extract_requests_get(self):
        """Test extracting requests.get() calls."""
        code = '''
import requests

def fetch_data():
    response = requests.get("http://api.example.com/data")
    return response.json()
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            calls = extract_http_clients(Path(f.name))

            assert len(calls) == 1
            assert calls[0].method == "GET"
            assert calls[0].target_url == "http://api.example.com/data"
            assert calls[0].library == "requests"

        os.unlink(f.name)

    def test_extract_requests_post(self):
        """Test extracting requests.post() calls."""
        code = '''
import requests

def send_data(data):
    response = requests.post("http://api.example.com/items", json=data)
    return response
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            calls = extract_http_clients(Path(f.name))

            assert len(calls) == 1
            assert calls[0].method == "POST"
            assert "items" in calls[0].target_url

        os.unlink(f.name)

    def test_extract_fstring_url(self):
        """Test extracting f-string URLs."""
        code = '''
import requests

BASE_URL = "http://api.example.com"

def fetch_item(item_id):
    response = requests.get(f"{BASE_URL}/items/{item_id}")
    return response.json()
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            calls = extract_http_clients(Path(f.name))

            assert len(calls) == 1
            # f-string URL should be partially extracted
            assert "/items/" in calls[0].target_url or "${" in calls[0].target_url

        os.unlink(f.name)

    def test_extract_variable_url(self):
        """Test extracting variable-based URLs."""
        code = '''
import requests

def fetch_from_config(api_url):
    response = requests.get(api_url)
    return response.json()
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            calls = extract_http_clients(Path(f.name))

            assert len(calls) == 1
            assert calls[0].target_variable == "${api_url}"

        os.unlink(f.name)

    def test_extract_httpx_calls(self):
        """Test extracting httpx calls."""
        code = '''
import httpx

def fetch_data():
    response = httpx.get("http://api.example.com/data")
    return response.json()
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            calls = extract_http_clients(Path(f.name))

            assert len(calls) == 1
            assert calls[0].library == "httpx"

        os.unlink(f.name)

    def test_extract_context(self):
        """Test that function context is captured."""
        code = '''
import requests

def my_function():
    response = requests.get("http://api.example.com/data")
    return response
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            calls = extract_http_clients(Path(f.name))

            assert len(calls) == 1
            assert calls[0].context == "my_function"

        os.unlink(f.name)


class TestCommunicationLink:
    """Test CommunicationLink dataclass."""

    def test_http_link(self):
        """Test HTTP communication link."""
        link = CommunicationLink(
            source="thor",
            target="vision-jetson.local:9091",
            link_type="http",
            direction="source_to_target",
            http_endpoint="/api/detections",
            http_method="GET",
        )

        assert link.link_type == "http"
        assert link.source == "thor"
        assert link.target == "vision-jetson.local:9091"

    def test_ros2_link(self):
        """Test ROS2 communication link."""
        link = CommunicationLink(
            source="motor_controller",
            target="navigation_node",
            link_type="ros2",
            direction="source_to_target",
            ros2_topic="/cmd_vel",
            ros2_msg_type="geometry_msgs/Twist",
        )

        assert link.link_type == "ros2"
        assert link.ros2_topic == "/cmd_vel"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        link = CommunicationLink(
            source="client",
            target="server",
            link_type="http",
            direction="source_to_target",
        )

        result = link.to_dict()

        assert result["source"] == "client"
        assert result["target"] == "server"
        assert result["link_type"] == "http"


class TestCommunicationMap:
    """Test CommunicationMap class."""

    def test_basic_creation(self):
        """Test basic map creation."""
        comm_map = CommunicationMap()

        assert len(comm_map.links) == 0
        assert len(comm_map.http_endpoints) == 0

    def test_add_http_endpoint(self):
        """Test adding HTTP endpoints."""
        comm_map = CommunicationMap()

        endpoint = HTTPEndpoint(
            path="/api/health",
            method="GET",
            file_path=Path("server.py"),
            line_number=1,
        )
        comm_map.add_http_endpoint(endpoint)

        assert len(comm_map.http_endpoints) == 1

    def test_add_http_client(self):
        """Test adding HTTP clients."""
        comm_map = CommunicationMap()

        client = HTTPClientCall(
            file_path=Path("client.py"),
            line_number=1,
            method="GET",
            target_url="http://example.com/api",
            library="requests",
        )
        comm_map.add_http_client(client)

        assert len(comm_map.http_clients) == 1

    def test_add_ros2_topic(self):
        """Test adding ROS2 topics."""
        comm_map = CommunicationMap()

        comm_map.add_ros2_topic(
            topic="/cmd_vel",
            publishers=["navigation_node"],
            subscribers=["motor_controller"],
            msg_type="geometry_msgs/Twist",
        )

        assert "/cmd_vel" in comm_map.ros2_topics
        assert len(comm_map.links) == 1
        assert comm_map.links[0].link_type == "ros2"

    def test_get_cross_system_links(self):
        """Test getting cross-system links."""
        comm_map = CommunicationMap()

        # Add HTTP link (cross-system)
        comm_map.links.append(CommunicationLink(
            source="thor",
            target="vision-jetson",
            link_type="http",
            direction="source_to_target",
        ))

        # Add ROS2 link (internal)
        comm_map.links.append(CommunicationLink(
            source="nav_node",
            target="motor",
            link_type="ros2",
            direction="source_to_target",
        ))

        cross_system = comm_map.get_cross_system_links()

        assert len(cross_system) == 1
        assert cross_system[0].link_type == "http"

    def test_summary(self):
        """Test map summary."""
        comm_map = CommunicationMap()

        # Add some data
        comm_map.links.append(CommunicationLink(
            source="a", target="b", link_type="http", direction="source_to_target"
        ))
        comm_map.links.append(CommunicationLink(
            source="c", target="d", link_type="ros2", direction="source_to_target"
        ))

        summary = comm_map.summary()

        assert summary["total_links"] == 2
        assert summary["http_links"] == 1
        assert summary["ros2_links"] == 1


class TestBuildCommunicationMap:
    """Test build_communication_map function."""

    def test_build_empty_map(self):
        """Test building empty map."""
        comm_map = build_communication_map(
            http_endpoints=[],
            http_clients=[],
        )

        assert len(comm_map.links) == 0

    def test_build_with_endpoints_and_clients(self):
        """Test building map with endpoints and clients."""
        endpoints = [
            HTTPEndpoint(
                path="/api/health",
                method="GET",
                file_path=Path("server.py"),
                line_number=1,
                inferred_host="vision-jetson.local:9091",
            ),
        ]

        clients = [
            HTTPClientCall(
                file_path=Path("client.py"),
                line_number=1,
                method="GET",
                target_url="http://vision-jetson.local:9091/api/health",
                library="requests",
                context="check_health",
            ),
        ]

        comm_map = build_communication_map(
            http_endpoints=endpoints,
            http_clients=clients,
        )

        assert len(comm_map.http_endpoints) == 1
        assert len(comm_map.http_clients) == 1
        assert comm_map.has_http_cross_jetson is True
