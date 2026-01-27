"""
RoboMind HTTP Module - HTTP/REST communication detection.

Day 10 implementation:
- endpoint_extractor.py - Detect Flask/FastAPI/aiohttp server endpoints
- client_extractor.py - Detect HTTP client calls (requests, aiohttp, httpx)
- communication_map.py - Build cross-system communication map
"""

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

__all__ = [
    "HTTPEndpoint",
    "HTTPEndpointExtractor",
    "extract_http_endpoints",
    "HTTPClientCall",
    "HTTPClientExtractor",
    "extract_http_clients",
    "CommunicationLink",
    "CommunicationMap",
    "build_communication_map",
]
