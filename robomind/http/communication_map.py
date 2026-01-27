"""
Communication Map - Build cross-system communication map.

This module builds a comprehensive map of how different system components
communicate, combining ROS2 topics and HTTP endpoints.

Features:
- Combine ROS2 and HTTP communication
- Identify cross-system vs internal communication
- Generate summary of communication patterns
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from robomind.http.endpoint_extractor import HTTPEndpoint
from robomind.http.client_extractor import HTTPClientCall

logger = logging.getLogger(__name__)


@dataclass
class CommunicationLink:
    """A communication link between components."""
    source: str  # Source component/host
    target: str  # Target component/host
    link_type: str  # "http" or "ros2"
    direction: str  # "bidirectional", "source_to_target", "target_to_source"

    # For HTTP links
    http_endpoint: Optional[str] = None
    http_method: Optional[str] = None

    # For ROS2 links
    ros2_topic: Optional[str] = None
    ros2_msg_type: Optional[str] = None

    # Metadata
    file_path: Optional[str] = None
    line_number: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "link_type": self.link_type,
            "direction": self.direction,
            "http_endpoint": self.http_endpoint,
            "http_method": self.http_method,
            "ros2_topic": self.ros2_topic,
            "ros2_msg_type": self.ros2_msg_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
        }


@dataclass
class CommunicationMap:
    """
    Complete communication map for a system.

    Combines ROS2 topics and HTTP endpoints to show how
    components communicate.
    """
    links: List[CommunicationLink] = field(default_factory=list)

    # Discovered endpoints and clients
    http_endpoints: List[HTTPEndpoint] = field(default_factory=list)
    http_clients: List[HTTPClientCall] = field(default_factory=list)

    # ROS2 info (from existing analysis)
    ros2_topics: List[str] = field(default_factory=list)
    ros2_publishers: Dict[str, List[str]] = field(default_factory=dict)  # topic -> nodes
    ros2_subscribers: Dict[str, List[str]] = field(default_factory=dict)  # topic -> nodes

    # Summary flags
    has_ros2_cross_jetson: bool = False
    has_http_cross_jetson: bool = True  # Default assumption for NEXUS-like systems

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "links": [link.to_dict() for link in self.links],
            "http_endpoints": [ep.to_dict() for ep in self.http_endpoints],
            "http_clients": [client.to_dict() for client in self.http_clients],
            "ros2_topics": self.ros2_topics,
            "has_ros2_cross_jetson": self.has_ros2_cross_jetson,
            "has_http_cross_jetson": self.has_http_cross_jetson,
            "summary": self.summary(),
        }

    def add_http_endpoint(self, endpoint: HTTPEndpoint):
        """Add an HTTP endpoint."""
        self.http_endpoints.append(endpoint)

    def add_http_client(self, client: HTTPClientCall):
        """Add an HTTP client call."""
        self.http_clients.append(client)

    def add_ros2_topic(
        self,
        topic: str,
        publishers: List[str],
        subscribers: List[str],
        msg_type: Optional[str] = None,
    ):
        """Add a ROS2 topic."""
        self.ros2_topics.append(topic)
        self.ros2_publishers[topic] = publishers
        self.ros2_subscribers[topic] = subscribers

        # Create links for this topic
        for pub in publishers:
            for sub in subscribers:
                self.links.append(CommunicationLink(
                    source=pub,
                    target=sub,
                    link_type="ros2",
                    direction="source_to_target",
                    ros2_topic=topic,
                    ros2_msg_type=msg_type,
                ))

    def build_http_links(self):
        """Build communication links from HTTP endpoints and clients."""
        # Try to match clients to endpoints
        for client in self.http_clients:
            host = client.get_host()
            if host:
                # Find matching endpoint
                matching_endpoint = None
                for endpoint in self.http_endpoints:
                    if endpoint.inferred_host == host:
                        matching_endpoint = endpoint
                        break

                # Create link
                source = client.context or Path(client.file_path).stem
                target = host

                self.links.append(CommunicationLink(
                    source=source,
                    target=target,
                    link_type="http",
                    direction="source_to_target",
                    http_endpoint=client.target_url,
                    http_method=client.method,
                    file_path=str(client.file_path),
                    line_number=client.line_number,
                ))

    def get_cross_system_links(self) -> List[CommunicationLink]:
        """Get links that cross system boundaries (different hosts)."""
        # For HTTP links, these are typically cross-system
        return [link for link in self.links if link.link_type == "http"]

    def get_internal_links(self) -> List[CommunicationLink]:
        """Get links that are internal to a system."""
        # For ROS2 links on same domain, these are internal
        return [link for link in self.links if link.link_type == "ros2"]

    def get_links_by_host(self, host: str) -> Dict[str, List[CommunicationLink]]:
        """Get all links involving a specific host."""
        result = {"incoming": [], "outgoing": [], "internal": []}

        for link in self.links:
            if link.source == host and link.target == host:
                result["internal"].append(link)
            elif link.source == host:
                result["outgoing"].append(link)
            elif link.target == host:
                result["incoming"].append(link)

        return result

    def summary(self) -> Dict:
        """Generate summary of communication map."""
        http_hosts = set()
        for client in self.http_clients:
            host = client.get_host()
            if host:
                http_hosts.add(host)

        return {
            "total_links": len(self.links),
            "http_links": len([l for l in self.links if l.link_type == "http"]),
            "ros2_links": len([l for l in self.links if l.link_type == "ros2"]),
            "http_endpoints": len(self.http_endpoints),
            "http_clients": len(self.http_clients),
            "http_target_hosts": list(http_hosts),
            "ros2_topics": len(self.ros2_topics),
            "cross_system_protocol": "http" if self.has_http_cross_jetson else "ros2",
        }


def build_communication_map(
    http_endpoints: List[HTTPEndpoint],
    http_clients: List[HTTPClientCall],
    ros2_topic_graph=None,  # TopicGraphResult from topic_extractor
) -> CommunicationMap:
    """
    Build a complete communication map.

    Args:
        http_endpoints: List of HTTP server endpoints
        http_clients: List of HTTP client calls
        ros2_topic_graph: Optional ROS2 topic graph

    Returns:
        CommunicationMap with all communication links
    """
    comm_map = CommunicationMap(
        http_endpoints=http_endpoints,
        http_clients=http_clients,
    )

    # Add ROS2 topics if provided
    if ros2_topic_graph:
        for topic in ros2_topic_graph.topics:
            pubs = [conn.publisher for conn in ros2_topic_graph.connections
                    if hasattr(conn, 'topic') and conn.topic == topic]
            subs = [conn.subscriber for conn in ros2_topic_graph.connections
                    if hasattr(conn, 'topic') and conn.topic == topic]

            if pubs or subs:
                comm_map.add_ros2_topic(
                    topic=topic,
                    publishers=list(set(pubs)),
                    subscribers=list(set(subs)),
                )

    # Build HTTP links
    comm_map.build_http_links()

    # Determine cross-system protocol
    if comm_map.http_clients:
        comm_map.has_http_cross_jetson = True

    return comm_map
