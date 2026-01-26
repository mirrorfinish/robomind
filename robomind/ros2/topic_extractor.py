"""
ROS2 Topic Extractor for RoboMind

Builds a graph of ROS2 topic connections from extracted node information.
Tracks which nodes publish to and subscribe from which topics.

Features:
- Topic connection graph
- Orphaned topic detection (no pub or no sub)
- Connection validation
- Message type consistency checking
"""

import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

from robomind.ros2.node_extractor import ROS2NodeInfo

logger = logging.getLogger(__name__)


@dataclass
class TopicConnection:
    """Information about a topic and its connections."""
    name: str
    msg_type: str
    publishers: List[str] = field(default_factory=list)  # Node names
    subscribers: List[str] = field(default_factory=list)  # Node names
    qos_profiles: Set[int] = field(default_factory=set)

    @property
    def is_connected(self) -> bool:
        """Check if topic has both publishers and subscribers."""
        return len(self.publishers) > 0 and len(self.subscribers) > 0

    @property
    def is_orphaned(self) -> bool:
        """Check if topic has no connections at all."""
        return len(self.publishers) == 0 and len(self.subscribers) == 0

    @property
    def has_publisher(self) -> bool:
        """Check if topic has at least one publisher."""
        return len(self.publishers) > 0

    @property
    def has_subscriber(self) -> bool:
        """Check if topic has at least one subscriber."""
        return len(self.subscribers) > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "msg_type": self.msg_type,
            "publishers": self.publishers,
            "subscribers": self.subscribers,
            "is_connected": self.is_connected,
            "publisher_count": len(self.publishers),
            "subscriber_count": len(self.subscribers),
        }


@dataclass
class ServiceConnection:
    """Information about a service and its connections."""
    name: str
    srv_type: str
    servers: List[str] = field(default_factory=list)  # Node names
    clients: List[str] = field(default_factory=list)  # Node names

    @property
    def is_connected(self) -> bool:
        """Check if service has both servers and clients."""
        return len(self.servers) > 0 and len(self.clients) > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "srv_type": self.srv_type,
            "servers": self.servers,
            "clients": self.clients,
            "is_connected": self.is_connected,
        }


@dataclass
class TopicGraphResult:
    """Results from building the topic connection graph."""
    topics: Dict[str, TopicConnection] = field(default_factory=dict)
    services: Dict[str, ServiceConnection] = field(default_factory=dict)
    nodes_processed: int = 0

    def get_connected_topics(self) -> List[TopicConnection]:
        """Get all topics that have both publishers and subscribers."""
        return [t for t in self.topics.values() if t.is_connected]

    def get_unconnected_topics(self) -> List[TopicConnection]:
        """Get topics missing either publishers or subscribers."""
        return [t for t in self.topics.values() if not t.is_connected]

    def get_publish_only_topics(self) -> List[TopicConnection]:
        """Get topics that only have publishers (no subscribers)."""
        return [t for t in self.topics.values()
                if t.has_publisher and not t.has_subscriber]

    def get_subscribe_only_topics(self) -> List[TopicConnection]:
        """Get topics that only have subscribers (no publishers)."""
        return [t for t in self.topics.values()
                if t.has_subscriber and not t.has_publisher]

    def get_topics_for_node(self, node_name: str) -> Dict[str, List[TopicConnection]]:
        """Get all topics a node interacts with."""
        publishes = []
        subscribes = []

        for topic in self.topics.values():
            if node_name in topic.publishers:
                publishes.append(topic)
            if node_name in topic.subscribers:
                subscribes.append(topic)

        return {
            "publishes": publishes,
            "subscribes": subscribes,
        }

    def summary(self) -> Dict:
        """Generate summary statistics."""
        connected = self.get_connected_topics()
        unconnected = self.get_unconnected_topics()

        return {
            "total_topics": len(self.topics),
            "connected_topics": len(connected),
            "unconnected_topics": len(unconnected),
            "publish_only": len(self.get_publish_only_topics()),
            "subscribe_only": len(self.get_subscribe_only_topics()),
            "total_services": len(self.services),
            "nodes_processed": self.nodes_processed,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary(),
            "topics": {name: t.to_dict() for name, t in self.topics.items()},
            "services": {name: s.to_dict() for name, s in self.services.items()},
        }


class TopicExtractor:
    """
    Build a topic connection graph from ROS2 node information.

    Aggregates publishers and subscribers from multiple nodes to show:
    - Which nodes publish to which topics
    - Which nodes subscribe to which topics
    - Topic message types
    - Connection status (connected, orphaned, etc.)

    Usage:
        extractor = TopicExtractor()
        for node in nodes:
            extractor.add_node(node)
        result = extractor.build()
        print(result.summary())
    """

    def __init__(self):
        self.topics: Dict[str, TopicConnection] = {}
        self.services: Dict[str, ServiceConnection] = {}
        self.nodes_processed = 0

    def add_node(self, node: ROS2NodeInfo):
        """
        Add a node's publishers and subscribers to the graph.

        Args:
            node: ROS2NodeInfo object with publisher/subscriber information
        """
        self.nodes_processed += 1

        # Process publishers
        for pub in node.publishers:
            topic_name = pub.topic
            if topic_name not in self.topics:
                self.topics[topic_name] = TopicConnection(
                    name=topic_name,
                    msg_type=pub.msg_type,
                )

            topic = self.topics[topic_name]
            if node.name not in topic.publishers:
                topic.publishers.append(node.name)
            topic.qos_profiles.add(pub.qos)

            # Update msg_type if we have a more specific one
            if topic.msg_type == "Unknown" and pub.msg_type != "Unknown":
                topic.msg_type = pub.msg_type

        # Process subscribers
        for sub in node.subscribers:
            topic_name = sub.topic
            if topic_name not in self.topics:
                self.topics[topic_name] = TopicConnection(
                    name=topic_name,
                    msg_type=sub.msg_type,
                )

            topic = self.topics[topic_name]
            if node.name not in topic.subscribers:
                topic.subscribers.append(node.name)
            topic.qos_profiles.add(sub.qos)

            if topic.msg_type == "Unknown" and sub.msg_type != "Unknown":
                topic.msg_type = sub.msg_type

        # Process services
        for srv in node.services:
            srv_name = srv.name
            if srv_name not in self.services:
                self.services[srv_name] = ServiceConnection(
                    name=srv_name,
                    srv_type=srv.srv_type,
                )

            service = self.services[srv_name]
            if node.name not in service.servers:
                service.servers.append(node.name)

        # Process service clients
        for client in node.service_clients:
            srv_name = client.name
            if srv_name not in self.services:
                self.services[srv_name] = ServiceConnection(
                    name=srv_name,
                    srv_type=client.srv_type,
                )

            service = self.services[srv_name]
            if node.name not in service.clients:
                service.clients.append(node.name)

        logger.debug(f"Added node {node.name}: {len(node.publishers)} pubs, "
                    f"{len(node.subscribers)} subs")

    def add_nodes(self, nodes: List[ROS2NodeInfo]):
        """Add multiple nodes to the graph."""
        for node in nodes:
            self.add_node(node)

    def build(self) -> TopicGraphResult:
        """
        Build and return the topic graph result.

        Returns:
            TopicGraphResult with all topic connections
        """
        return TopicGraphResult(
            topics=self.topics.copy(),
            services=self.services.copy(),
            nodes_processed=self.nodes_processed,
        )

    def get_topic(self, topic_name: str) -> Optional[TopicConnection]:
        """Get information about a specific topic."""
        return self.topics.get(topic_name)

    def find_message_type_mismatches(self) -> List[Dict]:
        """
        Find topics where publishers and subscribers might have different message types.

        This is a static analysis limitation - we can only detect if the
        extracted type strings differ.

        Returns:
            List of potential mismatches
        """
        # This would require tracking msg_type per publisher/subscriber
        # For now, return empty - would need enhanced data structures
        return []

    def build_adjacency_list(self) -> Dict[str, List[str]]:
        """
        Build an adjacency list for node-to-node connections via topics.

        Returns:
            Dict mapping node name to list of connected node names
        """
        adjacency: Dict[str, Set[str]] = {}

        for topic in self.topics.values():
            # Each publisher connects to each subscriber
            for pub_node in topic.publishers:
                if pub_node not in adjacency:
                    adjacency[pub_node] = set()

                for sub_node in topic.subscribers:
                    if pub_node != sub_node:  # Don't connect to self
                        adjacency[pub_node].add(sub_node)

        return {node: list(connected) for node, connected in adjacency.items()}


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from robomind.ros2.node_extractor import ROS2NodeExtractor

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python topic_extractor.py <project_path>")
        sys.exit(1)

    project_path = Path(sys.argv[1])

    # Find and parse all Python files
    node_extractor = ROS2NodeExtractor()
    topic_extractor = TopicExtractor()

    all_nodes = []

    for py_file in project_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        nodes = node_extractor.extract_from_file(py_file)
        all_nodes.extend(nodes)
        topic_extractor.add_nodes(nodes)

    result = topic_extractor.build()

    print("\n" + "=" * 60)
    print("TOPIC GRAPH SUMMARY")
    print("=" * 60)
    print(json.dumps(result.summary(), indent=2))

    print("\n" + "=" * 60)
    print("CONNECTED TOPICS")
    print("=" * 60)
    for topic in result.get_connected_topics():
        print(f"\n{topic.name} ({topic.msg_type})")
        print(f"  Publishers: {', '.join(topic.publishers)}")
        print(f"  Subscribers: {', '.join(topic.subscribers)}")

    print("\n" + "=" * 60)
    print("UNCONNECTED TOPICS")
    print("=" * 60)
    for topic in result.get_unconnected_topics():
        status = "pub only" if topic.has_publisher else "sub only"
        nodes = topic.publishers if topic.has_publisher else topic.subscribers
        print(f"  {topic.name} ({status}): {', '.join(nodes)}")
