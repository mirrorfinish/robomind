"""Load and query RoboMind system graph."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class SystemGraph:
    """Loads and queries the RoboMind system_graph.json."""

    def __init__(self, analysis_path: Optional[str] = None):
        """Initialize with path to analysis directory."""
        if analysis_path is None:
            analysis_path = os.environ.get(
                'ROBOMIND_ANALYSIS_PATH',
                os.path.expanduser('~/betaray_robomind_analysis')
            )
        self.analysis_path = Path(analysis_path)
        self.graph_path = self.analysis_path / 'system_graph.json'
        self._graph_data: Optional[Dict] = None
        self._context_data: Optional[Dict] = None

    def _load_graph(self) -> Dict:
        """Load the system graph from JSON."""
        if self._graph_data is None:
            if not self.graph_path.exists():
                raise FileNotFoundError(f"System graph not found: {self.graph_path}")
            with open(self.graph_path) as f:
                self._graph_data = json.load(f)
        return self._graph_data

    def _load_context(self) -> Dict:
        """Load the YAML context summary."""
        if self._context_data is None:
            context_path = self.analysis_path / 'system_context.yaml'
            if context_path.exists():
                import yaml
                with open(context_path) as f:
                    self._context_data = yaml.safe_load(f)
            else:
                self._context_data = {}
        return self._context_data

    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary of the system."""
        graph = self._load_graph()
        summary = graph.get('summary', {})
        return {
            'project': graph.get('metadata', {}).get('project_name', 'unknown'),
            'generated_at': graph.get('metadata', {}).get('generated_at', 'unknown'),
            'total_nodes': summary.get('ros2_nodes', 0),
            'total_topics': summary.get('topics', 0),
            'connected_topics': summary.get('connected_topics', 0),
            'total_http_endpoints': summary.get('http_endpoints', 0),
            'packages': summary.get('packages', []),
        }

    def query(self, pattern: str, limit: int = 20) -> Dict[str, Any]:
        """Search nodes, topics, and endpoints matching pattern."""
        graph = self._load_graph()
        pattern_lower = pattern.lower()
        pattern_re = re.compile(pattern_lower, re.IGNORECASE)

        results = {
            'ros2_nodes': [],
            'topics': [],
            'http_endpoints': [],
            'http_clients': [],
        }

        # Search ROS2 nodes (key is 'nodes' in system_graph.json)
        for node in graph.get('nodes', []):
            name = node.get('name', '')
            if pattern_re.search(name) or pattern_re.search(node.get('file_path', '')):
                results['ros2_nodes'].append({
                    'name': name,
                    'class': node.get('class_name', ''),
                    'file': node.get('file_path', ''),
                    'publishers': [p.get('topic') for p in node.get('publishers', [])],
                    'subscribers': [s.get('topic') for s in node.get('subscribers', [])],
                })
                if len(results['ros2_nodes']) >= limit:
                    break

        # Search topics (nested under graph['topics']['topics'] as dict)
        topics_data = graph.get('topics', {})
        topics_dict = topics_data.get('topics', {}) if isinstance(topics_data, dict) else {}
        for topic_name, topic_info in topics_dict.items():
            if pattern_re.search(topic_name):
                results['topics'].append({
                    'name': topic_name,
                    'msg_type': topic_info.get('msg_type', 'unknown'),
                    'publishers': topic_info.get('publishers', []),
                    'subscribers': topic_info.get('subscribers', []),
                    'connected': topic_info.get('is_connected', False),
                })
                if len(results['topics']) >= limit:
                    break

        # Search HTTP endpoints
        for endpoint in graph.get('http_endpoints', []):
            path = endpoint.get('path', '')
            if pattern_re.search(path) or pattern_re.search(endpoint.get('handler_name', '')):
                results['http_endpoints'].append({
                    'path': path,
                    'methods': endpoint.get('methods', []),
                    'handler': endpoint.get('handler_name', ''),
                    'file': endpoint.get('file_path', ''),
                    'inferred_host': endpoint.get('inferred_host', 'unknown'),
                    'inferred_port': endpoint.get('inferred_port', 'unknown'),
                })
                if len(results['http_endpoints']) >= limit:
                    break

        # Search HTTP clients
        for client in graph.get('http_clients', []):
            url = client.get('target_url', '')
            if pattern_re.search(url):
                results['http_clients'].append({
                    'url': url,
                    'method': client.get('method', 'GET'),
                    'file': client.get('file_path', ''),
                    'context': client.get('context', ''),
                })
                if len(results['http_clients']) >= limit:
                    break

        # Count total matches
        total = sum(len(v) for v in results.values())

        return {
            'pattern': pattern,
            'total_matches': total,
            'results': results,
        }

    def get_node_details(self, node_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific ROS2 node."""
        graph = self._load_graph()
        for node in graph.get('nodes', []):
            if node.get('name', '').lower() == node_name.lower():
                return {
                    'name': node.get('name'),
                    'class_name': node.get('class_name'),
                    'file_path': node.get('file_path'),
                    'publishers': node.get('publishers', []),
                    'subscribers': node.get('subscribers', []),
                    'services': node.get('services', []),
                    'timers': node.get('timers', []),
                    'parameters': node.get('parameters', []),
                }
        return None

    def get_topic_connections(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """Get publishers and subscribers for a topic."""
        graph = self._load_graph()
        topics_data = graph.get('topics', {})
        topics_dict = topics_data.get('topics', {}) if isinstance(topics_data, dict) else {}
        if topic_name in topics_dict:
            topic_info = topics_dict[topic_name]
            return {
                'name': topic_name,
                'msg_type': topic_info.get('msg_type'),
                'publishers': topic_info.get('publishers', []),
                'subscribers': topic_info.get('subscribers', []),
                'connected': topic_info.get('is_connected', False),
            }
        return None

    def get_http_endpoints(self) -> List[Dict[str, Any]]:
        """Get all HTTP endpoints for health checking."""
        graph = self._load_graph()
        endpoints = []
        for ep in graph.get('http_endpoints', []):
            host = ep.get('inferred_host', 'localhost')
            port = ep.get('inferred_port', 8080)
            # Only include endpoints with known hosts
            if host and port:
                endpoints.append({
                    'path': ep.get('path', '/'),
                    'host': host,
                    'port': port,
                    'methods': ep.get('methods', ['GET']),
                    'handler': ep.get('handler_name', ''),
                })
        return endpoints

    def get_coupling_pairs(self, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Get highly coupled node pairs."""
        graph = self._load_graph()
        coupling = graph.get('coupling', {}).get('pairs', [])
        return [
            {
                'source': p.get('source'),
                'target': p.get('target'),
                'score': p.get('score'),
            }
            for p in coupling
            if p.get('score', 0) >= min_score
        ]

    def get_orphaned_topics(self) -> Dict[str, List[str]]:
        """Get topics with missing publishers or subscribers."""
        graph = self._load_graph()
        topics_data = graph.get('topics', {})
        topics_dict = topics_data.get('topics', {}) if isinstance(topics_data, dict) else {}
        orphaned = {
            'no_publishers': [],
            'no_subscribers': [],
        }
        for name, topic_info in topics_dict.items():
            pubs = topic_info.get('publishers', [])
            subs = topic_info.get('subscribers', [])
            if not pubs and subs:
                orphaned['no_publishers'].append(name)
            if pubs and not subs:
                orphaned['no_subscribers'].append(name)
        return orphaned
