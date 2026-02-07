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

    def get_message_schema(self, type_name: str) -> Optional[Dict[str, Any]]:
        """Look up a message definition by type name (supports fuzzy matching)."""
        graph = self._load_graph()
        msg_defs = graph.get('message_definitions', {})
        if not msg_defs:
            return None

        # Exact match
        if type_name in msg_defs:
            return msg_defs[type_name]

        # Short name match (e.g. "LaserScan" -> "sensor_msgs/msg/LaserScan")
        type_lower = type_name.lower()
        for full_name, definition in msg_defs.items():
            short = full_name.split('/')[-1]
            if short.lower() == type_lower:
                return {**definition, '_matched_as': full_name}

        # Partial match
        matches = []
        for full_name, definition in msg_defs.items():
            if type_lower in full_name.lower():
                matches.append({**definition, '_matched_as': full_name})

        if len(matches) == 1:
            return matches[0]
        elif matches:
            return {
                'ambiguous': True,
                'matches': [m['_matched_as'] for m in matches],
                'hint': 'Use the full name for exact match',
            }

        return None

    def get_impact(self, target: str, target_type: str = "topic") -> Dict[str, Any]:
        """Analyze impact of a change using the graph data."""
        graph = self._load_graph()
        nodes = graph.get('nodes', [])
        topics_data = graph.get('topics', {})
        topics_dict = topics_data.get('topics', {}) if isinstance(topics_data, dict) else {}

        # Build publisher/subscriber indices from graph data
        topic_publishers: Dict[str, List[str]] = {}
        topic_subscribers: Dict[str, List[str]] = {}
        node_pub_topics: Dict[str, List[str]] = {}
        node_sub_topics: Dict[str, List[str]] = {}
        topic_types: Dict[str, str] = {}

        for node in nodes:
            name = node.get('name', '')
            pub_topics = []
            for pub in node.get('publishers', []):
                topic = pub.get('topic', '')
                if topic:
                    topic_publishers.setdefault(topic, []).append(name)
                    pub_topics.append(topic)
                    if pub.get('msg_type'):
                        topic_types[topic] = pub['msg_type']
            node_pub_topics[name] = pub_topics

            sub_topics = []
            for sub in node.get('subscribers', []):
                topic = sub.get('topic', '')
                if topic:
                    topic_subscribers.setdefault(topic, []).append(name)
                    sub_topics.append(topic)
                    if sub.get('msg_type') and topic not in topic_types:
                        topic_types[topic] = sub['msg_type']
            node_sub_topics[name] = sub_topics

        # Safety-critical topics
        critical_topics = {"/cmd_vel", "/emergency_stop", "/motor/command",
                          "/betaray/motors/cmd_vel", "/robot/cmd_vel",
                          "/betaray/emergency_stop"}

        def get_severity(topic: str) -> str:
            if topic in critical_topics:
                return "critical"
            if any(topic.startswith(p) for p in ("/betaray/debug/", "/betaray/log/",
                                                  "/diagnostics", "/rosout", "/parameter_events")):
                return "low"
            return "medium"

        result = {"target": target, "target_type": target_type,
                  "directly_affected": [], "cascade_affected": []}

        if target_type == "topic":
            sev = get_severity(target)
            for pub in topic_publishers.get(target, []):
                result["directly_affected"].append({
                    "name": pub, "kind": "node", "impact_type": "lost_publisher",
                    "severity": "critical" if sev == "critical" else "high",
                    "description": f"Publishes to {target}",
                })
            for sub in topic_subscribers.get(target, []):
                result["directly_affected"].append({
                    "name": sub, "kind": "node", "impact_type": "broken_subscriber",
                    "severity": "critical" if sev == "critical" else "high",
                    "description": f"Subscribes to {target}",
                })
            # Cascade
            affected = {i["name"] for i in result["directly_affected"]}
            for sub in topic_subscribers.get(target, []):
                for downstream_topic in node_pub_topics.get(sub, []):
                    if downstream_topic == target:
                        continue
                    for ds_sub in topic_subscribers.get(downstream_topic, []):
                        if ds_sub not in affected:
                            affected.add(ds_sub)
                            result["cascade_affected"].append({
                                "name": ds_sub, "kind": "node", "impact_type": "cascade",
                                "severity": "medium",
                                "description": f"Downstream of {sub} via {downstream_topic}",
                            })

        elif target_type == "node":
            for topic in node_pub_topics.get(target, []):
                other_pubs = [n for n in topic_publishers.get(topic, []) if n != target]
                for sub in topic_subscribers.get(topic, []):
                    if sub == target:
                        continue
                    if not other_pubs:
                        result["directly_affected"].append({
                            "name": sub, "kind": "node", "impact_type": "broken_subscriber",
                            "severity": get_severity(topic),
                            "description": f"Loses {topic} (no remaining publishers)",
                        })
                    else:
                        result["directly_affected"].append({
                            "name": sub, "kind": "node", "impact_type": "lost_publisher",
                            "severity": "low",
                            "description": f"Loses one publisher on {topic} ({len(other_pubs)} remaining)",
                        })

        elif target_type == "message_type":
            type_short = target.split("/")[-1]
            for topic, ttype in topic_types.items():
                if ttype == target or ttype.endswith(f"/{type_short}") or ttype == type_short:
                    for pub in topic_publishers.get(topic, []):
                        result["directly_affected"].append({
                            "name": pub, "kind": "node", "impact_type": "type_user",
                            "severity": "high",
                            "description": f"Publishes {target} on {topic}",
                        })
                    for sub in topic_subscribers.get(topic, []):
                        result["directly_affected"].append({
                            "name": sub, "kind": "node", "impact_type": "type_user",
                            "severity": "high",
                            "description": f"Subscribes to {target} on {topic}",
                        })

        # Deduplicate
        seen = set()
        deduped = []
        for item in result["directly_affected"]:
            key = (item["name"], item["impact_type"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        result["directly_affected"] = deduped

        # Summary counts
        all_items = result["directly_affected"] + result["cascade_affected"]
        by_severity = {}
        for item in all_items:
            by_severity[item["severity"]] = by_severity.get(item["severity"], 0) + 1
        result["summary"] = {
            "total_affected": len(all_items),
            "directly_affected": len(result["directly_affected"]),
            "cascade_affected": len(result["cascade_affected"]),
            "by_severity": by_severity,
        }

        return result

    def get_ai_services(self) -> Optional[Dict[str, Any]]:
        """Get AI/ML service analysis results."""
        graph = self._load_graph()
        ai_data = graph.get('ai_services')
        if not ai_data:
            return None
        return ai_data
