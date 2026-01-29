"""RoboMind MCP Server - Main entry point."""

import asyncio
import json
import os
import sys
from typing import Any

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .graph_loader import SystemGraph


# Initialize server and graph
server = Server("robomind")
graph = SystemGraph()


# NEXUS endpoints for health checking (from topology.yaml)
NEXUS_ENDPOINTS = [
    {"name": "thor-deep-reasoning", "host": "thor.local", "port": 8088, "path": "/api/health"},
    {"name": "thor-memory-api", "host": "thor.local", "port": 8089, "path": "/health"},
    {"name": "thor-face-recognition", "host": "thor.local", "port": 8090, "path": "/health"},
    {"name": "thor-vision-router", "host": "thor.local", "port": 8091, "path": "/health"},
    {"name": "thor-chromadb", "host": "localhost", "port": 8000, "path": "/api/v2/heartbeat"},
    {"name": "nav-canary", "host": "betaray-nav.local", "port": 8081, "path": "/health"},
    {"name": "ai-http", "host": "betaray-ai.local", "port": 8080, "path": "/health"},
    {"name": "vision-frame", "host": "vision-jetson.local", "port": 9091, "path": "/health"},
    {"name": "vllm-thought", "host": "localhost", "port": 30000, "path": "/health"},
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RoboMind tools."""
    return [
        Tool(
            name="query",
            description="Search the ROS2/HTTP system graph for nodes, topics, and endpoints matching a pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (e.g., 'motor', '/cmd_vel', 'tts')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results per category (default: 10)",
                        "default": 10
                    }
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="health",
            description="Check HTTP endpoint reachability for NEXUS services",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (default: 2)",
                        "default": 2
                    }
                }
            }
        ),
        Tool(
            name="summary",
            description="Get high-level summary of the analyzed system",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="topic",
            description="Get detailed info about a specific ROS2 topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Topic name (e.g., '/cmd_vel', '/betaray/tts/speak')"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="node",
            description="Get detailed info about a specific ROS2 node",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Node name (e.g., 'motor_controller', 'parallel_voice_node')"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="orphans",
            description="Get topics with missing publishers or subscribers (potential connectivity issues)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="coupling",
            description="Get highly coupled node pairs (potential refactoring targets)",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_score": {
                        "type": "number",
                        "description": "Minimum coupling score (0-1, default: 0.3)",
                        "default": 0.3
                    }
                }
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "query":
            pattern = arguments.get("pattern", "")
            limit = arguments.get("limit", 10)
            result = graph.query(pattern, limit=limit)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "health":
            timeout = arguments.get("timeout", 2)
            results = {}
            for ep in NEXUS_ENDPOINTS:
                url = f"http://{ep['host']}:{ep['port']}{ep['path']}"
                try:
                    resp = requests.get(url, timeout=timeout)
                    if resp.ok:
                        results[ep['name']] = {"status": "healthy", "code": resp.status_code}
                    else:
                        results[ep['name']] = {"status": "degraded", "code": resp.status_code}
                except requests.exceptions.Timeout:
                    results[ep['name']] = {"status": "timeout", "url": url}
                except requests.exceptions.ConnectionError:
                    results[ep['name']] = {"status": "unreachable", "url": url}
                except Exception as e:
                    results[ep['name']] = {"status": "error", "error": str(e)}

            # Count by status
            healthy = sum(1 for r in results.values() if r['status'] == 'healthy')
            total = len(results)

            output = {
                "summary": f"{healthy}/{total} services healthy",
                "services": results
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        elif name == "summary":
            result = graph.get_summary()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "topic":
            topic_name = arguments.get("name", "")
            result = graph.get_topic_connections(topic_name)
            if result:
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                return [TextContent(type="text", text=f"Topic '{topic_name}' not found in system graph")]

        elif name == "node":
            node_name = arguments.get("name", "")
            result = graph.get_node_details(node_name)
            if result:
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            else:
                return [TextContent(type="text", text=f"Node '{node_name}' not found in system graph")]

        elif name == "orphans":
            result = graph.get_orphaned_topics()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "coupling":
            min_score = arguments.get("min_score", 0.3)
            result = graph.get_coupling_pairs(min_score)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except FileNotFoundError as e:
        return [TextContent(type="text", text=f"Analysis not found. Run 'robomind analyze ~/betaray' first. Error: {e}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    """Entry point for the MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    # Check for --test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("RoboMind MCP Server - Test Mode")
        print(f"Analysis path: {graph.analysis_path}")
        try:
            summary = graph.get_summary()
            print(f"Loaded graph: {summary['total_nodes']} nodes, {summary['total_topics']} topics")
            print("MCP server ready!")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Run 'robomind analyze ~/betaray' to generate analysis first")
        sys.exit(0)

    run()
