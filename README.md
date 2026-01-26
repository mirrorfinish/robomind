# RoboMind

**Rapid Prototyping System for Autonomous ROS2 Robots**

Analyze, visualize, and accelerate development of ROS2 robotics projects. RoboMind scans your codebase, extracts ROS2 patterns (nodes, topics, services, parameters), and generates structured output for AI-assisted development and human understanding.

## Features

- **Project Scanning** - Discover Python files, ROS2 packages, launch files, configs
- **Python AST Parsing** - Extract classes, functions, imports, decorators
- **ROS2 Pattern Detection** - Publishers, subscribers, services, actions, parameters, timers
- **Launch File Analysis** - Parse `.launch.py` files for node sequences and arguments
- **Parameter Extraction** - Extract YAML config parameters
- **Dependency Graphs** - NetworkX-based system graph with coupling analysis
- **Multiple Outputs** - JSON (machine-readable), YAML (AI-optimized), HTML (interactive D3.js)
- **Distributed Analysis** - SSH to remote hosts for multi-machine systems

## Installation

```bash
# Clone the repository
git clone https://github.com/mirrorfinish/robomind.git
cd robomind

# Install in development mode
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Dependencies

Required:
- Python 3.10+
- click
- rich
- pyyaml
- networkx

Optional:
- paramiko (for SSH remote analysis)

## Quick Start

```bash
# Scan a project directory
robomind scan ~/my_robot_project

# Full analysis with all outputs
robomind analyze ~/my_robot_project -o ./analysis/

# Generate interactive visualization
robomind visualize ~/my_robot_project -o graph.html --open

# See all capabilities
robomind info
```

## Commands

### `robomind scan <path>`

Scan a project directory for Python files and ROS2 packages.

```bash
robomind scan ~/betaray --verbose
robomind scan ~/betaray --output scan_results.json
robomind scan ~/betaray --exclude "*/archive/*"  # Skip archived code
```

**Output:**
- ROS2 package count
- Python file count and lines of code
- Launch files and config files found

### `robomind analyze <path>`

Perform full analysis including ROS2 extraction, graph building, and export.

```bash
# Basic analysis
robomind analyze ~/betaray -o ./betaray_analysis/

# Specify output formats
robomind analyze ~/betaray -f json -f yaml -f html

# Exclude archived/backup directories
robomind analyze ~/betaray --exclude "*/archive/*" --exclude "*backup*"

# Include remote hosts (distributed systems)
robomind analyze ~/betaray \
    --remote robot@nav.local:~/betaray \
    --remote robot@ai.local:~/betaray \
    --key ~/.ssh/id_rsa

# Keep synced remote code for inspection
robomind analyze ~/betaray --remote robot@jetson.local --keep-remote
```

**Options:**
| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory (default: `robomind_analysis`) |
| `-f, --format` | Output formats: `json`, `yaml`, `html` (multiple allowed) |
| `-e, --exclude` | Glob patterns to exclude (e.g., `*/archive/*`) (multiple allowed) |
| `-r, --remote` | Remote host spec: `user@host:path` (multiple allowed) |
| `-k, --key` | SSH private key file |
| `--keep-remote` | Keep local copies of synced remote code |
| `-v, --verbose` | Verbose output with tracebacks |

### `robomind launch <path>`

Analyze ROS2 launch files and parameter configurations.

```bash
robomind launch ~/betaray --verbose
robomind launch ~/betaray --output launch_topology.json
```

**Extracts:**
- Node declarations with packages/executables
- Launch arguments
- Parameter files and overrides
- Conditional logic (if/unless)
- Timer delays

### `robomind graph <path>`

Build and analyze the system dependency graph.

```bash
robomind graph ~/betaray --coupling --verbose
robomind graph ~/betaray --output graph.json --graphml graph.graphml
```

**Features:**
- NetworkX-based system graph
- Coupling strength analysis
- Cycle detection
- Critical node identification (centrality)
- Export to GraphML for Gephi/Cytoscape

### `robomind visualize <path>`

Generate interactive HTML visualization.

```bash
robomind visualize ~/betaray -o graph.html --open
```

**Visualization Features:**
- D3.js force-directed layout
- Color-coded nodes by type (ROS2_NODE, TOPIC, SERVICE, etc.)
- Interactive zoom/pan
- Search and filter by type
- Click nodes for detailed info panel
- Standalone HTML (no server required)

### `robomind remote <hosts...>`

Test SSH connections and get remote ROS2 info.

```bash
# Test connections
robomind remote robot@nav.local robot@ai.local

# Get live ROS2 system info
robomind remote robot@jetson.local --ros2-info --verbose

# Use specific SSH key
robomind remote robot@nav.local --key ~/.ssh/robot_key
```

### `robomind info`

Show RoboMind capabilities and quick start guide.

## Output Formats

### JSON (`system_graph.json`)

Complete machine-readable system graph for AI tools and processing.

```json
{
  "metadata": {
    "project_name": "BetaRay",
    "generated_at": "2026-01-26T12:00:00",
    "tool": "RoboMind",
    "version": "0.1.0"
  },
  "summary": {
    "total_nodes": 23,
    "total_topics": 45,
    "total_services": 12
  },
  "nodes": [
    {
      "name": "motor_controller",
      "class_name": "MotorController",
      "file_path": "control/motor_controller.py",
      "publishers": [{"topic": "/motor_vel", "msg_type": "Float32"}],
      "subscribers": [{"topic": "/cmd_vel", "msg_type": "Twist"}]
    }
  ],
  "graph": {
    "nodes": [...],
    "edges": [...]
  },
  "coupling": {
    "pairs": [{"source": "nav", "target": "motor", "score": 0.72}]
  }
}
```

### YAML (`system_context.yaml`)

AI-optimized context file for use with Claude, GPT, etc. Token-efficient format.

```yaml
# System Context - Auto-generated by RoboMind
# ~200 tokens for full architecture overview

metadata:
  name: BetaRay
  version: auto-2026-01-26
  nodes: 23
  distributed_hosts:
    - thor
    - nav_orin
    - ai_orin

architecture:
  nav_orin:
    components: [odometry_publisher, motor_controller, navigation_node]
    node_count: 8
  ai_orin:
    components: [reasoning_engine, voice_coordinator, llm_interface]
    node_count: 6

nodes:
  motor_controller:
    class: MotorController
    file: motor_controller.py
    publishes:
      - topic: /motor_vel
        type: Float32
    subscribes:
      - topic: /cmd_vel
        type: Twist
    timers:
      - period: 0.02
        hz: 50.0
    parameters:
      - max_speed
      - acceleration_limit

topics:
  /cmd_vel:
    msg_type: Twist
    publishers: [navigation_node]
    subscribers: [motor_controller]
    connected: true
```

### YAML Summary (`CONTEXT_SUMMARY.yaml`)

Ultra-compact summary (~30 tokens) for quick LLM context.

```yaml
# Quick Reference - Auto-generated by RoboMind
# ~30 tokens

system: BetaRay
version: auto-2026-01-26
nodes: 23
topics: 45
connected: 32
```

### HTML (`visualization.html`)

Interactive D3.js graph visualization:
- **Node Colors:**
  - Blue: ROS2 Nodes
  - Yellow: Topics
  - Purple: Services
  - Green: Parameters
- **Features:**
  - Force-directed layout
  - Click nodes for details
  - Search by name
  - Filter by type
  - Zoom/pan

## Coupling Analysis

RoboMind calculates coupling strength between components based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Topic Connections | 40% | Number and frequency of shared topics |
| Shared Parameters | 30% | Parameters used by both components |
| Data Complexity | 20% | Message type complexity |
| Temporal Coupling | 10% | Timer synchronization |

**Coupling Levels:**
- **Critical (>0.7)**: Tightly coupled, changes affect both
- **High (0.5-0.7)**: Significant dependencies
- **Medium (0.3-0.5)**: Moderate coupling
- **Low (<0.3)**: Loosely coupled

## Distributed Systems

For multi-machine ROS2 systems (like Jetson clusters), RoboMind can analyze code across hosts:

```bash
robomind analyze ~/local_project \
    --remote robot@nav-jetson.local:~/betaray \
    --remote robot@ai-jetson.local:~/betaray \
    --remote voice@vision-jetson.local:~/betaray \
    --key ~/.ssh/id_rsa \
    --output ~/full_analysis \
    --keep-remote
```

**Workflow:**
1. Test SSH connections to all hosts
2. Rsync code from each remote to local temp directory
3. Analyze each codebase
4. Tag nodes with hardware target
5. Merge into unified system graph
6. Generate combined output

## Using with AI Assistants

### Claude/GPT Integration

1. Generate YAML context:
   ```bash
   robomind analyze ~/my_robot -f yaml
   ```

2. Copy `CONTEXT_SUMMARY.yaml` or `system_context.yaml` to your AI chat

3. Ask questions like:
   - "What components handle voice processing?"
   - "Which nodes publish to /cmd_vel?"
   - "How are the navigation and motor systems connected?"

### Benefits
- 10x faster context loading vs reading raw code
- Accurate architecture understanding
- Structured data for precise queries

## Development Status

| Feature | Status | Tests |
|---------|--------|-------|
| Project Scanner | ✅ Complete | 19 |
| Python AST Parser | ✅ Complete | 9 |
| ROS2 Node Extraction | ✅ Complete | 12 |
| Topic Graph Builder | ✅ Complete | 6 |
| Launch File Analysis | ✅ Complete | 27 |
| System Graph (NetworkX) | ✅ Complete | 23 |
| Coupling Analysis | ✅ Complete | 23 |
| JSON Exporter | ✅ Complete | 14 |
| YAML Exporter | ✅ Complete | 14 |
| HTML Visualization | ✅ Complete | 24 |
| SSH Remote Analysis | ✅ Complete | 35 |
| **Total** | **✅ Complete** | **192** |

## Architecture

```
robomind/
├── cli.py                  # Click-based CLI
├── __init__.py             # Version info
├── core/
│   ├── scanner.py          # Project file discovery
│   ├── parser.py           # Python AST parsing
│   └── graph.py            # NetworkX system graph
├── ros2/
│   ├── node_extractor.py   # ROS2 node analysis
│   ├── topic_extractor.py  # Topic graph builder
│   ├── launch_analyzer.py  # Launch file parsing
│   └── param_extractor.py  # YAML config extraction
├── analyzers/
│   └── coupling.py         # Coupling strength calculator
├── exporters/
│   ├── json_exporter.py    # JSON system graph
│   ├── yaml_exporter.py    # AI-optimized YAML
│   └── html_exporter.py    # D3.js visualization
├── remote/
│   └── ssh_analyzer.py     # SSH distributed analysis
└── templates/
    └── visualization.html  # D3.js template
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scanner.py -v

# Run with coverage
pytest tests/ --cov=robomind --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Related Projects

- [BetaRay](https://github.com/mirrorfinish/betaray) - The autonomous robot RoboMind was developed for
- [ROS2](https://docs.ros.org/) - Robot Operating System 2
- [D3.js](https://d3js.org/) - Visualization library used for HTML export
