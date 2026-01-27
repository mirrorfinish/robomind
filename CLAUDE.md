# CLAUDE.md - RoboMind

## Project Overview

RoboMind is a rapid prototyping system for autonomous ROS2 robots. It analyzes ROS2 codebases and generates structured output for AI-assisted development.

**Repository**: https://github.com/mirrorfinish/robomind
**Version**: 1.0.0
**Location**: `~/robomind`

## Quick Reference

```bash
# Full analysis with all outputs
robomind analyze ~/my_project -o ./analysis/

# Interactive visualization
robomind visualize ~/my_project -o graph.html --open

# Exclude archived/backup directories
robomind analyze ~/project --exclude "*/archive/*" --exclude "*backup*"

# Distributed system analysis
robomind analyze ~/project \
    --remote robot@nav.local:~/project \
    --remote robot@ai.local:~/project

# Validate against live ROS2 system
robomind validate ~/my_project
robomind validate ~/my_project --ssh robot@jetson.local

# Generate architecture report
robomind report ~/my_project -o ARCHITECTURE_REPORT.md

# Trace data flow between nodes
robomind trace ~/project --from sensor_node --to controller_node
robomind trace ~/project --topic /cmd_vel

# Test remote connections
robomind remote robot@jetson.local --ros2-info
```

## Architecture

```
robomind/
├── cli.py                  # Click-based CLI entry point
├── core/
│   ├── scanner.py          # File discovery (Python, launch, config)
│   ├── parser.py           # Python AST parsing
│   └── graph.py            # NetworkX system graph
├── ros2/
│   ├── node_extractor.py   # ROS2 node analysis
│   ├── topic_extractor.py  # Topic connection graph
│   ├── launch_analyzer.py  # Launch file parsing
│   └── param_extractor.py  # YAML config extraction
├── analyzers/
│   ├── coupling.py         # Component coupling strength
│   └── flow_tracer.py      # Data flow path tracing
├── validators/
│   └── live_validator.py   # Validate against running ROS2
├── reporters/
│   └── markdown_reporter.py # Generate markdown reports
├── exporters/
│   ├── json_exporter.py    # Machine-readable JSON
│   ├── yaml_exporter.py    # AI-optimized YAML
│   └── html_exporter.py    # D3.js visualization
├── remote/
│   └── ssh_analyzer.py     # SSH distributed analysis
└── templates/
    └── visualization.html  # D3.js template
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `robomind scan <path>` | Scan for Python files, packages |
| `robomind analyze <path>` | Full analysis with exports |
| `robomind launch <path>` | Analyze launch files |
| `robomind graph <path>` | Build dependency graph |
| `robomind visualize <path>` | Generate HTML visualization |
| `robomind validate <path>` | Compare against live ROS2 system |
| `robomind report <path>` | Generate markdown architecture report |
| `robomind trace <path>` | Trace data flow between nodes |
| `robomind remote <hosts>` | Test SSH connections |
| `robomind info` | Show capabilities |

## Output Files

```
robomind_analysis/
├── system_graph.json       # Complete machine-readable graph (for tools)
├── system_context.yaml     # Full AI context (~200-20K tokens)
├── CONTEXT_SUMMARY.yaml    # Quick reference (~30-150 tokens)
└── visualization.html      # Interactive D3.js graph
```

## Working with This Codebase

### Running Tests
```bash
cd ~/robomind
pytest tests/ -v              # All tests (248)
pytest tests/test_graph.py    # Specific module
```

### Key Classes

**Core:**
- `ProjectScanner` - Discovers files in project
- `PythonParser` - AST parsing for Python files
- `SystemGraph` - NetworkX-based dependency graph

**ROS2:**
- `ROS2NodeExtractor` - Extracts nodes, publishers, subscribers
- `TopicExtractor` - Builds topic connection graph
- `LaunchFileAnalyzer` - Parses launch.py files

**Analyzers:**
- `CouplingAnalyzer` - Calculates coupling strength between components
- `FlowTracer` - Traces data flow paths through the system

**Validators:**
- `LiveValidator` - Compares static analysis against running ROS2 system

**Reporters:**
- `MarkdownReporter` - Generates comprehensive markdown reports

**Exporters:**
- `JSONExporter` - Full system graph export
- `YAMLExporter` - AI-optimized context export
- `HTMLExporter` - D3.js visualization

**Remote:**
- `SSHAnalyzer` - Single host analysis via SSH
- `DistributedAnalyzer` - Multi-host parallel analysis

### Adding New Features

1. **New ROS2 pattern**: Add to `ros2/node_extractor.py`
2. **New export format**: Create in `exporters/`
3. **New analysis**: Add to `analyzers/`
4. **CLI command**: Add to `cli.py`

### Coupling Weights
```python
WEIGHT_TOPIC_CONNECTION = 0.40
WEIGHT_SHARED_PARAMETERS = 0.30
WEIGHT_DATA_COMPLEXITY = 0.20
WEIGHT_TEMPORAL_COUPLING = 0.10
```

## Integration with NEXUS

RoboMind was built to analyze the BetaRay/NEXUS robotics system. To analyze NEXUS:

```bash
# Analyze the full BetaRay codebase (excluding archived code)
robomind analyze ~/betaray -o ~/betaray_robomind_analysis \
    --exclude "*/archive/*" --exclude "*backup*"

# Results at ~/betaray_robomind_analysis/
# - 130 ROS2 nodes detected (active code only)
# - 204 topics, 65 connected
# - 361 publishers, 327 subscribers
# - 17 packages
```

## Development Notes

- Python 3.10+ required
- Uses NetworkX for graph operations
- D3.js v7 for visualization
- No external ROS2 dependencies (pure static analysis)
- SSH analysis uses subprocess (rsync + ssh), not paramiko
