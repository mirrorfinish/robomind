# RoboMind Feedback Specification v2

## Validation Context

This feedback is based on live validation against **NEXUS/BetaRay** - a production 4-Jetson distributed robot system running 16+ HTTP services on Thor with internal ROS2 on Nav Jetson.

**Validation Results**:
- RoboMind reported 130 nodes, 57 critical issues
- Live system runs ~19 ROS2 nodes on Nav Jetson
- **96% false positive rate** - only 2 actionable issues found

---

## Problem Statement

RoboMind v1.0 produces unusable output on production robotics systems because it:

1. **Analyzes ALL code** including `build/`, `install/`, archives, and experiments
2. **Counts duplicates as separate nodes** (same file in src/build/install = 3 nodes)
3. **Cannot detect HTTP communication** - misses the actual inter-system protocol
4. **No deployment awareness** - treats dead code the same as production code
5. **No runtime validation** - static analysis can't determine what actually runs

---

## Feature 1: Build Artifact Exclusion

### Problem
RoboMind found 76 nodes in "unknown" package - these are duplicate definitions:

```
betaray_motor_controller  ← src/betaray_motors/.../motor_controller_node.py
betaray_motor_controller  ← build/betaray_motors/.../motor_controller_node.py
betaray_motor_controller  ← install/betaray_motors/.../motor_controller_node.py
```

### Solution

**Default excludes** (always applied unless `--no-default-excludes`):

```python
DEFAULT_EXCLUDES = [
    "**/build/**",
    "**/install/**",
    "**/log/**",
    "**/__pycache__/**",
    "**/*.egg-info/**",
    "**/archive/**",
    "**/backup/**",
    "**/old/**",
    "**/deprecated/**",
    "**/*_backup.py",
    "**/*_old.py",
    "**/*_backup_*.py",
]
```

**Expected impact**: Reduces 130 nodes → ~54 nodes (58% reduction)

---

## Feature 2: Launch File Tracing

### Problem
RoboMind reports all nodes found in Python files. Only ~19 actually launch via `betaray_navigation_jetson.launch.py`.

### Solution

**Flag**: `--trace-launch <file.launch.py>`

```bash
robomind analyze ~/betaray \
  --trace-launch src/betaray_core/launch/betaray_navigation_jetson.launch.py
```

**Implementation**:

```python
def trace_launch_file(launch_file: Path) -> Set[str]:
    """Extract nodes that actually launch from a launch.py file."""
    deployed_nodes = set()

    # Parse Node() declarations
    # Node(package='betaray_motors', executable='motor_controller', name='motor_controller')

    # Follow IncludeLaunchDescription
    # Follow TimerAction wrapped nodes
    # Handle GroupAction
    # Handle conditional launches (IfCondition)

    return deployed_nodes
```

**Output enhancement**:

```yaml
nodes:
  motor_controller:
    package: betaray_motors
    deployment_status: DEPLOYED  # Found in launch file
    launch_delay: 0s

  guardian_bridge:
    package: betaray_guardian
    deployment_status: NOT_DEPLOYED  # Not in any launch file
    confidence: 0.1
```

**Expected impact**: Filters 54 nodes → 19 deployed nodes

---

## Feature 3: HTTP Communication Detection

### Problem
NEXUS uses HTTP for cross-Jetson communication. RoboMind only sees ROS2:

```
Thor (HTTP)  ←→  Nav Jetson (ROS2 internal)
             ←→  Vision Jetson (HTTP :9091)
             ←→  AI Jetson (HTTP :8080)
```

RoboMind reported Vision→AI ROS2 topics as "orphaned" - they're dead code because vision uses HTTP.

### Solution

**Detect HTTP servers** (Flask, FastAPI, aiohttp):

```python
HTTP_SERVER_PATTERNS = [
    r'@app\.route\([\'"](.+?)[\'"]\)',           # Flask
    r'@app\.(get|post|put|delete)\([\'"](.+?)[\'"]\)',  # FastAPI
    r'app\.add_api_route\([\'"](.+?)[\'"]\)',    # FastAPI alt
    r'web\.Application\(\).*add_routes',          # aiohttp
]
```

**Detect HTTP clients**:

```python
HTTP_CLIENT_PATTERNS = [
    r'requests\.(get|post|put|delete)\([\'"](.+?)[\'"]\)',
    r'aiohttp\.ClientSession\(\)\.get\(',
    r'httpx\.(get|post|AsyncClient)',
    r'urllib\.request\.urlopen\(',
]
```

**New output section**:

```yaml
http_communication:
  servers:
    - file: canary_http_server.py
      port: 8081
      endpoints:
        - path: /api/speak
          method: POST
        - path: /health
          method: GET

  clients:
    - file: unified_ai_processor.py:89
      target: http://vision-jetson.local:9091/detections

  cross_system_summary:
    protocol: HTTP
    ros2_cross_system: false
```

**Impact on findings**:

```yaml
findings:
  - id: VISION-001
    type: orphaned_subscriber
    topic: /detections
    severity: high → none
    reason: "HTTP endpoint detected for same data (vision-jetson:9091/detections)"
    status: SUPERSEDED_BY_HTTP
```

---

## Feature 4: Confidence Scoring

### Problem
All findings reported with equal weight. Dead code findings bury real issues.

### Solution

**Confidence calculation**:

```python
def calculate_confidence(finding: Finding, context: AnalysisContext) -> float:
    confidence = 0.5  # Base

    # Positive signals
    if finding.file in context.launch_deployed_files:
        confidence += 0.4
    if finding.file in context.systemd_deployed_files:
        confidence += 0.5
    if finding.has_matching_pub_sub:
        confidence += 0.2

    # Negative signals
    if "/archive/" in finding.file:
        confidence -= 0.4
    if "/build/" in finding.file or "/install/" in finding.file:
        confidence -= 0.5  # Duplicate
    if context.http_endpoint_exists_for_topic(finding.topic):
        confidence -= 0.3  # HTTP supersedes ROS2
    if finding.file not in context.git_files_modified_last_6_months:
        confidence -= 0.1

    return max(0.0, min(1.0, confidence))
```

**Output**:

```yaml
findings:
  - id: ESTOP-001
    topic: emergency_stop
    severity: critical
    confidence: 0.95
    factors:
      - "File in production launch": +0.4
      - "Motor controller is critical path": +0.1

  - id: GUARDIAN-001
    topic: /nexus/guardian/constitutional_status
    severity: high
    confidence: 0.15
    factors:
      - "Not in any launch file": -0.4
      - "HTTP Guardian service running on Thor": -0.3
    status: LIKELY_FALSE_POSITIVE
```

**Default filter**: `--min-confidence 0.5` (hide low-confidence findings)

---

## Feature 5: Systemd Service Discovery

### Problem
On Thor, 16 systemd services run the actual system. RoboMind doesn't know about them.

### Solution

**Flag**: `--discover-systemd <host>` or `--systemd-manifest <file>`

```bash
# Direct discovery (requires SSH)
robomind analyze ~/betaray --discover-systemd thor@localhost

# Or provide manifest
robomind analyze ~/betaray --systemd-manifest deployment.yaml
```

**Manifest format**:

```yaml
# deployment.yaml
hosts:
  thor:
    services:
      - name: thor-guardian
        exec: /home/thor/betaray/nexus_system/guardian_master_node.py
        protocol: http
        port: 8095

      - name: thor-voice-server
        exec: /home/thor/betaray/nexus_system/thor_voice_server.py
        protocol: http

  nav_jetson:
    services:
      - name: betaray-navigation
        exec: ros2 launch betaray_core betaray_navigation_jetson.launch.py
        protocol: ros2
        ros_domain_id: 0
```

**Impact**: Marks HTTP services as production, ROS2 code connecting to them as potentially dead.

---

## Feature 6: Runtime Validation Mode

### Problem
Static analysis cannot determine:
- Which nodes actually run
- Which topics have active publishers/subscribers
- Whether HTTP endpoints respond

### Solution

**Command**:

```bash
robomind validate ~/betaray_analysis \
  --ssh robot@betaray-nav.local \
  --http thor.local:8087,8095,9091
```

**Validation steps**:

```python
async def validate_runtime(analysis: Analysis, targets: List[Target]):
    results = ValidationResults()

    # ROS2 validation (via SSH)
    for ros_target in targets.ros2:
        nodes = await ssh_exec(ros_target, "ros2 node list")
        topics = await ssh_exec(ros_target, "ros2 topic list")

        for finding in analysis.findings:
            if finding.node in nodes:
                finding.runtime_status = "CONFIRMED"
            else:
                finding.runtime_status = "NOT_RUNNING"
                finding.confidence *= 0.2

    # HTTP validation
    for http_target in targets.http:
        health = await http_get(f"{http_target}/health")
        if health.ok:
            # Mark ROS2 findings for same service as likely dead
            ...

    return results
```

**Output**:

```yaml
runtime_validation:
  timestamp: 2026-01-26T20:30:00Z

  ros2_nodes:
    confirmed_running:
      - motor_controller
      - unified_ai_processor
      - tts_service
    not_found:
      - guardian_bridge      # Finding confidence: 0.95 → 0.19
      - reasoning_engine     # Finding confidence: 0.80 → 0.16

  http_endpoints:
    healthy:
      - thor:8087 (vLLM)
      - thor:8095 (Guardian)
      - vision-jetson:9091

  findings_updated: 47
  false_positives_detected: 43
```

---

## Feature 7: AI-Optimized Output

### Problem
Current output requires manual validation. LLMs can help but need structured context.

### Solution

**Flag**: `--format ai-context`

```yaml
# Optimized for LLM consumption
system_architecture:
  type: "distributed multi-jetson"
  cross_system_protocol: http
  internal_protocol: ros2

  hosts:
    thor:
      role: "main brain"
      services: [guardian, vllm, voice, memory, vision-router]
      protocol: http

    nav_jetson:
      role: "navigation"
      ros2_domain: 0
      deployed_nodes: 19

    vision_jetson:
      role: "vision"
      protocol: http_only
      port: 9091

actionable_findings:
  - id: ESTOP-001
    priority: 1
    summary: "Emergency stop uses inconsistent topic naming"
    file: motor_controller_node.py:63
    current: "create_subscription(Bool, 'emergency_stop', ...)"
    fix: "create_subscription(Bool, '/emergency_stop', ...)"
    risk: "E-stop may fail if node runs in namespace"
    confidence: 0.95
    runtime_confirmed: true

ignored_findings:
  count: 55
  reason: "Low confidence (<0.5) - likely dead code or duplicates"

dead_code_detected:
  - path: betaray-nexus-integration/
    reason: "ROS2↔NEXUS bridge - NEXUS uses HTTP directly"
  - path: betaray_ai/reasoning_engine_node.py
    reason: "Replaced by unified_ai_processor"
```

---

## Implementation Priority

| Phase | Feature | Effort | Impact |
|-------|---------|--------|--------|
| 1 | Build artifact exclusion | Low | High (58% noise reduction) |
| 2 | Confidence scoring | Low | High (surfaces real issues) |
| 3 | Launch file tracing | Medium | High (identifies deployed code) |
| 4 | HTTP detection | Medium | High (catches modern architectures) |
| 5 | Systemd discovery | Medium | Medium |
| 6 | Runtime validation | High | Very High (ground truth) |
| 7 | AI-optimized output | Low | Medium |

**Phase 1+2 alone would have reduced false positives from 96% to ~40%.**

---

## Verification Criteria

Re-run on BetaRay/NEXUS after implementing:

```bash
robomind analyze ~/betaray \
  --trace-launch src/betaray_core/launch/betaray_navigation_jetson.launch.py \
  --systemd-manifest ~/deployment.yaml \
  --min-confidence 0.5 \
  --format ai-context \
  -o ~/betaray_analysis_v2

robomind validate ~/betaray_analysis_v2 \
  --ssh robot@betaray-nav.local \
  --http localhost:8087,localhost:8095
```

**Success metrics**:

| Metric | Before | Target |
|--------|--------|--------|
| Nodes reported | 130 | ~19 (deployed only) |
| False positive rate | 96% | <20% |
| Actionable findings | 2 buried in 57 | 2 at top |
| HTTP endpoints detected | 0 | 10+ |
| Runtime validation | None | Integrated |

---

## Appendix: BetaRay/NEXUS Ground Truth

### Actually Running (Thor - 16 HTTP services)
```
thor-guardian.service          → guardian_master_node.py
thor-deep-reasoning.service    → vLLM :8087
thor-memory-api.service        → memory API
thor-voice-server.service      → thor_voice_server.py
thor-vision-router.service     → vision routing
thor-thought-loop-v2.service   → perception
thor-face-recognition.service  → face recognition
+ 9 more watchdogs/utilities
```

### Actually Running (Nav Jetson - 19 ROS2 nodes)
```
motor_controller, odometry_publisher, betaray_navigation_sensors,
memory_manager, rplidar_node, slam_toolbox, Nav2 stack,
tts_router, tts_service, unified_ai_processor, query_classifier,
ai_tools_service, voice_control, voice_coordinator, exploration_behavior,
system_coordinator, memory_system, robust_camera_launcher
```

### Dead Code (analyzed but not deployed)
```
betaray-nexus-integration/     → ROS2↔NEXUS bridge (unused)
reasoning_engine_node.py       → Replaced by unified_ai_processor
guardian_bridge_node.py        → Guardian is HTTP on Thor
constitutional_*.py            → Never deployed
vision_obstacle_detector.py    → Vision uses HTTP
76 duplicate nodes             → build/install artifacts
```

---

*Feedback based on live validation: 2026-01-26*
*System: NEXUS/BetaRay 4-Jetson distributed robot*
*Validation method: systemd service check, process list, HTTP health endpoints*
