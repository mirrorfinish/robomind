# RoboMind Feedback Specification

## Document Purpose

This specification provides developer feedback for improving RoboMind based on real-world validation against the NEXUS/BetaRay robotics system - a production 4-Jetson distributed robot architecture.

---

## Executive Summary

**Core Problem**: RoboMind v1.0 produced an 80% false-positive rate when analyzing BetaRay/NEXUS because it:

1. Analyzed dead/archived code alongside production code
2. Only detected ROS2 pub/sub, missing HTTP inter-Jetson communication
3. Reported all findings with equal weight regardless of confidence
4. Had no mechanism to validate against live running systems

**The one real finding** (emergency stop topic mismatch) was buried among 50+ false positives, making the report operationally useless without manual validation.

---

## Validation Results Summary

| Finding | RoboMind Severity | Actual Status | Production Risk |
|---------|-------------------|---------------|-----------------|
| Emergency Stop mismatch | CRITICAL | **REAL** | HIGH |
| Vision→AI pipeline | CRITICAL | DEAD CODE | None |
| Voice dialogue broken | HIGH | DEAD CODE | None |
| Guardian monitoring blind | HIGH | DEAD CODE | None |
| IMU/Odom fusion broken | HIGH | DEAD CODE | None |
| 46 topics missing slashes | MEDIUM | Mostly dead | Minimal |
| Duplicate nodes (4 copies) | INFO | FALSE POSITIVE | None |

**Only 1 of 7 major findings was a real production issue.**

---

## Root Cause Analysis

### Why RoboMind Got It Wrong

#### 1. No Understanding of HTTP Architecture

NEXUS evolved from pure ROS2 to HTTP-based inter-Jetson communication:

```
Thor (main brain)     → HTTP APIs :8087-8096, vLLM :30000
Nav Jetson            → ROS2 DOMAIN_ID=0 + Canary HTTP :8081
Vision Jetson         → HTTP :9091 (camera, YOLO) - NO ROS2 external
AI Jetson             → HTTP :8080 (LLM queries) - NO ROS2 external
Guardian              → HTTP-based Observatory polling - NO ROS2
```

**Cross-Jetson = HTTP. ROS2 is internal to each Jetson only.**

RoboMind only sees Python files with `create_subscription()` calls and assumes ROS2 is the communication layer.

#### 2. Analyzes ALL Code, Not Deployed Code

The codebase contains:
- `nexus_system/` - HTTP orchestration (production)
- `Betaray_Dual_v1/` - Mixed ROS2/HTTP (partially production)
- Archive folders, experiments, dead branches

RoboMind treated a deprecated `guardian_bridge.py` (dead code) the same as `motor_controller_node.py` (actively running).

#### 3. No Runtime Validation

Static analysis cannot determine:
- Which systemd services are enabled
- Which ROS2 nodes actually launch
- Whether topics have active publishers/subscribers

---

## Feature Requests

### Feature 1: Deployment Awareness

#### 1.1 Deployment Manifest Support

**Flag**: `--deployment-manifest <file>`

Accept a YAML file describing what actually runs:

```yaml
# deployment.yaml
jetsons:
  nav_jetson:
    hostname: betaray-nav.local
    ros_domain_id: 0
    systemd_services:
      - betaray-navigation-system
      - canary-http-server
    launch_files:
      - betaray_navigation_jetson.launch.py

  ai_jetson:
    hostname: betaray-ai.local
    systemd_services:
      - ai-http-server
    http_only: true  # No ROS2 external communication

  vision_jetson:
    hostname: vision-jetson.local
    systemd_services:
      - betaray-vision-system
    http_only: true

  thor:
    hostname: thor.local
    systemd_services:
      - thor-deep-reasoning
      - thor-guardian
      - thor-memory-api
    http_only: true
```

**Expected Behavior**:
- Only analyze code reachable from specified entry points
- Mark unreachable code as `deployment_status: not_deployed`
- Reduce noise by 70%+ on typical codebases

#### 1.2 Launch File Tracing

**Flag**: `--trace-launch <file.launch.py>`

Follow ROS2 launch composition to build actual node graph:

```bash
robomind analyze ~/betaray --trace-launch src/betaray_bringup/launch/robot.launch.py
```

**Implementation Requirements**:
1. Parse Python launch files (handle `LaunchDescription`, `Node`, `IncludeLaunchDescription`)
2. Follow `IncludeLaunchDescription` calls recursively
3. Extract `Node()` declarations with package/executable
4. Build dependency graph of what actually launches
5. Handle conditional launches (`IfCondition`, `UnlessCondition`)

#### 1.3 systemd Service Discovery

**Flag**: `--discover-systemd <pattern>`

Auto-detect deployed services:

```bash
robomind analyze ~/betaray --discover-systemd "/etc/systemd/system/betaray*.service"
```

**Implementation Requirements**:
1. Parse systemd service files for `ExecStart`
2. Map executables back to source files
3. Mark discovered entry points as `confirmed_deployed`
4. Handle `ros2 launch` commands in ExecStart

---

### Feature 2: HTTP/REST Communication Detection

#### 2.1 HTTP Endpoint Extraction

Detect Flask/FastAPI/aiohttp routes:

```python
# Patterns to detect:
@app.route('/api/detections')
@app.get('/health')
@router.post('/api/speak')
app.add_api_route('/api/tts/speak', ...)
```

**Expected Output**:
```yaml
http_endpoints:
  - path: /api/detections
    method: GET
    file: vision_server.py:45
    inferred_host: vision-jetson.local:9091

  - path: /api/tts/speak
    method: POST
    file: canary_http_server.py:120
    inferred_host: betaray-nav.local:8081
```

#### 2.2 HTTP Client Detection

Detect outbound HTTP calls:

```python
# Patterns to detect:
requests.get('http://vision-jetson.local:9091/detections')
requests.post(f'{VISION_URL}/api/capture')
aiohttp.ClientSession().get(url)
httpx.AsyncClient().post(...)
urllib.request.urlopen(...)
```

**Expected Output**:
```yaml
http_clients:
  - caller: unified_ai_processor.py:89
    target: http://vision-jetson.local:9091/detections
    method: GET

  - caller: voice_handler.py:156
    target_variable: CANARY_URL  # When URL is from env/config
    method: POST
```

#### 2.3 Cross-System Communication Map

New output section showing all inter-process communication:

```yaml
cross_system_communication:
  nav_jetson <-> thor:
    - type: http
      endpoint: POST /api/voice_query
      direction: nav -> thor

  vision_jetson <-> thor:
    - type: http
      endpoint: GET /detections
      direction: thor -> vision

  nav_jetson internal:
    - type: ros2
      topic: /cmd_vel
      publishers: [navigation_node]
      subscribers: [motor_controller]

  # Summary
  ros2_cross_jetson: false
  http_cross_jetson: true
```

---

### Feature 3: Confidence Scoring

#### 3.1 Finding Confidence Levels

Add confidence scores to all findings:

```yaml
findings:
  - id: TOPIC-001
    type: topic_mismatch
    severity: critical
    confidence: 0.95
    confidence_factors:
      - "Subscription found in deployed launch file": +0.4
      - "Publisher found in same package": +0.3
      - "Topic name matches exactly": +0.25
    evidence:
      - file: motor_controller_node.py:62
        code: "create_subscription(Bool, 'emergency_stop', ...)"

  - id: TOPIC-002
    type: orphaned_subscriber
    severity: high
    confidence: 0.25
    confidence_factors:
      - "No launch file references this node": -0.4
      - "File in archive/ directory": -0.3
      - "Last git commit 6 months ago": -0.05
    evidence:
      - file: archive/old_vision_node.py:34
```

#### 3.2 Confidence Calculation Matrix

| Factor | Confidence Impact |
|--------|-------------------|
| Referenced in launch file | +0.4 |
| Referenced in systemd service | +0.5 |
| In archive/backup/old directory | -0.4 |
| No imports from other project files | -0.2 |
| Last git commit > 6 months | -0.1 |
| Has matching publisher/subscriber | +0.3 |
| HTTP endpoint exists for same data | -0.3 (ROS2 finding may be dead) |
| File has `_old`, `_backup`, `_deprecated` suffix | -0.3 |
| In `__pycache__` or build directory | -0.5 |

#### 3.3 Confidence Thresholds

```yaml
output_filtering:
  default_minimum_confidence: 0.5
  flags:
    --min-confidence: <float>  # Show only findings above threshold
    --show-all: false          # Include low-confidence findings
    --show-dead-code: false    # Include likely dead code findings
```

---

### Feature 4: Runtime Validation Mode

#### 4.1 Live System Validation

**Command**:
```bash
robomind validate ~/betaray_analysis \
  --ssh robot@betaray-nav.local \
  --ssh robot@betaray-ai.local \
  --ssh voice_jetson@vision-jetson.local
```

**Behavior**:
1. SSH to each target system
2. Run `ros2 node list` and `ros2 topic list`
3. Run `ros2 topic info <topic> -v` for topics in findings
4. Check HTTP endpoints with `curl -I`
5. Compare against static analysis findings
6. Update findings with runtime status

**Expected Output**:
```yaml
runtime_validation:
  timestamp: 2026-01-26T10:30:00Z

  nodes:
    motor_controller_node:
      static_analysis: found
      runtime: RUNNING on nav_jetson
      status: CONFIRMED

    guardian_bridge_node:
      static_analysis: found
      runtime: NOT_RUNNING
      status: DEAD_CODE

  topics:
    /betaray/motors/emergency_stop:
      static_analysis: 2 publishers, 1 subscriber
      runtime: 1 publisher, 1 subscriber
      status: PARTIAL_MATCH

  http_endpoints:
    http://vision-jetson.local:9091/health:
      status: 200 OK
      response_time_ms: 45

  findings_updated:
    - id: TOPIC-001
      original_confidence: 0.95
      validated_confidence: 0.99
      runtime_status: CONFIRMED

    - id: TOPIC-002
      original_confidence: 0.75
      validated_confidence: 0.05
      runtime_status: FALSE_POSITIVE
```

#### 4.2 Continuous Monitoring Integration

**Flag**: `--export-prometheus`

Export metrics for Prometheus/Grafana:
```
# HELP robomind_nodes_expected Total nodes expected from static analysis
robomind_nodes_expected 45
# HELP robomind_nodes_running Nodes confirmed running at validation time
robomind_nodes_running 38
# HELP robomind_topics_connected Topics with matched pub/sub
robomind_topics_connected 67
# HELP robomind_topics_orphaned Topics missing publisher or subscriber
robomind_topics_orphaned 12
# HELP robomind_findings_confirmed Findings validated as real issues
robomind_findings_confirmed 5
# HELP robomind_findings_false_positive Findings determined to be false positives
robomind_findings_false_positive 23
```

---

### Feature 5: Smart Filtering

#### 5.1 Default Excludes

Apply sensible defaults automatically:

```bash
# Implicit excludes (unless --no-default-excludes)
--exclude "**/archive/**"
--exclude "**/backup/**"
--exclude "**/old/**"
--exclude "**/deprecated/**"
--exclude "**/*_old.py"
--exclude "**/*_backup.py"
--exclude "**/*_deprecated.py"
--exclude "**/test/**"
--exclude "**/tests/**"
--exclude "**/examples/**"
--exclude "**/__pycache__/**"
--exclude "**/build/**"
--exclude "**/install/**"
--exclude "**/log/**"
```

#### 5.2 Dead Code Detection Heuristics

Identify likely dead code:

```yaml
dead_code_indicators:
  strong:
    - File in archive/old/backup/deprecated directory
    - No imports from any other project file
    - Not in any launch file
    - Not in any systemd service

  moderate:
    - Last git commit > 6 months with no references
    - Contains TODO/FIXME mentioning "remove" or "deprecated"
    - Has HTTP endpoint for same data (ROS2 likely unused)

  weak:
    - No matching publisher/subscriber in project
    - Uses deprecated ROS2 APIs
```

**Output flag**: `--show-dead-code-candidates`

---

### Feature 6: Enhanced Output Formats

#### 6.1 AI-Optimized Context

New `--format ai-context` output designed for LLM consumption:

```yaml
# Optimized for LLM consumption
system_summary:
  architecture: "4-Jetson HTTP-based with internal ROS2"
  cross_jetson_protocol: http
  primary_ros2_domain: nav_jetson (DOMAIN_ID=0)

deployment_status:
  confirmed_running:
    - motor_controller_node
    - canary_http_server
    - betaray_navigation_node
  likely_dead:
    - guardian_bridge
    - unified_ai_node_v1
    - old_voice_handler

actionable_findings:
  - id: ESTOP-001
    summary: "Emergency stop subscription uses relative topic name"
    file: motor_controller_node.py:62
    current_code: "create_subscription(Bool, 'emergency_stop', ...)"
    suggested_fix: "create_subscription(Bool, '/betaray/motors/emergency_stop', ...)"
    confidence: 0.95
    runtime_confirmed: true
    risk_level: HIGH
    effort_level: LOW

non_actionable_findings:
  - id: VISION-001
    summary: "Vision pipeline ROS2 topics orphaned"
    reason_non_actionable: "Code is dead - vision uses HTTP :9091"
    confidence: 0.15
```

#### 6.2 Diff-Friendly Output

**Flag**: `--format diff`

For tracking changes between analyses:

```diff
# robomind diff old_analysis/ new_analysis/
+ topic: /nexus/emergency_stop (NEW)
- topic: /betaray/emergency_stop (REMOVED)
~ node: motor_controller (MODIFIED: +1 subscription)
= finding: ESTOP-001 (UNCHANGED)
! finding: VISION-001 (confidence 0.75 -> 0.15, now non-actionable)
```

#### 6.3 Sarif Output

**Flag**: `--format sarif`

For IDE/CI integration with standard Static Analysis Results Interchange Format.

---

## Implementation Priority

| Phase | Features | Effort | Impact |
|-------|----------|--------|--------|
| 1 | Default excludes, confidence scores | Low | High |
| 2 | Launch file tracing, deployment manifest | Medium | High |
| 3 | HTTP endpoint detection | Medium | High |
| 4 | Runtime validation mode | High | Very High |
| 5 | AI-optimized output, Prometheus export | Medium | Medium |

**Recommendation**: Phases 1-3 would have prevented 80% of false positives in the BetaRay analysis.

---

## Verification Criteria

After implementing features, re-run on BetaRay/NEXUS codebase:

```bash
robomind analyze ~/betaray \
  --deployment-manifest ~/betaray/deployment.yaml \
  --trace-launch ~/betaray/Betaray_Dual_v1/jetson_navigation/betaray_nav_ws/src/betaray_bringup/launch/robot.launch.py \
  --format ai-context \
  -o ~/betaray_analysis_v2

# Validate against live system
robomind validate ~/betaray_analysis_v2 \
  --ssh robot@betaray-nav.local
```

**Success Criteria**:

| Metric | Current | Target |
|--------|---------|--------|
| False positive rate | ~80% | <20% |
| HTTP endpoints detected | 0% | 100% |
| Confidence correlation with runtime | N/A | >0.8 |
| AI context actionability | Low | High (no additional exploration needed) |

---

## The Real Issue That RoboMind Found

For reference, the one legitimate finding:

**Location**: `motor_controller_node.py:62-66`

```python
self.stop_sub = self.create_subscription(
    Bool, 'emergency_stop', self.emergency_stop_callback, 10)  # NO SLASH
self.voice_emergency_sub = self.create_subscription(
    Bool, '/betaray/motors/emergency_stop', self.voice_emergency_callback, 1)  # WITH SLASH
```

**Risk**: Bare `emergency_stop` is namespace-relative. If the node launches in a namespace, the subscription becomes `/<namespace>/emergency_stop`, potentially causing emergency stop failures.

**Mitigation Already Present**: The dual-subscription means voice e-stop works via the second subscription. The risk is if something publishes to bare `emergency_stop` expecting it to work universally.

**Recommended Fix**: Add leading slash for consistency:
```python
self.stop_sub = self.create_subscription(
    Bool, '/emergency_stop', self.emergency_stop_callback, 10)
```

---

## Appendix: NEXUS Architecture Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THOR (Main Brain)                           │
│  HTTP APIs: 8087-8096 | vLLM: 30000 | Guardian: Observatory HTTP    │
└──────────────────┬─────────────────┬─────────────────┬──────────────┘
                   │ HTTP            │ HTTP            │ HTTP
                   ▼                 ▼                 ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│    Nav Jetson       │ │  Vision Jetson  │ │      AI Jetson          │
│ ROS2 DOMAIN_ID=0    │ │  HTTP :9091     │ │   HTTP :8080            │
│ Canary HTTP :8081   │ │  (no ROS2 ext)  │ │   (no ROS2 ext)         │
│                     │ │                 │ │                         │
│ ┌─────────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────────────┐ │
│ │ motor_controller│ │ │ │ YOLO server │ │ │ │  LLM query server   │ │
│ │ navigation_node │ │ │ │ camera_node │ │ │ │  embedding_server   │ │
│ │ canary_http     │ │ │ └─────────────┘ │ │ └─────────────────────┘ │
│ └─────────────────┘ │ │                 │ │                         │
└─────────────────────┘ └─────────────────┘ └─────────────────────────┘

Key: Cross-Jetson = HTTP only. ROS2 is internal to Nav Jetson.
```

---

*Document generated: 2026-01-26*
*Based on: RoboMind v1.0 analysis of NEXUS/BetaRay codebase*
*Validated against: Live 4-Jetson robot system*
