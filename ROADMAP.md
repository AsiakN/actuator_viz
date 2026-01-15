# actuator-viz Roadmap

## Milestone 1: Core Analysis Engine

**Goal**: A working CLI that answers "is my actuator layout controllable?"

### Deliverables
- [x] `ActuatorConfig` dataclass with validation
- [x] YAML/JSON parser for generic config format
- [x] Effectiveness matrix computation
- [x] SVD-based controllability analysis (rank, condition number)
- [x] Issue detection (weak axes, redundant actuators, coupling)
- [x] CLI with text output (`actuator-viz config.yaml`)
- [x] Non-zero exit code when not fully controllable (CI integration)

### Success Criteria
```bash
$ actuator-viz examples/rov_8_thruster.yaml

Analyzing "UUV Reconbot" (8 actuators)...

Effectiveness Matrix Rank: 6/6 ✓
Condition Number: 2.34 ✓
Controllability: FULL

Warnings:
  - Yaw authority 43% lower than average

$ echo $?
0
```

---

## Milestone 2: Visualization

**Goal**: Generate visual reports that show *where* problems are.

### Deliverables
- [ ] Effectiveness heatmap (actuators × DOF)
- [ ] 3D thruster layout plot (positions + force vectors)
- [ ] Combined HTML report generation
- [ ] PNG export option
- [ ] `--output report.html` CLI flag

### Success Criteria
- Single HTML file with embedded plots, shareable via email/Slack
- Heatmap clearly shows which actuators affect which DOF
- 3D plot lets user verify positions match their physical layout

---

## Milestone 3: PX4 Integration

**Goal**: Zero-friction onboarding for PX4 users.

### Deliverables
- [ ] PX4 airframe file parser
- [ ] Auto-detection (try parsers until one succeeds)
- [ ] `--px4` flag for explicit format selection
- [ ] Example: parse stock PX4 airframes (quad_x, hexa_x, etc.)

### Success Criteria
```bash
$ actuator-viz /path/to/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/4001_quad_x
```
Works out of the box.

---

## Milestone 4: ArduPilot Integration

**Goal**: Support ArduSub/ArduCopter users.

### Deliverables
- [ ] ArduPilot parameter file parser (MOT_*, FRAME_CLASS, etc.)
- [ ] ArduSub frame type mapping (BlueROV2, custom, etc.)
- [ ] `--ardupilot` flag

### Success Criteria
```bash
$ actuator-viz bluerov2.param
```
Correctly identifies 6-thruster vectored layout.

---

## Milestone 5: Failure Mode Analysis

**Goal**: Answer "what if actuator N fails?"

### Deliverables
- [ ] `--failure N` flag to simulate actuator N offline
- [ ] `--failure-all` to test all single-actuator failures
- [ ] Degraded controllability report
- [ ] Identify critical actuators (failure = loss of control)

### Success Criteria
```bash
$ actuator-viz config.yaml --failure-all

Failure Analysis:
  Actuator 1 offline: 6/6 DOF ✓ (condition: 3.1)
  Actuator 2 offline: 6/6 DOF ✓ (condition: 2.9)
  Actuator 3 offline: 5/6 DOF ✗ (loses yaw)  ← CRITICAL
  ...
```

---

## Milestone 6: Web UI

**Goal**: Browser-based interface for non-CLI users.

### Deliverables
- [ ] Streamlit app with file upload
- [ ] Live config editor with syntax highlighting
- [ ] Interactive 3D view (rotate, zoom)
- [ ] Preset examples dropdown
- [ ] Download report button
- [ ] `actuator-viz --web` to launch

### Success Criteria
- User can drag-drop a YAML file and see results without terminal
- Works for design reviews on shared screen

---

## Milestone 7: Polish & Release

**Goal**: Ready for public use.

### Deliverables
- [ ] PyPI package (`pip install actuator-viz`)
- [ ] Example configs (quadcopter, hexacopter, ROV, hexapod, robot arm)
- [ ] JSON Schema for config validation
- [ ] Comprehensive test suite
- [ ] GitHub Actions CI
- [ ] Documentation site or detailed README

### Success Criteria
- New user can install and run on their config in under 5 minutes
- Tests pass on Python 3.9, 3.10, 3.11, 3.12

---

## Future Ideas (Post-Release)

- STL/OBJ mesh loading for vehicle body visualization
- Inverse visualization (show force allocation for a commanded wrench)
- Optimization suggestions ("move actuator 3 by X to improve condition number")
- Export to PX4/ArduPilot format (reverse direction)
- Real-time telemetry overlay (connect to vehicle, show actual vs. commanded)
- Docker container for zero-install usage
- VS Code extension

---

## Current Status

**Completed**: Milestone 1 (Core Analysis Engine) ✓

**Active Milestone**: 2 (Visualization)

Last updated: 2025-01
