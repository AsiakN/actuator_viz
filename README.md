# actuator-viz

**Visualize and analyze multi-actuator control systems before you build them.**

actuator-viz helps robotics engineers understand whether their actuator layout will actually work. Define your thruster, motor, or joint positions in a simple config file, and instantly see:

- Is the system fully controllable?
- Where are the control authority weak spots?
- How will actuators interact (coupling effects)?
- What happens if one fails?

## The Problem

You've designed a vehicle with multiple actuators—thrusters on an ROV, motors on a drone, joints on a robot arm. You placed them based on intuition, CAD constraints, or copied an existing design. But:

- Can you actually command all degrees of freedom independently?
- Is one axis barely controllable while another is over-actuated?
- Will commanding pitch accidentally induce roll?
- If a motor fails, can you still maintain control?

These questions are hard to answer by staring at CAD models or running mental math. You usually discover problems during testing—after the hardware is built.

## The Solution

actuator-viz computes the **effectiveness matrix** from your actuator geometry and visualizes what it means:

```
┌─────────────────────────────────────────────────────────┐
│  3D Layout        │  Effectiveness Matrix (Heatmap)    │
│                   │                                     │
│    ◇───────◇      │   Actuator  1   2   3   4   5   6  │
│   /│       │\     │   ────────────────────────────────  │
│  ◇─┼───────┼─◇    │   Surge   [███|   |███|   |   |   ]│
│    │   ⊕   │      │   Sway    [   |███|   |███|   |   ]│
│  ◇─┼───────┼─◇    │   Heave   [   |   |   |   |███|███]│
│   \│       │/     │   Roll    [▓▓▓|▓▓▓|   |   |   |   ]│
│    ◇───────◇      │   Pitch   [   |   |▓▓▓|▓▓▓|   |   ]│
│                   │   Yaw     [▓▓▓|   |▓▓▓|   |▓▓▓|   ]│
└─────────────────────────────────────────────────────────┘

✓ Full rank (6/6 DOF controllable)
✓ Condition number: 2.3 (well-conditioned)
⚠ Yaw authority is 40% lower than other axes
```

## Use Cases

### Validate a New Design
You're designing an 8-thruster ROV. Before machining mounts or ordering hardware, run your thruster positions through actuator-viz. Discover that two thrusters are nearly coplanar and provide redundant force—fix the geometry in CAD, not in the water.

### Debug Control Problems
Your hexacopter yaws when you command pure pitch. The effectiveness heatmap reveals a motor axis is 3° off from where you thought. Fix the mechanical alignment or compensate in your mixer.

### Compare Design Options
Should you go with 6 thrusters or 8? Symmetric X-pattern or asymmetric? Generate reports for each layout, compare control authority quantitatively, make data-driven decisions in design reviews.

### Analyze Failure Modes
For safety-critical applications: what happens when motor 3 fails? Simulate the n-1 configuration, verify the remaining actuators can still stabilize the vehicle for a controlled descent or return-to-home.

### Learn Control Allocation
Students and hobbyists: add and remove actuators interactively, watch the effectiveness matrix change, build intuition for how geometry translates to controllability.

## Target Platforms

actuator-viz works with any multi-actuator system:

- **Underwater vehicles** (ROVs, AUVs) with vectored thrusters
- **Multirotors** (quadcopters, hexacopters, octocopters)
- **Fixed-wing VTOL** with tilt-rotors or separate hover/cruise motors
- **Legged robots** (hexapods, quadrupeds) analyzing joint torque authority
- **Robot arms** with redundant joints
- **Marine surface vessels** with multiple propellers/azimuth thrusters

## Quick Start

```bash
# Install
pip install actuator-viz

# Analyze a configuration
actuator-viz my_vehicle.yaml

# Launch interactive web UI
actuator-viz --web

# Generate HTML report
actuator-viz my_vehicle.yaml --output report.html
```

Define your actuators in YAML:

```yaml
name: "My ROV"
frame: "NED"

actuators:
  - id: 1
    name: "Front-Right Vertical"
    position: [0.3, 0.2, 0.0]    # meters from CoG
    axis: [0, 0, -1]              # thrust direction

  - id: 2
    name: "Front-Left Vertical"
    position: [0.3, -0.2, 0.0]
    axis: [0, 0, -1]

  # ... more actuators
```

## Features

- **Multi-format input**: Native YAML/JSON, PX4 airframe files, ArduPilot parameters
- **Controllability analysis**: Rank, condition number, singular value decomposition
- **3D visualization**: See actuator positions and force vectors in space
- **Effectiveness heatmap**: Understand which actuators contribute to which DOF
- **Control authority chart**: Compare force/torque capacity across axes
- **Failure simulation**: Analyze degraded configurations with actuators removed
- **CLI + Web UI + Python API**: Use however fits your workflow

## License

MIT
