# Actuator Effectiveness Visualizer - Standalone Tool

## Project Vision

A generic, open-source tool for visualizing and analyzing multi-actuator control allocation systems. Target users: robotics developers working with ROVs, drones, hexapods, robot arms, or any system with multiple actuators.

**Name**: `actuator-viz`
**License**: MIT
**Distribution**: Standalone GitHub repository

---

## Architecture Overview

```
actuator-viz/
├── src/actuator_viz/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point
│   ├── web.py                 # Web UI (Streamlit)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── effectiveness.py   # Matrix computation
│   │   ├── analysis.py        # SVD, controllability
│   │   └── geometry.py        # 3D transforms, frames
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract parser interface
│   │   ├── json_yaml.py       # Generic JSON/YAML
│   │   ├── px4.py             # PX4 airframe files
│   │   └── ardupilot.py       # ArduSub/ArduPilot
│   ├── visualizers/
│   │   ├── __init__.py
│   │   ├── thruster_3d.py     # 3D layout plot
│   │   ├── heatmap.py         # Effectiveness matrix
│   │   ├── authority.py       # Control authority bars
│   │   └── report.py          # Combined HTML report
│   └── schemas/
│       └── config.schema.json # JSON Schema for validation
├── tests/
├── examples/
│   ├── rov_8_thruster.yaml
│   ├── quadcopter.yaml
│   ├── hexapod.yaml
│   └── px4_airframe_example
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Key Components

### 1. Generic Config Schema (JSON/YAML)

```yaml
# examples/rov_8_thruster.yaml
name: "UUV Reconbot"
frame: "ENU"  # or NED, body-fixed
units: "meters"

actuators:
  - id: 0
    name: "Bow Starboard Vertical"
    type: "thruster"
    position: [-0.424, 0.134, -0.117]
    axis: [0, -0.707, 0.707]
    coefficient: 1.0
    moment_ratio: 0.0

  - id: 1
    name: "Bow Port Vertical"
    # ...

# Optional: vehicle geometry for visualization
geometry:
  type: "box"
  dimensions: [1.0, 0.4, 0.3]
  # or: mesh_file: "vehicle.stl"
```

### 2. Parser Interface

```python
class ConfigParser(ABC):
    @abstractmethod
    def parse(self, source: str | Path) -> ActuatorConfig:
        """Parse config from file or string."""
        pass

    @abstractmethod
    def can_parse(self, source: str | Path) -> bool:
        """Check if this parser handles the format."""
        pass
```

Parsers auto-detect format based on file extension and content.

### 3. CLI Interface

```bash
# Basic usage
actuator-viz config.yaml

# Output options
actuator-viz config.yaml --output report.html
actuator-viz config.yaml --format png --output plots/

# Parse PX4 airframe
actuator-viz --px4 path/to/airframe

# Analysis only (no viz)
actuator-viz config.yaml --analyze-only

# Launch web UI
actuator-viz --web
actuator-viz --web --port 8080
```

### 4. Web UI (Streamlit)

Features:
- File upload (drag & drop YAML/JSON/airframe)
- Live config editor with syntax highlighting
- Interactive 3D view
- Download report as HTML/PNG
- Preset examples (quadcopter, hexacopter, ROV, etc.)

### 5. Python API

```python
from actuator_viz import ActuatorConfig, analyze, visualize

# Load config
config = ActuatorConfig.from_yaml("my_robot.yaml")

# Compute effectiveness
result = analyze(config)
print(result.rank, result.controllable, result.condition_number)

# Generate visualization
fig = visualize.thruster_3d(config)
fig.show()

# Full report
visualize.generate_report(config, result, "report.html")
```

---

## Implementation Phases

### Phase 1: Core Refactor (Foundation)
- [ ] Create new repo structure with `pyproject.toml`
- [ ] Extract core math to `effectiveness.py` and `analysis.py`
- [ ] Define `ActuatorConfig` dataclass with validation
- [ ] Create JSON Schema for config validation
- [ ] Write generic JSON/YAML parser

### Phase 2: Parser Ecosystem
- [ ] Abstract `ConfigParser` base class
- [ ] Port PX4 airframe parser
- [ ] Add ArduPilot/ArduSub frame parser
- [ ] Auto-detection logic (try parsers until one succeeds)

### Phase 3: Visualization Module
- [ ] Port existing Plotly visualizations
- [ ] Add vehicle geometry rendering (box, cylinder, STL mesh)
- [ ] Support coordinate frame options (ENU/NED)
- [ ] PNG export option

### Phase 4: CLI
- [ ] Build CLI with `click` or `typer`
- [ ] Add `--web` flag to launch Streamlit
- [ ] Add `--analyze-only` for CI/scripting
- [ ] Exit codes for CI integration (non-zero if not controllable)

### Phase 5: Web UI
- [ ] Streamlit app with file upload
- [ ] Interactive config editor
- [ ] Preset examples dropdown
- [ ] Download buttons for HTML/PNG

### Phase 6: Polish & Release
- [ ] Write README with examples
- [ ] Add example configs (quadcopter, hexacopter, ROV, hexapod)
- [ ] Write tests for parsers and core math
- [ ] GitHub Actions CI
- [ ] MIT License

---

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.21",
    "plotly>=5.0",
    "pyyaml>=6.0",
    "jsonschema>=4.0",
    "typer>=0.9",      # CLI
    "rich>=13.0",      # Pretty terminal output
]

[project.optional-dependencies]
web = ["streamlit>=1.28"]
dev = ["pytest", "pytest-cov", "black", "ruff"]
```

---

## Differentiators from Existing Tools

| Feature | This Tool | QGroundControl | PlotJuggler |
|---------|-----------|----------------|-------------|
| Offline analysis | Yes | No (runtime) | Yes |
| Multi-format input | Yes | PX4 only | Logs only |
| 3D visualization | Yes | Limited | No |
| Control analysis | Yes (SVD) | No | No |
| Web UI | Yes | No | No |
| Python API | Yes | No | No |

---

## Files to Create (New Repo)

This will be a **new standalone repository**, not modifications to reconbot_ground_control.

Starting point: Extract and generalize code from:
- `effectiveness_calculator.py` → `src/actuator_viz/core/`
- `effectiveness_visualizer.py` → `src/actuator_viz/visualizers/`

---

## Code to Reuse from Reconbot

The following functions from `reconbot_ground_control/` can be extracted and generalized:

### From `effectiveness_calculator.py`:
| Function | Target Location | Changes Needed |
|----------|-----------------|----------------|
| `cross_product()` | `core/geometry.py` | None |
| `compute_effectiveness_matrix()` | `core/effectiveness.py` | Accept `ActuatorConfig` dataclass |
| `analyze_controllability()` | `core/analysis.py` | Return `AnalysisResult` dataclass |
| `detect_issues()` | `core/analysis.py` | Make issue checks configurable |
| `parse_airframe_file()` | `parsers/px4.py` | Refactor to `ConfigParser` interface |
| `get_uuv_reconbot_config()` | `examples/` | Convert to YAML example file |

### From `effectiveness_visualizer.py`:
| Function | Target Location | Changes Needed |
|----------|-----------------|----------------|
| `create_3d_thruster_plot()` | `visualizers/thruster_3d.py` | Add geometry mesh support |
| `create_effectiveness_heatmap()` | `visualizers/heatmap.py` | Parameterize DOF labels |
| `create_control_authority_chart()` | `visualizers/authority.py` | None |
| `generate_visualization_report()` | `visualizers/report.py` | Template-based HTML |

---

## Estimated Effort

| Phase | Effort | Description |
|-------|--------|-------------|
| Phase 1: Core | 2-3 hours | Repo setup, dataclasses, schema |
| Phase 2: Parsers | 2-3 hours | PX4, ArduPilot, auto-detect |
| Phase 3: Visualization | 1-2 hours | Port existing + add geometry |
| Phase 4: CLI | 1-2 hours | Typer CLI with all flags |
| Phase 5: Web UI | 2-3 hours | Streamlit app |
| Phase 6: Polish | 2-3 hours | README, examples, tests, CI |

**Total**: ~12-16 hours for MVP

---

## Future Enhancements (Post-MVP)

- STL/OBJ mesh loading for vehicle body
- Animated thrust visualization (show forces in motion)
- Optimization suggestions (detect and propose fixes)
- Export to PX4/ArduPilot format (reverse direction)
- Docker container for zero-install usage
- PyPI publishing for `pip install actuator-viz`
