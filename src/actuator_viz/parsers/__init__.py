"""
Configuration parsers for various actuator config formats.

Supported formats:
- Generic JSON/YAML (actuator-viz native format)
- PX4 airframe files
- ArduPilot/ArduSub configuration
"""

from .base import ConfigParser, ParserRegistry, get_registry, register_parser, parse_config
from .json_yaml import JsonYamlParser, load_yaml, load_json, parse_yaml_string
from .px4 import PX4Parser, parse_px4_airframe, generate_px4_params
from .ardupilot import (
    ArduPilotParser,
    ArduSubFrame,
    parse_ardupilot,
    get_ardusub_frame,
    list_ardusub_frames,
)

__all__ = [
    # Base
    "ConfigParser",
    "ParserRegistry",
    "get_registry",
    "register_parser",
    "parse_config",
    # JSON/YAML
    "JsonYamlParser",
    "load_yaml",
    "load_json",
    "parse_yaml_string",
    # PX4
    "PX4Parser",
    "parse_px4_airframe",
    "generate_px4_params",
    # ArduPilot
    "ArduPilotParser",
    "ArduSubFrame",
    "parse_ardupilot",
    "get_ardusub_frame",
    "list_ardusub_frames",
]
