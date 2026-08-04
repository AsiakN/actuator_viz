"""
Configuration parsers for various actuator config formats.

Supported formats:
- Generic JSON/YAML (actuator-viz native format)
- PX4 airframe files
- ArduPilot/ArduSub configuration
"""

from .ardupilot import (
    ArduPilotParser,
    ArduSubFrame,
    get_ardusub_frame,
    list_ardusub_frames,
    parse_ardupilot,
)
from .base import ConfigParser, ParserRegistry, get_registry, parse_config, register_parser
from .json_yaml import JsonYamlParser, load_json, load_yaml, parse_yaml_string
from .px4 import PX4Parser, generate_px4_params, parse_px4_airframe

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
