"""
Abstract base class for configuration parsers.

All parsers should inherit from ConfigParser and implement
the parse() and can_parse() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..core.models import ActuatorConfig

# Robust float matcher shared by the param-file parsers: optional sign, an
# integer / decimal / leading-dot mantissa, and an optional exponent. Unlike a
# loose ``[\d.]+`` this never captures a bare ``.`` (which crashed float()).
NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


class ConfigParser(ABC):
    """
    Abstract base class for configuration parsers.

    Parsers convert various file formats into ActuatorConfig objects.
    Each parser should:
    1. Implement can_parse() to check if it can handle a given source
    2. Implement parse() to convert the source to ActuatorConfig
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this parser."""
        pass

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this parser handles (e.g., ['.yaml', '.yml'])."""
        pass

    @abstractmethod
    def can_parse(self, source: str | Path) -> bool:
        """
        Check if this parser can handle the given source.

        Should check file extension and/or content structure.

        Args:
            source: File path or string content

        Returns:
            True if this parser can handle the source
        """
        pass

    @abstractmethod
    def parse(self, source: str | Path) -> ActuatorConfig:
        """
        Parse configuration from file or string.

        Args:
            source: File path or string content

        Returns:
            ActuatorConfig object

        Raises:
            ValueError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        pass

    def parse_file(self, path: str | Path) -> ActuatorConfig:
        """
        Parse configuration from a file path.

        Args:
            path: Path to configuration file

        Returns:
            ActuatorConfig object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return self.parse(path)

    def parse_string(self, content: str) -> ActuatorConfig:
        """
        Parse configuration from a string.

        Args:
            content: Configuration content as string

        Returns:
            ActuatorConfig object
        """
        return self.parse(content)


class ParserRegistry:
    """
    Registry of available configuration parsers.

    Provides auto-detection of file format and parser selection.
    """

    def __init__(self):
        self._parsers: list[ConfigParser] = []

    def register(self, parser: ConfigParser) -> None:
        """
        Register a parser.

        Args:
            parser: Parser instance to register
        """
        self._parsers.append(parser)

    def unregister(self, parser_name: str) -> bool:
        """
        Unregister a parser by name.

        Args:
            parser_name: Name of parser to remove

        Returns:
            True if parser was found and removed
        """
        for i, parser in enumerate(self._parsers):
            if parser.name == parser_name:
                del self._parsers[i]
                return True
        return False

    def get_parser(self, source: str | Path) -> ConfigParser | None:
        """
        Find a parser that can handle the given source.

        Tries parsers in registration order until one succeeds.

        Args:
            source: File path or string content

        Returns:
            Parser that can handle the source, or None
        """
        for parser in self._parsers:
            if parser.can_parse(source):
                return parser
        return None

    def parse(self, source: str | Path) -> ActuatorConfig:
        """
        Parse configuration using auto-detected parser(s).

        Every parser whose ``can_parse`` accepts the source is tried in
        registration order. If one accepts the format but fails to parse it
        (e.g. an over-eager detector matched a file it can't actually turn into
        an actuator layout), the next candidate is tried and the errors are
        aggregated — so an ambiguous file gets a clear diagnosis instead of the
        first candidate's hard failure.

        Args:
            source: File path or string content

        Returns:
            ActuatorConfig object

        Raises:
            ValueError: If no parser recognizes the source, or every parser that
                recognized it failed to parse it.
        """
        candidates = [p for p in self._parsers if p.can_parse(source)]
        if not candidates:
            raise ValueError(
                f"No parser recognized this source. Registered parsers: "
                f"{[p.name for p in self._parsers]}"
            )

        errors: list[str] = []
        for parser in candidates:
            try:
                return parser.parse(source)
            except ValueError as err:
                # This parser claimed the format but couldn't parse it; record
                # why and let the next candidate try. FileNotFoundError and the
                # like are real I/O failures and propagate.
                errors.append(f"{parser.name}: {err}")

        tried = ", ".join(p.name for p in candidates)
        detail = " | ".join(errors)
        raise ValueError(
            f"Recognized the file but could not build an actuator layout from it "
            f"(tried: {tried}). {detail}"
        )

    def list_parsers(self) -> list[str]:
        """List names of registered parsers."""
        return [p.name for p in self._parsers]

    def get_supported_extensions(self) -> list[str]:
        """Get all supported file extensions."""
        extensions = []
        for parser in self._parsers:
            extensions.extend(parser.extensions)
        return list(set(extensions))


# Global registry instance
_registry = ParserRegistry()


def get_registry() -> ParserRegistry:
    """Get the global parser registry."""
    return _registry


def register_parser(parser: ConfigParser) -> None:
    """Register a parser in the global registry."""
    _registry.register(parser)


def parse_config(source: str | Path) -> ActuatorConfig:
    """
    Parse configuration using auto-detected parser from global registry.

    Args:
        source: File path or string content

    Returns:
        ActuatorConfig object
    """
    return _registry.parse(source)
