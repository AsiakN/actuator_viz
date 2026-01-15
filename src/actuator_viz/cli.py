"""
Command-line interface for actuator-viz.

Usage:
    actuator-viz config.yaml              # Analyze configuration
    actuator-viz config.yaml --verbose    # Show effectiveness matrix
    actuator-viz config.yaml --output report.html  # Generate report (future)
    actuator-viz --web                    # Launch web UI (future)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__, parse_config, analyze
from .core.analysis import compute_control_authority
from .core.effectiveness import get_dof_names

# Create Typer app
app = typer.Typer(
    name="actuator-viz",
    help="Visualize and analyze multi-actuator control allocation systems.",
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"actuator-viz {__version__}")
        raise typer.Exit()


@app.command()
def cli(
    config_file: Annotated[
        Optional[Path],
        typer.Argument(help="Path to configuration file (YAML, JSON, PX4 airframe, or ArduPilot params)")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output including effectiveness matrix")
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=version_callback, is_eager=True, help="Show version and exit")
    ] = False,
):
    """
    Analyze actuator configuration for controllability.

    Computes the effectiveness matrix and checks if the system can control
    all 6 degrees of freedom (roll, pitch, yaw, x, y, z).

    Exit codes:
        0 = Fully controllable
        1 = Not fully controllable or error
    """
    if config_file is None:
        console.print("[yellow]Usage:[/yellow] actuator-viz <config-file>")
        console.print("\nRun [cyan]actuator-viz --help[/cyan] for more options.")
        raise typer.Exit(0)

    try:
        # Parse configuration
        config = parse_config(config_file)

        # Run analysis
        result = analyze(config)

        # Print results
        print_report(config, result, verbose)

        # Exit with appropriate code
        if not result.controllable:
            raise typer.Exit(1)

    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {config_file}")
        raise typer.Exit(1)
    except ValueError as e:
        error_console.print(f"[red]Error:[/red] Invalid configuration: {e}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def print_report(config, result, verbose: bool = False):
    """Print analysis report to console."""

    # Header
    console.print()
    console.print(Panel(
        f"[bold]{config.name}[/bold]\n"
        f"{config.n_actuators} actuators · {config.frame.value} frame",
        title="actuator-viz",
        border_style="blue",
    ))

    # Controllability status
    console.print()
    if result.controllable:
        console.print("[green]✓[/green] [bold]FULLY CONTROLLABLE[/bold]")
    else:
        console.print("[red]✗[/red] [bold]NOT FULLY CONTROLLABLE[/bold]")

    # Key metrics
    console.print()
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value")

    metrics_table.add_row("Rank", f"{result.rank}/6")

    # Condition number with color coding
    cn = result.condition_number
    if cn < 10:
        cn_style = "green"
    elif cn < 50:
        cn_style = "yellow"
    else:
        cn_style = "red"
    metrics_table.add_row("Condition Number", f"[{cn_style}]{cn:.2f}[/{cn_style}]")

    console.print(metrics_table)

    # Control authority bar chart
    console.print()
    console.print("[bold]Control Authority[/bold]")

    authority = compute_control_authority(result.effectiveness_matrix)
    max_authority = max(authority.values()) if authority else 1

    dof_names = get_dof_names()
    for dof in dof_names:
        value = authority.get(dof, 0)
        bar_width = int((value / max_authority) * 30) if max_authority > 0 else 0
        bar = "█" * bar_width

        # Color based on relative authority
        ratio = value / max_authority if max_authority > 0 else 0
        if ratio > 0.7:
            color = "green"
        elif ratio > 0.4:
            color = "yellow"
        else:
            color = "red"

        console.print(f"  {dof:>6}  [{color}]{bar:<30}[/{color}] {value:.2f}")

    # Issues
    if result.issues:
        console.print()
        console.print("[bold yellow]Issues[/bold yellow]")
        for issue in result.issues:
            console.print(f"  [yellow]⚠[/yellow] {issue}")

    # Verbose: show effectiveness matrix
    if verbose:
        print_effectiveness_matrix_rich(config, result.effectiveness_matrix)

    console.print()


def print_effectiveness_matrix_rich(config, effectiveness):
    """Print effectiveness matrix with rich formatting."""
    console.print()
    console.print("[bold]Effectiveness Matrix[/bold]")

    dof_names = get_dof_names()

    # Create table
    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("DOF", style="dim", width=8)

    for actuator in config.actuators:
        # Truncate long names
        name = actuator.name[:8] if len(actuator.name) > 8 else actuator.name
        table.add_column(name, justify="right", width=10)

    # Add rows
    for i, dof in enumerate(dof_names):
        row = [dof]
        for j in range(config.n_actuators):
            val = effectiveness[i, j]
            if abs(val) < 1e-6:
                row.append("[dim]0[/dim]")
            elif val > 0:
                row.append(f"[green]{val:.3f}[/green]")
            else:
                row.append(f"[red]{val:.3f}[/red]")
        table.add_row(*row)

    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
