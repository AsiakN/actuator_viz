"""
Command-line interface for actuator-viz.

Usage:
    actuator-viz config.yaml              # Analyze configuration
    actuator-viz config.yaml --verbose    # Show effectiveness matrix
    actuator-viz config.yaml --output report.html  # Generate HTML report
    actuator-viz config.yaml --failure-all         # Sweep single-actuator failures
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__, analyze, parse_config
from .core.analysis import (
    analyze_all_failures,
    compute_control_authority,
    simulate_failure,
)
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
        Path | None,
        typer.Argument(
            help="Path to configuration file (YAML, JSON, PX4 airframe, or ArduPilot params)"
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output including effectiveness matrix"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write an interactive HTML report to this path"),
    ] = None,
    failure: Annotated[
        int | None, typer.Option("--failure", help="Simulate actuator with this ID going offline")
    ] = None,
    failure_all: Annotated[
        bool, typer.Option("--failure-all", help="Simulate every single-actuator failure in turn")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
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

        # Optionally write an HTML report
        if output is not None:
            write_html_report(config, result, output)

        # Optionally run failure-mode analysis
        if failure_all:
            impacts = analyze_all_failures(config)
            print_failure_analysis(impacts, comprehensive=True)
        elif failure is not None:
            actuator = config.get_actuator(failure)
            if actuator is None:
                error_console.print(
                    f"[red]Error:[/red] No actuator with ID {failure}. "
                    f"Available IDs: {', '.join(str(a.id) for a in config.actuators)}"
                )
                raise typer.Exit(1)
            index = config.actuators.index(actuator)
            print_failure_analysis([simulate_failure(config, index)])

        # Exit with appropriate code
        if not result.controllable:
            raise typer.Exit(1)

    except FileNotFoundError:
        error_console.print(f"[red]Error:[/red] File not found: {config_file}")
        raise typer.Exit(1) from None
    except ValueError as e:
        error_console.print(f"[red]Error:[/red] Invalid configuration: {e}")
        raise typer.Exit(1) from None
    except typer.Exit:
        # Intentional exit (e.g. non-controllable config) — not an error.
        raise
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


def write_html_report(config, result, output_path: Path) -> None:
    """
    Write an interactive HTML report for the analysis.

    Adapts the modern ActuatorConfig/AnalysisResult dataclasses to the
    dict-based contract of the visualizer module, then delegates to it.
    """
    try:
        from .visualizers import generate_visualization_report
    except ImportError:
        error_console.print(
            "[red]Error:[/red] Plotly is required for HTML reports. "
            "Install with: [cyan]pip install 'plotly>=5.0'[/cyan]"
        )
        raise typer.Exit(1) from None

    generate_visualization_report(
        rotors=config.to_rotor_list(),
        effectiveness=result.effectiveness_matrix,
        controllability_result=result.to_dict(),
        issues=result.issues,
        output_path=str(output_path),
        title=f"{config.name} — Effectiveness Report",
        geometry=config.geometry,
    )

    console.print(f"\n[green]✓[/green] Report written to [cyan]{output_path}[/cyan]")


def print_failure_analysis(impacts, comprehensive: bool = False) -> None:
    """
    Print single-actuator failure analysis as a table.

    Args:
        impacts: List of FailureImpact to display.
        comprehensive: True if every actuator was tested (``--failure-all``),
            which licenses a system-wide fault-tolerance conclusion.
    """
    console.print()
    console.print("[bold]Single-Actuator Failure Analysis[/bold]")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Actuator")
    table.add_column("DOF", justify="center")
    table.add_column("Condition", justify="right")
    table.add_column("Impact")

    for impact in impacts:
        # DOF column: rank/6 with a pass/fail glyph
        if impact.controllable:
            dof = "[green]6/6 ✓[/green]"
        else:
            dof = f"[red]{impact.rank}/6 ✗[/red]"

        # Condition number
        cn = impact.condition_number
        if cn == float("inf"):
            condition = "[red]∞[/red]"
        else:
            condition = f"{cn:.2f}"

        # Impact column
        if impact.critical:
            impact_text = (
                f"[red]loses {', '.join(impact.lost_dofs)}[/red]  "
                f"[bold red]← CRITICAL[/bold red]"
            )
        else:
            impact_text = "[green]nominal[/green]"

        table.add_row(
            str(impact.actuator_id),
            impact.actuator_name,
            dof,
            condition,
            impact_text,
        )

    console.print(table)

    # Summary line
    critical = [i for i in impacts if i.critical]
    console.print()
    if comprehensive:
        # Every actuator was tested — we can make a system-wide claim.
        if critical:
            names = ", ".join(str(i.actuator_id) for i in critical)
            console.print(
                f"[bold red]⚠[/bold red] {len(critical)} of {len(impacts)} actuators "
                f"are critical (failure loses control): IDs {names}"
            )
        else:
            console.print(
                f"[green]✓[/green] All {len(impacts)} actuators are non-critical — "
                "the system tolerates any single failure."
            )
    else:
        # Only the requested actuator(s) were tested.
        for impact in impacts:
            if impact.critical:
                console.print(
                    f"[bold red]⚠[/bold red] Actuator {impact.actuator_id} is critical — "
                    f"its failure loses control of {', '.join(impact.lost_dofs)}."
                )
            else:
                console.print(
                    f"[green]✓[/green] Actuator {impact.actuator_id} is non-critical — "
                    "the system tolerates its failure."
                )


def print_report(config, result, verbose: bool = False):
    """Print analysis report to console."""

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]{config.name}[/bold]\n"
            f"{config.n_actuators} actuators · {config.frame.value} frame",
            title="actuator-viz",
            border_style="blue",
        )
    )

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
