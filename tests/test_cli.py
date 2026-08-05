"""
End-to-end CLI tests via Typer's CliRunner.

These lock in the exit-code contract (0 = controllable, 1 = not / error) and
the report/failure surfaces so behavior is observable from the outside.
"""

from __future__ import annotations

from typer.testing import CliRunner

from actuator_viz.cli import app

runner = CliRunner()


def test_controllable_config_exits_zero(examples_dir):
    result = runner.invoke(app, [str(examples_dir / "rov_8_thruster.yaml")])
    assert result.exit_code == 0
    assert "FULLY CONTROLLABLE" in result.output


def test_under_actuated_config_exits_one(examples_dir):
    result = runner.invoke(app, [str(examples_dir / "px4_quadcopter")])
    assert result.exit_code == 1


def test_missing_file_exits_one():
    result = runner.invoke(app, ["nope_does_not_exist.yaml"])
    assert result.exit_code == 1


def test_no_argument_shows_usage_and_exits_zero():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_output_flag_writes_html(examples_dir, tmp_path):
    report = tmp_path / "report.html"
    result = runner.invoke(
        app, [str(examples_dir / "rov_8_thruster.yaml"), "--output", str(report)]
    )
    assert result.exit_code == 0
    assert report.exists()
    assert report.read_text().lstrip().startswith("<!DOCTYPE html>")


def test_output_report_is_self_contained(examples_dir, tmp_path):
    # The report must render offline: no external <script src>/CDN dependency,
    # with the Plotly bundle inlined instead.
    report = tmp_path / "report.html"
    runner.invoke(
        app, [str(examples_dir / "rov_8_thruster.yaml"), "--output", str(report)]
    )
    html = report.read_text()
    assert "<script src=" not in html          # no external scripts
    assert 'src="https://cdn.plot.ly' not in html  # no CDN Plotly loader
    assert "Plotly.newPlot" in html            # charts are wired
    assert len(html) > 1_000_000               # the inlined bundle is present


def test_failure_all_reports_and_stays_zero_for_tolerant_rov(examples_dir):
    result = runner.invoke(
        app, [str(examples_dir / "rov_8_thruster.yaml"), "--failure-all"]
    )
    assert result.exit_code == 0
    assert "Failure Analysis" in result.output
    assert "non-critical" in result.output


def test_failure_invalid_id_exits_one(examples_dir):
    result = runner.invoke(
        app, [str(examples_dir / "rov_8_thruster.yaml"), "--failure", "99"]
    )
    assert result.exit_code == 1


def test_version_flag(examples_dir):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "actuator-viz" in result.output
