"""cerebra-triage CLI entry point.

Usage:
    cerebra-triage <input_dir> [--output-dir <dir>] [--study-id <id>]

Prints the structured JSON report to stdout and saves the PNG overlay to
--output-dir (default: /tmp).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog
import typer

from cerebra.pipeline import run_triage

app = typer.Typer(name="cerebra-triage", add_completion=False, pretty_exceptions_enable=False)

log = structlog.get_logger()


@app.command()
def triage(
    input_dir: Path = typer.Argument(..., help="Directory with BraTS NIfTI or DICOM series"),
    output_dir: Path = typer.Option(Path("/tmp"), "--output-dir", "-o", help="Output directory for PNG"),
    study_id: str | None = typer.Option(None, "--study-id", help="Optional stable study identifier"),
) -> None:
    """Run Cerebra triage on a brain MRI study and print the JSON report."""
    if not input_dir.is_dir():
        typer.echo(f"Error: {input_dir} is not a directory", err=True)
        raise typer.Exit(code=1)

    try:
        report = run_triage(
            input_path=input_dir,
            output_dir=output_dir,
            study_id=study_id,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        log.exception("cli.triage.error", exc=str(exc))
        raise typer.Exit(code=2) from exc

    print(json.dumps(report.model_dump(), indent=2))
    typer.echo(f"\nOverlay saved to: {report.visualization_path}", err=True)
