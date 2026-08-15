"""``argus fix`` — emit a coding-agent-ready fix prompt for a run's root cause."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from argus.fix_prompt import FixPromptError, build_fix_prompt_for_record
from argus.storage import load_run

# Errors and status go to stderr so `argus fix <id> > fix.md` stays clean.
console = Console(stderr=True)


def fix_run(
    run_id: str,
    *,
    node: Optional[str] = None,
    output: Optional[str] = None,
    sanitized: bool = False,
) -> None:
    """Build the fix prompt and write it to stdout or a file."""
    try:
        record = load_run(run_id)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    try:
        result = build_fix_prompt_for_record(
            record, node=node, sanitized=sanitized
        )
    except FixPromptError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    if output:
        path = Path(output)
        try:
            path.write_text(result.prompt, encoding="utf-8")
        except OSError as e:
            console.print(f"[red]Error:[/red] could not write {path}: {e}")
            raise typer.Exit(1) from e
        console.print(
            f"  [green]Fix prompt for[/green] [cyan]{result.node}[/cyan] "
            f"[green]written to[/green] [cyan]{path}[/cyan]"
        )
        return

    # Plain write, not console.print — rich would wrap and re-style the
    # markdown, and this output is meant to be pasted verbatim.
    #
    # The prompt contains em-dashes and arrows. On Windows a redirected
    # stdout uses the locale encoding (cp1252), not UTF-8, so a bare print()
    # raises UnicodeEncodeError on exactly the documented `argus fix <id> >
    # fix.md` invocation. Write UTF-8 bytes to the underlying buffer when the
    # stream cannot encode the text itself.
    try:
        sys.stdout.write(result.prompt)
    except UnicodeEncodeError:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            raise
        buffer.write(result.prompt.encode("utf-8"))
        buffer.flush()
