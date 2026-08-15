"""argus init — write Cursor and Claude project skills for the ARGUS debug loop."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from rich.console import Console

from argus.cli import print_footer
from argus.storage import resolve_project_root

console = Console()

SKILL_NAME = "argus-debug"
TEMPLATE_RESOURCE = "data/skills/argus-debug/SKILL.md"
RELATIVE_TARGETS = (
    Path(".cursor") / "skills" / SKILL_NAME / "SKILL.md",
    Path(".claude") / "skills" / SKILL_NAME / "SKILL.md",
)


def skill_template() -> str:
    """Load the bundled SKILL.md shipped with argus-agents."""
    ref = importlib.resources.files("argus").joinpath(TEMPLATE_RESOURCE)
    return ref.read_text(encoding="utf-8")


def _write_skill(path: Path, body: str, *, force: bool) -> str:
    """Write ``body`` to ``path``. Returns 'wrote' | 'unchanged' | 'skipped'."""
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == body:
            return "unchanged"
        if not force:
            return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return "wrote"


def init_skills(*, force: bool = False, start: Path | None = None) -> dict[str, str]:
    """Write Cursor and Claude project skills under the ARGUS project root.

    Idempotent: identical files are left alone. Differing files are skipped
    unless ``force`` is True.

    Returns a map of relative POSIX paths to action
    (``wrote`` / ``unchanged`` / ``skipped``).
    """
    root = resolve_project_root(start)
    body = skill_template()
    results: dict[str, str] = {}
    for rel in RELATIVE_TARGETS:
        action = _write_skill(root / rel, body, force=force)
        results[rel.as_posix()] = action
    return results


def init_skills_cmd(*, force: bool = False) -> None:
    """CLI entry: write skills and print what happened."""
    root = resolve_project_root()
    results = init_skills(force=force)

    console.print()
    header = "argus init"
    console.print(f"  [bold italic]{header}[/bold italic]")
    console.print()
    console.print(f"  [dim]project root[/dim]  {root}")
    console.print()

    skipped = 0
    for rel, action in results.items():
        if action == "wrote":
            icon = "[bold green]✓[/bold green]"
            label = "wrote"
        elif action == "unchanged":
            icon = "[bold green]✓[/bold green]"
            label = "unchanged"
        else:
            icon = "[bold yellow]⚠[/bold yellow]"
            label = "exists — skipped (use --force to overwrite)"
            skipped += 1
        console.print(f"  {icon}  {rel}  [dim]{label}[/dim]")

    console.print()
    console.print(
        "  Commit these files so later chats know ARGUS exists and read "
        "[bold].argus/runs/<id>.json[/bold] instead of guessing from logs."
    )
    if any(p.startswith(".claude/") for p in results):
        console.print(
            "  [dim]If .claude/ is gitignored, add "
            "!.claude/skills/argus-debug/ so the Claude skill is tracked.[/dim]"
        )
    console.print()
    console.print("  [dim]next[/dim]")
    console.print(
        "    attach with [bold]ArgusWatcher.attach(graph)[/bold], run the pipeline, then:"
    )
    console.print("    [bold]argus show last[/bold]")
    console.print()
    if skipped:
        console.print(
            "  [yellow]Existing skill files were left in place.[/yellow] "
            "Re-run with [bold]--force[/bold] to replace them."
        )
        console.print()
    print_footer()
