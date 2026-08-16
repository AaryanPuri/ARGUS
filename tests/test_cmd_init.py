"""VAR-108: argus init writes Cursor and Claude project skills."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli.cmd_init import (
    RELATIVE_TARGETS,
    SKILL_NAME,
    init_skills,
    skill_template,
)
from argus.cli.main import app

_KEYWORDS = (
    "LangGraph",
    "pipeline failed",
    "silent failure",
    "agent run",
    "ARGUS",
)

_SETUP_TRIGGERS = (
    "attach",
    "wire ARGUS",
    "add monitoring",
    "ArgusWatcher",
    "setup",
    "empty documents",
    ".argus/runs",
)


def _project_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv("ARGUS_DIR", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path.resolve()


@pytest.mark.unit
def test_skill_template_ships_in_package():
    body = skill_template()
    assert body.startswith("---\n")
    assert f"name: {SKILL_NAME}" in body
    desc_block = body.split("---", 2)[1]
    for kw in _KEYWORDS:
        assert kw in desc_block, f"frontmatter description missing {kw!r}"
    for trigger in _SETUP_TRIGGERS:
        assert trigger in desc_block, f"frontmatter description missing {trigger!r}"
    assert "ArgusWatcher.attach" in body
    assert "find the graph" in body.lower()
    assert "Do not convert plain-dict state to TypedDict" in body
    assert "pip install argus-agents" in body
    assert "not `argus`" in body or "not argus" in body
    assert "graph.compile()" in body
    assert ".argus/runs" in body
    assert "argus show" in body
    assert "argus replay" in body
    assert "semantic_judge=False" in body
    assert "guessing from logs" in body
    assert "argus login" in body
    assert "not required" in body.lower() or "optional" in body.lower()


@pytest.mark.unit
def test_init_writes_cursor_and_claude_skills(tmp_path, monkeypatch):
    root = _project_root(tmp_path, monkeypatch)
    results = init_skills()
    assert set(results) == {p.as_posix() for p in RELATIVE_TARGETS}
    assert all(action == "wrote" for action in results.values())
    expected = skill_template()
    for rel in RELATIVE_TARGETS:
        path = root / rel
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == expected


@pytest.mark.unit
def test_init_is_idempotent(tmp_path, monkeypatch):
    _project_root(tmp_path, monkeypatch)
    first = init_skills()
    second = init_skills()
    assert all(v == "wrote" for v in first.values())
    assert all(v == "unchanged" for v in second.values())


@pytest.mark.unit
def test_init_skips_customized_skill_without_force(tmp_path, monkeypatch):
    root = _project_root(tmp_path, monkeypatch)
    init_skills()
    custom = root / RELATIVE_TARGETS[0]
    custom.write_text("custom skill\n", encoding="utf-8")
    results = init_skills()
    assert results[RELATIVE_TARGETS[0].as_posix()] == "skipped"
    assert results[RELATIVE_TARGETS[1].as_posix()] == "unchanged"
    assert custom.read_text(encoding="utf-8") == "custom skill\n"


@pytest.mark.unit
def test_init_force_overwrites_customized_skill(tmp_path, monkeypatch):
    root = _project_root(tmp_path, monkeypatch)
    init_skills()
    custom = root / RELATIVE_TARGETS[0]
    custom.write_text("custom skill\n", encoding="utf-8")
    results = init_skills(force=True)
    assert results[RELATIVE_TARGETS[0].as_posix()] == "wrote"
    assert custom.read_text(encoding="utf-8") == skill_template()


@pytest.mark.unit
def test_init_writes_to_project_root_not_cwd(tmp_path, monkeypatch):
    root = _project_root(tmp_path, monkeypatch)
    nested = root / "notebooks"
    nested.mkdir()
    monkeypatch.chdir(nested)
    init_skills()
    assert (root / RELATIVE_TARGETS[0]).is_file()
    assert not (nested / ".cursor").exists()


@pytest.mark.unit
def test_cli_init_help_and_write(tmp_path, monkeypatch):
    root = _project_root(tmp_path, monkeypatch)
    runner = CliRunner()
    help_result = runner.invoke(app, ["init", "--help"])
    assert help_result.exit_code == 0, help_result.output
    # Rich help wraps / colorizes on CI (80-col Linux); strip ANSI and newlines.
    help_plain = re.sub(r"\x1b\[[0-9;]*m", "", help_result.output).replace("\n", "")
    assert "--force" in help_plain

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert "argus init" in result.output
    assert ".cursor/skills/argus-debug/SKILL.md" in result.output
    assert ".claude/skills/argus-debug/SKILL.md" in result.output
    assert "argus show last" in result.output
    assert (root / RELATIVE_TARGETS[0]).is_file()


@pytest.mark.unit
def test_readme_and_guide_document_argus_init():
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "README.md").read_text(encoding="utf-8")
    guide = (repo / "website/app/guide/guide-content.tsx").read_text(encoding="utf-8")
    assert "argus init" in readme
    assert ".cursor/skills/argus-debug" in readme
    assert ".claude/skills/argus-debug" in readme
    assert "skill already contains the setup prompt" in readme
    assert "fallback" in readme.lower()
    assert "argus init" in guide
    assert ".cursor/skills/argus-debug" in guide
    assert "skill already contains the setup prompt" in guide
    assert "fallback" in guide.lower()
