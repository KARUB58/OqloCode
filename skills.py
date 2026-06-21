"""Oqlo Code — user-extensible Skill registry.

This is what makes Oqlo Code a *free, open* automation system: anyone can drop a
JSON skill definition into ``~/.oqlo/skills/`` (or the local ``./skills/`` dir)
and immediately invoke it from the CLI with ``/skill run <name>`` — no code
changes required. Skills are shell-command templates with named, typed
parameters.

A skill file looks like::

    {
      "name": "blender_render",
      "description": "Render the current .blend to a PNG.",
      "command": "blender -b {blend} -o {out} -f 1",
      "params": {"blend": "Path to .blend file", "out": "Output path"},
      "tags": ["blender", "render"]
    }

Skills are executed in a subprocess with the operator's own permissions; they are
*not* exposed to the LLM by default, keeping the trust boundary explicit.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

USER_SKILL_DIR = Path.home() / ".oqlo" / "skills"
LOCAL_SKILL_DIR = Path.cwd() / "skills"


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    command: str
    params: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = ""

    def render(self, args: dict[str, str]) -> str:
        """Fill the command template, shell-quoting every supplied value."""
        safe = {k: shlex.quote(str(v)) for k, v in args.items()}
        missing = [p for p in self.params if p not in safe]
        if missing:
            raise KeyError(f"missing parameters: {', '.join(missing)}")
        return self.command.format(**safe)


@dataclass(slots=True)
class SkillRunResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    command: str


class SkillRegistry:
    """Loads and runs user-defined skills from disk."""

    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self.dirs = [USER_SKILL_DIR, LOCAL_SKILL_DIR, *(extra_dirs or [])]
        self.skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> int:
        """(Re)scan all skill directories. Returns the number of skills loaded."""
        self.skills.clear()
        for d in self.dirs:
            if not d.is_dir():
                continue
            for path in sorted(d.glob("*.json")):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    skill = Skill(
                        name=raw["name"],
                        description=raw.get("description", ""),
                        command=raw["command"],
                        params=raw.get("params", {}),
                        tags=raw.get("tags", []),
                        source=str(path),
                    )
                    self.skills[skill.name] = skill
                except (KeyError, json.JSONDecodeError, OSError):
                    # Skip malformed skill files rather than failing startup.
                    continue
        return len(self.skills)

    def add(self, skill: Skill, *, to_user_dir: bool = True) -> Path:
        """Persist a new skill to disk and register it."""
        target_dir = USER_SKILL_DIR if to_user_dir else LOCAL_SKILL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{skill.name}.json"
        path.write_text(
            json.dumps(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "command": skill.command,
                    "params": skill.params,
                    "tags": skill.tags,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        skill.source = str(path)
        self.skills[skill.name] = skill
        return path

    def remove(self, name: str) -> bool:
        skill = self.skills.pop(name, None)
        if skill and skill.source:
            Path(skill.source).unlink(missing_ok=True)
            return True
        return False

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def search(self, term: str) -> list[Skill]:
        term = term.lower()
        return [
            s
            for s in self.skills.values()
            if term in s.name.lower()
            or term in s.description.lower()
            or any(term in t.lower() for t in s.tags)
        ]

    async def run(self, name: str, args: dict[str, str]) -> SkillRunResult:
        skill = self.get(name)
        if skill is None:
            return SkillRunResult(False, 127, "", f"unknown skill: {name}", "")
        try:
            command = skill.render(args)
        except KeyError as exc:
            return SkillRunResult(False, 2, "", str(exc), skill.command)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return SkillRunResult(
            ok=proc.returncode == 0,
            exit_code=proc.returncode or 0,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
            command=command,
        )


def seed_example_skills() -> int:
    """Write a couple of starter skills so new users have working examples."""
    registry = SkillRegistry()
    examples = [
        Skill(
            name="blender_render_frame",
            description="Headlessly render a single frame of a .blend file to PNG.",
            command="blender -b {blend} -o {out}# -f {frame}",
            params={"blend": ".blend file", "out": "output path prefix",
                    "frame": "frame number"},
            tags=["blender", "render"],
        ),
        Skill(
            name="unity_build_log",
            description="Tail the most recent Unity Editor log.",
            command="tail -n {lines} {logfile}",
            params={"lines": "number of lines", "logfile": "path to Editor.log"},
            tags=["unity", "logs"],
        ),
        Skill(
            name="open_in_cursor",
            description="Open a folder in the Cursor IDE.",
            command="cursor {folder}",
            params={"folder": "directory to open"},
            tags=["cursor", "ide"],
        ),
    ]
    for s in examples:
        if s.name not in registry.skills:
            registry.add(s)
    return len(examples)
