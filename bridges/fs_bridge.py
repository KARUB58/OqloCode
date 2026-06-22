"""Oqlo Code — local filesystem bridge.

Gives the LLM the ability to read, write, and edit files on the host
filesystem without requiring any external application to be running.
All paths are resolved as absolute (expanduser + resolve) so relative
paths work from wherever the CLI is launched.
"""

from __future__ import annotations

from pathlib import Path

from .result import ToolResult

_MAX_READ_CHARS: int = 12_000  # hard cap to keep tool results inside token budget


class FSBridge:
    """Read / write / edit / list local files."""

    # ------------------------------------------------------------------ #
    # read_file
    # ------------------------------------------------------------------ #
    async def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        try:
            target = Path(path).expanduser().resolve()
            if not target.exists():
                return ToolResult(
                    ok=False, tool="read_file",
                    summary=f"File not found: {target}",
                    error=f"No such file or directory: {target}",
                )
            if target.is_dir():
                return ToolResult(
                    ok=False, tool="read_file",
                    summary=f"{target} is a directory — use list_directory instead",
                    error="Path is a directory.",
                )
            content = target.read_text(encoding=encoding, errors="replace")
            truncated = len(content) > _MAX_READ_CHARS
            if truncated:
                content = content[:_MAX_READ_CHARS]
            lines = len(content.splitlines())
            suffix = (
                f"\n\n[TRUNCATED — showing first {_MAX_READ_CHARS:,} chars of {target}]"
                if truncated else ""
            )
            return ToolResult(
                ok=True, tool="read_file",
                summary=f"Read {target.name} ({lines} lines{', truncated' if truncated else ''})",
                data={"path": str(target), "content": content + suffix, "lines": lines},
            )
        except OSError as exc:
            return ToolResult(ok=False, tool="read_file", summary=str(exc), error=str(exc))

    # ------------------------------------------------------------------ #
    # write_file
    # ------------------------------------------------------------------ #
    async def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        try:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            target.write_text(content, encoding=encoding)
            lines = len(content.splitlines())
            action = "updated" if existed else "created"
            return ToolResult(
                ok=True, tool="write_file",
                summary=f"File {action}: {target.name} ({lines} lines)",
                data={"path": str(target), "lines": lines, "action": action},
            )
        except OSError as exc:
            return ToolResult(ok=False, tool="write_file", summary=str(exc), error=str(exc))

    # ------------------------------------------------------------------ #
    # edit_file
    # ------------------------------------------------------------------ #
    async def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        try:
            target = Path(path).expanduser().resolve()
            if not target.exists():
                return ToolResult(
                    ok=False, tool="edit_file",
                    summary=f"File not found: {target}",
                    error=f"No such file: {target}",
                )
            original = target.read_text(encoding=encoding, errors="replace")
            count = original.count(old_string)
            if count == 0:
                preview = original[:400].replace("\n", "↵")
                return ToolResult(
                    ok=False, tool="edit_file",
                    summary="old_string not found in file",
                    error=(
                        f"The exact string was not found in {target.name}. "
                        f"File preview: {preview!r}"
                    ),
                )
            if count > 1:
                return ToolResult(
                    ok=False, tool="edit_file",
                    summary=f"old_string matches {count} locations — add more context to make it unique",
                    error=(
                        f"old_string appears {count} times in {target.name}. "
                        "Include more surrounding lines so the match is unambiguous."
                    ),
                )
            updated = original.replace(old_string, new_string, 1)
            target.write_text(updated, encoding=encoding)
            lines = len(updated.splitlines())
            return ToolResult(
                ok=True, tool="edit_file",
                summary=f"Edited {target.name} ({lines} lines after edit)",
                data={"path": str(target), "lines": lines, "replacements": 1},
            )
        except OSError as exc:
            return ToolResult(ok=False, tool="edit_file", summary=str(exc), error=str(exc))

    # ------------------------------------------------------------------ #
    # list_directory
    # ------------------------------------------------------------------ #
    async def list_directory(
        self,
        path: str,
        max_entries: int = 200,
    ) -> ToolResult:
        try:
            target = Path(path).expanduser().resolve()
            if not target.exists():
                return ToolResult(
                    ok=False, tool="list_directory",
                    summary=f"Path not found: {target}",
                    error=f"No such path: {target}",
                )
            if not target.is_dir():
                return ToolResult(
                    ok=False, tool="list_directory",
                    summary=f"{target} is a file — use read_file",
                    error="Path is a file, not a directory.",
                )
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            shown = entries[:max_entries]
            lines: list[str] = []
            for e in shown:
                if e.is_dir():
                    lines.append(f"[dir]  {e.name}/")
                else:
                    try:
                        size = e.stat().st_size
                        lines.append(f"[file] {e.name}  ({size:,} B)")
                    except OSError:
                        lines.append(f"[file] {e.name}")
            if len(entries) > max_entries:
                lines.append(f"… and {len(entries) - max_entries} more entries (not shown)")
            listing = "\n".join(lines)
            return ToolResult(
                ok=True, tool="list_directory",
                summary=f"Listed {target}: {len(shown)} entries",
                data={"path": str(target), "listing": listing, "total": len(entries)},
            )
        except OSError as exc:
            return ToolResult(ok=False, tool="list_directory", summary=str(exc), error=str(exc))
