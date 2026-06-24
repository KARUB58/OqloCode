"""Cursor / Antigravity bridge — applies patches and opens workspaces.

Cursor exposes a CLI (``cursor``) for opening folders/files. This bridge uses it
when present and otherwise performs the equivalent file-system operation
directly. Patching is done with ``git apply`` so diffs round-trip cleanly.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import config
from .result import ToolResult


class CursorBridge:
    """Drives the Cursor IDE workspace and applies git-diff patches."""

    def __init__(self) -> None:
        self.workspace = config.BRIDGES.cursor_workspace
        self.preset: str = "default"

    def _resolve(self, file_path: str) -> Path:
        base = self.workspace or Path.cwd()
        return (base / file_path).resolve()

    async def patch_workspace_file(
        self, file_path: str, unified_diff: str
    ) -> ToolResult:
        target = self._resolve(file_path)
        if not target.exists():
            return ToolResult(
                ok=False,
                tool="patch_workspace_file",
                summary="Patch target missing",
                error=f"FileNotFoundError: {target} does not exist.",
            )
        if shutil.which("git") is None:
            return ToolResult(
                ok=False,
                tool="patch_workspace_file",
                summary="git unavailable",
                error="git is required to apply unified diffs but was not found.",
            )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".diff", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(unified_diff)
            diff_path = fh.name
        proc = await asyncio.create_subprocess_exec(
            "git", "apply", "--whitespace=nowarn", diff_path,
            cwd=str(self.workspace or Path.cwd()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        Path(diff_path).unlink(missing_ok=True)
        if proc.returncode != 0:
            return ToolResult(
                ok=False,
                tool="patch_workspace_file",
                summary="git apply failed",
                error=stderr.decode().strip() or "git apply returned non-zero.",
            )
        return ToolResult(
            ok=True,
            tool="patch_workspace_file",
            summary=f"Applied patch to {file_path}.",
            data={"file": file_path},
        )

    async def set_antigravity_model_preset(
        self, preset: str, open_workspace: str | None = None
    ) -> ToolResult:
        # Terminal-only mode: Oqlo never launches a GUI application. We record
        # the preset and resolve/validate the workspace path, reporting it as
        # text instead of spawning the Cursor window.
        self.preset = preset
        workspace = "none"
        if open_workspace:
            ws = Path(open_workspace).expanduser()
            exists = "exists" if ws.exists() else "missing"
            workspace = f"{ws} ({exists}) — not opened (terminal-only mode)"
        return ToolResult(
            ok=True,
            tool="set_antigravity_model_preset",
            summary=f"Antigravity preset set to '{preset}' (terminal-only).",
            data={"preset": preset, "workspace": workspace},
        )
