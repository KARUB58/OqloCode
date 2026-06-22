#!/usr/bin/env python3
"""Oqlo Code — the orchestrator terminal.

An interactive, async CLI that turns LLMs into runtime executives driving
Blender, Unity, and Cursor through pre-built Skills. Free and self-hostable:
bring your own keys, drop in your own skills, run your own pipelines.

Run with::

    python main.py

Type ``/help`` for the full command list. Plain text is sent to the
orchestrator; lines beginning with ``/`` are operator commands.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Awaitable, Callable

import config
from bridges import (
    BridgeRouter,
    BlenderBridge,
    UnityBridge,
)
from bridges.blender_bridge import BLENDER_ADDON_TEMPLATE
from bridges.unity_bridge import UNITY_EDITOR_SCRIPT_TEMPLATE
from llm_router import Conversation, LLMRouter
from orchestrator import Orchestrator, StepRecord
from skills import Skill, SkillRegistry, seed_example_skills
from tools_manifest import ALL_TOOLS
from memory_rag import MemoryRAG
from ast_memory import ASTMemory
from telemetry import Telemetry, TelemetryThrottler
from swarm_bus import SwarmCoordinator
from intent import Intent, classify_intent
from workspace import WorkspaceManager, CrossTask

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    from rich import box
except Exception:  # pragma: no cover
    print("Oqlo Code requires the 'rich' library. Install with: pip install rich")
    sys.exit(1)


console = Console()

# Big, bold ASCII wordmark. Rendered in purple (magenta) at startup.
BANNER = r"""
  ██████╗  ██████╗ ██╗      ██████╗      ██████╗ ██████╗ ██████╗ ███████╗
 ██╔═══██╗██╔═══██╗██║     ██╔═══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
 ██║   ██║██║   ██║██║     ██║   ██║    ██║     ██║   ██║██║  ██║█████╗
 ██║   ██║██║▄▄ ██║██║     ██║   ██║    ██║     ██║   ██║██║  ██║██╔══╝
 ╚██████╔╝╚██████╔╝███████╗╚██████╔╝    ╚██████╗╚██████╔╝██████╔╝███████╗
  ╚═════╝  ╚══▀▀═╝ ╚══════╝ ╚═════╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
"""

SUBTITLE = "        ⟪ The Agentic OS Orchestrator ⟫   ·   terminal-only   ·   BYOK"


class OqloCLI:
    """Interactive terminal wrapping the router, bridges, and skill registry."""

    def __init__(self) -> None:
        self.policy = config.get_routing_policy()
        self.router = LLMRouter(
            chain=config.build_priority_chain(self.policy, only_available=True),
            on_event=self._on_router_event,
        )
        self.bridges = BridgeRouter()
        self.conv = Conversation(config.SYSTEM_PROMPT)
        self.skills = SkillRegistry()
        # Next-gen engines.
        self.memory = MemoryRAG()
        self.ast_memory = ASTMemory()
        self.telemetry = Telemetry()
        self.throttle = TelemetryThrottler(telemetry=self.telemetry)
        self.auto_intent = True  # Module 1: hybrid chat/agentic routing.
        self.verbose = True
        self.running = True
        self._last_events: list[str] = []
        # Active file/directory session (/start <path> sets one of these).
        self.active_file: "Path | None" = None
        self.active_dir: "Path | None" = None  # directory workspace
        # Multi-project workspace module.
        self.wm = WorkspaceManager()
        self.commands: dict[str, Callable[[list[str]], Awaitable[None]]] = {
            "start": self.cmd_start,
            "help": self.cmd_help,
            "model": self.cmd_model,
            "api": self.cmd_api,
            "local": self.cmd_local,
            "swarm": self.cmd_swarm,
            "memory": self.cmd_memory,
            "telemetry": self.cmd_telemetry,
            "budget": self.cmd_budget,
            "models": self.cmd_models,
            "chain": self.cmd_chain,
            "policy": self.cmd_policy,
            "providers": self.cmd_providers,
            "cost": self.cmd_cost,
            "tokensave": self.cmd_tokensave,
            "plan": self.cmd_plan,
            "tools": self.cmd_tools,
            "status": self.cmd_status,
            "history": self.cmd_history,
            "save": self.cmd_save,
            "load": self.cmd_load,
            "clear": self.cmd_clear,
            "reset": self.cmd_clear,
            "export": self.cmd_export,
            "blender": self.cmd_blender,
            "unity": self.cmd_unity,
            "cursor": self.cmd_cursor,
            "skill": self.cmd_skill,
            "addon": self.cmd_addon,
            "config": self.cmd_config,
            "verbose": self.cmd_verbose,
            "write": self.cmd_write,
            "file": self.cmd_file,
            "autosave": self.cmd_autosave,
            "workspace": self.cmd_workspace,
            "ws": self.cmd_workspace,
            "quit": self.cmd_quit,
            "exit": self.cmd_quit,
        }

    # ------------------------------------------------------------------ #
    # Router event hook -> live status line
    # ------------------------------------------------------------------ #
    def _on_router_event(self, event: str, fields: dict[str, Any]) -> None:
        if not self.verbose:
            return
        if event == "attempt":
            free_note = (
                " [dim italic](:free tier — shared queue, may take 30-90 s)[/]"
                if ":free" in fields.get("model", "") else ""
            )
            console.print(
                f"[dim]→ trying [bold]{fields['model']}[/]{free_note} "
                f"(attempt {fields['attempt']})[/dim]"
            )
        elif event == "skip":
            console.print(
                f"[yellow]⤳ skipping {fields['model']}: {fields['reason']}[/yellow]"
            )
        elif event == "fallback":
            console.print(
                f"[red]✗ {fields['model']} failed "
                f"(status {fields.get('status')}): "
                f"{fields.get('message', '')}[/red] [dim]→ falling back[/dim]"
            )
        elif event == "success":
            console.print(
                f"[green]✓ {fields['model']}[/] "
                f"[dim]{fields['latency_ms']}ms · "
                f"in {fields['in_tok']} / out {fields['out_tok']} tok · "
                f"${fields['cost']:.6f} · {fields['tool_calls']} tool call(s)[/dim]"
            )

    # ------------------------------------------------------------------ #
    # Orchestration step hook
    # ------------------------------------------------------------------ #
    def _on_step(self, rec: StepRecord) -> None:
        if rec.kind == "plan" and rec.text:
            console.print(Panel(rec.text, title="🗺️  plan (token-saver)",
                                border_style="magenta", box=box.ROUNDED))
        elif rec.kind == "memory" and rec.text:
            console.print(Panel(rec.text, title="🧩 memory recall",
                                border_style="blue", box=box.ROUNDED))
        elif rec.kind == "freeze":
            console.print(Panel(rec.text, title="⛔ financial freeze (HITL)",
                                border_style="red", box=box.HEAVY))
        elif rec.kind == "assistant" and rec.text:
            console.print(Panel(rec.text, title=f"🧠 {rec.model}",
                                border_style="cyan", box=box.ROUNDED))
        elif rec.kind == "tool":
            tag = "[yellow]SIM[/]" if rec.simulated else (
                "[green]OK[/]" if rec.ok else "[red]ERR[/]")
            console.print(f"  🔧 [bold]{rec.tool}[/] {tag} — {rec.text}")
            for k, v in rec.data.items():
                console.print(f"     [dim]{k}:[/] {v}")
        elif rec.kind == "heal":
            console.print(f"  [magenta]🩹 {rec.text}[/magenta]")
        elif rec.kind == "final" and rec.text:
            console.print(Panel(rec.text, title="✅ result",
                                border_style="green", box=box.ROUNDED))

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    async def cmd_help(self, _: list[str]) -> None:
        t = Table(title="Oqlo Code — Commands", box=box.SIMPLE_HEAVY,
                  header_style="bold cyan")
        t.add_column("Command")
        t.add_column("Description")
        sections = {
            "Getting started": [
                ("/start oqlocode", "Show the banner and full system status."),
                ("/help", "Show this command reference."),
                ("/status", "Active model, policy, bridges, token settings."),
            ],
            "Providers & routing (priority: 1.antigravity 2.cursor 3.codex 4.api)": [
                ("/model <slug>", "Override the active engine live (any OpenRouter or "
                                  "native slug), e.g. /model nvidia/nemotron-3-ultra:free "
                                  "or /model anthropic/claude-4.6-opus. /model reset to revert."),
                ("/api list", "Show API key status for every cloud provider."),
                ("/api <provider> <key>", "Bind a key live, e.g. /api openrouter sk-..."),
                ("/api <provider> clear", "Remove a bound key."),
                ("/local list", "Show local executive activation state."),
                ("/local <antigravity|cursor> on|off", "Activate/deactivate a local agent."),
                ("/providers", "BYOK status per provider."),
                ("/models", "Model catalogue with cost & availability."),
                ("/chain", "Current ordered fallback chain."),
                ("/policy [name]", "local_first / max_intelligence / cost_optimization."),
            ],
            "Token saving": [
                ("/tokensave on|off", "Toggle aggressive trimming + lower output caps."),
                ("/plan on|off", "Toggle the cheap planning pass before execution."),
                ("/cost", "Session token usage & USD spend."),
            ],
            "Next-gen engines": [
                ("/swarm <task>", "Run the 4-agent swarm (Architect/Blender/Unity/QA)."),
                ("/memory ...", "RAG: stats/search <q>/add <text>/forget <id>/ast."),
                ("/telemetry", "Live CPU/RAM/GPU-VRAM + throttle stats."),
                ("/budget [limit <usd>|resume]", "Financial velocity guard ($/min)."),
            ],
            "Hybrid chat (plain text auto-routes)": [
                ("just type a question", "Chat goes straight to the model — no pipeline."),
                ("just type an instruction", "Actionable work spins up the agent pipeline."),
            ],
            "Multi-project workspaces (/ws alias works too)": [
                ("/workspace add <name> <path> [type]",
                 "Register a workspace root (Blender|Unity|Cursor|Generic). "
                 "E.g. /workspace add server /home/user/api Generic"),
                ("/workspace list", "Show all registered workspaces."),
                ("/workspace switch <name>", "Set the active workspace context."),
                ("/workspace remove <name>", "Unregister a workspace."),
                ("/workspace index [name|all]",
                 "Walk Python files, extract AST symbols, detect cross-workspace "
                 "dependencies. Flags stale symbols when a hash changes."),
                ("/workspace symbols [ws] [term]",
                 "Search the cross-workspace symbol registry."),
                ("/workspace stale",
                 "Show symbols flagged stale by a cross-workspace hash change."),
                ("/workspace deps",
                 "List all detected cross-workspace dependency links."),
                ("/workspace resolve [ws]:path",
                 "Resolve a VFFS path expression to an absolute path."),
                ("/workspace sync ws:task [ws:task ...]",
                 "Run a cross-project pipeline with dependency locking. "
                 "E.g. /workspace sync server:compile client:types client:tests"),
                ("/workspace status", "Full workspace + symbol registry overview."),
            ],
            "File / directory session (enter full path)": [
                ("/start /path/to/file.py [task]",
                 "Open a single-file session — subsequent messages are "
                 "automatically given the file's content. "
                 "E.g. /start /home/user/project/app.py add logging"),
                ("/start /path/to/directory [task]",
                 "Open a directory session — shows files, resolves "
                 "filenames mentioned in messages against that dir. "
                 "E.g. /start /home/user/OqloCode"),
                ("/start clear", "End the active file/directory session."),
                ("/write [/path/to/file]",
                 "Write the last code block from the conversation back to "
                 "the active file (or any path you specify)."),
                ("/file show|clear",
                 "Show info about the active session or clear it."),
                ("/autosave on|off|threshold <N>",
                 "Toggle auto-save (default on). Changes ≤ threshold lines "
                 "are saved silently; larger changes ask for approval."),
            ],
            "Bridges (terminal-only — never launches apps)": [
                ("/tools", "LLM-accessible bridge tools."),
                ("/blender <bpy>", "Run a bpy snippet directly via the bridge."),
                ("/unity logs [sev]", "Fetch Unity editor logs."),
                ("/cursor open <path>", "Set workspace path (reported, not opened)."),
                ("/addon <blender|unity>", "Print the in-app bridge addon/script."),
                ("/export", "List exported FBX artifacts."),
            ],
            "Skills & session": [
                ("/skill ...", "list/search/run/add/remove/reload/examples."),
                ("/history", "Show the conversation buffer."),
                ("/save <file>  •  /load <file>", "Persist / restore a session."),
                ("/clear  •  /reset", "Reset the conversation buffer."),
                ("/config", "Show resolved configuration."),
                ("/verbose <on|off>", "Toggle router/step detail."),
                ("/quit  •  /exit", "Leave Oqlo Code."),
            ],
        }
        for section, rows in sections.items():
            t.add_row(f"[bold magenta]{section}[/]", "")
            for c, d in rows:
                t.add_row(c, d)
        console.print(t)
        console.print(
            "[dim]Any line without a leading '/' is sent to the orchestrator.[/dim]"
        )

    # --- start / banner / file-or-directory session ----------------------- #
    async def cmd_start(self, args: list[str]) -> None:
        # /start  OR  /start oqlocode  → show banner + status (existing behaviour)
        if not args or args[0].lower() == "oqlocode":
            console.print(Text(BANNER, style="bold magenta"))
            console.print(f"[magenta]{SUBTITLE}[/magenta]\n")
            await self.cmd_status([])
            console.print("\n[dim]Type /help for all commands.[/dim]")
            return

        # /start clear → close the active session
        if args[0].lower() == "clear":
            self.active_file = None
            self.active_dir = None
            console.print("[green]Session cleared.[/green]")
            return

        # /start <full-path> [initial task...]
        from pathlib import Path as _Path
        # Normalise Windows-style backslashes if pasted from Explorer.
        raw = args[0].replace("\\", "/")
        target = _Path(raw).expanduser().resolve()
        task = " ".join(args[1:])

        if not target.exists():
            console.print(
                f"[red]Path not found:[/red] {target}\n"
                f"[dim]Current working directory: {_Path.cwd()}[/dim]\n"
                "[dim]Tip: use the full/absolute path, "
                "e.g. /start /home/user/project/main.py[/dim]"
            )
            return

        # ── DIRECTORY ─────────────────────────────────────────────────────
        if target.is_dir():
            self.active_dir = target
            self.active_file = None
            # List the files for the user.
            entries = sorted(target.iterdir(), key=lambda p: (p.is_dir(), p.name))
            t = Table(title=f"📁 {target}", box=box.SIMPLE, header_style="bold cyan")
            t.add_column("Name")
            t.add_column("Type")
            t.add_column("Size", justify="right")
            for e in entries[:40]:
                kind = "[blue]dir[/]" if e.is_dir() else "file"
                size = f"{e.stat().st_size:,} B" if e.is_file() else ""
                t.add_row(e.name, kind, size)
            if len(entries) > 40:
                t.add_row(f"…and {len(entries)-40} more", "", "")
            console.print(t)
            console.print(Panel(
                f"[bold cyan]{target}[/]\n\n"
                "Directory session open — mention any filename and I'll resolve it "
                "from this location automatically.\n"
                "[dim]Example: düzenle main.py içine logging ekle[/dim]\n"
                "[dim]/start clear  to end · /file show for status[/dim]",
                title="📁 Directory Session Started",
                border_style="magenta", box=box.ROUNDED,
            ))
            if task:
                await self._dispatch_with_file(task)
            return

        # ── SINGLE FILE ───────────────────────────────────────────────────
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            console.print(f"[red]Cannot read {target}: {exc}[/red]")
            return

        self.active_file = target
        self.active_dir = target.parent  # also set parent dir for context
        lines = content.splitlines()
        console.print(Panel(
            f"[bold cyan]{target}[/]\n"
            f"[dim]{len(lines)} lines · {len(content):,} chars[/dim]\n\n"
            "Just type your instructions — file is auto-injected every turn.\n"
            "[dim]/start clear  to end · /write to save last code block back[/dim]",
            title="📄 File Session Started",
            border_style="magenta", box=box.ROUNDED,
        ))
        if task:
            await self._dispatch_with_file(task)

    async def _dispatch_with_file(self, task: str) -> None:
        """Dispatch a task with active file/directory content injected as context."""
        cap = 8_000
        context_lines: list[str] = []

        if self.active_file and self.active_file.exists():
            try:
                content = self.active_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                console.print(f"[yellow]Could not re-read active file: {exc}[/yellow]")
                content = ""
            snippet = content[:cap]
            if len(content) > cap:
                snippet += f"\n…[{len(content) - cap:,} chars not shown]"
            context_lines.append(
                f"[Active file: {self.active_file}]\n```\n{snippet}\n```"
            )
        elif self.active_dir and self.active_dir.exists():
            # Directory session: give the model the directory listing plus
            # content of any file mentioned in the task.
            from pathlib import Path as _Path
            listing = "\n".join(
                e.name for e in sorted(self.active_dir.iterdir())
                if not e.name.startswith(".")
            )
            context_lines.append(
                f"[Active directory: {self.active_dir}]\nFiles:\n{listing}"
            )
            # Try to find a file mentioned by name in the task and inject it.
            for word in task.replace(",", " ").split():
                candidate = self.active_dir / word
                if candidate.is_file():
                    try:
                        fc = candidate.read_text(encoding="utf-8", errors="replace")
                        snippet = fc[:cap]
                        if len(fc) > cap:
                            snippet += f"\n…[{len(fc)-cap:,} chars not shown]"
                        context_lines.append(
                            f"[File content: {candidate}]\n```\n{snippet}\n```"
                        )
                    except OSError:
                        pass
                    break

        augmented = "\n\n".join(context_lines) + f"\n\nTask: {task}"
        intent = classify_intent(task)
        if intent is Intent.CHAT:
            console.print("[dim]↳ intent: chat (direct model, no pipeline)[/dim]")
            await self._chat(augmented)
        else:
            console.print("[dim]↳ intent: agentic (multi-agent pipeline)[/dim]")
            await self._agentic(augmented)

        # After any file-session response, check whether to auto-save.
        await self._maybe_auto_save(task)

    # --- auto-save logic -------------------------------------------------- #
    # Threshold: code blocks with more lines than this trigger an approval
    # prompt; smaller blocks (or explicit /autosave off) skip the prompt.
    _AUTO_SAVE_THRESHOLD: int = 40

    async def _maybe_auto_save(self, hint: str = "") -> None:
        """Find the last code block in the conversation and save it.

        * Lines ≤ threshold  → auto-save silently (green tick).
        * Lines > threshold  → show a preview and ask [y/N] before writing.
        * No active_file     → nothing to do (directory sessions skip auto-save).
        """
        import re

        if not getattr(self, "_autosave_enabled", True):
            return  # user disabled auto-save
        if not self.active_file:
            return  # directory sessions don't auto-save (no single target)

        # Find the most recent assistant message that contains a code block.
        code: str | None = None
        for m in reversed(self.conv.messages):
            if m.role == "assistant" and m.text:
                blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", m.text)
                if blocks:
                    code = max(blocks, key=len)
                    break
        if code is None or not code.strip():
            return

        lines = code.splitlines()
        target = self.active_file

        # Determine whether the change is "important" (needs approval).
        existing_lines = 0
        if target.exists():
            try:
                existing_lines = len(
                    target.read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError:
                pass
        new_lines = len(lines)
        # Important if: new file, or block is large, or replaces > 60% of file.
        is_important = (
            not target.exists()
            or new_lines > self._AUTO_SAVE_THRESHOLD
            or (existing_lines > 0
                and abs(new_lines - existing_lines) / existing_lines > 0.6)
        )

        if not is_important:
            # Small change — write silently.
            try:
                target.write_text(code, encoding="utf-8")
                console.print(
                    f"[dim green]✓ auto-saved → {target.name} "
                    f"({new_lines} lines)[/dim green]"
                )
            except OSError as exc:
                console.print(f"[red]Auto-save failed: {exc}[/red]")
            return

        # Large / important change — show a diff summary and ask.
        console.print(Panel(
            f"[bold]{target}[/]\n"
            f"Current: [dim]{existing_lines} lines[/dim] → "
            f"New: [bold]{new_lines} lines[/bold]\n\n"
            + "\n".join(lines[:12])
            + ("\n[dim]…[/dim]" if new_lines > 12 else ""),
            title="💾 Save confirmation required",
            border_style="yellow", box=box.ROUNDED,
        ))
        try:
            answer = (
                await asyncio.to_thread(
                    input, "  Save to file? [y/N] → "
                )
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer in ("y", "yes", "evet", "e"):
            try:
                target.write_text(code, encoding="utf-8")
                console.print(
                    f"[green]✓ Saved → {target} ({new_lines} lines).[/green]"
                )
            except OSError as exc:
                console.print(f"[red]Save failed: {exc}[/red]")
        else:
            console.print(
                "[dim]Not saved. Use /write to save manually later.[/dim]"
            )

    # --- /write — apply last code block to active file -------------------- #
    async def cmd_write(self, args: list[str]) -> None:
        """Write the last code block from the conversation to the active file."""
        import re
        from pathlib import Path as _Path
        target = _Path(args[0]).expanduser() if args else self.active_file
        if not target:
            console.print(
                "[red]No active file. Use /start <filename> first, "
                "or /write <filename>.[/red]"
            )
            return
        for m in reversed(self.conv.messages):
            if m.role == "assistant" and m.text:
                blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", m.text)
                if blocks:
                    largest = max(blocks, key=len)
                    target.write_text(largest, encoding="utf-8")
                    console.print(
                        f"[green]✓ Written to {target} "
                        f"({len(largest.splitlines())} lines).[/green]"
                    )
                    return
        console.print("[yellow]No code block found in recent conversation.[/yellow]")

    # --- /file — show / clear active session ------------------------------ #
    async def cmd_file(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "show"
        if sub == "clear":
            self.active_file = None
            self.active_dir = None
            console.print("[green]Session cleared.[/green]")
        elif sub in ("show", "status") or not args:
            if self.active_file:
                try:
                    content = self.active_file.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    lines = content.splitlines()
                    console.print(Panel(
                        f"[bold]{self.active_file}[/]\n"
                        f"[dim]{len(lines)} lines · {len(content):,} chars[/dim]",
                        title="📄 Active File",
                        border_style="magenta", box=box.ROUNDED,
                    ))
                except OSError as exc:
                    console.print(f"[red]{exc}[/red]")
            elif self.active_dir:
                entries = sorted(self.active_dir.iterdir(),
                                 key=lambda p: (p.is_dir(), p.name))
                names = [
                    ("[blue]" + e.name + "/[/]" if e.is_dir() else e.name)
                    for e in entries[:30]
                ]
                console.print(Panel(
                    f"[bold]{self.active_dir}[/]\n\n"
                    + "  ".join(names)
                    + (f"\n[dim]…and {len(entries)-30} more[/]"
                       if len(entries) > 30 else ""),
                    title="📁 Active Directory",
                    border_style="magenta", box=box.ROUNDED,
                ))
            else:
                console.print(
                    "[dim]No active session. "
                    "Use /start /full/path/to/file.py  or  "
                    "/start /full/path/to/directory[/dim]"
                )
        else:
            console.print("[red]Usage: /file show|clear[/red]")

    # --- /autosave — toggle auto-save feature ------------------------------ #
    async def cmd_autosave(self, args: list[str]) -> None:
        """Toggle or configure the auto-save feature.

        Usage:
          /autosave            → show status
          /autosave on|off     → enable / disable
          /autosave threshold <N>  → set line threshold (default 40)
        """
        sub = args[0].lower() if args else "status"

        if sub in ("on", "enable"):
            self._autosave_enabled = True
            console.print(
                f"[green]✓ Auto-save ON[/green]  "
                f"(threshold: {self._AUTO_SAVE_THRESHOLD} lines — "
                "changes above that size will ask for approval)"
            )

        elif sub in ("off", "disable"):
            self._autosave_enabled = False
            console.print(
                "[yellow]Auto-save OFF.[/yellow]  "
                "Use [bold]/write[/bold] to save manually."
            )

        elif sub == "threshold":
            if len(args) < 2 or not args[1].isdigit():
                console.print("[red]Usage: /autosave threshold <number>[/red]")
                return
            self.__class__._AUTO_SAVE_THRESHOLD = int(args[1])
            console.print(
                f"[green]✓ Threshold set to {self._AUTO_SAVE_THRESHOLD} lines.[/green]"
            )

        else:  # status
            enabled = getattr(self, "_autosave_enabled", True)
            status_str = "[green]ON[/green]" if enabled else "[yellow]OFF[/yellow]"
            active_path = self.active_file or self.active_dir
            console.print(Panel(
                f"Auto-save: {status_str}\n"
                f"Threshold: [bold]{self._AUTO_SAVE_THRESHOLD}[/bold] lines "
                "(above → approval prompt, below → silent save)\n"
                f"Active target: [dim]{active_path or 'none'}[/dim]\n\n"
                "[dim]Commands:[/dim]\n"
                "  /autosave on|off\n"
                "  /autosave threshold <N>\n"
                "  /write [path]  — save last code block manually",
                title="💾 Auto-Save",
                border_style="cyan", box=box.ROUNDED,
            ))

    # ------------------------------------------------------------------ #
    # /workspace — multi-project workspace module
    # ------------------------------------------------------------------ #
    async def cmd_workspace(self, args: list[str]) -> None:  # noqa: C901
        """Virtual Federated File System + cross-boundary symbol resolver."""
        sub = args[0].lower() if args else "list"

        # ── /workspace list ──────────────────────────────────────────────
        if sub == "list":
            ws_list = self.wm.vffs.all()
            if not ws_list:
                console.print(
                    "[dim]No workspaces registered. "
                    "Use /workspace add <name> <path> [type][/dim]"
                )
                return
            t = Table(title="Registered Workspaces (VFFS)",
                      box=box.ROUNDED, header_style="bold cyan")
            t.add_column("Name")
            t.add_column("Root Path")
            t.add_column("Type")
            t.add_column("AST")
            t.add_column("Active")
            active = self.wm.state.active_workspace_context
            for ws in ws_list:
                t.add_row(
                    ws.name,
                    str(ws.root),
                    ws.environment_type,
                    "[green]on[/]" if ws.ast_enabled else "[red]off[/]",
                    "[green]◀[/]" if ws.name == active else "",
                )
            console.print(t)
            stale = self.wm.symbols.stale_symbols()
            if stale:
                console.print(
                    f"[yellow]⚠ {len(stale)} stale cross-workspace symbol(s). "
                    "Run /workspace index to refresh.[/yellow]"
                )

        # ── /workspace add <name> <path> [type] ─────────────────────────
        elif sub == "add":
            if len(args) < 3:
                console.print(
                    "[red]Usage: /workspace add <name> <path> "
                    "[Blender|Unity|Cursor|Generic][/red]"
                )
                return
            name, raw_path = args[1], args[2]
            env_type = args[3].capitalize() if len(args) > 3 else "Generic"
            from pathlib import Path as _P
            root = _P(raw_path.replace("\\", "/")).expanduser().resolve()
            if not root.exists():
                console.print(f"[red]Path does not exist: {root}[/red]")
                return
            profile = self.wm.add_workspace(name, str(root), env_type)
            console.print(Panel(
                f"[bold cyan]{profile.name}[/]  →  {profile.root}\n"
                f"Type: {profile.environment_type}  ·  "
                f"AST indexing: {'enabled' if profile.ast_enabled else 'disabled'}",
                title="✅ Workspace Registered",
                border_style="green", box=box.ROUNDED,
            ))

        # ── /workspace remove <name> ─────────────────────────────────────
        elif sub in ("remove", "rm"):
            if len(args) < 2:
                console.print("[red]Usage: /workspace remove <name>[/red]")
                return
            if self.wm.remove_workspace(args[1]):
                console.print(f"[green]Workspace '{args[1]}' removed.[/green]")
            else:
                console.print(f"[red]Workspace '{args[1]}' not found.[/red]")

        # ── /workspace switch <name> ─────────────────────────────────────
        elif sub in ("switch", "use"):
            if len(args) < 2:
                console.print("[red]Usage: /workspace switch <name>[/red]")
                return
            if self.wm.switch_workspace(args[1]):
                ws = self.wm.vffs.get(args[1])
                console.print(
                    f"[green]Switched to workspace '{args[1]}' "
                    f"({ws.root if ws else '?'})[/green]"
                )
            else:
                console.print(f"[red]Workspace '{args[1]}' not found.[/red]")

        # ── /workspace index [name|all] ──────────────────────────────────
        elif sub == "index":
            name = args[1] if len(args) > 1 else None
            if name and name != "all":
                result = self.wm.index_workspace(name)
                if "error" in result:
                    console.print(f"[red]{result['error']}[/red]")
                else:
                    console.print(Panel(
                        f"[bold]{result['workspace']}[/]\n"
                        f"Files scanned: {result.get('py_files', 0)}\n"
                        f"Symbols added: {result.get('symbols_added', 0)}\n"
                        f"Symbols updated: {result.get('symbols_updated', 0)}\n"
                        f"Stale flags: {result.get('stale_flags', 0)}",
                        title="🔍 AST Index Result",
                        border_style="cyan", box=box.ROUNDED,
                    ))
            else:
                results = self.wm.index_all()
                for r in results:
                    if "error" in r:
                        console.print(f"[red]{r['error']}[/red]")
                    else:
                        console.print(
                            f"[green]✓ {r['workspace']}[/] "
                            f"{r.get('py_files',0)} files · "
                            f"+{r.get('symbols_added',0)} sym "
                            f"~{r.get('symbols_updated',0)} upd"
                        )

        # ── /workspace symbols [name] [term] ─────────────────────────────
        elif sub in ("symbols", "sym"):
            ws_filter = args[1] if len(args) > 1 else None
            term = args[2] if len(args) > 2 else ""
            hits = self.wm.symbols.search(term, workspace=ws_filter)
            if not hits:
                console.print("[dim]No matching symbols.[/dim]")
                return
            t = Table(title="Symbol Registry", box=box.SIMPLE,
                      header_style="bold cyan")
            t.add_column("FQN")
            t.add_column("Kind")
            t.add_column("Signature")
            for s in hits[:30]:
                t.add_row(s.fqn, s.kind, s.signature)
            console.print(t)
            st = self.wm.symbols.stats()
            console.print(
                f"[dim]Total: {st['total_symbols']} symbols · "
                f"{st['dependency_links']} links · "
                f"{st['cross_workspace_links']} cross-ws · "
                f"{st['stale_flags']} stale[/dim]"
            )

        # ── /workspace stale ─────────────────────────────────────────────
        elif sub == "stale":
            stale = self.wm.symbols.stale_symbols()
            if not stale:
                console.print("[green]No stale cross-workspace symbols.[/green]")
                return
            console.print(Panel(
                "\n".join(s.fqn for s in stale),
                title=f"⚠ {len(stale)} Stale Symbol(s)",
                border_style="yellow", box=box.ROUNDED,
            ))
            console.print(
                "[dim]These depend on symbols that changed in another workspace. "
                "Re-run /workspace index after updating them.[/dim]"
            )

        # ── /workspace deps ──────────────────────────────────────────────
        elif sub == "deps":
            links = self.wm.symbols.cross_workspace_deps()
            if not links:
                console.print("[dim]No cross-workspace dependency links.[/dim]")
                return
            t = Table(title="Cross-Workspace Dependencies",
                      box=box.SIMPLE, header_style="bold cyan")
            t.add_column("Source")
            t.add_column("Kind")
            t.add_column("Target")
            for lnk in links[:40]:
                t.add_row(lnk.source_fqn, lnk.kind, lnk.target_fqn)
            console.print(t)

        # ── /workspace resolve <path-expr> ───────────────────────────────
        elif sub == "resolve":
            if len(args) < 2:
                console.print(
                    "[red]Usage: /workspace resolve [name]:path/to/file[/red]"
                )
                return
            resolved = self.wm.resolve(" ".join(args[1:]))
            exists = "[green]exists[/]" if resolved.exists() else "[red]not found[/]"
            console.print(f"{resolved}  {exists}")

        # ── /workspace sync — cross-project pipeline demo ────────────────
        elif sub == "sync":
            if len(args) < 2:
                console.print(
                    "[red]Usage: /workspace sync <task> [→ <dep_task>...]\n"
                    "Example: /workspace sync server:compile client:types[/red]"
                )
                return
            self.wm.sync.clear()
            task_ids: list[str] = []
            for spec in args[1:]:
                ws_part, _, desc = spec.partition(":")
                ws_part = ws_part.strip()
                if not ws_part:
                    ws_part = self.wm.state.active_workspace_context
                tid = self.wm.sync.add_task(
                    ws_part,
                    desc.strip() or spec,
                    depends_on=[task_ids[-1]] if task_ids else [],
                )
                task_ids.append(tid)
            console.print(Panel(
                self.wm.sync.dag_summary(),
                title="⛓ Cross-Project Task DAG",
                border_style="yellow", box=box.ROUNDED,
            ))

            async def _dummy_executor(task: CrossTask) -> str:
                await asyncio.sleep(0.05)
                return f"[simulated] {task.description} complete"

            results = await self.wm.sync.run_all(
                executor=_dummy_executor,
                on_update=lambda t: console.print(
                    f"  [dim]{t.workspace}[/] [{t.state}] {t.description}"
                ),
            )
            done = sum(1 for t in results.values() if t.state == "done")
            failed = sum(1 for t in results.values() if t.state == "failed")
            color = "green" if not failed else "red"
            console.print(Panel(
                self.wm.sync.dag_summary(),
                title=f"⛓ Pipeline Done — {done} ok / {failed} failed",
                border_style=color, box=box.ROUNDED,
            ))

        # ── /workspace status ─────────────────────────────────────────────
        elif sub == "status":
            ctx = self.wm.context_block()
            st = self.wm.symbols.stats()
            body = (
                ctx or "[dim]No workspaces registered.[/dim]"
            ) + f"\n\n[dim]Symbol index: {st['total_symbols']} total · "
            f"{st['cross_workspace_links']} cross-ws deps[/dim]"
            console.print(Panel(body, title="📐 Workspace Status",
                                border_style="magenta", box=box.ROUNDED))

        else:
            console.print(
                "[red]Usage: /workspace <list|add|remove|switch|index|"
                "symbols|stale|deps|resolve|sync|status>[/red]"
            )

    # --- API key binding -------------------------------------------------- #
    def _rebuild_chain(self) -> None:
        self.router.chain = config.build_priority_chain(
            self.policy, only_available=True
        ) or [config.MODELS_BY_SLUG["codex-local"]]
        self.router.active_model = self.router.chain[0]

    async def cmd_api(self, args: list[str]) -> None:
        if not args or args[0] == "list":
            t = Table(title="API Providers (priority tier 4)", box=box.ROUNDED,
                      header_style="bold cyan")
            t.add_column("Provider")
            t.add_column("Key")
            t.add_column("Base URL")
            for name, prov in (
                ("openrouter", config.Provider.OPENROUTER),
                ("anthropic", config.Provider.ANTHROPIC),
                ("openai", config.Provider.OPENAI),
                ("gemini", config.Provider.GEMINI),
            ):
                ep = config.ENDPOINTS[prov]
                key = "[green]set[/]" if ep.is_configured else "[red]missing[/]"
                t.add_row(name, key, ep.base_url)
            console.print(t)
            console.print("[dim]Bind with: /api <provider> <key>[/dim]")
            return
        name = args[0].lower()
        prov = config.API_PROVIDER_ALIASES.get(name)
        if prov is None:
            console.print(f"[red]Unknown API provider '{name}'. "
                          "Use openrouter|anthropic|openai|gemini.[/red]")
            return
        if len(args) < 2:
            console.print(f"[red]Usage: /api {name} <key>  (or 'clear')[/red]")
            return
        if args[1] == "clear":
            config.set_runtime_key(prov, None)
            console.print(f"[green]Cleared {name} key.[/green]")
        else:
            key = args[1]
            config.set_runtime_key(prov, key)
            masked = key[:6] + "…" + key[-4:] if len(key) > 12 else "••••"
            console.print(f"[green]Bound {name} key ({masked}).[/green]")
        self._rebuild_chain()
        await self.cmd_chain([])

    # --- local executive activation -------------------------------------- #
    async def cmd_local(self, args: list[str]) -> None:
        locals_ = {
            "antigravity": config.Provider.ANTIGRAVITY,
            "cursor": config.Provider.CURSOR,
            "codex": config.Provider.CODEX,
        }
        if not args or args[0] == "list":
            t = Table(title="Local Executives (priority 1-3)", box=box.ROUNDED,
                      header_style="bold cyan")
            t.add_column("Agent")
            t.add_column("Priority")
            t.add_column("Active")
            for name, prov in locals_.items():
                rank = config.MODELS_BY_SLUG[
                    {"antigravity": "antigravity-agent",
                     "cursor": "cursor-agent",
                     "codex": "codex-local"}[name]].priority_rank
                state = ("[green]on[/]" if config.is_local_active(prov)
                         else "[yellow]defer[/]")
                t.add_row(name, str(rank), state)
            console.print(t)
            console.print("[dim]Codex auto-answers offline when no API key is "
                          "set. Toggle others with /local <name> on|off[/dim]")
            return
        name = args[0].lower()
        prov = locals_.get(name)
        if prov is None or len(args) < 2 or args[1] not in ("on", "off"):
            console.print("[red]Usage: /local <antigravity|cursor|codex> on|off[/red]")
            return
        config.set_local_active(prov, args[1] == "on")
        console.print(f"[green]{name} {'activated' if args[1]=='on' else 'deactivated'}.[/green]")
        self._rebuild_chain()

    # --- Module 1: live model override ----------------------------------- #
    async def cmd_model(self, args: list[str]) -> None:
        if not args:
            cur = self.router.override_model
            if cur:
                console.print(f"Active override: [bold]{cur.slug}[/] "
                              f"({cur.provider.value})")
            else:
                console.print("No override. Using the priority chain: "
                              f"[bold]{self.router.active_model.name}[/]")
            console.print("[dim]Usage: /model <provider/model:tier>  · "
                          "/model reset  · examples:\n"
                          "  /model anthropic/claude-4.6-opus\n"
                          "  /model nvidia/nemotron-3-ultra-550b-a55b:free\n"
                          "  /model gpt-5.5-pro[/dim]")
            return
        if args[0] in ("reset", "clear", "default"):
            self.router.clear_override()
            console.print("[green][SYSTEM] Model override cleared. "
                          "Reverting to the priority chain.[/green]")
            return
        slug = args[0]
        model = config.make_model_for_slug(slug)
        if not model.is_available:
            ep = model.endpoint
            console.print(
                f"[yellow][SYSTEM] '{slug}' routes via {model.provider.value} "
                f"but no key is set. Bind one: /api {model.provider.value} "
                f"<key>[/yellow]")
        self.router.set_override(model)
        console.print(
            f"[bold green][SYSTEM] Target engine successfully switched to: "
            f"{slug}[/bold green] [dim]({model.provider.value})[/dim]")

    # --- token-saving toggles -------------------------------------------- #
    async def cmd_tokensave(self, args: list[str]) -> None:
        b = config.TOKEN_BUDGET
        if args and args[0] in ("on", "off"):
            b.saver_mode = args[0] == "on"
        t = Table(title="Token Saver", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("Setting")
        t.add_column("Value", justify="right")
        t.add_row("Saver mode", "[green]on[/]" if b.saver_mode else "[red]off[/]")
        t.add_row("Output cap (saver)", str(b.saver_output_tokens))
        t.add_row("History window (msgs)", str(b.max_history_messages))
        t.add_row("Tool result cap (chars)", str(b.max_tool_chars))
        t.add_row("Plan-first", "[green]on[/]" if b.plan_first else "[red]off[/]")
        console.print(t)

    async def cmd_plan(self, args: list[str]) -> None:
        b = config.TOKEN_BUDGET
        if args and args[0] in ("on", "off"):
            b.plan_first = args[0] == "on"
        console.print(f"Plan-first: [bold]{'on' if b.plan_first else 'off'}[/]")

    # --- Engine 5: multi-agent swarm ------------------------------------- #
    async def cmd_swarm(self, args: list[str]) -> None:
        intent = " ".join(args)
        if not intent:
            console.print("[red]Usage: /swarm <task description>[/red]")
            return
        console.print(Panel(f"Dispatching swarm: {intent}",
                            title="🐝 swarm", border_style="yellow", box=box.ROUNDED))
        coord = SwarmCoordinator(
            self.bridges, self.memory, throttle=self.throttle,
            on_event=lambda ev: console.print(
                f"  [dim]{ev.source:>9}[/] ▸ [bold]{ev.topic}[/]"
                + (f" [dim]{ev.payload.get('detail','')}[/]"
                   if ev.payload.get('detail') else "")),
        )
        out = await coord.run(intent)
        color = "green" if out["ok"] else "red"
        body = (f"ok: {out['ok']}\npipeline: {out['pipeline_id']}\n"
                f"errors: {out['errors'] or 'none'}")
        console.print(Panel(body, title="🐝 swarm result",
                            border_style=color, box=box.ROUNDED))

    # --- Engine 6: long-term memory -------------------------------------- #
    async def cmd_memory(self, args: list[str]) -> None:
        sub = args[0] if args else "stats"
        if sub == "stats":
            s = self.memory.stats()
            t = Table(title="Semantic Memory (RAG)", box=box.ROUNDED,
                      header_style="bold cyan")
            t.add_column("Metric")
            t.add_column("Value")
            for k, v in s.items():
                t.add_row(k, str(v))
            console.print(t)
        elif sub == "search" and len(args) > 1:
            hits = self.memory.query(" ".join(args[1:]), top_k=5)
            if not hits:
                console.print("[dim]No matches.[/dim]")
                return
            for h in hits:
                console.print(f"[green]{h.score:.3f}[/] [{h.record.kind}] "
                              f"{h.record.text[:120]}")
        elif sub == "add" and len(args) > 1:
            rec = self.memory.remember(" ".join(args[1:]), kind="note",
                                       tags=["manual"])
            console.print(f"[green]Remembered {rec.id}.[/green]")
        elif sub == "forget" and len(args) > 1:
            ok = self.memory.forget(args[1])
            console.print("[green]Forgotten.[/green]" if ok else "[red]Not found.[/red]")
        elif sub == "ast":
            s = self.ast_memory.stats()
            t = Table(title="AST Incremental Memory", box=box.ROUNDED,
                      header_style="bold cyan")
            t.add_column("qualname")
            t.add_column("kind")
            t.add_column("v", justify="right")
            t.add_column("signature")
            for u in list(self.ast_memory.units.values())[:25]:
                t.add_row(u.qualname, u.kind, str(u.version), u.signature)
            console.print(t)
            console.print(f"[dim]{s['units']} units · {s['versioned']} versioned "
                          f"· {s['path']}[/dim]")
        else:
            console.print("[red]Usage: /memory stats|search <q>|add <text>|"
                          "forget <id>|ast[/red]")

    # --- Engine 7: telemetry --------------------------------------------- #
    async def cmd_telemetry(self, _: list[str]) -> None:
        s = self.telemetry.snapshot()
        t = Table(title="Host Telemetry", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("Resource")
        t.add_column("Value", justify="right")
        t.add_row("CPU", f"{s.cpu_percent:.1f}%")
        t.add_row("Memory", f"{s.mem_percent:.1f}% "
                            f"({s.mem_used_mb:.0f}/{s.mem_total_mb:.0f} MB)")
        t.add_row("GPU VRAM", s.vram_label)
        t.add_row("Load avg (1m)",
                  str(s.load_avg_1m) if s.load_avg_1m is not None else "n/a")
        t.add_row("Throttle events", str(self.throttle.throttle_events))
        console.print(t)

    # --- Engine 9: financial velocity guard ------------------------------ #
    async def cmd_budget(self, args: list[str]) -> None:
        guard = self.router.cost_guard
        if args and args[0] == "resume":
            guard.reset_freeze()
            console.print("[green]Freeze cleared. Execution may resume.[/green]")
            return
        if len(args) >= 2 and args[0] == "limit":
            try:
                guard.max_usd_per_min = float(args[1])
                console.print(f"[green]Limit set to ${guard.max_usd_per_min:.2f}/min.[/green]")
            except ValueError:
                console.print("[red]Usage: /budget limit <usd_per_min>[/red]")
            return
        frozen, msg = guard.check()
        t = Table(title="Financial Velocity Guard", box=box.ROUNDED,
                  header_style="bold cyan")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Spend velocity", f"${guard.velocity_per_min():.4f}/min")
        t.add_row("Limit", f"${guard.max_usd_per_min:.2f}/min")
        t.add_row("Session total", f"${guard.total_usd:.6f}")
        t.add_row("State", "[red]FROZEN[/]" if frozen or guard.frozen else "[green]ok[/]")
        console.print(t)
        console.print("[dim]/budget limit <usd> · /budget resume[/dim]")

    async def cmd_models(self, _: list[str]) -> None:
        t = Table(title="Model Catalogue", box=box.ROUNDED,
                  header_style="bold cyan")
        for col in ("Model", "Slug", "Provider", "In $/1M", "Out $/1M",
                    "Intel", "Tier", "Available"):
            t.add_column(col)
        for m in config.MODEL_CATALOGUE:
            avail = "[green]yes[/]" if m.is_available else "[red]no[/]"
            t.add_row(
                m.name, m.slug, m.provider.value,
                f"{m.input_cost_per_1m:g}", f"{m.output_cost_per_1m:g}",
                str(m.intelligence_tier), str(m.fallback_tier), avail,
            )
        console.print(t)

    async def cmd_chain(self, _: list[str]) -> None:
        chain = self.router.chain
        t = Table(title=f"Active Fallback Chain ({self.policy.value})",
                  box=box.ROUNDED, header_style="bold cyan")
        t.add_column("#")
        t.add_column("Model")
        t.add_column("Provider")
        t.add_column("Role")
        for i, m in enumerate(chain):
            role = "primary" if i == 0 else f"fallback {i}"
            marker = " [green]◀ active[/]" if m is self.router.active_model else ""
            t.add_row(str(i), m.name + marker, m.provider.value, role)
        console.print(t)

    async def cmd_policy(self, args: list[str]) -> None:
        if not args:
            console.print(f"Current policy: [bold]{self.policy.value}[/]")
            console.print("Options: local_first, max_intelligence, cost_optimization")
            return
        try:
            self.policy = config.RoutingPolicy(args[0])
        except ValueError:
            console.print(f"[red]Unknown policy '{args[0]}'.[/red]")
            return
        self.router.chain = config.build_priority_chain(
            self.policy, only_available=True
        ) or [config.MODELS_BY_SLUG["codex-local"]]
        console.print(f"[green]Policy set to {self.policy.value}.[/green] "
                      "Re-run /chain to see the new order.")

    async def cmd_providers(self, _: list[str]) -> None:
        t = Table(title="Providers (BYOK)", box=box.ROUNDED,
                  header_style="bold cyan")
        t.add_column("Provider")
        t.add_column("Base URL")
        t.add_column("Key")
        for prov, ep in config.ENDPOINTS.items():
            key = "[green]set[/]" if ep.is_configured else "[red]missing[/]"
            t.add_row(prov.value, ep.base_url, key)
        console.print(t)

    async def cmd_cost(self, _: list[str]) -> None:
        u = self.router.session_usage
        t = Table(title="Session Consumption", box=box.ROUNDED,
                  header_style="bold cyan")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Input tokens", f"{u.input_tokens:,}")
        t.add_row("Output tokens", f"{u.output_tokens:,}")
        t.add_row("Total tokens", f"{u.input_tokens + u.output_tokens:,}")
        t.add_row("Estimated spend", f"${self.router.session_cost_usd:.6f}")
        t.add_row("Last active model", self.router.active_model.name)
        console.print(t)

    async def cmd_tools(self, _: list[str]) -> None:
        t = Table(title="LLM Bridge Tools", box=box.ROUNDED,
                  header_style="bold cyan")
        t.add_column("Tool")
        t.add_column("Bridge")
        t.add_column("Description")
        for spec in ALL_TOOLS:
            t.add_row(spec.name, spec.bridge, spec.description)
        console.print(t)

    async def cmd_status(self, _: list[str]) -> None:
        blender_ok = await _probe_blender(self.bridges.blender)
        unity_ok = await _probe_unity(self.bridges.unity)
        t = Table(title="Oqlo Status", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("Subsystem")
        t.add_column("State")
        b = config.TOKEN_BUDGET
        api_on = "[green]yes[/]" if config.any_api_key_configured() else "[red]no[/]"
        override = self.router.override_model
        t.add_row("Routing policy", self.policy.value)
        t.add_row("Priority", "1.antigravity → 2.cursor → 3.codex → 4.api")
        t.add_row("Active model", self.router.active_model.name)
        t.add_row("Model override",
                  f"[bold]{override.slug}[/] ({override.provider.value})"
                  if override else "[dim]none (priority chain)[/]")
        t.add_row("Hybrid intent router",
                  "[green]on[/]" if self.auto_intent else "[red]off[/]")
        t.add_row("Chain length", str(len(self.router.chain)))
        t.add_row("API key configured", api_on)
        t.add_row("Mode", "terminal-only (never launches apps)")
        t.add_row("Token saver", "[green]on[/]" if b.saver_mode else "[red]off[/]")
        t.add_row("Plan-first", "[green]on[/]" if b.plan_first else "[red]off[/]")
        snap = self.telemetry.snapshot()
        t.add_row("Host CPU / MEM",
                  f"{snap.cpu_percent:.0f}% / {snap.mem_percent:.0f}%"
                  + (f" · VRAM {snap.vram_label}" if snap.vram_percent else ""))
        t.add_row("Memory cache",
                  f"{self.memory.stats()['records']} docs · "
                  f"{self.ast_memory.stats()['units']} AST units")
        t.add_row("Session spend",
                  f"${self.router.session_cost_usd:.6f} "
                  f"({self.router.session_usage.input_tokens + self.router.session_usage.output_tokens:,} tok)")
        t.add_row("Budget guard",
                  f"${self.router.cost_guard.max_usd_per_min:.2f}/min limit"
                  + (" [red](FROZEN)[/]" if self.router.cost_guard.frozen else ""))
        t.add_row("Blender bridge",
                  "[green]live[/]" if blender_ok else "[yellow]offline (simulated)[/]")
        t.add_row("Unity bridge",
                  "[green]live[/]" if unity_ok else "[yellow]offline (simulated)[/]")
        t.add_row("Loaded skills", str(len(self.skills.skills)))
        t.add_row("Messages in buffer", str(len(self.conv.messages)))
        console.print(t)

    async def cmd_history(self, _: list[str]) -> None:
        if not self.conv.messages:
            console.print("[dim]Conversation is empty.[/dim]")
            return
        for m in self.conv.messages:
            if m.role == "tool":
                console.print(f"[dim]tool[{m.tool_name}]:[/] {m.text[:200]}")
            elif m.role == "assistant":
                tc = f" (+{len(m.tool_calls)} tool calls)" if m.tool_calls else ""
                console.print(f"[cyan]assistant:[/] {m.text[:200]}{tc}")
            else:
                console.print(f"[white]{m.role}:[/] {m.text[:200]}")

    async def cmd_save(self, args: list[str]) -> None:
        import json
        path = args[0] if args else "oqlo_session.json"
        data = {
            "system": self.conv.system_prompt,
            "messages": [
                {
                    "role": m.role,
                    "text": m.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in m.tool_calls
                    ],
                    "tool_call_id": m.tool_call_id,
                    "tool_name": m.tool_name,
                }
                for m in self.conv.messages
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        console.print(f"[green]Saved {len(self.conv.messages)} messages to {path}.[/green]")

    async def cmd_load(self, args: list[str]) -> None:
        import json
        from llm_router import Message, ToolCall
        if not args:
            console.print("[red]Usage: /load <file>[/red]")
            return
        try:
            with open(args[0], encoding="utf-8") as fh:
                data = json.load(fh)
        except OSError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        self.conv = Conversation(data.get("system", config.SYSTEM_PROMPT))
        for m in data.get("messages", []):
            self.conv.messages.append(
                Message(
                    role=m["role"],
                    text=m.get("text", ""),
                    tool_calls=[
                        ToolCall(tc["id"], tc["name"], tc["arguments"])
                        for tc in m.get("tool_calls", [])
                    ],
                    tool_call_id=m.get("tool_call_id"),
                    tool_name=m.get("tool_name"),
                )
            )
        console.print(f"[green]Loaded {len(self.conv.messages)} messages.[/green]")

    async def cmd_clear(self, _: list[str]) -> None:
        # Clear the screen and flush the transient short-term chat buffer, while
        # preserving the long-term semantic + AST memory matrices.
        console.clear()
        self.conv = Conversation(config.SYSTEM_PROMPT)
        console.print("[green]Screen cleared and short-term context flushed. "
                      "Long-term memory preserved "
                      f"({self.memory.stats()['records']} docs, "
                      f"{self.ast_memory.stats()['units']} AST units).[/green]")

    async def cmd_export(self, _: list[str]) -> None:
        d = config.BRIDGES.fbx_export_dir
        if not d.is_dir():
            console.print(f"[dim]No exports yet ({d}).[/dim]")
            return
        files = sorted(d.glob("*"))
        if not files:
            console.print(f"[dim]No exports yet ({d}).[/dim]")
            return
        t = Table(title=f"Exports — {d}", box=box.SIMPLE)
        t.add_column("File")
        t.add_column("Size", justify="right")
        for f in files:
            t.add_row(f.name, f"{f.stat().st_size:,} B")
        console.print(t)

    async def cmd_blender(self, args: list[str]) -> None:
        code = " ".join(args)
        if not code:
            console.print("[red]Usage: /blender <bpy code>[/red]")
            return
        result = await self.bridges.blender.execute_bpy_command(code)
        self._print_tool_result(result)

    async def cmd_unity(self, args: list[str]) -> None:
        if args and args[0] == "logs":
            sev = args[1] if len(args) > 1 else "error"
            result = await self.bridges.unity.get_editor_logs(severity=sev)
            self._print_tool_result(result)
        else:
            console.print("[red]Usage: /unity logs [all|warning|error][/red]")

    async def cmd_cursor(self, args: list[str]) -> None:
        if len(args) >= 2 and args[0] == "open":
            result = await self.bridges.cursor.set_antigravity_model_preset(
                preset=self.bridges.cursor.preset, open_workspace=args[1]
            )
            self._print_tool_result(result)
        else:
            console.print("[red]Usage: /cursor open <path>[/red]")

    async def cmd_skill(self, args: list[str]) -> None:
        if not args:
            args = ["list"]
        sub, rest = args[0], args[1:]
        if sub == "list":
            self._render_skills(list(self.skills.skills.values()))
        elif sub == "search" and rest:
            self._render_skills(self.skills.search(rest[0]))
        elif sub == "reload":
            n = self.skills.reload()
            console.print(f"[green]Reloaded {n} skills.[/green]")
        elif sub == "examples":
            n = seed_example_skills()
            self.skills.reload()
            console.print(f"[green]Seeded {n} example skills.[/green]")
        elif sub == "remove" and rest:
            ok = self.skills.remove(rest[0])
            console.print("[green]Removed.[/green]" if ok else "[red]Not found.[/red]")
        elif sub == "run" and rest:
            await self._run_skill(rest)
        elif sub == "add":
            await self._add_skill_interactive()
        else:
            console.print(
                "[red]Usage: /skill list|search <t>|run <name> k=v...|"
                "add|remove <name>|reload|examples[/red]"
            )

    async def _run_skill(self, rest: list[str]) -> None:
        name = rest[0]
        kv: dict[str, str] = {}
        for token in rest[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                kv[k] = v
        console.print(f"[dim]running skill '{name}'...[/dim]")
        res = await self.skills.run(name, kv)
        console.print(f"[dim]$ {res.command}[/dim]")
        if res.stdout:
            console.print(res.stdout.rstrip())
        if res.stderr:
            console.print(f"[red]{res.stderr.rstrip()}[/red]")
        tag = "[green]ok[/]" if res.ok else f"[red]exit {res.exit_code}[/]"
        console.print(f"-> {tag}")

    async def _add_skill_interactive(self) -> None:
        console.print("[cyan]Define a new skill (blank name to cancel).[/cyan]")
        name = (await _ainput("  name: ")).strip()
        if not name:
            return
        desc = (await _ainput("  description: ")).strip()
        command = (await _ainput("  command (use {param} placeholders): ")).strip()
        params_raw = (await _ainput("  params (comma-separated names): ")).strip()
        params = {p.strip(): "" for p in params_raw.split(",") if p.strip()}
        path = self.skills.add(
            Skill(name=name, description=desc, command=command, params=params)
        )
        console.print(f"[green]Saved skill to {path}.[/green]")

    def _render_skills(self, skills: list[Skill]) -> None:
        if not skills:
            console.print("[dim]No skills. Try '/skill examples'.[/dim]")
            return
        t = Table(title="Skills", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("Name")
        t.add_column("Params")
        t.add_column("Description")
        for s in skills:
            t.add_row(s.name, ", ".join(s.params) or "-", s.description)
        console.print(t)

    async def cmd_addon(self, args: list[str]) -> None:
        which = args[0] if args else ""
        if which == "blender":
            console.print(Syntax(BLENDER_ADDON_TEMPLATE, "python",
                                 theme="monokai", line_numbers=True))
        elif which == "unity":
            console.print(Syntax(UNITY_EDITOR_SCRIPT_TEMPLATE, "csharp",
                                 theme="monokai", line_numbers=True))
        else:
            console.print("[red]Usage: /addon <blender|unity>[/red]")

    async def cmd_config(self, _: list[str]) -> None:
        t = Table(title="Resolved Config", box=box.ROUNDED,
                  header_style="bold cyan")
        t.add_column("Key")
        t.add_column("Value")
        t.add_row("Routing policy", self.policy.value)
        t.add_row("Blender WS", config.BRIDGES.blender_ws_url)
        t.add_row("Unity REST", config.BRIDGES.unity_rest_url)
        t.add_row("Cursor workspace", str(config.BRIDGES.cursor_workspace or "(cwd)"))
        t.add_row("FBX export dir", str(config.BRIDGES.fbx_export_dir))
        console.print(t)

    async def cmd_verbose(self, args: list[str]) -> None:
        self.verbose = (args and args[0] == "on") or (not args and not self.verbose)
        console.print(f"Verbose: [bold]{'on' if self.verbose else 'off'}[/]")

    async def cmd_quit(self, _: list[str]) -> None:
        self.running = False

    # ------------------------------------------------------------------ #
    def _print_tool_result(self, result: Any) -> None:
        tag = "[yellow]SIM[/]" if result.simulated else (
            "[green]OK[/]" if result.ok else "[red]ERR[/]")
        console.print(f"{tag} {result.summary}")
        if result.error:
            console.print(f"[red]{result.error}[/red]")
        for k, v in result.data.items():
            console.print(f"  [dim]{k}:[/] {v}")

    # ------------------------------------------------------------------ #
    async def _chat(self, line: str) -> None:
        """INTENT_CHAT fast path — direct, tool-free conversational turn."""
        self.conv.user(line)
        try:
            resp = await self.router.chat(self.conv)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive.
            # Remove the orphaned user message so the next turn is clean.
            if self.conv.messages and self.conv.messages[-1].role == "user":
                self.conv.messages.pop()
            override = self.router.override_model
            if override:
                console.print(
                    f"[red]✗ '{override.slug}' failed — all fallbacks exhausted.[/red]\n"
                    f"[yellow]→ Bind a key:  /api {override.provider.value} <key>\n"
                    "→ Or reset:    /model reset[/yellow]"
                )
            else:
                console.print(f"[red]All providers exhausted: {exc}[/red]")
            return
        self.conv.assistant(resp.text, [])
        if not resp.text.strip():
            console.print(Panel(
                "[dim]The model returned an empty response.\n"
                "Free-tier models sometimes do this under load — try again, "
                "or use /model reset to return to the priority chain.[/dim]",
                title=f"💬 {resp.model.name} [yellow](empty)[/yellow]",
                border_style="yellow", box=box.ROUNDED,
            ))
        else:
            console.print(Panel(resp.text,
                                title=f"💬 {resp.model.name}",
                                border_style="cyan", box=box.ROUNDED))

    async def _agentic(self, line: str) -> None:
        orch = Orchestrator(
            self.router, self.bridges, self.conv,
            on_step=self._on_step, memory=self.memory,
            ast_memory=self.ast_memory,
        )
        try:
            await orch.run(line)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive.
            console.print(f"[red]Orchestration error: {exc}[/red]")

    async def dispatch(self, line: str) -> None:
        # Module 1: slash commands are intercepted before any LLM node.
        if line.startswith("/"):
            parts = line[1:].split()
            if not parts:
                return
            cmd, args = parts[0].lower(), parts[1:]
            handler = self.commands.get(cmd)
            if handler is None:
                console.print(f"[red]Unknown command '/{cmd}'. Try /help.[/red]")
                return
            await handler(args)
            return

        # When a file session is active, auto-inject file content as context.
        if self.active_file:
            await self._dispatch_with_file(line)
            return

        # Module 1: hybrid intent routing for plain text.
        if not self.auto_intent:
            await self._agentic(line)
            return
        intent = classify_intent(line)
        if intent is Intent.CHAT:
            console.print("[dim]↳ intent: chat (direct model, no pipeline)[/dim]")
            await self._chat(line)
        else:
            console.print("[dim]↳ intent: agentic (multi-agent pipeline)[/dim]")
            await self._agentic(line)

    async def repl(self) -> None:
        await self.cmd_start([])
        console.print(
            "\n[dim]Plain text runs the orchestrator. "
            "Type /start oqlocode anytime. Ctrl-C or /quit to exit.[/dim]"
        )
        while self.running:
            try:
                # Show active path in the prompt when a session is open.
                if self.active_file:
                    ctx = f":{self.active_file.name}"
                elif self.active_dir:
                    ctx = f":{self.active_dir.name}/"
                else:
                    ctx = ""
                line = (await _ainput(f"\n[oqlo{ctx}] › ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            await self.dispatch(line)
        await self.router.aclose()
        await self.bridges.aclose()
        console.print("\n[magenta]Oqlo Code shutting down. Goodbye.[/magenta]")


# ---------------------------------------------------------------------- #
# Async helpers
# ---------------------------------------------------------------------- #
async def _ainput(prompt: str) -> str:
    """Non-blocking input() via a worker thread."""
    return await asyncio.to_thread(input, prompt)


async def _probe_blender(bridge: BlenderBridge) -> bool:
    reply = await bridge._send("ping", {})  # noqa: SLF001 - intentional probe.
    return reply is not None


async def _probe_unity(bridge: UnityBridge) -> bool:
    reply = await bridge._post("/ping", {})  # noqa: SLF001 - intentional probe.
    return reply is not None


def main() -> None:
    cli = OqloCLI()
    try:
        asyncio.run(cli.repl())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
