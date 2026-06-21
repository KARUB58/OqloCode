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
        self.verbose = True
        self.running = True
        self._last_events: list[str] = []
        self.commands: dict[str, Callable[[list[str]], Awaitable[None]]] = {
            "start": self.cmd_start,
            "help": self.cmd_help,
            "api": self.cmd_api,
            "local": self.cmd_local,
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
            console.print(
                f"[dim]→ trying [bold]{fields['model']}[/] "
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

    # --- start / banner --------------------------------------------------- #
    async def cmd_start(self, _: list[str]) -> None:
        console.print(Text(BANNER, style="bold magenta"))
        console.print(f"[magenta]{SUBTITLE}[/magenta]\n")
        await self.cmd_status([])
        console.print("\n[dim]Type /help for all commands.[/dim]")

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
        t.add_row("Routing policy", self.policy.value)
        t.add_row("Priority", "1.antigravity → 2.cursor → 3.codex → 4.api")
        t.add_row("Active model", self.router.active_model.name)
        t.add_row("Chain length", str(len(self.router.chain)))
        t.add_row("API key configured", api_on)
        t.add_row("Mode", "terminal-only (never launches apps)")
        t.add_row("Token saver", "[green]on[/]" if b.saver_mode else "[red]off[/]")
        t.add_row("Plan-first", "[green]on[/]" if b.plan_first else "[red]off[/]")
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
        self.conv = Conversation(config.SYSTEM_PROMPT)
        console.print("[green]Conversation reset.[/green]")

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
    async def dispatch(self, line: str) -> None:
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
        else:
            orch = Orchestrator(
                self.router, self.bridges, self.conv, on_step=self._on_step
            )
            try:
                await orch.run(line)
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive.
                console.print(f"[red]Orchestration error: {exc}[/red]")

    async def repl(self) -> None:
        await self.cmd_start([])
        console.print(
            "\n[dim]Plain text runs the orchestrator. "
            "Type /start oqlocode anytime. Ctrl-C or /quit to exit.[/dim]"
        )
        while self.running:
            try:
                line = (await _ainput("\n[oqlo] › ")).strip()
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
