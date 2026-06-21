"""Oqlo Code — Multi-Project Workspace Module.

Three interlocking engines:

1. **Virtual Federated File System (VFFS)**
   Registers a named dictionary of workspace roots and resolves paths across
   them using ``[name]:relative/path`` notation.  Replaces the single-project-
   root restriction so the orchestrator can address files in any registered
   workspace transparently.

2. **Cross-Boundary Symbol Resolution**
   Per-workspace AST indexing feeds a ``GlobalSymbolRegistry``.  Every indexed
   symbol carries a structural hash; when a hash changes the registry performs a
   BFS over the dependency graph and flags all downstream cross-workspace
   consumers as stale — e.g. a backend schema change in the ``server`` workspace
   automatically flags dependent files in the ``client`` workspace.

3. **Inter-Project State Synchronizer**
   Extends the async DAG scheduler with cross-workspace task locks.  Tasks
   carry a ``workspace`` tag and a ``depends_on`` list of foreign task-IDs;
   the executor awaits the upstream Event before running the downstream task.
"""

from __future__ import annotations

import ast as _ast
import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKSPACE_STATE_PATH = Path.home() / ".oqlo" / "workspaces.json"
SYMBOL_REGISTRY_PATH = Path.home() / ".oqlo" / "symbol_registry.json"

ENVIRONMENT_TYPES = frozenset({"Blender", "Unity", "Cursor", "Generic"})


# --------------------------------------------------------------------------- #
# Data models (stdlib dataclasses — no extra dependency)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class WorkspaceProfile:
    """A single registered workspace root."""

    name: str
    root_path: str
    environment_type: str = "Generic"   # Blender | Unity | Cursor | Generic
    ast_enabled: bool = True
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return Path(self.root_path).expanduser().resolve()

    def resolve(self, relative: str) -> Path:
        return self.root / relative.lstrip("/\\")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_path": self.root_path,
            "environment_type": self.environment_type,
            "ast_enabled": self.ast_enabled,
            "created_at": self.created_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkspaceProfile":
        return cls(
            name=d["name"],
            root_path=d["root_path"],
            environment_type=d.get("environment_type", "Generic"),
            ast_enabled=d.get("ast_enabled", True),
            created_at=d.get("created_at", time.time()),
            tags=d.get("tags", []),
        )


@dataclass(slots=True)
class GlobalState:
    """Persisted state: workspace registry + active context pointer."""

    workspaces: dict[str, WorkspaceProfile] = field(default_factory=dict)
    active_workspace_context: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspaces": {k: v.to_dict() for k, v in self.workspaces.items()},
            "active_workspace_context": self.active_workspace_context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GlobalState":
        return cls(
            workspaces={
                k: WorkspaceProfile.from_dict(v)
                for k, v in d.get("workspaces", {}).items()
            },
            active_workspace_context=d.get("active_workspace_context", "default"),
        )


# --------------------------------------------------------------------------- #
# Engine 1 — Virtual Federated File System
# --------------------------------------------------------------------------- #
class VirtualFederatedFileSystem:
    """Federated view over multiple named workspace roots.

    Path expression syntax
    ----------------------
    ``[name]:relative/path``  → workspace root / relative path
    ``/absolute/path``        → raw filesystem path (cross-workspace reference)
    ``relative/path``         → resolved under the active workspace
    """

    def __init__(self, state: GlobalState) -> None:
        self.state = state

    # --- registry --------------------------------------------------------- #
    def register(self, profile: WorkspaceProfile) -> None:
        self.state.workspaces[profile.name] = profile

    def remove(self, name: str) -> bool:
        return bool(self.state.workspaces.pop(name, None))

    def get(self, name: str) -> WorkspaceProfile | None:
        return self.state.workspaces.get(name)

    def all(self) -> list[WorkspaceProfile]:
        return list(self.state.workspaces.values())

    # --- resolution ------------------------------------------------------- #
    def resolve(self, path_expr: str) -> Path:
        """Resolve a VFFS expression to an absolute ``Path``."""
        # Strip optional [] around workspace name and handle [ws]:rel form.
        clean = path_expr.strip()
        if clean.startswith("["):
            bracket_end = clean.find("]:")
            if bracket_end != -1:
                ws_name = clean[1:bracket_end]
                rel = clean[bracket_end + 2:]
                ws = self.get(ws_name)
                if ws:
                    return ws.root / rel.lstrip("/\\")
        # bare ws:rel form (no brackets), but guard against Windows drive C:
        if ":" in clean and len(clean.split(":")[0]) > 1:
            ws_name, rel = clean.split(":", 1)
            ws = self.get(ws_name)
            if ws:
                return ws.root / rel.lstrip("/\\")
        p = Path(clean)
        if p.is_absolute():
            return p
        active = self.active_workspace()
        if active:
            return active.root / clean
        return Path.cwd() / clean

    def active_workspace(self) -> WorkspaceProfile | None:
        return self.get(self.state.active_workspace_context)

    def switch(self, name: str) -> bool:
        if name in self.state.workspaces:
            self.state.active_workspace_context = name
            return True
        return False

    def list_workspace_files(
        self,
        name: str,
        glob: str = "**/*",
        max_entries: int = 200,
    ) -> list[Path]:
        ws = self.get(name)
        if not ws or not ws.root.is_dir():
            return []
        return [
            p for p in list(ws.root.glob(glob))[:max_entries]
            if not any(part.startswith(".") for part in p.parts)
        ]


# --------------------------------------------------------------------------- #
# Engine 2 — Global Symbol Registry + Cross-Boundary Dependency Tracking
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SymbolRef:
    """An indexed symbol from a specific workspace file."""

    workspace: str
    file_path: str      # relative to workspace root
    qualname: str       # e.g. "ClassName.method"
    kind: str           # "function" | "class" | "method"
    signature: str = ""
    struct_hash: str = ""
    updated_at: float = field(default_factory=time.time)

    @property
    def fqn(self) -> str:
        """Fully-qualified name: ``[workspace]::file::qualname``"""
        return f"[{self.workspace}]::{self.file_path}::{self.qualname}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "file_path": self.file_path,
            "qualname": self.qualname,
            "kind": self.kind,
            "signature": self.signature,
            "struct_hash": self.struct_hash,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SymbolRef":
        return cls(
            workspace=d["workspace"],
            file_path=d["file_path"],
            qualname=d["qualname"],
            kind=d.get("kind", "function"),
            signature=d.get("signature", ""),
            struct_hash=d.get("struct_hash", ""),
            updated_at=d.get("updated_at", time.time()),
        )


@dataclass(slots=True)
class DependencyLink:
    """A directed dependency edge: source_fqn depends on target_fqn."""

    source_fqn: str
    target_fqn: str
    kind: str = "import"     # import | call | inherit
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fqn": self.source_fqn,
            "target_fqn": self.target_fqn,
            "kind": self.kind,
            "detected_at": self.detected_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DependencyLink":
        return cls(
            source_fqn=d["source_fqn"],
            target_fqn=d["target_fqn"],
            kind=d.get("kind", "import"),
            detected_at=d.get("detected_at", time.time()),
        )


class GlobalSymbolRegistry:
    """Cross-workspace symbol index with structural-hash dependency propagation.

    Workflow
    --------
    1. ``index_workspace`` walks Python files and calls ``upsert`` for each
       extracted symbol.
    2. If a symbol's ``struct_hash`` changed, ``_propagate_stale`` does a BFS
       over ``deps`` and marks every downstream consumer as stale.
    3. ``stale_symbols()`` surfaces the flagged set so the orchestrator can
       notify the user or re-run affected tasks.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or SYMBOL_REGISTRY_PATH
        self.symbols: dict[str, SymbolRef] = {}
        self.deps: list[DependencyLink] = []
        self._stale: set[str] = set()
        self.load()

    # --- persistence ------------------------------------------------------ #
    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.symbols = {
                fqn: SymbolRef.from_dict(d)
                for fqn, d in raw.get("symbols", {}).items()
            }
            self.deps = [DependencyLink.from_dict(d) for d in raw.get("deps", [])]
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "symbols": {fqn: s.to_dict() for fqn, s in self.symbols.items()},
                    "deps": [d.to_dict() for d in self.deps],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- symbol management ------------------------------------------------ #
    def upsert(self, ref: SymbolRef) -> list[str]:
        """Insert or update a symbol. Returns list of newly-staled FQNs."""
        old = self.symbols.get(ref.fqn)
        staled: list[str] = []
        if old and old.struct_hash and old.struct_hash != ref.struct_hash:
            staled = self._propagate_stale(ref.fqn)
        self.symbols[ref.fqn] = ref
        return staled

    def _propagate_stale(self, changed_fqn: str) -> list[str]:
        """BFS: flag all symbols that (transitively) depend on changed_fqn."""
        staled: list[str] = []
        frontier = [changed_fqn]
        visited: set[str] = set()
        while frontier:
            fqn = frontier.pop(0)
            if fqn in visited:
                continue
            visited.add(fqn)
            for dep in self.deps:
                if dep.target_fqn == fqn and dep.source_fqn not in visited:
                    if dep.source_fqn not in self._stale:
                        self._stale.add(dep.source_fqn)
                        staled.append(dep.source_fqn)
                    frontier.append(dep.source_fqn)
        return staled

    def add_dependency(
        self, source_fqn: str, target_fqn: str, kind: str = "import"
    ) -> None:
        if not any(
            d.source_fqn == source_fqn and d.target_fqn == target_fqn
            for d in self.deps
        ):
            self.deps.append(DependencyLink(source_fqn, target_fqn, kind))
            self.save()

    def stale_symbols(self) -> list[SymbolRef]:
        return [self.symbols[fqn] for fqn in self._stale if fqn in self.symbols]

    def clear_stale(self) -> None:
        self._stale.clear()

    def search(
        self, term: str, workspace: str | None = None
    ) -> list[SymbolRef]:
        term = term.lower()
        return [
            s for s in self.symbols.values()
            if (not workspace or s.workspace == workspace)
            and (
                term in s.qualname.lower()
                or term in s.file_path.lower()
                or term in s.signature.lower()
            )
        ]

    def cross_workspace_deps(self) -> list[DependencyLink]:
        """Return only dependency links that cross workspace boundaries."""
        return [
            d for d in self.deps
            if (
                d.source_fqn.split("::")[0] != d.target_fqn.split("::")[0]
            )
        ]

    def stats(self) -> dict[str, Any]:
        by_ws: dict[str, int] = {}
        for s in self.symbols.values():
            by_ws[s.workspace] = by_ws.get(s.workspace, 0) + 1
        return {
            "total_symbols": len(self.symbols),
            "by_workspace": by_ws,
            "dependency_links": len(self.deps),
            "cross_workspace_links": len(self.cross_workspace_deps()),
            "stale_flags": len(self._stale),
        }


def index_workspace_ast(
    profile: WorkspaceProfile,
    registry: GlobalSymbolRegistry,
) -> dict[str, Any]:
    """Walk a workspace root, extract AST symbols, and upsert into the registry.

    Also detects simple import-based cross-workspace references by scanning
    ``import`` statements and matching them against existing registry FQNs.
    """
    if not profile.ast_enabled:
        return {"skipped": "ast_enabled=False"}

    root = profile.root
    if not root.is_dir():
        return {"error": f"Root not found: {root}"}

    added = updated = stale_total = 0
    py_files = list(root.rglob("*.py"))

    for py_file in py_files:
        try:
            code = py_file.read_text(encoding="utf-8", errors="replace")
            tree = _ast.parse(code)
        except (OSError, SyntaxError):
            continue

        rel = str(py_file.relative_to(root))
        # Extract top-level defs and class methods.
        for node in tree.body:
            _index_node(node, profile.name, rel, registry,
                        added_counter=[added], updated_counter=[updated],
                        stale_counter=[stale_total])
            if isinstance(node, _ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        _index_node(
                            sub, profile.name, rel, registry,
                            qualname_prefix=node.name + ".",
                            added_counter=[added],
                            updated_counter=[updated],
                            stale_counter=[stale_total],
                        )

        # Detect cross-workspace import hints (best-effort).
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, _ast.Import)
                    else ([node.module or ""] if node.module else [])
                )
                for imported_name in names:
                    for fqn, sym in registry.symbols.items():
                        if (sym.workspace != profile.name
                                and imported_name.endswith(sym.qualname)):
                            source_fqn = f"[{profile.name}]::{rel}::{imported_name}"
                            registry.add_dependency(source_fqn, fqn, kind="import")

    registry.save()
    return {
        "workspace": profile.name,
        "py_files": len(py_files),
        "symbols_added": added,
        "symbols_updated": updated,
        "stale_flags": stale_total,
    }


def _index_node(
    node: _ast.AST,
    ws_name: str,
    rel_path: str,
    registry: GlobalSymbolRegistry,
    *,
    qualname_prefix: str = "",
    added_counter: list[int],
    updated_counter: list[int],
    stale_counter: list[int],
) -> None:
    if not isinstance(
        node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)
    ):
        return
    qualname = qualname_prefix + node.name
    kind = "class" if isinstance(node, _ast.ClassDef) else "function"
    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args]
        sig = f"def {qualname}({', '.join(args)})"
    else:
        bases = [getattr(b, "id", "…") for b in node.bases]
        sig = f"class {qualname}({', '.join(bases)})"
    dump = _ast.dump(node, annotate_fields=False, include_attributes=False)
    shash = hashlib.sha256(dump.encode()).hexdigest()[:16]
    existing = registry.symbols.get(f"[{ws_name}]::{rel_path}::{qualname}")
    ref = SymbolRef(
        workspace=ws_name,
        file_path=rel_path,
        qualname=qualname,
        kind=kind,
        signature=sig,
        struct_hash=shash,
        updated_at=time.time(),
    )
    staled = registry.upsert(ref)
    if staled:
        stale_counter[0] += len(staled)
    if existing:
        updated_counter[0] += 1
    else:
        added_counter[0] += 1


# --------------------------------------------------------------------------- #
# Engine 3 — Inter-Project State Synchronizer
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CrossTask:
    """A unit of work bound to a specific workspace in a cross-project DAG."""

    task_id: str
    workspace: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    state: str = "pending"          # pending | running | done | failed
    result: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class InterProjectSynchronizer:
    """Async DAG executor with cross-workspace pipeline locks.

    Example
    -------
    ::

        sync = InterProjectSynchronizer()
        t1 = sync.add_task("server", "compile API schema")
        t2 = sync.add_task("client", "regenerate TS types", depends_on=[t1])
        t3 = sync.add_task("client", "run unit tests",    depends_on=[t2])
        results = await sync.run_all(executor=my_fn)

    Task B in ``[client]`` waits until Task A in ``[server]`` has set its
    ``asyncio.Event`` — no shared state mutation occurs before the lock clears.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, CrossTask] = {}
        self._events: dict[str, asyncio.Event] = {}

    def add_task(
        self,
        workspace: str,
        description: str,
        depends_on: list[str] | None = None,
    ) -> str:
        tid = f"{workspace}:{uuid.uuid4().hex[:6]}"
        self._tasks[tid] = CrossTask(
            task_id=tid,
            workspace=workspace,
            description=description,
            depends_on=depends_on or [],
        )
        self._events[tid] = asyncio.Event()
        return tid

    def clear(self) -> None:
        self._tasks.clear()
        self._events.clear()

    def _topo_order(self) -> list[CrossTask]:
        """Kahn's algorithm topological sort; raises on cycle."""
        in_deg: dict[str, int] = {tid: 0 for tid in self._tasks}
        for t in self._tasks.values():
            for dep in t.depends_on:
                in_deg[t.task_id] = in_deg.get(t.task_id, 0) + 1
        queue = [tid for tid, d in in_deg.items() if d == 0]
        order: list[CrossTask] = []
        while queue:
            tid = queue.pop(0)
            order.append(self._tasks[tid])
            for t in self._tasks.values():
                if tid in t.depends_on:
                    in_deg[t.task_id] -= 1
                    if in_deg[t.task_id] == 0:
                        queue.append(t.task_id)
        if len(order) != len(self._tasks):
            raise ValueError("Dependency cycle detected in cross-project task graph.")
        return order

    async def run_all(
        self,
        executor: Callable[[CrossTask], Awaitable[str]],
        on_update: Callable[[CrossTask], None] | None = None,
    ) -> dict[str, CrossTask]:
        order = self._topo_order()

        async def _run_one(task: CrossTask) -> None:
            for dep_id in task.depends_on:
                evt = self._events.get(dep_id)
                if evt:
                    await evt.wait()
                dep = self._tasks.get(dep_id)
                if dep and dep.state == "failed":
                    task.state = "failed"
                    task.result = f"Blocked: upstream task '{dep_id}' failed."
                    task.finished_at = time.time()
                    self._events[task.task_id].set()
                    if on_update:
                        on_update(task)
                    return
            task.state = "running"
            if on_update:
                on_update(task)
            try:
                task.result = await executor(task)
                task.state = "done"
            except Exception as exc:  # noqa: BLE001
                task.result = str(exc)
                task.state = "failed"
            task.finished_at = time.time()
            self._events[task.task_id].set()
            if on_update:
                on_update(task)

        await asyncio.gather(*(_run_one(t) for t in order))
        return dict(self._tasks)

    def dag_summary(self) -> str:
        if not self._tasks:
            return "  (no tasks)"
        try:
            order = self._topo_order()
        except ValueError:
            order = list(self._tasks.values())
        icons = {"pending": "⏳", "running": "▶", "done": "✓", "failed": "✗"}
        lines = []
        for t in order:
            deps = (" ← " + ", ".join(t.depends_on)) if t.depends_on else ""
            icon = icons.get(t.state, "?")
            elapsed = (
                f" ({t.finished_at - t.created_at:.1f}s)"
                if t.finished_at else ""
            )
            lines.append(
                f"  {icon} [{t.workspace}] {t.task_id}: "
                f"{t.description}{deps}{elapsed}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Top-level WorkspaceManager — wires all three engines together
# --------------------------------------------------------------------------- #
class WorkspaceManager:
    """Single entry-point for the multi-project workspace module.

    Owns the :class:`GlobalState`, :class:`VirtualFederatedFileSystem`,
    :class:`GlobalSymbolRegistry`, and :class:`InterProjectSynchronizer`.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = state_path or WORKSPACE_STATE_PATH
        self.state = self._load_state()
        self.vffs = VirtualFederatedFileSystem(self.state)
        self.symbols = GlobalSymbolRegistry()
        self.sync = InterProjectSynchronizer()

    def _load_state(self) -> GlobalState:
        if self.state_path.is_file():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                return GlobalState.from_dict(raw)
            except (OSError, json.JSONDecodeError):
                pass
        return GlobalState()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2), encoding="utf-8"
        )

    # --- workspace CRUD --------------------------------------------------- #
    def add_workspace(
        self,
        name: str,
        root_path: str,
        environment_type: str = "Generic",
        ast_enabled: bool = True,
        tags: list[str] | None = None,
    ) -> WorkspaceProfile:
        env = environment_type.capitalize()
        if env not in ENVIRONMENT_TYPES:
            env = "Generic"
        profile = WorkspaceProfile(
            name=name,
            root_path=str(Path(root_path).expanduser().resolve()),
            environment_type=env,
            ast_enabled=ast_enabled,
            tags=tags or [],
        )
        self.vffs.register(profile)
        if len(self.state.workspaces) == 1:
            self.state.active_workspace_context = name
        self.save()
        return profile

    def remove_workspace(self, name: str) -> bool:
        removed = self.vffs.remove(name)
        if removed:
            if self.state.active_workspace_context == name:
                remaining = list(self.state.workspaces)
                self.state.active_workspace_context = (
                    remaining[0] if remaining else "default"
                )
            # Remove symbols for this workspace.
            self.symbols.symbols = {
                fqn: s for fqn, s in self.symbols.symbols.items()
                if s.workspace != name
            }
            self.symbols.deps = [
                d for d in self.symbols.deps
                if not (d.source_fqn.startswith(f"[{name}]")
                        or d.target_fqn.startswith(f"[{name}]"))
            ]
            self.symbols.save()
            self.save()
        return removed

    def switch_workspace(self, name: str) -> bool:
        ok = self.vffs.switch(name)
        if ok:
            self.save()
        return ok

    # --- indexing --------------------------------------------------------- #
    def index_workspace(self, name: str) -> dict[str, Any]:
        profile = self.vffs.get(name)
        if not profile:
            return {"error": f"Workspace '{name}' not found."}
        return index_workspace_ast(profile, self.symbols)

    def index_all(self) -> list[dict[str, Any]]:
        return [self.index_workspace(name) for name in self.state.workspaces]

    # --- cross-workspace path resolution ---------------------------------- #
    def resolve(self, path_expr: str) -> Path:
        return self.vffs.resolve(path_expr)

    # --- context block for LLM prompt injection --------------------------- #
    def context_block(self, query: str | None = None) -> str:
        """Return a compact workspace summary suitable for LLM prompt injection."""
        ws_list = self.vffs.all()
        if not ws_list:
            return ""
        active = self.state.active_workspace_context
        lines = ["[Workspace Registry]"]
        for ws in ws_list:
            marker = " ◀ active" if ws.name == active else ""
            lines.append(
                f"  • {ws.name}{marker}: {ws.root}  [{ws.environment_type}]"
            )
        stale = self.symbols.stale_symbols()
        if stale:
            lines.append(
                f"\n[⚠ Stale symbols ({len(stale)}) — cross-workspace deps may be broken]"
            )
            for s in stale[:5]:
                lines.append(f"  • {s.fqn}")
        return "\n".join(lines)
