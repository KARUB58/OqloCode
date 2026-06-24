"""Oqlo Code — Engine 5: Multi-Agent Swarm Topology & Local Event Bus.

Execution is deconstructed into four asynchronous agents that coordinate purely
through a local Pub/Sub :class:`EventBus` (built on ``asyncio.Queue``):

* :class:`ArchitectAgent` — parses intent, builds the DAG, tracks lifecycle.
* :class:`BlenderAgent`   — modeling, procedural materials, FBX export.
* :class:`UnityAgent`     — scene composition, C# components, editor state.
* :class:`QAAgent`        — captures errors, runs pre-flight QA, drives healing,
  and commits successful patterns to long-term memory.

The swarm runs fully offline using the bridges in simulation mode, so the whole
topology is demonstrable with nothing installed. Telemetry throttling and the
memory cache are injected so each agent is resource-aware and learns over runs.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bridges import BridgeRouter
from memory_rag import MemoryRAG
from sandbox import SandboxValidator
from telemetry import TelemetryThrottler


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Event:
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class _Subscription:
    name: str
    patterns: tuple[str, ...]
    queue: "asyncio.Queue[Event]"


class EventBus:
    """A minimal async topic-based Pub/Sub bus.

    Topics are dot-delimited (``blender.render.complete``); subscribers match
    with shell-style globs (``blender.*``, ``*.error``, ``*``).
    """

    def __init__(self, on_publish: Callable[[Event], None] | None = None) -> None:
        self._subs: list[_Subscription] = []
        self.history: list[Event] = []
        self.on_publish = on_publish

    def subscribe(self, name: str, patterns: list[str]) -> "asyncio.Queue[Event]":
        q: "asyncio.Queue[Event]" = asyncio.Queue()
        self._subs.append(_Subscription(name, tuple(patterns), q))
        return q

    def publish(self, topic: str, payload: dict[str, Any] | None = None,
                *, source: str = "system") -> Event:
        ev = Event(topic=topic, payload=payload or {}, source=source)
        self.history.append(ev)
        if self.on_publish:
            self.on_publish(ev)
        for sub in self._subs:
            if any(fnmatch.fnmatch(topic, p) for p in sub.patterns):
                sub.queue.put_nowait(ev)
        return ev


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
class Agent:
    """Base class: subscribe to topics, consume events in a background task."""

    NAME = "agent"
    TOPICS: list[str] = []

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.queue = bus.subscribe(self.NAME, self.TOPICS)
        self._task: asyncio.Task[None] | None = None
        self.running = False

    def start(self) -> None:
        self.running = True
        self._task = asyncio.create_task(self._loop(), name=f"{self.NAME}-loop")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self.running:
            ev = await self.queue.get()
            try:
                await self.handle(ev)
            except Exception as exc:  # noqa: BLE001 - report, never die.
                self.bus.publish(
                    "swarm.agent.error",
                    {"agent": self.NAME, "error": f"{type(exc).__name__}: {exc}",
                     "cause_topic": ev.topic},
                    source=self.NAME,
                )

    async def handle(self, ev: Event) -> None:  # pragma: no cover - overridden.
        raise NotImplementedError

    def emit(self, topic: str, payload: dict[str, Any]) -> None:
        self.bus.publish(topic, payload, source=self.NAME)


class ArchitectAgent(Agent):
    """Parses intent, builds the DAG, and tracks pipeline lifecycle."""

    NAME = "architect"
    TOPICS = ["task.submitted", "*.complete", "*.error", "swarm.*"]

    def __init__(self, bus: EventBus, memory: MemoryRAG) -> None:
        super().__init__(bus)
        self.memory = memory
        self.pipelines: dict[str, dict[str, Any]] = {}

    def build_dag(self, intent: str) -> list[dict[str, Any]]:
        """Decompose intent into an ordered task DAG (keyword heuristic)."""
        low = intent.lower()
        dag: list[dict[str, Any]] = []
        if any(k in low for k in ("blender", "model", "mesh", "3d", "asset",
                                  "street", "building", "fbx")):
            dag.append({"step": "blender.build", "needs": []})
        if any(k in low for k in ("unity", "import", "scene", "prefab",
                                  "camera", "c#", "csharp", "script")):
            dag.append({"step": "unity.compose", "needs": ["blender.build"]})
        if not dag:  # default minimal pipeline.
            dag = [{"step": "blender.build", "needs": []},
                   {"step": "unity.compose", "needs": ["blender.build"]}]
        return dag

    async def handle(self, ev: Event) -> None:
        if ev.topic == "task.submitted":
            pid = ev.payload["pipeline_id"]
            intent = ev.payload["intent"]
            dag = self.build_dag(intent)
            self.pipelines[pid] = {"intent": intent, "dag": dag, "done": set(),
                                   "failed": False}
            self.emit("swarm.dag.built", {"pipeline_id": pid, "dag": dag})
            # Kick the first stage.
            self.emit("blender.build",
                      {"pipeline_id": pid, "intent": intent})
        elif ev.topic.endswith(".complete") and "pipeline_id" in ev.payload:
            pid = ev.payload["pipeline_id"]
            p = self.pipelines.get(pid)
            if not p:
                return
            p["done"].add(ev.topic)
            if ev.topic == "unity.compose.complete" or \
                    ev.topic == "blender.build.complete" and \
                    not any(d["step"] == "unity.compose" for d in p["dag"]):
                self.emit("task.complete",
                          {"pipeline_id": pid, "intent": p["intent"]})
        elif ev.topic.endswith(".error") and "pipeline_id" in ev.payload:
            pid = ev.payload.get("pipeline_id")
            if pid in self.pipelines:
                self.pipelines[pid]["failed"] = True


class BlenderAgent(Agent):
    """Models assets, applies materials, exports FBX — all via the bridge."""

    NAME = "blender"
    TOPICS = ["blender.build", "blender.heal"]

    BPY_TEMPLATE = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cube_add(size=2)\n"
        "obj = bpy.context.active_object\n"
        "mat = bpy.data.materials.new(name='OqloProc')\n"
        "obj.data.materials.append(mat)\n"
    )

    def __init__(self, bus: EventBus, bridges: BridgeRouter,
                 sandbox: SandboxValidator, memory: MemoryRAG,
                 throttle: TelemetryThrottler) -> None:
        super().__init__(bus)
        self.bridges = bridges
        self.sandbox = sandbox
        self.memory = memory
        self.throttle = throttle

    async def handle(self, ev: Event) -> None:
        pid = ev.payload.get("pipeline_id")
        intent = ev.payload.get("intent", "")
        healing = ev.topic == "blender.heal"
        # Engine 6: reuse a proven snippet if memory has one — but on a heal
        # attempt, the cached path already failed, so fall back to the known-good
        # template instead of repeating the same mistake.
        code = self.BPY_TEMPLATE
        if not healing and intent:
            hits = self.memory.query(intent, top_k=1, kind="code")
            if hits:
                code = hits[0].record.text

        # Engine 10: pre-flight QA before touching live Blender.
        report = await self.sandbox.validate_python(code)
        self.emit("qa.preflight", {"pipeline_id": pid, "language": "python",
                                   "ok": report.ok, "detail": report.summary()})
        if not report.ok:
            self.emit("blender.error", {"pipeline_id": pid,
                                        "error": report.summary()})
            return

        # Engine 7: wait if the host is choked before heavy work.
        await self.throttle.wait_until_clear()
        result = await self.bridges.blender.execute_bpy_command(code, "swarm build")
        if not result.ok:
            self.emit("blender.error",
                      {"pipeline_id": pid, "error": result.error or "bpy failed"})
            return

        fbx = await self.bridges.blender.export_active_scene_to_fbx(
            f"swarm_{(pid or 'x')[:8]}")
        self.emit("blender.render.complete",
                  {"pipeline_id": pid, "intent": intent,
                   "fbx_path": fbx.data.get("fbx_path", ""),
                   "snippet": code})
        self.emit("blender.build.complete",
                  {"pipeline_id": pid, "intent": intent})


class UnityAgent(Agent):
    """Imports FBX, composes scene, generates & 'compiles' C# components."""

    NAME = "unity"
    TOPICS = ["blender.render.complete", "unity.heal"]

    CS_TEMPLATE = (
        "using UnityEngine;\n\n"
        "public class CameraPan : MonoBehaviour\n{\n"
        "    public float speed = 2.0f;\n"
        "    void Update()\n    {\n"
        "        transform.Translate(Vector3.right * speed * Time.deltaTime);\n"
        "    }\n}\n"
    )

    def __init__(self, bus: EventBus, bridges: BridgeRouter,
                 sandbox: SandboxValidator, memory: MemoryRAG) -> None:
        super().__init__(bus)
        self.bridges = bridges
        self.sandbox = sandbox
        self.memory = memory

    async def handle(self, ev: Event) -> None:
        pid = ev.payload.get("pipeline_id")
        fbx = ev.payload.get("fbx_path", "")
        if fbx:
            imp = await self.bridges.unity.import_asset_to_project(fbx)
            if not imp.ok:
                self.emit("unity.compile.error",
                          {"pipeline_id": pid, "error": imp.error or "import failed"})
                return

        # Engine 10: structural pre-flight on the C# before writing it.
        report = self.sandbox.validate_csharp(self.CS_TEMPLATE)
        self.emit("qa.preflight", {"pipeline_id": pid, "language": "csharp",
                                   "ok": report.ok, "detail": report.summary()})
        if not report.ok:
            self.emit("unity.compile.error",
                      {"pipeline_id": pid, "error": report.summary()})
            return

        cs = await self.bridges.unity.create_csharp_script("CameraPan",
                                                           self.CS_TEMPLATE)
        if not cs.ok:
            self.emit("unity.compile.error",
                      {"pipeline_id": pid, "error": cs.error or "compile failed"})
            return
        self.emit("unity.compose.complete",
                  {"pipeline_id": pid,
                   "script": cs.data.get("script_path", ""),
                   "snippet": self.CS_TEMPLATE})


class QAAgent(Agent):
    """Captures errors, drives self-healing, and learns from successes."""

    NAME = "qa"
    TOPICS = ["*.error", "*.complete", "qa.preflight"]

    def __init__(self, bus: EventBus, memory: MemoryRAG,
                 max_heals: int = 2) -> None:
        super().__init__(bus)
        self.memory = memory
        self.max_heals = max_heals
        self.heal_counts: dict[str, int] = {}
        self.errors: list[str] = []

    async def handle(self, ev: Event) -> None:
        if ev.topic.endswith(".error"):
            pid = ev.payload.get("pipeline_id", "?")
            err = ev.payload.get("error", "unknown")
            self.errors.append(f"[{ev.source}] {err}")
            key = f"{pid}:{ev.source}"
            count = self.heal_counts.get(key, 0)
            if count < self.max_heals:
                self.heal_counts[key] = count + 1
                # Initiate a self-heal loop by re-dispatching a heal event.
                heal_topic = f"{ev.source}.heal"
                self.emit(heal_topic,
                          {"pipeline_id": pid, "system_error": err,
                           "intent": ev.payload.get("intent", ""),
                           "attempt": count + 1})
            else:
                self.emit("task.error",
                          {"pipeline_id": pid, "error": err, "exhausted": True})
        elif ev.topic.endswith(".complete") and "snippet" in ev.payload:
            # Engine 6: commit the proven snippet to long-term memory.
            kind = "code" if ev.source == "blender" else "csharp"
            self.memory.remember(
                ev.payload["snippet"], kind=kind,
                metadata={"pipeline_id": ev.payload.get("pipeline_id")},
                tags=[ev.source, "proven"])


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
class SwarmCoordinator:
    """Wires the bus + agents and runs a single task to completion."""

    def __init__(
        self,
        bridges: BridgeRouter,
        memory: MemoryRAG,
        *,
        throttle: TelemetryThrottler | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.bus = EventBus(on_publish=on_event)
        self.memory = memory
        self.sandbox = SandboxValidator()
        self.throttle = throttle or TelemetryThrottler()
        self.architect = ArchitectAgent(self.bus, memory)
        self.blender = BlenderAgent(self.bus, bridges, self.sandbox, memory,
                                    self.throttle)
        self.unity = UnityAgent(self.bus, bridges, self.sandbox, memory)
        self.qa = QAAgent(self.bus, memory)
        self.agents = [self.architect, self.blender, self.unity, self.qa]

    async def run(self, intent: str, *, timeout: float = 30.0) -> dict[str, Any]:
        done = self.bus.subscribe("coordinator", ["task.complete", "task.error"])
        for a in self.agents:
            a.start()
        pid = f"pipe_{uuid.uuid4().hex[:8]}"
        self.bus.publish("task.submitted",
                         {"pipeline_id": pid, "intent": intent})
        outcome: dict[str, Any]
        try:
            ev = await asyncio.wait_for(done.get(), timeout=timeout)
            outcome = {"ok": ev.topic == "task.complete",
                       "pipeline_id": pid, "payload": ev.payload,
                       "errors": list(self.qa.errors)}
        except asyncio.TimeoutError:
            outcome = {"ok": False, "pipeline_id": pid,
                       "payload": {"error": "swarm timed out"},
                       "errors": list(self.qa.errors)}
        finally:
            for a in self.agents:
                await a.stop()
        return outcome
