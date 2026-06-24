"""Oqlo Code — Engine 7 (Telemetry & Throttling) + Engine 9 (Cost Guard).

Two resource governors that keep autonomous runs from harming the host machine
or the operator's wallet:

* :class:`Telemetry` / :class:`TelemetryThrottler` — sample CPU, RAM, and (when
  available) GPU VRAM, and delay async work while the host is choked.
* :class:`FinancialVelocityGuard` — track real-time $/min API spend and freeze
  the execution graph for Human-In-The-Loop review if it runs away.

Everything degrades gracefully: ``psutil`` and ``nvidia-smi`` are optional. With
neither present, telemetry falls back to ``os.getloadavg`` / ``/proc`` and simply
reports ``None`` for VRAM.
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional.
    psutil = None  # type: ignore[assignment]


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Telemetry sampling
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SystemSnapshot:
    cpu_percent: float
    mem_percent: float
    mem_used_mb: float
    mem_total_mb: float
    vram_percent: float | None
    vram_used_mb: float | None
    vram_total_mb: float | None
    load_avg_1m: float | None
    timestamp: float = field(default_factory=time.time)

    @property
    def vram_label(self) -> str:
        if self.vram_percent is None:
            return "n/a"
        return f"{self.vram_percent:.0f}% ({self.vram_used_mb:.0f}/{self.vram_total_mb:.0f} MB)"


def _read_gpu() -> tuple[float, float] | None:
    """Return (used_mb, total_mb) via nvidia-smi, or None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        first = out.stdout.strip().splitlines()[0]
        used, total = (float(x) for x in first.split(","))
        return used, total
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Telemetry:
    """Samples host resource utilisation."""

    def snapshot(self) -> SystemSnapshot:
        cpu = mem_pct = used_mb = total_mb = 0.0
        load = None
        if psutil is not None:
            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            mem_pct = vm.percent
            used_mb = (vm.total - vm.available) / 1024 / 1024
            total_mb = vm.total / 1024 / 1024
        else:
            # Pure-stdlib fallback.
            try:
                load = os.getloadavg()[0]
                cpu = min(100.0, load / (os.cpu_count() or 1) * 100)
            except (OSError, AttributeError):
                load = None
            used_mb, total_mb, mem_pct = _read_meminfo()

        gpu = _read_gpu()
        vram_pct = vram_used = vram_total = None
        if gpu is not None:
            vram_used, vram_total = gpu
            vram_pct = (vram_used / vram_total * 100) if vram_total else None

        if load is None and psutil is not None:
            try:
                load = os.getloadavg()[0]
            except (OSError, AttributeError):
                load = None

        return SystemSnapshot(
            cpu_percent=round(cpu, 1),
            mem_percent=round(mem_pct, 1),
            mem_used_mb=round(used_mb, 1),
            mem_total_mb=round(total_mb, 1),
            vram_percent=round(vram_pct, 1) if vram_pct is not None else None,
            vram_used_mb=round(vram_used, 1) if vram_used is not None else None,
            vram_total_mb=round(vram_total, 1) if vram_total is not None else None,
            load_avg_1m=round(load, 2) if load is not None else None,
        )


def _read_meminfo() -> tuple[float, float, float]:
    """Parse /proc/meminfo (Linux). Returns (used_mb, total_mb, percent)."""
    try:
        info: dict[str, float] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 1024  # kB -> MB
        total = info.get("MemTotal", 0.0)
        avail = info.get("MemAvailable", total)
        used = total - avail
        pct = (used / total * 100) if total else 0.0
        return used, total, pct
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0, 0.0


# --------------------------------------------------------------------------- #
# Dynamic throttling (Engine 7)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ThrottleLimits:
    cpu_percent: float = 92.0
    mem_percent: float = 90.0
    vram_percent: float = 90.0
    poll_seconds: float = 0.75
    max_wait_seconds: float = 30.0


class TelemetryThrottler:
    """Delays async tasks while the host is choked.

    Usable two ways::

        throttle = TelemetryThrottler()

        @throttle                         # decorator
        async def render(...): ...

        await throttle.wait_until_clear() # inline gate
    """

    def __init__(
        self,
        limits: ThrottleLimits | None = None,
        telemetry: Telemetry | None = None,
        on_throttle: Callable[[SystemSnapshot], None] | None = None,
    ) -> None:
        self.limits = limits or ThrottleLimits()
        self.telemetry = telemetry or Telemetry()
        self.on_throttle = on_throttle
        self.throttle_events = 0

    def _is_choked(self, snap: SystemSnapshot) -> bool:
        if snap.cpu_percent >= self.limits.cpu_percent:
            return True
        if snap.mem_percent >= self.limits.mem_percent:
            return True
        if snap.vram_percent is not None and \
                snap.vram_percent >= self.limits.vram_percent:
            return True
        return False

    async def wait_until_clear(self) -> None:
        waited = 0.0
        while waited < self.limits.max_wait_seconds:
            snap = self.telemetry.snapshot()
            if not self._is_choked(snap):
                return
            self.throttle_events += 1
            if self.on_throttle:
                self.on_throttle(snap)
            await asyncio.sleep(self.limits.poll_seconds)
            waited += self.limits.poll_seconds

    def __call__(
        self, func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            await self.wait_until_clear()
            return await func(*args, **kwargs)

        return wrapper


# --------------------------------------------------------------------------- #
# Predictive Financial Velocity Guard (Engine 9)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CostSample:
    at: float
    amount_usd: float


class FinancialVelocityGuard:
    """Tracks $/min spend velocity and freezes runaway execution.

    When spend over the trailing ``window_seconds`` exceeds
    ``max_usd_per_min``, :meth:`check` returns ``frozen=True`` and an alert.
    The orchestrator should then halt and defer to the Human-In-The-Loop
    gatekeeper rather than continuing to burn money.
    """

    def __init__(
        self,
        max_usd_per_min: float = 2.0,
        window_seconds: float = 60.0,
    ) -> None:
        self.max_usd_per_min = max_usd_per_min
        self.window_seconds = window_seconds
        self.samples: deque[CostSample] = deque()
        self.total_usd = 0.0
        self.frozen = False

    def record(self, amount_usd: float, *, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        if amount_usd > 0:
            self.samples.append(CostSample(now, amount_usd))
            self.total_usd += amount_usd
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0].at < cutoff:
            self.samples.popleft()

    def velocity_per_min(self, *, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        self._evict(now)
        if not self.samples:
            return 0.0
        spent = sum(s.amount_usd for s in self.samples)
        span = max(now - self.samples[0].at, 1e-6)
        return spent / span * 60.0

    def check(self, *, now: float | None = None) -> tuple[bool, str]:
        """Return (frozen, message). Sets ``self.frozen`` once tripped."""
        velocity = self.velocity_per_min(now=now)
        if velocity > self.max_usd_per_min:
            self.frozen = True
            return (
                True,
                f"FINANCIAL FREEZE: spend velocity ${velocity:.2f}/min exceeds "
                f"limit ${self.max_usd_per_min:.2f}/min. Handing to HITL.",
            )
        return False, f"spend velocity ${velocity:.2f}/min (ok)"

    def reset_freeze(self) -> None:
        self.frozen = False
