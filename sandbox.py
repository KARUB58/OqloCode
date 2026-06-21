"""Oqlo Code — Engine 10: Local Sandboxed Pre-Flight QA Zone.

Before any generated script reaches a live application, the QA layer validates it
here so a syntax slip never crashes a running Blender/Unity session.

* Python (``bpy``) snippets: compiled for syntax, AST-scanned for obviously
  dangerous calls, then *mock-executed* in a separate, resource-limited
  subprocess where ``bpy`` (and friends) are replaced by no-op stubs. This proves
  the control flow and imports hold up without touching real software.
* C# scripts: structurally validated (balanced braces, a class declaration,
  using directives) since no compiler is assumed present.

The subprocess is wrapped with CPU-time and address-space ``resource`` limits
(POSIX) and a wall-clock timeout, so a runaway snippet cannot hang the host.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import textwrap
from dataclasses import dataclass, field

# Calls we refuse to let through pre-flight even as a mock.
_DANGEROUS = {
    ("os", "system"), ("subprocess", "run"), ("subprocess", "Popen"),
    ("subprocess", "call"), ("shutil", "rmtree"), ("os", "remove"),
    ("os", "removedirs"), ("os", "unlink"),
}
_DANGEROUS_NAMES = {"eval", "exec", "compile", "__import__"}


@dataclass(slots=True)
class SandboxReport:
    ok: bool
    language: str
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        parts = [f"[pre-flight {head}/{self.language}]"]
        parts.extend(f"✗ {m}" for m in self.messages)
        parts.extend(f"⚠ {w}" for w in self.warnings)
        return " ".join(parts) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------- #
# Mock harness injected ahead of bpy snippets in the subprocess.
# --------------------------------------------------------------------------- #
_MOCK_HARNESS = textwrap.dedent(
    '''
    import sys, types

    class _Mock(types.ModuleType):
        def __init__(self, name="mock"):
            super().__init__(name)
        def __getattr__(self, item):
            child = _Mock(item)
            setattr(self, item, child)
            return child
        def __call__(self, *a, **k):
            return _Mock("call")
        def __iter__(self):
            return iter(())
        def __getitem__(self, k):
            return _Mock("item")
        def __setitem__(self, k, v):
            pass

    for _name in ("bpy", "bmesh", "mathutils", "bpy_extras"):
        sys.modules[_name] = _Mock(_name)
    '''
).strip()


class SandboxValidator:
    """Validates generated code before it is shipped to a live app."""

    def __init__(self, timeout_seconds: float = 5.0,
                 mem_limit_mb: int = 256) -> None:
        self.timeout_seconds = timeout_seconds
        self.mem_limit_mb = mem_limit_mb

    # --- Python / bpy ----------------------------------------------------- #
    def _static_scan(self, code: str) -> tuple[bool, list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, [f"SyntaxError: {exc.msg} (line {exc.lineno})"], []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _DANGEROUS_NAMES:
                    warnings.append(f"use of '{func.id}' flagged")
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    pair = (func.value.id, func.attr)
                    if pair in _DANGEROUS:
                        errors.append(f"blocked call {pair[0]}.{pair[1]}()")
        return (not errors), errors, warnings

    async def validate_python(self, code: str) -> SandboxReport:
        ok, errors, warnings = self._static_scan(code)
        if not ok:
            return SandboxReport(False, "python", errors, warnings)

        # Mock-execute in an isolated subprocess.
        program = f"{_MOCK_HARNESS}\n\n{_PREAMBLE_LIMITS}\n\n{code}\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c", program,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                proc.kill()
                return SandboxReport(
                    False, "python",
                    [f"mock execution exceeded {self.timeout_seconds}s timeout"],
                    warnings)
        except OSError as exc:
            return SandboxReport(False, "python",
                                 [f"sandbox spawn failed: {exc}"], warnings)

        if proc.returncode != 0:
            tail = stderr.decode(errors="replace").strip().splitlines()[-3:]
            return SandboxReport(False, "python",
                                 ["mock run raised: " + " | ".join(tail)],
                                 warnings)
        return SandboxReport(True, "python", [], warnings)

    # --- C# --------------------------------------------------------------- #
    def validate_csharp(self, source: str) -> SandboxReport:
        errors: list[str] = []
        warnings: list[str] = []
        if source.count("{") != source.count("}"):
            errors.append("unbalanced braces { }")
        if source.count("(") != source.count(")"):
            errors.append("unbalanced parentheses ( )")
        if "class " not in source and "struct " not in source:
            errors.append("no class/struct declaration found")
        if "using UnityEngine" not in source:
            warnings.append("missing 'using UnityEngine;'")
        # crude statement-termination heuristic
        for i, line in enumerate(source.splitlines(), 1):
            s = line.strip()
            if s and not s.endswith((";", "{", "}", ")", ",")) \
                    and not s.startswith(("//", "#", "[", "using", "namespace",
                                          "public", "private", "protected",
                                          "internal", "class", "struct", "void",
                                          "if", "else", "for", "while", "switch")):
                warnings.append(f"line {i} may be missing a semicolon")
                break
        return SandboxReport(not errors, "csharp", errors, warnings)


# Resource limits applied at the top of the mock subprocess (POSIX only).
_PREAMBLE_LIMITS = textwrap.dedent(
    '''
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        _mb = 256 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (_mb, _mb))
    except Exception:
        pass
    '''
).strip()
