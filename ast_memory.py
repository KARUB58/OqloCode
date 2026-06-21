"""Oqlo Code — Module 2: AST Incremental Memory.

NOTE: the upgrade message named this module but did not include its detailed
spec, so this is a reasonable, self-contained interpretation: a code memory that
stores Python at the granularity of its **AST units** (functions and classes)
rather than as opaque text blobs, and updates *incrementally*.

Why AST-granular + incremental:

* **No duplication.** Re-ingesting a near-identical snippet only rewrites the
  units whose structure actually changed (compared by a normalized structural
  hash), so memory does not bloat with copies — minimizing storage and the
  number of tokens later injected back into prompts.
* **Sharper recall.** You can retrieve a single proven function by name instead
  of dragging a whole file along with it.
* **Versioned.** Each changed unit bumps a version and keeps a short history.

Pairs with the TF-IDF :class:`memory_rag.MemoryRAG` for free-text recall; this
one is the structured, code-aware tier.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_AST_PATH = Path.home() / ".oqlo" / "memory" / "ast_index.json"


def _structural_hash(node: ast.AST) -> str:
    """Hash of the node's structure, ignoring formatting/whitespace."""
    dump = ast.dump(node, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class CodeUnit:
    qualname: str                # e.g. "ClassName.method" or "function_name"
    kind: str                    # "function" | "class"
    source: str
    struct_hash: str
    signature: str = ""
    version: int = 1
    updated_at: float = field(default_factory=time.time)
    history: list[str] = field(default_factory=list)  # prior struct hashes
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualname": self.qualname, "kind": self.kind, "source": self.source,
            "struct_hash": self.struct_hash, "signature": self.signature,
            "version": self.version, "updated_at": self.updated_at,
            "history": self.history, "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CodeUnit":
        return cls(
            qualname=d["qualname"], kind=d.get("kind", "function"),
            source=d.get("source", ""), struct_hash=d.get("struct_hash", ""),
            signature=d.get("signature", ""), version=d.get("version", 1),
            updated_at=d.get("updated_at", time.time()),
            history=d.get("history", []), tags=d.get("tags", []),
        )


@dataclass(slots=True)
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        if self.error:
            return f"[ast-memory] parse failed: {self.error}"
        return (f"[ast-memory] +{len(self.added)} new, "
                f"~{len(self.updated)} updated, "
                f"={len(self.unchanged)} unchanged")


class ASTMemory:
    """Incremental, AST-granular store of proven Python code units."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_AST_PATH
        self.units: dict[str, CodeUnit] = {}
        self.load()

    # --- persistence ------------------------------------------------------ #
    def load(self) -> int:
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.units = {
                    u["qualname"]: CodeUnit.from_dict(u) for u in raw
                }
            except (json.JSONDecodeError, OSError, KeyError):
                self.units = {}
        return len(self.units)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([u.to_dict() for u in self.units.values()], indent=2),
            encoding="utf-8",
        )

    # --- extraction ------------------------------------------------------- #
    @staticmethod
    def _signature(node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            return f"def {node.name}({', '.join(args)})"
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", "…") for b in node.bases]
            return f"class {node.name}({', '.join(bases)})"
        return ""

    def _extract(self, code: str) -> list[tuple[str, str, ast.AST]]:
        """Return (qualname, kind, node) for each top-level def/class (+ methods)."""
        tree = ast.parse(code)
        try:
            lines = code.splitlines()

            def src(node: ast.AST) -> str:
                seg = ast.get_source_segment(code, node)
                if seg is not None:
                    return seg
                # Fallback for older behavior.
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                return "\n".join(lines[start:end])
        except Exception:  # pragma: no cover
            def src(node: ast.AST) -> str:  # type: ignore[misc]
                return ast.dump(node)

        units: list[tuple[str, str, ast.AST]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                units.append((node.name, "function", node))
            elif isinstance(node, ast.ClassDef):
                units.append((node.name, "class", node))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        units.append((f"{node.name}.{sub.name}", "function", sub))
        # Attach a source resolver attribute for ingest to use.
        self._src_resolver = src  # type: ignore[attr-defined]
        return units

    # --- public API ------------------------------------------------------- #
    def ingest(self, code: str, *, tags: list[str] | None = None) -> IngestReport:
        """Incrementally fold a code snippet into the store."""
        report = IngestReport()
        try:
            units = self._extract(code)
        except SyntaxError as exc:
            report.error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
            return report

        src = self._src_resolver  # type: ignore[attr-defined]
        changed = False
        for qualname, kind, node in units:
            shash = _structural_hash(node)
            existing = self.units.get(qualname)
            if existing is None:
                self.units[qualname] = CodeUnit(
                    qualname=qualname, kind=kind, source=src(node),
                    struct_hash=shash, signature=self._signature(node),
                    tags=tags or [],
                )
                report.added.append(qualname)
                changed = True
            elif existing.struct_hash != shash:
                existing.history.append(existing.struct_hash)
                existing.history = existing.history[-5:]
                existing.source = src(node)
                existing.struct_hash = shash
                existing.signature = self._signature(node)
                existing.version += 1
                existing.updated_at = time.time()
                for t in (tags or []):
                    if t not in existing.tags:
                        existing.tags.append(t)
                report.updated.append(qualname)
                changed = True
            else:
                report.unchanged.append(qualname)
        if changed:
            self.save()
        return report

    def get(self, qualname: str) -> CodeUnit | None:
        return self.units.get(qualname)

    def search(self, term: str) -> list[CodeUnit]:
        term = term.lower()
        return [
            u for u in self.units.values()
            if term in u.qualname.lower()
            or term in u.signature.lower()
            or any(term in t.lower() for t in u.tags)
        ]

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for u in self.units.values():
            kinds[u.kind] = kinds.get(u.kind, 0) + 1
        return {
            "units": len(self.units),
            "by_kind": kinds,
            "versioned": sum(1 for u in self.units.values() if u.version > 1),
            "path": str(self.path),
        }
