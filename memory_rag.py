"""Oqlo Code — Engine 6: Semantic Graph RAG & Long-Term Memory.

A local, zero-dependency semantic cache. It retains successful code snippets,
aesthetic theme configs, and UI layouts across project instances and surfaces the
most relevant prior solutions for each new task, so the swarm reuses proven
structural patterns instead of regenerating them from scratch.

Retrieval uses TF-IDF vectors with cosine similarity, implemented in pure Python
(NumPy is used automatically if present, purely for speed). The index persists to
``~/.oqlo/memory/index.json``.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - optional accelerator.
    _np = None  # type: ignore[assignment]


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
DEFAULT_MEMORY_PATH = Path.home() / ".oqlo" / "memory" / "index.json"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass(slots=True)
class MemoryRecord:
    id: str
    text: str
    kind: str  # e.g. "code", "theme", "layout"
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    uses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "kind": self.kind,
            "metadata": self.metadata, "tags": self.tags,
            "created_at": self.created_at, "uses": self.uses,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=d["id"], text=d["text"], kind=d.get("kind", "code"),
            metadata=d.get("metadata", {}), tags=d.get("tags", []),
            created_at=d.get("created_at", time.time()), uses=d.get("uses", 0),
        )


@dataclass(slots=True)
class RetrievalHit:
    score: float
    record: MemoryRecord


class MemoryRAG:
    """A persistent TF-IDF semantic cache."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_MEMORY_PATH
        self.records: list[MemoryRecord] = []
        # Cached term frequencies per record id (token -> count).
        self._tf: dict[str, Counter[str]] = {}
        self._idf: dict[str, float] = {}
        self.load()

    # --- persistence ------------------------------------------------------ #
    def load(self) -> int:
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.records = [MemoryRecord.from_dict(r) for r in raw]
            except (json.JSONDecodeError, OSError, KeyError):
                self.records = []
        self._reindex()
        return len(self.records)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([r.to_dict() for r in self.records], indent=2),
            encoding="utf-8",
        )

    # --- indexing --------------------------------------------------------- #
    def _reindex(self) -> None:
        self._tf = {r.id: Counter(tokenize(r.text)) for r in self.records}
        n_docs = len(self.records) or 1
        df: Counter[str] = Counter()
        for counts in self._tf.values():
            df.update(counts.keys())
        self._idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {
            term: (cnt / total) * self._idf.get(term, math.log(len(self.records) + 1) + 1.0)
            for term, cnt in counts.items()
        }

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        if _np is not None:
            na = float(_np.linalg.norm(_np.fromiter(a.values(), dtype=float)))
            nb = float(_np.linalg.norm(_np.fromiter(b.values(), dtype=float)))
        else:
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    # --- public API ------------------------------------------------------- #
    def remember(
        self,
        text: str,
        *,
        kind: str = "code",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        rec = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:10]}",
            text=text, kind=kind,
            metadata=metadata or {}, tags=tags or [],
        )
        self.records.append(rec)
        self._reindex()
        self.save()
        return rec

    def query(
        self, text: str, *, top_k: int = 3, min_score: float = 0.05,
        kind: str | None = None,
    ) -> list[RetrievalHit]:
        q_vec = self._vector(Counter(tokenize(text)))
        hits: list[RetrievalHit] = []
        for rec in self.records:
            if kind is not None and rec.kind != kind:
                continue
            score = self._cosine(q_vec, self._vector(self._tf[rec.id]))
            if score >= min_score:
                hits.append(RetrievalHit(score=round(score, 4), record=rec))
        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:top_k]
        for h in top:  # usage tracking informs future eviction policies.
            h.record.uses += 1
        if top:
            self.save()
        return top

    def context_block(self, text: str, *, top_k: int = 3) -> str:
        """Render the best prior solutions as a compact context string.

        Token-saving: only the highest-similarity snippets are injected, so the
        model gets proven patterns without re-deriving them.
        """
        hits = self.query(text, top_k=top_k)
        if not hits:
            return ""
        lines = ["[MEMORY: proven patterns from past projects]"]
        for h in hits:
            tag = f" ({', '.join(h.record.tags)})" if h.record.tags else ""
            lines.append(f"- ({h.score:.2f}) [{h.record.kind}]{tag} "
                         f"{h.record.text[:240]}")
        return "\n".join(lines)

    def forget(self, record_id: str) -> bool:
        before = len(self.records)
        self.records = [r for r in self.records if r.id != record_id]
        if len(self.records) != before:
            self._reindex()
            self.save()
            return True
        return False

    def stats(self) -> dict[str, Any]:
        kinds: Counter[str] = Counter(r.kind for r in self.records)
        return {
            "records": len(self.records),
            "vocab": len(self._idf),
            "by_kind": dict(kinds),
            "path": str(self.path),
            "numpy": _np is not None,
        }
