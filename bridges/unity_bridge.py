"""Unity bridge — talks to a running Unity Editor over a local REST endpoint.

An editor script (see :data:`UNITY_EDITOR_SCRIPT_TEMPLATE`) hosts an
``HttpListener`` named ``OqloUnityBridge`` on ``http://127.0.0.1:8088``. This
module imports FBX assets, writes/compiles C# scripts, and pulls console logs.
When Unity is offline, file-system side effects (copying assets, writing scripts)
are still performed against a local project directory and flagged as simulated.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Any

import config
from .result import ToolResult

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


UNITY_EDITOR_SCRIPT_TEMPLATE: str = textwrap.dedent(
    """
    // Place under Assets/Editor/OqloUnityBridge.cs in your Unity project.
    using System.Net;
    using System.Threading;
    using System.IO;
    using UnityEditor;
    using UnityEngine;

    [InitializeOnLoad]
    public static class OqloUnityBridge
    {
        static OqloUnityBridge()
        {
            var listener = new HttpListener();
            listener.Prefixes.Add("http://127.0.0.1:8088/");
            listener.Start();
            var t = new Thread(() => {
                while (true) {
                    var ctx = listener.GetContext();
                    var path = ctx.Request.Url.AbsolutePath;
                    string body;
                    using (var r = new StreamReader(ctx.Request.InputStream))
                        body = r.ReadToEnd();
                    // Dispatch on path: /import, /script, /logs, /refresh
                    var resp = OqloDispatcher.Handle(path, body);
                    var buf = System.Text.Encoding.UTF8.GetBytes(resp);
                    ctx.Response.OutputStream.Write(buf, 0, buf.Length);
                    ctx.Response.Close();
                }
            });
            t.IsBackground = true;
            t.Start();
        }
    }
    """
).strip()


class UnityBridge:
    """Async client for the OqloUnityBridge editor REST endpoint."""

    def __init__(self, rest_url: str | None = None) -> None:
        self.rest_url = (rest_url or config.BRIDGES.unity_rest_url).rstrip("/")
        # Fallback local "project" used when no live editor is reachable.
        self.local_project = config.BRIDGES.fbx_export_dir.parent / "unity_project"
        self._client: Any = None

    def _http(self) -> Any:
        if httpx is None:
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        client = self._http()
        if client is None:
            return None
        try:
            resp = await client.post(f"{self.rest_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception:  # noqa: BLE001 - offline editor -> simulated path.
            return None

    def _assets_dir(self, subdir: str) -> Path:
        d = self.local_project / "Assets" / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- tools ------------------------------------------------------------ #
    async def import_asset_to_project(
        self,
        source_path: str,
        target_subdir: str = "Imported",
        instantiate_in_scene: bool = False,
    ) -> ToolResult:
        src = Path(source_path)
        if not src.exists():
            return ToolResult(
                ok=False,
                tool="import_asset_to_project",
                summary="Source asset not found",
                error=f"FileNotFoundError: {source_path} does not exist.",
            )
        reply = await self._post(
            "/import",
            {
                "source": str(src),
                "subdir": target_subdir,
                "instantiate": instantiate_in_scene,
            },
        )
        if reply is None:
            dest = self._assets_dir(target_subdir) / src.name
            shutil.copy2(src, dest)
            rel = dest.relative_to(self.local_project)
            return ToolResult(
                ok=True,
                tool="import_asset_to_project",
                summary="Copied asset into local project (Unity not running).",
                data={"asset_path": str(rel), "guid": "simulated"},
                simulated=True,
            )
        return ToolResult(
            ok=True,
            tool="import_asset_to_project",
            summary="Imported asset into Unity project.",
            data={
                "asset_path": reply.get("asset_path", ""),
                "guid": reply.get("guid", ""),
            },
        )

    async def create_csharp_script(
        self,
        class_name: str,
        source: str,
        target_subdir: str = "Scripts",
        attach_to: str | None = None,
    ) -> ToolResult:
        reply = await self._post(
            "/script",
            {
                "class_name": class_name,
                "source": source,
                "subdir": target_subdir,
                "attach_to": attach_to,
            },
        )
        if reply is None:
            dest = self._assets_dir(target_subdir) / f"{class_name}.cs"
            dest.write_text(source, encoding="utf-8")
            rel = dest.relative_to(self.local_project)
            return ToolResult(
                ok=True,
                tool="create_csharp_script",
                summary="Wrote C# script to local project (Unity not running).",
                data={"script_path": str(rel), "attached_to": attach_to or "none"},
                simulated=True,
            )
        if not reply.get("ok", True):
            return ToolResult(
                ok=False,
                tool="create_csharp_script",
                summary="Unity rejected the C# script",
                error=reply.get("error", "compilation error"),
            )
        return ToolResult(
            ok=True,
            tool="create_csharp_script",
            summary="Created C# script and recompiled.",
            data={"script_path": reply.get("script_path", ""),
                  "attached_to": attach_to or "none"},
        )

    async def get_editor_logs(
        self, severity: str = "error", max_lines: int = 100
    ) -> ToolResult:
        reply = await self._post(
            "/logs", {"severity": severity, "max_lines": max_lines}
        )
        if reply is None:
            return ToolResult(
                ok=True,
                tool="get_editor_logs",
                summary="No live Unity editor; no logs available (simulated).",
                data={"log_count": 0, "logs": ""},
                simulated=True,
            )
        return ToolResult(
            ok=True,
            tool="get_editor_logs",
            summary=f"Fetched Unity {severity} logs.",
            data={
                "log_count": reply.get("count", 0),
                "logs": reply.get("logs", ""),
            },
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
