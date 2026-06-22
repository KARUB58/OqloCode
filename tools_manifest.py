"""Oqlo Code — canonical tool manifest.

Tools are declared once in a provider-neutral form (``name`` / ``description`` /
JSON-Schema ``parameters``) and rendered on demand into the two wire dialects
the router speaks:

* OpenAI / OpenRouter / Gemini -> ``{"type": "function", "function": {...}}``
* Anthropic                    -> ``{"name", "description", "input_schema"}``

Keeping a single source of truth is what lets the router swap providers
mid-stream without the model losing access to a single tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A provider-neutral tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (object).
    bridge: str  # which bridge module fulfills it: blender|unity|cursor

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Helper to build a strict JSON-Schema object node."""
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Blender skills
# --------------------------------------------------------------------------- #
EXECUTE_BPY_COMMAND = ToolSpec(
    name="execute_bpy_command",
    description=(
        "Execute Python `bpy` code inside the live Blender instance over the "
        "local WebSocket bridge. Use for mesh creation, modifiers, and "
        "procedural texturing. Returns stdout plus any traceback."
    ),
    bridge="blender",
    parameters=_obj(
        {
            "code": {
                "type": "string",
                "description": "Complete, runnable bpy Python snippet.",
            },
            "description": {
                "type": "string",
                "description": "One-line summary of what this code builds.",
            },
        },
        required=["code"],
    ),
)

EXPORT_ACTIVE_SCENE_TO_FBX = ToolSpec(
    name="export_active_scene_to_fbx",
    description=(
        "Export the current Blender scene (or selected objects) to an FBX file "
        "and return the absolute path for downstream import into Unity."
    ),
    bridge="blender",
    parameters=_obj(
        {
            "filename": {
                "type": "string",
                "description": "Base filename without extension, e.g. 'street_block'.",
            },
            "selected_only": {
                "type": "boolean",
                "description": "Export only selected objects instead of the whole scene.",
                "default": False,
            },
            "apply_modifiers": {
                "type": "boolean",
                "description": "Bake modifiers into the exported mesh.",
                "default": True,
            },
        },
        required=["filename"],
    ),
)


# --------------------------------------------------------------------------- #
# Unity skills
# --------------------------------------------------------------------------- #
IMPORT_ASSET_TO_PROJECT = ToolSpec(
    name="import_asset_to_project",
    description=(
        "Copy an external asset (typically an FBX from Blender) into the Unity "
        "project's Assets folder and trigger an AssetDatabase import. Returns "
        "the in-project asset path and GUID."
    ),
    bridge="unity",
    parameters=_obj(
        {
            "source_path": {
                "type": "string",
                "description": "Absolute path on disk of the file to import.",
            },
            "target_subdir": {
                "type": "string",
                "description": "Folder under Assets/ to place it in.",
                "default": "Imported",
            },
            "instantiate_in_scene": {
                "type": "boolean",
                "description": "Instantiate the imported model into the open scene.",
                "default": False,
            },
        },
        required=["source_path"],
    ),
)

CREATE_CSHARP_SCRIPT = ToolSpec(
    name="create_csharp_script",
    description=(
        "Write a C# MonoBehaviour (or plain class) into the Unity project and "
        "let the editor recompile. Optionally attach it to a scene object."
    ),
    bridge="unity",
    parameters=_obj(
        {
            "class_name": {
                "type": "string",
                "description": "C# class name; the file is named to match.",
            },
            "source": {
                "type": "string",
                "description": "Full C# source for the file.",
            },
            "target_subdir": {
                "type": "string",
                "description": "Folder under Assets/ for the script.",
                "default": "Scripts",
            },
            "attach_to": {
                "type": "string",
                "description": "Optional scene GameObject name to attach the component to.",
            },
        },
        required=["class_name", "source"],
    ),
)

GET_EDITOR_LOGS = ToolSpec(
    name="get_editor_logs",
    description=(
        "Fetch the most recent Unity Editor console output, including any "
        "compilation errors. Feed the result back to yourself to self-heal."
    ),
    bridge="unity",
    parameters=_obj(
        {
            "severity": {
                "type": "string",
                "enum": ["all", "warning", "error"],
                "default": "error",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of log lines to return.",
                "default": 100,
            },
        },
        required=[],
    ),
)


# --------------------------------------------------------------------------- #
# Cursor / Antigravity skills
# --------------------------------------------------------------------------- #
PATCH_WORKSPACE_FILE = ToolSpec(
    name="patch_workspace_file",
    description=(
        "Apply a unified git-diff patch to a file in the Cursor workspace. Use "
        "to fix compilation errors surfaced by get_editor_logs."
    ),
    bridge="cursor",
    parameters=_obj(
        {
            "file_path": {
                "type": "string",
                "description": "Workspace-relative path of the file to patch.",
            },
            "unified_diff": {
                "type": "string",
                "description": "A valid unified diff (git apply compatible).",
            },
        },
        required=["file_path", "unified_diff"],
    ),
)

SET_ANTIGRAVITY_MODEL_PRESET = ToolSpec(
    name="set_antigravity_model_preset",
    description=(
        "Select the Cursor 'Antigravity' coding checkpoint / model preset used "
        "for inline edits in the open workspace."
    ),
    bridge="cursor",
    parameters=_obj(
        {
            "preset": {
                "type": "string",
                "description": "Named preset, e.g. 'structural-refactor' or 'fast-edit'.",
            },
            "open_workspace": {
                "type": "string",
                "description": "Optional absolute path to open in Cursor first.",
            },
        },
        required=["preset"],
    ),
)


# --------------------------------------------------------------------------- #
# Filesystem tools — read / write / edit / list local files
# --------------------------------------------------------------------------- #
READ_FILE = ToolSpec(
    name="read_file",
    description=(
        "Read the contents of any local file and return them as text. "
        "Use this to inspect source code, configs, or data before editing. "
        "Output is capped at 12,000 characters; large files are truncated."
    ),
    bridge="fs",
    parameters=_obj(
        {
            "path": {
                "type": "string",
                "description": "Absolute or home-relative (~) path to the file.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding (default 'utf-8').",
                "default": "utf-8",
            },
        },
        required=["path"],
    ),
)

WRITE_FILE = ToolSpec(
    name="write_file",
    description=(
        "Write (or overwrite) a local file with the given content. "
        "Parent directories are created automatically. "
        "Use this to save generated code or modified files to disk."
    ),
    bridge="fs",
    parameters=_obj(
        {
            "path": {
                "type": "string",
                "description": "Absolute or home-relative (~) destination path.",
            },
            "content": {
                "type": "string",
                "description": "Full text content to write to the file.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding (default 'utf-8').",
                "default": "utf-8",
            },
        },
        required=["path", "content"],
    ),
)

EDIT_FILE = ToolSpec(
    name="edit_file",
    description=(
        "Make a surgical in-place edit to a local file by replacing an exact "
        "string with a new one. The old_string must match exactly once — if it "
        "appears multiple times, include extra surrounding lines to make it unique. "
        "Prefer this over write_file for targeted changes to existing files."
    ),
    bridge="fs",
    parameters=_obj(
        {
            "path": {
                "type": "string",
                "description": "Absolute or home-relative (~) path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": (
                    "The exact text to find and replace. Must be unique in the file. "
                    "Include surrounding lines if needed to disambiguate."
                ),
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace old_string with.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding (default 'utf-8').",
                "default": "utf-8",
            },
        },
        required=["path", "old_string", "new_string"],
    ),
)

LIST_DIRECTORY = ToolSpec(
    name="list_directory",
    description=(
        "List the files and subdirectories inside a local directory. "
        "Returns names, types (file/dir), and file sizes. "
        "Use this to explore a project before reading or editing specific files."
    ),
    bridge="fs",
    parameters=_obj(
        {
            "path": {
                "type": "string",
                "description": "Absolute or home-relative (~) directory path.",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum number of entries to return (default 200).",
                "default": 200,
            },
        },
        required=["path"],
    ),
)


ALL_TOOLS: Final[tuple[ToolSpec, ...]] = (
    EXECUTE_BPY_COMMAND,
    EXPORT_ACTIVE_SCENE_TO_FBX,
    IMPORT_ASSET_TO_PROJECT,
    CREATE_CSHARP_SCRIPT,
    GET_EDITOR_LOGS,
    PATCH_WORKSPACE_FILE,
    SET_ANTIGRAVITY_MODEL_PRESET,
    READ_FILE,
    WRITE_FILE,
    EDIT_FILE,
    LIST_DIRECTORY,
)

TOOLS_BY_NAME: Final[dict[str, ToolSpec]] = {t.name: t for t in ALL_TOOLS}


def tools_for_dialect(dialect: str) -> list[dict[str, Any]]:
    """Render every tool into the requested wire dialect (``openai``/``anthropic``)."""
    if dialect == "anthropic":
        return [t.to_anthropic() for t in ALL_TOOLS]
    return [t.to_openai() for t in ALL_TOOLS]
