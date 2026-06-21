# Oqlo Code — The Agentic OS Orchestrator

Oqlo Code is a **free, self-hostable** asynchronous CLI that turns LLMs into
*runtime executives*. Instead of asking a model to write entire scripts from
scratch, Oqlo gives it a catalogue of pre-built automation **Skills** and lets it
orchestrate them across local creative applications — **Blender**, **Unity**, and
**Cursor** — in one continuous, self-healing session.

Bring your own keys, drop in your own skills, run your own pipelines. Nothing is
locked behind a service.

> **Note on model names.** The default catalogue ships with placeholder slugs
> (`claude-4.6-opus`, `gpt-5.5-pro`, `gemini-3.5-flash`) representing the "top
> tier" of each provider. These are **configuration, not constants** — edit
> `config.py` (or your `.env`) to point them at whatever model IDs your accounts
> actually expose. The router never hard-codes a slug.

---

## Why it's reliable: the Universal Gateway

The heart of Oqlo is a **state-preserving multi-provider router** (`llm_router.py`).
It keeps a single *canonical* conversation buffer and, at send time, translates
both the messages **and** the tool manifest into whichever provider dialect is
active:

| Dialect      | Providers                          |
|--------------|------------------------------------|
| `openai`     | OpenAI, OpenRouter                 |
| `anthropic`  | Anthropic Messages API             |
| `gemini`     | Google Gemini / Vertex             |
| `cursor`     | Local terminal fallback (no network) |

When a provider returns **429 (rate limit)**, **503**, or times out, the router
captures the exact buffer, translates tool-use structures to the next provider's
schema, and resumes mid-stream — **no message state is lost**. The chain always
ends at the local Cursor fallback, so the loop never hard-crashes, even offline.

Routing is governed by a policy:

- `max_intelligence` — order by capability (default)
- `cost_optimization` — order by blended token price (cheapest first)

---

## Features

1. **Universal BYOK Gateway** — load any subset of OpenRouter / Anthropic / OpenAI
   / Gemini keys; unconfigured providers are skipped automatically.
2. **Cross-Application Pipeline** — pipe artifacts (FBX paths, asset GUIDs, log
   dumps) from Blender → Unity → Cursor in a single request.
3. **Self-Healing Loop** — tool errors are wrapped as `[SYSTEM ERROR: <traceback>]`
   and fed back so the active model can correct and re-issue the call.
4. **Graceful Simulation** — if Blender/Unity aren't running, bridges still do the
   safe file-system work and flag results as `[SIMULATED]`, so you can develop
   and test the whole flow with nothing installed.
5. **User-Extensible Skills** — drop a JSON file in `~/.oqlo/skills/` and invoke it
   instantly. No code changes.

---

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the keys you have
python main.py
```

Requires Python 3.11+ (3.12 recommended).

---

## Commands

Type `/help` inside the CLI. Highlights:

| Command | Description |
|---|---|
| `/status` | Active model, policy, and bridge reachability |
| `/models` | Model catalogue with cost & availability |
| `/chain` | Current ordered fallback chain |
| `/policy [name]` | Show/set routing policy |
| `/providers` | BYOK key status |
| `/cost` | Session token usage & USD spend |
| `/tools` | LLM-accessible bridge tools |
| `/blender <bpy>` | Run a bpy snippet directly |
| `/unity logs [sev]` | Fetch Unity editor logs |
| `/cursor open <path>` | Open a folder in Cursor |
| `/skill ...` | `list / search / run / add / remove / reload / examples` |
| `/addon <blender\|unity>` | Print the in-app bridge addon/script |
| `/save` · `/load` · `/clear` | Session persistence |

Any line **without** a leading `/` is sent to the orchestrator.

### Example compound request

```
[oqlo] › Create an apocalyptic cyberpunk street block: generate the 3D
         structures in Blender, export them, import into Unity, write a
         camera-pan C# script, and open the project in Cursor.
```

The orchestrator plans this as a sequence of tool calls across all three
bridges, piping the FBX path from Blender into Unity and the project path into
Cursor — falling over to the next provider transparently if the primary model
rate-limits mid-run.

---

## Connecting real applications

The bridges are clients; the apps host the listeners. Print the ready-to-install
glue with:

- `/addon blender` → a Blender addon hosting a WebSocket command server on
  `ws://127.0.0.1:9876`.
- `/addon unity` → a Unity editor script hosting `OqloUnityBridge` on
  `http://127.0.0.1:8088`.

Until those are installed and running, every bridge call runs in simulation mode.

---

## Project layout

```
config.py            BYOK config, ModelProfile catalogue, PRIORITY_CHAIN, policies
llm_router.py        Multi-provider async router + canonical↔dialect translation
tools_manifest.py    Provider-neutral tool specs (rendered per dialect)
orchestrator.py      The agent loop + self-healing controller
skills.py            User-extensible JSON skill registry
main.py              Rich-powered interactive terminal
bridges/
  result.py          Shared ToolResult type
  blender_bridge.py  WebSocket client + in-app addon template
  unity_bridge.py    REST client + in-app editor script template
  cursor_bridge.py   git-apply patcher + workspace launcher
```

---

## License

See [LICENSE](./LICENSE).
