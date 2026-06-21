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

## Provider priority (token-saving by design)

Oqlo tries **free local executives first** and only spends paid API tokens as a
last resort:

```
1. Antigravity   (local agent — no token cost)
2. Cursor        (local agent — no token cost)
3. Codex         (local executive / offline safety net)
4. API           (OpenRouter / Anthropic / OpenAI / Gemini — costs tokens)
```

Antigravity and Cursor *defer* (fall through) until you activate them with
`/local <name> on`. Codex is the offline safety net: it answers locally when no
API key is set so the loop never crashes, and steps aside the moment you add a
key so the API tier does the real work.

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
| `local`      | Antigravity / Cursor / Codex (no network) |

When a provider returns **429 (rate limit)**, **503**, or times out, the router
captures the exact buffer, translates tool-use structures to the next provider's
schema, and resumes mid-stream — **no message state is lost**.

Routing is governed by a policy:

- `local_first` — honor the 1-4 priority above (default)
- `max_intelligence` — order by capability
- `cost_optimization` — order by blended token price (cheapest first)

### Bind API keys live — no restart

```
/api openrouter sk-or-...      # bind a key at runtime
/api anthropic   sk-ant-...
/api list                      # see status
/api gemini clear              # remove
```

## Token-saving features

Everything runs in the **terminal** — Oqlo never launches a GUI app. On top of
local-first routing, it minimizes spend with:

1. **Planning pass** (`/plan on`) — a cheap, output-capped plan is drafted first
   so the (expensive) execution loop stays focused and runs fewer iterations.
2. **History trimming** — only the last *N* messages are sent upstream; the full
   buffer is kept locally for fallback and persistence.
3. **Tool-result truncation** — bulky tool outputs are clipped before being fed
   back to the model.
4. **Saver mode** (`/tokensave on`) — caps output tokens per turn.

Tune everything live with `/tokensave` and `/plan`.

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

## Next-gen engines

Oqlo Code ships a distributed, hardware-aware, self-optimizing layer on top of
the orchestrator:

| Engine | Module | What it does |
|---|---|---|
| **5 · Multi-Agent Swarm** | `swarm_bus.py` | An async Pub/Sub event bus drives four agents — `ArchitectAgent` (intent → DAG, lifecycle), `BlenderAgent`, `UnityAgent`, and `QAAgent` (errors, healing, learning). Run with `/swarm <task>`. |
| **6 · Semantic RAG Memory** | `memory_rag.py` | A zero-dependency TF-IDF vector store (NumPy optional) persisted to `~/.oqlo/memory/`. Proven code/themes/layouts are cached and recalled into each new task. `/memory`. |
| **7 · Resource Telemetry** | `telemetry.py` | Samples CPU / RAM / GPU-VRAM (`psutil`/`nvidia-smi` optional, `/proc` fallback). `TelemetryThrottler` delays async work while the host is choked. `/telemetry`. |
| **8 · Context Slicing & Caching** | `llm_router.py` | History trimming, tool-output truncation, and Anthropic/OpenAI prompt-cache markers on the stable system prompt + tool manifest. |
| **9 · Financial Velocity Guard** | `telemetry.py` | Tracks `$/min` spend; if a runaway loop exceeds the limit (default `$2.00/min`) it freezes the graph and hands off to Human-In-The-Loop. `/budget`. |
| **10 · Sandboxed Pre-Flight QA** | `sandbox.py` | Validates generated `bpy`/C# before it touches a live app: AST scan + mock-execution in a resource-limited subprocess (Python), structural checks (C#). |

How they compose in a normal run: relevant memory is recalled → a cheap plan is
drafted → each tool result that succeeds is remembered → the cost guard checks
spend velocity before every model turn → telemetry can throttle heavy steps. In
`/swarm` mode the QA agent additionally pre-flights every snippet and drives the
heal loop over the event bus.

---

## Hybrid daily-driver mode

Oqlo is both a macro-task executor and a conversational assistant. Every plain
line first passes through a **zero-token intent classifier** (`intent.py`):

* `INTENT_CHAT` — questions / chit-chat go **straight to the active model** with
  no tools and no DAG, so casual use is cheap and instant.
* `INTENT_AGENTIC` — actionable instructions spin up the full multi-agent
  pipeline.

The classifier is heuristic (verbs × domain nouns), so it adds no latency or
spend. You'll see a `↳ intent: chat|agentic` tag on each line.

### Live model override

```
/model anthropic/claude-4.6-opus
/model nvidia/nemotron-3-ultra-550b-a55b:free   # any OpenRouter slug
/model gpt-5.5-pro                              # native provider slug
/model reset                                    # back to the priority chain
```

`/model <slug>` switches the active engine immediately and **bypasses the
priority chain until reset** — `provider/model:tier` slugs route via OpenRouter;
bare slugs map to their native provider. `/clear` wipes the screen and the
short-term chat buffer but **preserves long-term memory** (RAG + AST).

## Module 2 — AST Incremental Memory

> The upgrade brief named this module but didn't include its detailed spec, so
> `ast_memory.py` is a self-contained interpretation, open to revision.

Instead of storing code as opaque text, it parses Python into **AST units**
(functions, classes, methods) and folds new snippets in **incrementally**: a unit
is added, updated (version bumped, with history), or skipped as unchanged — keyed
by a normalized structural hash. This avoids duplicate blobs, enables
function-level recall, and keeps the tokens later injected into prompts minimal.
View it with `/memory ast`.

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
| `/start oqlocode` | Show the banner and full system status |
| `/status` | Active model, override, policy, bridges, spend |
| `/model <slug>` | Override the active engine live (`/model reset` to revert) |
| `/api <provider> <key>` | Bind a cloud key live (`/api list` to view) |
| `/local <name> on\|off` | Activate/deactivate a local executive |
| `/models` | Model catalogue with cost & availability |
| `/chain` | Current ordered fallback chain |
| `/policy [name]` | `local_first` / `max_intelligence` / `cost_optimization` |
| `/providers` | BYOK key status |
| `/tokensave on\|off` | Toggle aggressive token trimming |
| `/plan on\|off` | Toggle the cheap planning pass |
| `/cost` | Session token usage & USD spend |
| `/swarm <task>` | Run the 4-agent swarm pipeline |
| `/memory ...` | RAG memory: `stats / search / add / forget` |
| `/telemetry` | Live CPU / RAM / GPU-VRAM stats |
| `/budget [limit\|resume]` | Financial velocity guard ($/min) |
| `/tools` | LLM-accessible bridge tools |
| `/blender <bpy>` | Run a bpy snippet directly |
| `/unity logs [sev]` | Fetch Unity editor logs |
| `/cursor open <path>` | Set workspace path (reported, never opened) |
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
orchestrator.py      The agent loop + self-healing + memory/cost-guard wiring
intent.py            Module 1: zero-token chat/agentic intent classifier
skills.py            User-extensible JSON skill registry
swarm_bus.py         Engine 5: async event bus + 4 specialized agents
memory_rag.py        Engine 6: TF-IDF semantic long-term memory
ast_memory.py        Module 2: AST-granular incremental code memory
telemetry.py         Engines 7 & 9: host telemetry, throttling, cost guard
sandbox.py           Engine 10: pre-flight QA sandbox for bpy/C#
main.py              Rich-powered interactive terminal
bridges/
  result.py          Shared ToolResult type
  blender_bridge.py  WebSocket client + in-app addon template
  unity_bridge.py    REST client + in-app editor script template
  cursor_bridge.py   git-apply patcher (terminal-only, never launches the app)
```

---

## License

See [LICENSE](./LICENSE).
