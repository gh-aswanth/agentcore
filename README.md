# AgentCore

A ReAct agent backend: JWT-authenticated sessions with a per-session tool
allowlist, a background LLM↔tool loop that persists every step, two-tier memory
(Redis + pgvector), and a live SSE trace with replay.

FastAPI · SQLAlchemy 2 (async) · PostgreSQL 16 + pgvector · Redis 7 · Celery 5 · OpenAI SDK v1

## Quickstart

```bash
cp .env.example .env          # then put a real OPENAI_API_KEY in it
docker compose up             # db, redis, api, worker — migrations run automatically
curl localhost:8000/health    # {"status":"ok"}
open http://localhost:8000/scalar
```

## API reference

| URL | |
|---|---|
| `/` | Scalar reference — grouped by tag, with a built-in request client |
| `/openapi.json` | the schema all three render |

`POST /auth/register` returns a bearer token; paste it into *Authorize* and every
request is scoped to that user. Scalar persists it across page reloads.

Scalar's bundle loads from a CDN in the browser, so `/scalar` needs client-side
internet; `/docs` works without it.

`API_PORT=8001 docker compose up` if port 8000 is already taken on your machine.

### Data and hot reload

`docker compose up` loads `docker-compose.override.yml` automatically, which
mounts `./app` into the api and worker read-only and runs both under a file
watcher. Edit a file on the host and the change is live in about two seconds — no
rebuild.

```bash
docker compose up                        # dev: live reload
docker compose -f docker-compose.yml up  # exactly what ships in the image
docker compose build                     # still needed after a dependency change
```

Postgres and Redis write to `./.data/` on the host (gitignored), so state
survives a restart **and** `docker compose down -v`:

| | |
|---|---|
| `docker compose down` | keeps data |
| `docker compose down -v` | keeps data — it lives in `./.data`, not a volume |
| `rm -rf .data` | the only way to start clean |

Redis runs with `--appendonly yes`, so short-term memory windows and run event
streams survive a restart too.

## Local development

Dependencies are managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml`
declares them and `uv.lock` pins the exact resolution used by both your machine
and the image.

```bash
uv sync                       # creates .venv from the lockfile (Python 3.12, fetched if absent)
uv run uvicorn app.main:app --reload
uv add <package>              # updates pyproject.toml and uv.lock together
uv export --format requirements-txt > requirements.txt   # if you need a pip-style file
```

`uv run <cmd>` syncs before it runs, so the environment can never drift from the
lockfile.

## Tests

Run entirely offline — the single LLM seam (`app.agent.planner.chat`) is mocked,
so no network and no API key are needed.

```bash
OPENAI_API_KEY=test-key uv run pytest
```

The runtime image ships without test dependencies, so the suite runs from the
Dockerfile's `dev` stage instead:

```bash
docker compose --profile test build tests    # the profile is skipped by a plain build
docker compose --profile test run --rm tests
```

## Try it

```bash
API=http://localhost:8000

TOKEN=$(curl -s -X POST $API/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"hunter2hunter2"}' | jq -r .access_token)

SESSION=$(curl -s -X POST $API/sessions -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Research Assistant","tools_enabled":["web_search","calculator","remember_fact"]}' | jq -r .id)

RUN=$(curl -s -X POST $API/sessions/$SESSION/run -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is 15% of the current population of Dubai, and remember that fact for me?"}' | jq -r .run_id)

curl -N $API/runs/$RUN/stream -H "Authorization: Bearer $TOKEN"   # live trace, replays from the start
curl -s $API/runs/$RUN/steps  -H "Authorization: Bearer $TOKEN" | jq
curl -s "$API/memory/search?q=Dubai" -H "Authorization: Bearer $TOKEN" | jq
```

Connecting to the stream *after* submitting is fine — see NOTES.md §4.

The stream carries the whole run as it happens:

```
event: tool_call_delta   [web_search#0] '{"query": "current populati'
event: tool_call_delta   [web_search#0] 'on of Dubai"}'
event: llm_call          iter=0 tools=['web_search'] reasoning_tokens=64
event: tool_call         web_search({"query": "current population of Dubai"})
event: tool_result       ... estimated at 3,787,000.
...
event: content_delta     "15% of Dubai's population (~"
event: content_delta     '3,787,000) is about 568,050.'
event: final_answer      ...
event: done              completed
```

`*_delta` frames are the model's output as it is generated; the un-suffixed
frames are the persisted trace. Clients can bind per-type handlers
(`es.addEventListener('content_delta', …)`) rather than parsing every payload.

## How the agent reasons

Every tool's schema carries a required `thought`, and every value-carrying tool a
required `grounding` — both injected by the `@tool` decorator and stripped before
the function runs. The ReAct "thought" is therefore a field of the action rather
than prose to be parsed, and each step of the trace explains itself:

```
  tool_call    calculator
               thought   : Taking 15% of a number I have not actually read yet.
               grounding : assumed
  tool_result  Error: you marked these arguments as assumed, which means you invented values...

  tool_call    calculator
               thought   : The search returned 3,787,000; using the real figure.
               grounding : tool_result
  tool_result  568050.0
```

A call the model labels `assumed` is refused before the tool runs, so an invented
value cannot silently propagate. When the agent finishes with a question rather
than an answer, the run ends as `needs_input` with a `needs_input` step carrying
it — not `completed`, because the question is unanswered, and not `failed`,
because nothing broke.

## Thinking mode

`CHAT_MODEL` defaults to a reasoning model with `REASONING_EFFORT=medium`. Effort
is sent only when the model supports it, so `CHAT_MODEL=gpt-4o-mini` silently
drops the parameter rather than failing. Reasoning tokens are billed separately
from visible output, so they are tracked per run and per call:

```bash
curl -s $API/runs/$RUN/status -H "Authorization: Bearer $TOKEN" | jq
# { "tokens_used": 480, "reasoning_tokens": 256, "step_count": 11, ... }
```

Set `STREAM_TOKENS=false` to fall back to non-streaming calls; the SSE step
frames are unaffected.

## Endpoints

| Method | Path | |
|---|---|---|
| POST | `/auth/register` | 201 + token |
| POST | `/auth/token` | OAuth2 password form |
| GET/POST | `/sessions` | list / create (422 on an unknown tool) |
| GET/DELETE | `/sessions/{id}` | detail with run summaries / delete |
| POST | `/sessions/{id}/run` | **202**, returns `run_id` immediately |
| GET | `/runs/{id}/status` | status, tokens, step count |
| GET | `/runs/{id}/steps` | the full ordered trace |
| GET | `/runs/{id}/stream` | SSE, replay + live tail |
| DELETE | `/runs/{id}` | cancel (409 if already terminal) |
| GET | `/memory`, `/memory/search?q=` | list / semantic search |
| DELETE | `/memory/{id}` | |

## Two ReAct examples, for comparison

Both are standalone scripts against `gpt-4o-mini` — they import nothing from
`app/` and need only the `openai` package. Same cycle, different transport.

| | `react_scratchpad.py` | `react_tool_calling.py` |
|---|---|---|
| Action | `Action: search` in prose | a native `tool_call` |
| Parsing | a regex over the reply | none — a typed object arrives |
| Arguments | one untyped string | a JSON object, schema-checked |
| Thought | a `Thought:` line | a required `thought` parameter on every tool |
| Stop sequence | required | not needed |
| Two tools at once | impossible | parallel tool calls |
| Malformed output | breaks the run | cannot be represented |

### `react_scratchpad.py` — the 2022 text format

The original: a `Thought / Action / Action Input / Observation` scratchpad parsed
with a regex, with four dummy tools.

```bash
uv run python examples/react_scratchpad.py
uv run python examples/react_scratchpad.py --show-prompt -q "How tall is Everest in feet?"
```

When the request is ambiguous the agent asks you, and your answer becomes the
next Observation:

```
$ uv run python examples/react_scratchpad.py -q "What is the weather there?"

      Thought: The question does not say which city. I should not guess — I will ask.
       Action: ask_user
 Action Input: Which city do you want the weather for?

               Which city do you want the weather for?
   your reply> Dubai
  Observation: Dubai

      Thought: The user said a city. I will look up its weather.
       Action: get_weather
 Action Input: Dubai
  Observation: Dubai: 34C, clear skies, humidity 55%.

 Final Answer: In Dubai it is currently 34C with clear skies and 55% humidity.
```

`ask_user` is offered only when stdin is a terminal — pass `--no-interactive`
(or pipe input) and it is withheld, so the model is never handed a tool that
cannot work. Declining (Ctrl-D) returns an observation saying so rather than
crashing the run.

```
     Question: What is 15% of the current population of Dubai, and what is the weather there?

      Thought: I need the population of Dubai before I can take 15% of it.
       Action: web_lookup
 Action Input: population of Dubai
  Observation: Error: unknown tool 'web_lookup'. Available tools: search, calculator, ...

      Thought: That tool does not exist. The list says to use `search` instead.
       Action: search
 Action Input: Dubai population
  Observation: Dubai's population is estimated at 3,787,000 as of 2025.

      Thought: The population is 3,787,000. Now I take 15% of it.
       Action: calculator
 Action Input: 0.15 * 3787000
  Observation: 568050.0

 Final Answer: 15% of Dubai's population (3,787,000) is 568,050. It is currently 34C and clear there.
```

### `react_tool_calling.py` — native tool calling

Same trace to read, but nothing is parsed. The interesting question it answers:
*if the model no longer writes "Thought:", where does the thought go?*

Relying on `message.content` alongside `tool_calls` does not work — `gpt-4o-mini`
usually returns `content=null` when it emits tool calls. So every tool's schema
gets a **required `thought` parameter**, injected by the decorator and stripped
before the tool runs. The model cannot make a well-formed call without it, so the
guarantee comes from the schema rather than from asking nicely in the prompt —
and with parallel calls, each action carries its own.

```bash
uv run python examples/react_tool_calling.py
uv run python examples/react_tool_calling.py --show-schemas --strict
uv run python examples/react_tool_calling.py --no-parallel
```

```
      Thought: I do not know Dubai's population, so I must look it up before any arithmetic.
       Action: search
 Action Input: {"query": "Dubai population"}
  Observation: Dubai's population is estimated at 3,787,000 as of 2025.

      Thought: The user asked about the current figure, so I want today's date alongside it.
       Action: get_current_time
 Action Input: {}
  Observation: 2026-08-19 18:47 UTC

    Reasoning: Let me compute the percentage now.
      Thought: I have 3,787,000 and need 15% of it.
       Action: compute
 Action Input: {"expression": "0.15 * 3787000"}
  Observation: Error: unknown tool 'compute'. Available tools: calculator, get_current_time, ...
```

Those first two actions came back in **one** message — parallel calls, each with
its own thought. `Reasoning:` is the optional `content` channel when the model
happens to narrate; `Thought:` is the guaranteed one.

`--strict` sends Structured Outputs schemas (`strict: true`,
`additionalProperties: false`, every property required, optionals as nullable
unions) so the argument shape is guaranteed rather than hoped for. It is opt-in
because it changes what the API accepts.

Both scripts are the contrast to `app/agent/runner.py`, which runs the same cycle
in production — see NOTES.md §6.

## Adding a tool

Write the function. That is the whole procedure — the decorator derives the
OpenAI schema from the signature and the description from the docstring.

```python
# app/agent/tools.py
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city}: 34°C, clear skies, humidity 55%."
```

Then add `"get_weather"` to a session's `tools_enabled`. It appears in the
`POST /sessions` enum in `/docs` automatically — the request type is derived
from the registry, so there is no schema to keep in sync.

## Layout

```
app/
├── main.py              app factory, lifespan (Redis pool), error envelope
├── core/                config · security (JWT) · dependencies (DI aliases)
├── db/                  async engine + Alembic migrations
├── models/              User · AgentSession · AgentRun · RunStep · LongTermMemory
├── schemas/             Pydantic v2 request/response models
├── routers/             auth · sessions · runs · memory · stream
├── agent/               runner (the loop) · tools · planner · prompts · memory_manager · events
├── tasks/               Celery app + execute_agent_run
└── tests/               offline suite, mocked LLM

examples/
├── react_scratchpad.py    standalone ReAct, 2022 text format (regex-parsed)
└── react_tool_calling.py  standalone ReAct, native tool calls (thought in the schema)
```

`pyproject.toml` + `uv.lock` at the root; `alembic.ini` points at
`app/db/migrations`.

## Image

One multi-stage `Dockerfile`, three targets:

| Target | Contains | Used by |
|---|---|---|
| `builder` | uv, binutils, the lockfile | discarded — nothing ships from it |
| `runtime` | base image + venv + app, non-root, healthcheck | `api`, `worker` |
| `dev` | runtime + uv + the test group | `tests` profile |

243 MB on disk / 78 MB compressed, down from 331 MB / 110 MB single-stage: the
uv binary, the compiler toolchain, the apt cache and the wheels' debug symbols
all stay in the builder.

Design decisions, deviations and known limitations are in **NOTES.md**.
