# NOTES

---

## 1 · Agent loop design

`app/agent/runner.py`. A ReAct cycle — reason, act, observe, repeat — where the
reasoning is a field of the action rather than prose to be parsed.

### The cycle

1. **Build context.** System prompt → recalled long-term facts (only if the
   vector search returned rows) → the last 10 conversational turns, chronological
   → the new user message. Tool schemas come from `session.tools_enabled` filtered
   through `TOOL_REGISTRY`, so a tool renamed since the session was created
   degrades that session instead of crashing its runs.
2. **Iterate.** Call the LLM with `tool_choice="auto"`; record the step; dispatch
   every tool it asked for; feed each result back as a `tool` message; repeat.
3. **Finalise.** Persist the answer, flip the status, append the turn to
   short-term memory, publish `done`.

### Two exits, and the difference matters

| | Trigger | `final_answer` |
|---|---|---|
| **Natural stop** | the model replies with content and no `tool_calls` — it is signalling it has enough | the content |
| **Circuit breaker** | the iteration counter reaches `MAX_AGENT_ITERATIONS` (10) | the sentinel `"Max iterations reached"` |

The cap is not a stopping *strategy*; it is a guard against a model that loops
calling the same tool forever. The sentinel is deliberate — if it fires,
something went wrong, and that should stay visible rather than be papered over
with a fabricated answer.

A third outcome sits between them. "No tool calls" conflates *here is the answer*
with *I need something you did not give me*, and only the first should end a run.
A reply that ends in a question mark records a `needs_input` step and finishes as
`RunStatus.NEEDS_INPUT` — terminal, but neither `COMPLETED` (the question is
unanswered) nor `FAILED` (nothing broke).

### Errors are observations, not exceptions

`tools.dispatch` never raises. An unknown tool name, malformed JSON arguments, a
wrong keyword, a timeout, any other exception — all become a *string* handed back
as an ordinary tool result:

```
Error: unknown tool 'teleport'. Available tools: calculator, get_current_datetime, ...
```

The model reads the list and corrects on the next iteration. The loop never
crashes, which is the ReAct pattern's real strength: a failure is just another
observation.

### Where the "thought" went

ReAct means the reasoning/acting *cycle*, not the 2022 paper's text format. The
original had the model write `Thought: … / Action: search[…] / Observation: …` as
prose that a regex parsed, because completion APIs had no notion of a tool call.
Native tool calling replaced the parsing.

So the prompt (`app/agent/prompts.py`) contains no `Thought:` scaffolding —
asking a tool-calling model for it makes it *describe* an action instead of
taking one, and the loop would see a message with no `tool_calls`, treat that as
the natural stop, and terminate on iteration 1 with a plan instead of an answer.

Instead, the `@tool` decorator injects two required parameters into every schema:

| Parameter | On | Purpose |
|---|---|---|
| `thought` | every tool, first | the model's reasoning for *this* call |
| `grounding` | tools that take values | `user` / `tool_result` / `assumed` — where those values came from |

Both are stripped before the function runs, so tools never see them, and adding a
tool is still writing one decorated function. `thought` makes every `tool_call`
step in the durable trace explain itself. `grounding` exists because a prompt rule
demonstrably does not hold: asked to "search for python then add 3 numbers",
`gpt-4o-mini` invented `5 + 10 + 15` and reported the sum as if given. Requiring
the model to *classify* the provenance of its own arguments, and refusing any call
it labels `assumed`, turns an instruction into a claim the runtime checks. A
verified run shows it working: `0.15 * 3000000` marked `assumed` → refused →
retried as `0.15 * 3787000` marked `tool_result`.

It is not proof against a model that mislabels an invention as `user` — but a
false claim then sits in the trace instead of passing silently, which is the most
a self-report can buy.

### Every step is persisted *and* published, in one call

```python
async def _record(run_id, step_type, payload):
    db.add(RunStep(...)); await db.commit()
    await events.publish_event(redis, run_id, {...})
```

Coupling them is a correctness decision, not tidiness: it makes it structurally
impossible for the Postgres trace and the live SSE stream to diverge. A test
asserts the published sequence equals the persisted one.

Token-level deltas (`content_delta`, `tool_call_delta`) are the exception —
transport-only, like `done`, because the assembled message they compose is what
becomes the step. The trace stays a trace.

### Four invariants, each a real failure mode

| Invariant | Symptom if broken |
|---|---|
| Assistant message appended before its tool results | OpenAI 400 on iteration 2 |
| One `tool` message per `tool_call`, ids matching | OpenAI 400 on iteration 2 |
| Iterate **all** `tool_calls`, not `[0]` | parallel calls silently dropped |
| `model_dump(exclude_none=True)` | `"function_call": null` rejected by some SDK versions |

---

## 2 · Memory strategy

`app/agent/memory_manager.py`. Two tiers with different lifetimes, different
scopes and different retrieval models.

| | Short-term | Long-term |
|---|---|---|
| Store | Redis list | Postgres + pgvector |
| Scope key | `session_id` | `user_id` |
| Content | conversational turns | discrete facts |
| Retrieval | recency, last 10 | cosine similarity (`<=>`), top 3 |
| Written | end of every completed run | only when `remember_fact` fires |
| Lifetime | 7-day TTL, 20-turn cap | permanent until deleted |

**The scoping difference is the design point.** Long-term memory is user-scoped so
facts follow the user *across* sessions; conversational context stays
session-isolated. Two sessions of the same user share remembered facts but not
chat history — which is what you want, because a fact is about the person while a
transcript is about a conversation.

**`reversed()` is load-bearing.** `LPUSH` stores newest at index 0, so `LRANGE`
returns newest-first. Feeding reverse-chronological history to an LLM raises
nothing and degrades every answer, so it has its own test.

**No ANN index, deliberately.** At assessment scale the `WHERE user_id = ?` filter
plus a sequential scan is sub-millisecond and *exact*; an approximate index would
trade recall for nothing. The trigger for
`USING hnsw (embedding vector_cosine_ops)` is ~100k rows.

The non-obvious consequence to plan for: **ANN indexes search globally and filter
afterwards**, so `WHERE user_id = ?` can return fewer than `LIMIT k` rows, or fall
back to a scan. Mitigations are pgvector 0.8 iterative index scans, a partial
index per high-volume tenant, or partitioning `long_term_memory` by `user_id`.
Getting to the filter interaction is the part most answers miss.

**Isolation is in the query, not after it.** The `user_id` filter is applied
before the vector ordering, never as a post-fetch ownership check.

---

## 3 · User isolation

Ownership is a `WHERE` clause everywhere:

```python
select(AgentSession).where(
    AgentSession.id == session_id,
    AgentSession.user_id == user.id,
)
```

Runs and steps carry no `user_id`; they reach it by joining `agent_sessions`. A
miss returns **404, not 403** — a 403 confirms the id exists and turns the API
into an enumeration oracle. Verified end to end: user B gets 404 on user A's
session, run status, steps, stream and cancel.

---

## 4 · Streaming

Events go over a **Redis Stream** (`XADD` / `XREAD`), and `app/agent/events.py`
owns the whole protocol — key schema, wire format, retention, cursor semantics.
The worker is its only writer, the API its only reader, and neither imports the
other. Inlining `XADD` in the runner and `XREAD` in the router would put one
protocol in two layers of two processes, where the key format and the cursor
convention drift independently and the resulting bug is invisible: the client
just silently receives nothing.

**Why Streams over pub/sub.** Pub/sub is fire-and-forget. `POST /run` and
`GET /stream` are two separate round trips, so the worker starts publishing
before the client subscribes — the late-connect gap is the *normal* path, not an
edge case. With `XREAD` from cursor `"0"`, backlog and live tail are one call:
there is no replay mode, no live mode, and therefore no window in which an event
falls between them. The cursor default is the named constant `BEGINNING`; `"$"`
means new-entries-only and would silently reinstate the exact bug Streams fix.

**Reconnect resume is one line.** The SSE `id:` field carries the Redis entry id,
so the `Last-Event-ID` header a browser `EventSource` replays on reconnect drops
straight in as the `XREAD` cursor. The transport's cursor and the protocol's
cursor are the same value.

**Plain `XREAD`, never `XREADGROUP`.** Groups divide entries between competing
consumers; viewers of a trace broadcast, they do not compete. Two clients on one
run each receive the complete sequence.

Retention is short on purpose (1h TTL, `MAXLEN ~ 10000`, TTL refreshed on every
write so a long run cannot expire mid-flight). The stream is a delivery buffer;
`run_steps` is the record. A completed run whose stream has expired is replayed
from Postgres rather than hanging on a key that will never receive another entry.

---

## 5 · Background execution

`POST /run` inserts a `QUEUED` row, calls `.delay()`, returns **202** — measured
at 37–65 ms for runs that take seconds.

**Idempotency is an atomic claim, not read-then-write.** A plain `SELECT` then
`UPDATE` races: two workers both read `QUEUED` and both execute.

```python
select(AgentRun).where(AgentRun.id == run_id).with_for_update()
```

**Celery's delivery defaults are wrong for this workload**, and correcting either
of the two important ones needs code, not just a flag:

| Setting | Default | Here | Why |
|---|---|---|---|
| `worker_prefetch_multiplier` | 4 | **1** | a `--concurrency=2` worker reserves 8 messages; these tasks run for minutes, so the extras queue behind the one in progress while another worker idles |
| `task_acks_late` | False | **True** | acking on receipt means a worker killed mid-run takes the message with it, and the run sits at RUNNING forever holding a concurrency slot |
| `task_reject_on_worker_lost` | False | **True** | requeue a child killed outright rather than silently failing the task |
| `task_soft_time_limit` / `task_time_limit` | none | **900 / 960** | a wedged run must not hold a slot indefinitely; soft first so the run finishes honestly |
| `visibility_timeout` | 3600 | **1260** | *derived* — Redis redelivers anything unacked after this, so below the hard limit it hands a live task to a second worker |

`acks_late` is actively harmful without a takeover path: the claim treated any
`RUNNING` row as a duplicate, so the redelivery it buys would arrive and be
discarded, leaving the run stuck. The fix uses a property of Celery rather than
new state — **the task id survives a redelivery**, so `RUNNING` under *our own*
`celery_task_id` means the previous attempt died, while a different id means
another worker holds it right now.

`reject_on_worker_lost` then introduces a poison-message loop, capped by
`INCR run:{id}:attempts` at `MAX_RUN_ATTEMPTS`.

**Cancellation is real revocation.** `DELETE /runs/{id}` calls
`revoke(task_id, terminate=True, signal="SIGUSR1")`, which raises
`SoftTimeLimitExceeded` *inside* the worker rather than killing the process — the
row is finished honestly and the SSE client still gets `done`. Verified: a run
stopped at 6 steps and was still at 7 twenty-two seconds later.

Where that handler goes was the whole problem. SIGUSR1 is raised at an arbitrary
bytecode boundary, so catching it around the agent loop misses every other
instant; and if it lands in the event loop's own machinery it propagates straight
out of `asyncio.run()`. It is handled in both places, with cleanup safe to run
twice.

The soft time limit collides with that: both surface as `SoftTimeLimitExceeded`,
so an overrun would have been recorded as CANCELLED — reading as though the user
asked to stop. The `run:{id}:cancel` flag distinguishes them.

**Rate limiting** claims one of ten per-user slots with an atomic Redis `INCR`,
429 with `Retry-After` when they are gone. Redis is the fast path; Postgres stays
the source of truth and is consulted when the counter says the user is full,
because a counter held outside the data it describes drifts, and a worker killed
mid-run would otherwise lock the user out permanently. Slot release is idempotent
via a per-run `SET NX` marker — the reconciliation only corrects an over-count, so
a double release would silently grant a permanent extra slot.

**Event loops.** uvicorn picks uvloop on its own; the worker called `asyncio.run`
directly and was still on the stdlib selector loop — the process that matters
most, since the agent loop is almost entirely awaited I/O. Both are on uvloop now,
and the worker states which loop it booted on.

---

## 6 · Deliberate deviations

**`calculator` does not use `ast.literal_eval`.** The brief specifies it, but
`literal_eval` evaluates *literals*, not expressions — `literal_eval("2+2")`
raises `ValueError`; the `+`/`-` allowance exists only to reconstruct complex
literals like `1+2j`. A literal reading produces a tool that fails on every
input, including the reference scenario's `0.15 * 3787000`. Implemented instead:
`ast.parse(expr, mode="eval")` plus a recursive walk permitting only `Constant`,
`BinOp` and `UnaryOp` with an operator whitelist. Same safety property — no
`eval`, no name resolution, no attribute access, no calls — with a tool that
works.

**Sync tools are moved to a thread before being timed out.** The plan suggests
detecting async with `inspect.isawaitable` on the *return value*, but you cannot
time-box a call you have already made: an 8-second `time.sleep` ran to completion
and blocked the event loop. Sync tools now go through `asyncio.to_thread` first.
The honest limitation is that the thread cannot be cancelled and runs to
completion in the background — the timeout guarantees the *agent* stops waiting,
not that the work stops.

**Tool-allowlist validation is a `Literal` type, not a `field_validator`.** The
allowlist is built from the registry at import, so the same 422 comes with the
error located at the offending *index*, the rejected value echoed, and a real
`enum` in the OpenAPI schema. Trade-off: the literal is frozen at import, so a
tool registered after `app.schemas.agent` loads would not be accepted — fine
while every tool is defined at module level.

**Alembic runs in async mode** rather than swapping in a sync driver, keeping
`DATABASE_URL` as the single source of truth and dropping psycopg2 entirely.

**Dependencies are managed with uv**, so `requirements.txt` is replaced by
`pyproject.toml` + `uv.lock`. A pinned requirements file fixes only direct
dependencies and lets the transitive tree drift; the lockfile pins the full
resolution, and the image builds from it with `uv sync --locked`, which fails
loudly rather than quietly resolving something other than what was tested.

**Two modules added to the prescribed structure**, each with a reason:
`agent/events.py`, because the transport spans a process boundary and the layout
has no home for it; and `agent/prompts.py`, because the operating instructions are
a property of the runtime rather than of the runner.

**`examples/`** holds two standalone ReAct implementations that import nothing
from `app/`: `react_scratchpad.py` is the 2022 text format with a regex-parsed
scratchpad, and `react_tool_calling.py` is the same cycle over native tool calls.
They exist to make the prompt decision above demonstrable rather than asserted.

---

## 7 · What I would improve with more time

Ordered by what I think would matter most.

1. **Token-budgeted context window.** The 20-turn cap is the weakest thing in the
   memory design: turn count is a poor proxy for tokens, and one long tool result
   can blow the window while twenty short turns barely fill it. I would count with
   tiktoken, evict oldest until under budget, and summarise evicted turns into a
   rolling session summary rather than dropping them. This also makes the ~1,200
   token system prompt visible as a cost competing with conversation history.

2. **A clarification round trip.** `needs_input` currently records that the agent
   needs something and stops. The useful version resumes: a
   `POST /runs/{id}/answer` that appends the reply and re-queues the run, so the
   agent can ask mid-task the way the interactive examples do. That is the single
   biggest functional gap.

3. **Self-hosted Scalar bundle.** Swagger and ReDoc are disabled, and Scalar loads
   from a CDN — so there is currently no offline documentation UI at all.
   `/openapi.json` is still served, but that is a workaround, not a fix.

4. **Loop detection on top of the iteration counter.** If the same tool is called
   with identical arguments twice in a row, inject a system message saying that
   call already returned that result. The counter catches runaway loops only after
   ten wasted LLM calls.

5. **HNSW once row counts justify it**, with the ANN/filter interaction in §2
   handled rather than discovered.

6. **Per-session `reasoning_effort` and model.** A cheap lookup session and a hard
   analysis session should not share one global setting.

7. **Grounding beyond self-report.** `grounding` relies on the model classifying
   honestly. Checking numeric literals in tool arguments against the conversation
   and earlier tool results would catch a mislabelled invention rather than trust
   the label.

8. **Backpressure on the SSE stream.** Nothing bounds a client that reads slower
   than the agent emits; `MAXLEN ~ 10000` is the only limit.

9. **Real load testing.** Every latency number here is single-user on a laptop
   against a scripted LLM. The concurrency limit, the prefetch setting and the
   pool sizes are all reasoned rather than measured under contention.

---

## 8 · Known limitations

- **`asyncio.run` per Celery task** creates and disposes an event loop per run.
  Fine at second-scale; at throughput this wants a persistent loop or an
  async-native queue (`arq`, `taskiq`).
- **`lazy='selectin'`** on the run relationships eager-loads the full trace, which
  is why `/runs/{id}/status` uses a `func.count()` projection rather than
  `len(run.steps)` — clients poll it every second.
- **1-hour stream TTL** means very old runs fall back to the Postgres replay,
  which is correct but flushes the whole trace at once rather than incrementally.
- **`summarise_text` shares the 5-second tool timeout**, which is tight for an
  LLM-backed tool. Per-tool timeouts would be better than one global setting.
- **pgvector behaviour is exercised against Postgres, not in the offline suite.**
  `Vector(1536)` has no SQLite equivalent, so the unit tests assert the isolation
  guarantee on the compiled SQL and cover short-term memory against fakeredis.
- **Async logging costs ~5.3× per record** (9.4µs → 50.2µs measured). Against
  stdout that is pure overhead; it earns its cost when the sink is a socket or a
  shipper applying backpressure.

---

## 9 · Reference trace

Session `Research Assistant`, tools `[web_search, calculator, remember_fact]`.
Message: *"What is 15% of the current population of Dubai, and remember that fact
for me?"* — captured from a live `docker compose` stack via `GET /runs/{id}/steps`.

| Iter | `step_type` | Content |
|---|---|---|
| 0 | `llm_call` | `tool_calls: ["web_search"]` |
| 0 | `tool_call` | `web_search({"query": "current population of Dubai"})` · thought: *"I do not know Dubai's population, so I must look it up first."* · grounding: `user` |
| 0 | `tool_result` | `…estimated at 3,787,000.` |
| 1 | `llm_call` | `tool_calls: ["calculator"]` |
| 1 | `tool_call` | `calculator({"expression": "0.15 * 3787000"})` · thought: *"Taking 15% of the 3,787,000 the search returned."* · grounding: `tool_result` |
| 1 | `tool_result` | `568050.0` |
| 2 | `llm_call` | `tool_calls: ["remember_fact"]` |
| 2 | `tool_call` | `remember_fact({"fact": "15% of Dubai's population is about 568,050"})` · grounding: `tool_result` |
| 2 | `tool_result` | `Stored: …` |
| 3 | `llm_call` | plain text, no tool calls |
| 3 | `final_answer` | `"15% of Dubai's population (~3,787,000) is about 568,050. I've remembered that."` |
| — | `done` | `status: completed` — transport only, not persisted as a step |

`GET /runs/{id}/status` → `{"status": "completed", "tokens_used": 480,
"reasoning_tokens": 256, "step_count": 11}`. The stored fact comes back from
`GET /memory/search?q=how many people live in Dubai`, at
`vector_dims(embedding) = 1536`.

SSE frames arrived spread across the run rather than in one burst at close, and a
client connecting *after* the run finished still received the complete trace from
the stream backlog — which under pub/sub would have been empty.

> The trace was produced against a scripted stand-in for the OpenAI API so the
> whole path — API → Celery → runner → tools → pgvector → Redis Streams → SSE —
> could be verified without shipping a key. With a real `OPENAI_API_KEY` the same
> request runs unchanged; nothing in the application is aware of the difference.

---

## 10 · Full request trace

1. `POST /sessions/{id}/run` → **PG** scoped session lookup → **Redis** claim a
   concurrency slot → **PG** `INSERT` run (`queued`) → **Redis** Celery enqueue →
   `202`.
2. `GET /runs/{id}/stream` → **PG** ownership check via join → **Redis** `XREAD`
   from cursor `0` (or `Last-Event-ID`), returning any backlog immediately.
3. Worker consumes → **PG** `SELECT … FOR UPDATE`, claim, flip to `running`,
   record the task id.
4. Context build → **Redis** `LRANGE` short-term → **OpenAI** embed the user
   message → **PG** pgvector top-3.
5. Per iteration → **OpenAI** chat → **PG** `INSERT run_step` → **Redis** `XADD` +
   `EXPIRE` (pipelined) → the blocked `XREAD` returns → SSE frame.
6. `remember_fact` → **OpenAI** embed → **PG** `INSERT long_term_memory`.
7. Finalise → **PG** `UPDATE` final_answer / `completed` → **Redis** `LPUSH` +
   `LTRIM` the turn → **Redis** `XADD` done → **Redis** release the slot.
8. Consumer sees `done` → generator breaks → connection closes. The stream key
   survives for an hour, so a reconnect still replays.

## 11 · Shared resources under concurrency

The Postgres pool, the Redis instance, Celery worker slots and the OpenAI rate
limit are all shared. None are correctness problems — they are capacity problems.
Correctness comes from three places: ownership in the `WHERE` clause (§3), Redis
keys namespaced by UUID so no cross-tenant collision is possible, and
`with_for_update()` for the one genuine race, duplicate run execution (§5).
