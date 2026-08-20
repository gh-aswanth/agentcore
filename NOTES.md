# Notes

## The agent loop

`app/agent/runner.py`. Standard ReAct: build the context, call the model, run
whatever tools it asks for, feed the results back, repeat.

Context goes in a fixed order — system prompt, any long-term facts the vector
search turned up, the last 10 turns from Redis in chronological order, then the
new message. Tool schemas come from `session.tools_enabled` filtered through the
registry, so an old session naming a tool that no longer exists still runs
instead of blowing up.

The loop ends one of two ways. Either the model replies with text and no tool
calls, which is it telling you it's done, or it hits 10 iterations. In the second
case `final_answer` is set to the literal string "Max iterations reached" rather
than whatever the model last said. The cap isn't a stopping strategy, it's a
circuit breaker, and if it fires I want that obvious.

I added a third ending. "No tool calls" was conflating "here's your answer" with
"I need something you didn't give me", and only the first should finish a run. If
the final reply is a question, the run records a `needs_input` step and ends as
`NEEDS_INPUT` — not `COMPLETED`, because nothing was answered, and not `FAILED`,
because nothing broke.

`dispatch` never raises. Unknown tool, malformed JSON, wrong keyword, timeout —
they all come back as a string:

```
Error: unknown tool 'teleport'. Available tools: calculator, get_current_datetime, ...
```

which goes to the model as an ordinary tool result. It reads the list and picks a
real tool next iteration. Errors are just more observations, which is the part of
ReAct that actually earns its keep.

There's no `Thought:` scaffolding in the prompt. The 2022 paper had the model
write its reasoning as prose you'd parse with a regex, because completion APIs
had no concept of a tool call. With native tool calling, asking for that prose is
counterproductive: the model describes an action instead of taking one, the loop
sees a message with no `tool_calls`, treats it as the natural stop, and you get a
plan instead of an answer on iteration 1.

Instead the `@tool` decorator injects two arguments into every schema. `thought`
is required on all of them and holds the model's reason for that specific call, so
every `tool_call` row in the trace explains itself. `grounding` goes on tools that
take values and is one of `user` / `tool_result` / `assumed` — where the values
came from. A call marked `assumed` is refused before the tool runs.

That second one exists because the prompt rule wasn't enough. Asked to "search for
python then add 3 numbers", gpt-4o-mini made up `5 + 10 + 15` and reported the sum
as though I'd given it those numbers. Making the model classify its own arguments
turns "don't invent values" into something the runtime can check. It's not
airtight — it could label an invention `user` — but then the lie is in the trace
instead of invisible.

Both arguments are stripped before the function is called, so tools never see
them and adding a tool is still just writing the function.

Steps are persisted and published in the same method:

```python
async def _record(run_id, step_type, payload):
    db.add(RunStep(...)); await db.commit()
    await events.publish_event(redis, run_id, {...})
```

That's deliberate rather than tidy. Coupling them means the Postgres trace and the
SSE stream can't drift apart. Token deltas are the exception — they're
transport-only, since the message they add up to is what becomes the step.

Four things that break this loop if you get them wrong, all of which I hit at some
point: the assistant message has to be appended before its tool results; every
`tool_call` needs exactly one `tool` message with a matching id; you have to
iterate all the tool calls rather than `[0]`, because gpt-4o-mini emits parallel
ones routinely; and `model_dump` needs `exclude_none=True` or a null
`function_call` key sneaks into the next request.

## Memory

Two tiers, and the split matters more than either half.

| | Short-term | Long-term |
|---|---|---|
| Where | Redis list | Postgres + pgvector |
| Keyed by | `session_id` | `user_id` |
| Holds | conversation turns | discrete facts |
| Retrieval | last 10, by recency | top 3, cosine distance |
| Written | end of every run | only by `remember_fact` |
| Lifetime | 7 days, 20 turns | until deleted |

Long-term memory is scoped to the user, short-term to the session. So facts follow
someone between sessions while chat history doesn't, which is what you want: a
fact is about the person, a transcript is about one conversation.

One real trap: `LPUSH` puts the newest turn at index 0, so `LRANGE` hands it back
newest-first. Feed that to a model in that order and every answer quietly gets
worse — no exception, no error. `read_short_term` reverses it, and there's a test
pinning that specifically because nothing else would catch it.

No vector index, on purpose. At a few hundred rows the `user_id` filter plus a
sequential scan is sub-millisecond and exact; an approximate index would cost
recall and buy nothing. I'd add HNSW somewhere around 100k rows.

The thing to know before adding one: ANN indexes search globally and apply the
filter afterwards, so `WHERE user_id = ?` can come back with fewer than the three
rows you asked for, or quietly fall back to a scan. pgvector 0.8's iterative scans,
a partial index per heavy tenant, or partitioning by `user_id` are the ways out.

The `user_id` filter is applied before the ordering, not as an ownership check
after the fetch. Same rule everywhere else in the API: ownership is in the `WHERE`
clause, and someone else's resource is a 404, not a 403, because 403 tells you the
id exists.

## What I'd do next

**Token-budgeted context window.** The 20-turn cap is the weakest part of the
memory design. Turn count is a bad proxy for tokens — one long tool result can
blow the window while twenty short turns barely touch it. I'd count with tiktoken,
evict oldest until under budget, and roll evicted turns into a running summary
instead of dropping them.

**Finish the clarification round trip.** `needs_input` currently records that the
agent needs something and stops there. A `POST /runs/{id}/answer` that appends the
reply and requeues the run would make it actually useful. Biggest functional gap.

**Loop detection.** If the same tool gets called with identical arguments twice
running, tell the model it already has that result. The iteration cap only catches
this after ten wasted calls.

**Per-session model and reasoning effort.** A quick lookup and a hard analysis
shouldn't share one global setting.

**Load testing.** Every timing in here is one user on a laptop against a scripted
model. The concurrency limit, the Celery prefetch setting and the pool sizes are
reasoned, not measured under contention.

## Decisions worth flagging

**Redis Streams rather than pub/sub for the event transport.** `POST /run` and
`GET /stream` are separate round trips, so the worker starts publishing before the
client subscribes. With pub/sub those events are gone. Reading a stream from
cursor `0` gives you the backlog and the live tail in one call, and since the SSE
`id:` is the Redis entry id, `Last-Event-ID` reconnect works with no extra state.

**`lazy='raise'` on the run relationships rather than an eager loader.**
`selectin` is not an N+1 — it batches — but it fires on *every* query that loads
the parent, whatever the SELECT list says. With it, `/runs/{id}/status` emitted a
second statement pulling every step and its payload on each 1 Hz poll, and
`GET /sessions` pulled every run of every session plus all their steps. Reading
a trace is what one endpoint does, not a property of loading a run, so the
relationships raise and the two places that want children query for them: a
`func.count()` projection for `step_count`, a `LIMIT 20` for session detail.
`passive_deletes=True` hands the delete cascade to the FK's `ON DELETE CASCADE`,
which is what lets the collections stay unloaded on the delete path too.

## Known limitations

- `asyncio.run` per Celery task builds and tears down an event loop each time.
  Fine at this scale; at throughput I'd want a persistent loop or `arq`/`taskiq`.


## Reference run

Session `Research Assistant` with `[web_search, calculator, remember_fact]`,
message *"What is 15% of the current population of Dubai, and remember that fact
for me?"*, taken from `GET /runs/{id}/steps` on a running stack:

```
llm_call      tool_calls: [web_search]
tool_call     web_search({"query": "current population of Dubai"})
              thought: "I do not know Dubai's population, so I must look it up first."
              grounding: user
tool_result   ...estimated at 3,787,000.
llm_call      tool_calls: [calculator]
tool_call     calculator({"expression": "0.15 * 3787000"})
              thought: "Taking 15% of the 3,787,000 the search returned."
              grounding: tool_result
tool_result   568050.0
llm_call      tool_calls: [remember_fact]
tool_call     remember_fact({"fact": "15% of Dubai's population is about 568,050"})
tool_result   Stored: ...
llm_call      (plain text, no tool calls)
final_answer  "15% of Dubai's population (~3,787,000) is about 568,050. I've
               remembered that."
done          status: completed
```fix it
